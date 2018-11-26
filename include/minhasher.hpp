#ifndef CARE_MINHASHER_HPP
#define CARE_MINHASHER_HPP

#include "options.hpp"
#include "hpc_helpers.cuh"
#include "util.hpp"

#include "ntHash/nthash.hpp"

#include <cstdint>
#include <memory>
#include <map>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <numeric>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <array>
#include <fstream>
#include <cassert>

#ifdef __NVCC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/iterator/constant_iterator.h>

#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>

#include "thrust_custom_allocators.hpp"
#endif

namespace care{

    namespace minhasherdetail{
        template<class T> struct max_k;
        template<> struct max_k<std::uint8_t>{static constexpr int value = 4;};
        template<> struct max_k<std::uint16_t>{static constexpr int value = 8;};
        template<> struct max_k<std::uint32_t>{static constexpr int value = 16;};
        template<> struct max_k<std::uint64_t>{static constexpr int value = 32;};

        class TransformException : public std::exception {
            int line;
            const char* msg;
            std::uint64_t value;
        public:
            TransformException(const char* msg, int line, std::uint64_t value = 0) : std::exception(), line(line), msg(msg), value(value)
            {
            }

            int getLine() const{
                return line;
            };

            std::uint64_t getValue() const{
                return value;
            };

            virtual const char* what() const noexcept{
                return msg;
            }
        };

        template<class Key_t, class Value_t, class Index_t>
        void cpu_transformation(std::vector<Key_t>& keys, std::vector<Value_t>& values, std::vector<Index_t>& countsPrefixSum){
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








		/*
		 * hash map to map keys to indices using linear probing
		 */
		template<class Key_t, class Index_t>
		struct KeyIndexMap{
			std::pair<Key_t, Index_t> EmptySlot{0, std::numeric_limits<Index_t>::max()};

			std::uint64_t murmur_hash_3_uint64_t(std::uint64_t x) const{
				x ^= x >> 33;
				x *= 0xff51afd7ed558ccd;
				x ^= x >> 33;
				x *= 0xc4ceb9fe1a85ec53;
				x ^= x >> 33;

				return x;
			}

			std::vector<std::pair<Key_t, Index_t>> keyToIndexMap;
			std::uint64_t size;

            std::size_t numBytes() const{
                return keyToIndexMap.size() * sizeof(std::pair<Key_t, Index_t>);
            }

			KeyIndexMap() : KeyIndexMap(0){}
			KeyIndexMap(std::uint64_t size) : size(size){
				if(size == std::numeric_limits<Index_t>::max())
					throw std::runtime_error("KeyIndexMap: too many keys!");

				keyToIndexMap.resize(size, KeyIndexMap::EmptySlot);
			}

            bool operator==(const KeyIndexMap& rhs) const{
                if(size != rhs.size)
                    return false;
                if(keyToIndexMap != rhs.keyToIndexMap)
                    return false;
                return true;
            }

            bool operator!=(const KeyIndexMap& rhs) const{
                return !(*this == rhs);
            }

			void insert(Key_t key, Index_t value) noexcept{
				std::uint64_t probes = 1;
				std::uint64_t pos = murmur_hash_3_uint64_t(key) % size;
				while(keyToIndexMap[pos] != KeyIndexMap::EmptySlot){
					pos = (pos + 1) % size;
					probes++;
				}
				keyToIndexMap[pos].first = key;
				keyToIndexMap[pos].second = value;
			}

			Index_t get(Key_t key) const noexcept{
				std::uint64_t probes = 1;
				std::uint64_t pos = murmur_hash_3_uint64_t(key) % size;
				while(keyToIndexMap[pos].first != key){
					pos = (pos + 1) % size;
					probes++;
				}
				return keyToIndexMap[pos].second;
			}

			void clear() noexcept{
				keyToIndexMap.clear();
			}

			void destroy() noexcept{
				clear();
				keyToIndexMap.shrink_to_fit();
			}
		};

		template<class key_t, class value_t, class index_t>
		struct KeyValueMapFixedSize{
			using Key_t = key_t;
			using Value_t = value_t;
			using Index_t = index_t;

			static constexpr bool resultsAreSorted = true;

			Index_t size;
			Index_t nKeys;
			Index_t nValues;
			bool noMoreWrites;
            bool canUseGpu = false;
			std::vector<Key_t> keys;
			std::vector<Value_t> values;
			std::vector<Index_t> countsPrefixSum;
            std::vector<int> deviceIds;

			double load = 0.5;
			KeyIndexMap<Key_t, Index_t> keyIndexMap;

            KeyValueMapFixedSize() : KeyValueMapFixedSize(0, {}){
			}

			KeyValueMapFixedSize(Index_t size_, const std::vector<int>& deviceIds_)
                : size(size_), nKeys(size_), nValues(size_), noMoreWrites(false), deviceIds(deviceIds_){
				keys.resize(size);
				values.resize(size);
			}

            KeyValueMapFixedSize(const KeyValueMapFixedSize& other){
                *this = other;
            }

            KeyValueMapFixedSize(KeyValueMapFixedSize&& other){
                *this = std::move(other);
            }

            KeyValueMapFixedSize& operator=(const KeyValueMapFixedSize& other) noexcept{
                size = other.size;
                nKeys = other.nKeys;
                nValues = other.nValues;
                noMoreWrites = other.noMoreWrites;
                keys = other.keys;
                values = other.values;
                countsPrefixSum = other.countsPrefixSum;
                return *this;
            }

            KeyValueMapFixedSize& operator=(KeyValueMapFixedSize&& other) noexcept{
                size = other.size;
                nKeys = other.nKeys;
                nValues = other.nValues;
                noMoreWrites = other.noMoreWrites;
                keys = std::move(other.keys);
                values = std::move(other.values);
                countsPrefixSum = std::move(other.countsPrefixSum);
                return *this;
            }

            bool operator==(const KeyValueMapFixedSize& rhs) const{
                if(size != rhs.size)
                    return false;
                if(nKeys != rhs.nKeys)
                    return false;
                if(nValues != rhs.nValues)
                    return false;
                if(noMoreWrites != rhs.noMoreWrites)
                    return false;
                if(keys != rhs.keys)
                    return false;
                if(values != rhs.values)
                    return false;
                if(countsPrefixSum != rhs.countsPrefixSum)
                    return false;
                return true;
            }

            bool operator!=(const KeyValueMapFixedSize& rhs) const{
                return !(*this == rhs);
            }


            void writeToStream(std::ofstream& outstream) const{
                bool resultsAreSorted_towrite = resultsAreSorted;
                outstream.write(reinterpret_cast<const char*>(&resultsAreSorted_towrite), sizeof(bool));
                outstream.write(reinterpret_cast<const char*>(&size), sizeof(Index_t));
                outstream.write(reinterpret_cast<const char*>(&nKeys), sizeof(Index_t));
                outstream.write(reinterpret_cast<const char*>(&nValues), sizeof(Index_t));
                outstream.write(reinterpret_cast<const char*>(&noMoreWrites), sizeof(bool));
                outstream.write(reinterpret_cast<const char*>(&canUseGpu), sizeof(bool));

                assert(nKeys == keys.size());
                assert(nValues == values.size());

                for(const auto& key : keys)
                    outstream.write(reinterpret_cast<const char*>(&key), sizeof(Key_t));
                for(const auto& val : values)
                    outstream.write(reinterpret_cast<const char*>(&val), sizeof(Value_t));

                std::size_t nCounts = countsPrefixSum.size();
                outstream.write(reinterpret_cast<const char*>(&nCounts), sizeof(std::size_t));
                for(const auto& count : countsPrefixSum)
                    outstream.write(reinterpret_cast<const char*>(&count), sizeof(Index_t));
            }

            void readFromStream(std::ifstream& instream){
                bool sorted;
                instream.read(reinterpret_cast<char*>(&sorted), sizeof(bool));
                assert(sorted == resultsAreSorted);

                bool canUseGpu_loaded;
                instream.read(reinterpret_cast<char*>(&size), sizeof(Index_t));
                instream.read(reinterpret_cast<char*>(&nKeys), sizeof(Index_t));
                instream.read(reinterpret_cast<char*>(&nValues), sizeof(Index_t));
                instream.read(reinterpret_cast<char*>(&noMoreWrites), sizeof(bool));
                instream.read(reinterpret_cast<char*>(&canUseGpu_loaded), sizeof(bool));

                keys.resize(nKeys);
                values.resize(nValues);
                countsPrefixSum.resize(nKeys+1);

                for(auto& key : keys)
                    instream.read(reinterpret_cast<char*>(&key), sizeof(Key_t));
                for(auto& val : values)
                    instream.read(reinterpret_cast<char*>(&val), sizeof(Value_t));

                std::size_t nCounts = countsPrefixSum.size();
                instream.read(reinterpret_cast<char*>(&nCounts), sizeof(std::size_t));
                countsPrefixSum.resize(nCounts);
                for(auto& count : countsPrefixSum)
                    instream.read(reinterpret_cast<char*>(&count), sizeof(Index_t));
            }

            std::size_t numBytes() const{
                return keys.size() * sizeof(Key_t)
                    + keys.size() * sizeof(Key_t)
                    + values.size() * sizeof(Value_t)
                    + countsPrefixSum.size() * sizeof(Index_t)
                    + keyIndexMap.numBytes();
            }



			void resize(Index_t size_){
				assert(!noMoreWrites);

				size = size_;
				nValues = size_;
				keys.resize(size);
				values.resize(size);
			}

			void clear() noexcept{
				size = 0;
				nKeys = 0;
				nValues = 0;
				noMoreWrites = false;
				keys.clear();
				values.clear();
				countsPrefixSum.clear();
				keyIndexMap.clear();
			}

			void destroy() noexcept{
				clear();
				keys.shrink_to_fit();
				values.shrink_to_fit();
				countsPrefixSum.shrink_to_fit();
				keyIndexMap.shrink_to_fit();
			}

			bool add(Key_t key, Value_t value, Index_t index) noexcept{
				if(index >= size){
					std::cout << "KeyValueMapFixedSize: want to set index " << index << " but size is " << size << '\n';
					return false;
				}

				if(!noMoreWrites){
					keys[index] = key;
					values[index] = value;
					return true;
				}else{
					std::cout << "KeyValueMapFixedSize: want to set " << index << " but no more writes allowed\n";
					return false;
				}
			}

			//Must call transform() beforehand !!!
			std::vector<Value_t> get(Key_t key) noexcept{
				//TIMERSTARTCPU(binarysearch);
				auto range = std::equal_range(keys.begin(), keys.end(), key);
				if(range.first == keys.end()) return {};

				Index_t index = std::distance(keys.begin(), range.first);
				//TIMERSTOPCPU(binarysearch);

				//TIMERSTARTCPU(probing);
				//Index_t index = keyIndexMap.get(key);
				//TIMERSTOPCPU(probing);
				//assert(index == index2);
				return {&values[countsPrefixSum[index]], &values[countsPrefixSum[index+1]]};
			}

            //Must call transform() beforehand !!!
			std::pair<const Value_t*, const Value_t*> get_ranged(Key_t key) noexcept{
				//TIMERSTARTCPU(binarysearch);
				auto range = std::equal_range(keys.begin(), keys.end(), key);
				if(range.first == keys.end()) return {};

				Index_t index = std::distance(keys.begin(), range.first);

				return {&values[countsPrefixSum[index]], &values[countsPrefixSum[index+1]]};
			}

			void transform(){
				if(noMoreWrites) return;

				if(size == 0) return;

#ifdef __NVCC__
    		    if(deviceIds.size() == 0){
#endif

                    cpu_transformation(keys, values, countsPrefixSum);

#ifdef __NVCC__
                }else{
                    bool success = GPUTransformation<false>::execute(keys, values, countsPrefixSum, deviceIds);

                    if(!success){
                        std::cerr << "Fallback to managed memory transformation\n";
                        success = GPUTransformation<true>::execute(keys, values, countsPrefixSum, deviceIds);
                    }

                    if(!success){
                        std::cerr << "Fallback to cpu transformation\n";
                        cpu_transformation(keys, values, countsPrefixSum);
                    }

                    nKeys = keys.size();
                }
#endif

                noMoreWrites = true;


				/*keyIndexMap = KeyIndexMap(nKeys / load);
				for(Index_t i = 0; i < nKeys; i++){
					keyIndexMap.insert(keys[i], i);
				}
				for(Index_t i = 0; i < nKeys; i++){
					assert(keyIndexMap.get(keys[i]) == i);
				}*/
			}
		};
    }

template<class Key_t, class ReadId_t>
struct Minhasher {
	static_assert(std::is_integral<Key_t>::value, "Minhasher Key_t must be integral");
	static_assert(std::is_integral<ReadId_t>::value, "Minhasher ReadId_t must be integral");

    using Index_t = ReadId_t; //read id type
    using Value_t = Index_t; //Value type for hashmap
    using Result_t = Index_t; // Return value for minhash query
    using Map_t = minhasherdetail::KeyValueMapFixedSize<Key_t, Value_t, Index_t>; //internal map type

	static constexpr int bits_key = sizeof(Key_t) * 8;
	static constexpr std::uint64_t key_mask = (std::uint64_t(1) << (bits_key - 1)) | ((std::uint64_t(1) << (bits_key - 1)) - 1);
    static constexpr std::uint64_t max_read_num = std::numeric_limits<Index_t>::max();
    static constexpr int maximum_number_of_maps = 16;
    static constexpr int maximum_kmer_length = minhasherdetail::max_k<Key_t>::value;

	// the actual maps
	std::vector<std::unique_ptr<Map_t>> minhashTables;
	MinhashOptions minparams;
	ReadId_t nReads;
    bool canUseGpu = false;
    std::vector<int> deviceIds;
    bool allowUVM = false;

    Minhasher() : Minhasher(MinhashOptions{2,16}){}

    Minhasher(const MinhashOptions& parameters)
		: Minhasher(parameters, {})
	{
	}

	Minhasher(const MinhashOptions& parameters, const std::vector<int>& deviceIds_)
		: minparams(parameters), nReads(0), deviceIds(deviceIds_)
	{
		if(maximum_number_of_maps < minparams.maps)
			throw std::runtime_error("Minhasher: Maximum number of maps is "
									+ std::to_string(maximum_number_of_maps) + "!");

		if(maximum_kmer_length < minparams.k){
			throw std::runtime_error("Minhasher is configured for maximum kmer length of "
									+ std::to_string(maximum_kmer_length) + "!");
		}
	}

    bool operator==(const Minhasher& rhs) const{
        if(minparams != rhs.minparams)
            return false;
        if(nReads != rhs.nReads)
            return false;
        if(minhashTables.size() != rhs.minhashTables.size())
            return false;
        for(std::size_t i = 0; i < minhashTables.size(); i++){
            if(*minhashTables[i] != *rhs.minhashTables[i])
                return false;
        }
        return true;
    }

    bool operator!=(const Minhasher& rhs) const{
        return !(*this == rhs);
    }

	struct Handle{
		std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;
	};

    std::size_t numBytes() const{
        //return minhashTables[0]->numBytes() * minhashTables.size();
        std::size_t result = 0;
        for(const auto& m : minhashTables)
            result += m->numBytes();
        return result;
    }



    void saveToFile(const std::string& filename) const{
        std::ofstream outstream(filename, std::ios::binary);

        int bits_key_tosave = bits_key;
        std::uint64_t key_mask_tosave = key_mask;
        std::uint64_t max_read_num_tosave = max_read_num;
        int maximum_number_of_maps_tosave = maximum_number_of_maps;
        int maximum_kmer_length_tosave = maximum_kmer_length;

        outstream.write(reinterpret_cast<const char*>(&bits_key_tosave), sizeof(int));
        outstream.write(reinterpret_cast<const char*>(&key_mask_tosave), sizeof(std::uint64_t));
        outstream.write(reinterpret_cast<const char*>(&max_read_num_tosave), sizeof(std::uint64_t));
        outstream.write(reinterpret_cast<const char*>(&maximum_number_of_maps_tosave), sizeof(int));
        outstream.write(reinterpret_cast<const char*>(&maximum_kmer_length_tosave), sizeof(int));

        outstream.write(reinterpret_cast<const char*>(&minparams), sizeof(MinhashOptions));
        outstream.write(reinterpret_cast<const char*>(&nReads), sizeof(ReadId_t));
        outstream.write(reinterpret_cast<const char*>(&canUseGpu), sizeof(bool));
        for(const auto& tableptr : minhashTables)
            tableptr->writeToStream(outstream);
    }

    void loadFromFile(const std::string& filename){
        std::ifstream instream(filename, std::ios::binary);
        if(!instream)
            throw std::runtime_error("Cannot load hashtable from file " + filename);

        int bits_key_loaded;
    	std::uint64_t key_mask_loaded;
        std::uint64_t max_read_num_loaded;
        int maximum_number_of_maps_loaded;
        int maximum_kmer_length_loaded;

        instream.read(reinterpret_cast<char*>(&bits_key_loaded), sizeof(int));
        instream.read(reinterpret_cast<char*>(&key_mask_loaded), sizeof(std::uint64_t));
        instream.read(reinterpret_cast<char*>(&max_read_num_loaded), sizeof(std::uint64_t));
        instream.read(reinterpret_cast<char*>(&maximum_number_of_maps_loaded), sizeof(int));
        instream.read(reinterpret_cast<char*>(&maximum_kmer_length_loaded), sizeof(int));

        assert(bits_key == bits_key_loaded);
        assert(key_mask == key_mask_loaded);
        assert(max_read_num == max_read_num_loaded);
        assert(maximum_number_of_maps == maximum_number_of_maps_loaded);
        assert(maximum_kmer_length == maximum_kmer_length_loaded);

        MinhashOptions minparams_loaded;
        ReadId_t nReads_loaded;
        bool canUseGpu_loaded;

        instream.read(reinterpret_cast<char*>(&minparams_loaded), sizeof(MinhashOptions));
        instream.read(reinterpret_cast<char*>(&nReads_loaded), sizeof(ReadId_t));
        instream.read(reinterpret_cast<char*>(&canUseGpu_loaded), sizeof(bool));

        assert(minparams == minparams_loaded);
        assert(nReads == nReads_loaded);

        minhashTables.resize(minparams.maps);

		for (int i = 0; i < minparams.maps; ++i) {
			minhashTables[i].reset();
			minhashTables[i].reset(new Map_t(nReads, deviceIds));
		}

        for(auto& tableptr : minhashTables)
            tableptr->readFromStream(instream);
    }

	void init(std::uint64_t nReads_){
		if(nReads_ == 0) throw std::runtime_error("Minhasher::init cannnot be called with argument 0");
		if(nReads_-1 > max_read_num)
			throw std::runtime_error("Minhasher::init: Minhasher is configured for only" + std::to_string(max_read_num) + " reads, not " + std::to_string(nReads_) + "!!!");

		nReads = nReads_;

		minhashTables.resize(minparams.maps);

		for (int i = 0; i < minparams.maps; ++i) {
			minhashTables[i].reset();
			minhashTables[i].reset(new Map_t(nReads, deviceIds));
		}
	}

	void clear(){
		minhashTables.clear();
		nReads = 0;
	}

	void destroy(){
		clear();
		minhashTables.shrink_to_fit();
	}

	void insertSequence(const std::string& sequence, ReadId_t readnum){
		if(readnum >= nReads)
			throw std::runtime_error("Minhasher::insertSequence: read number too large. " + std::to_string(readnum) + " > " + std::to_string(nReads));

		// we do not consider reads which are shorter than k
		if(sequence.size() < unsigned(minparams.k))
			return;

		std::uint64_t hashValues[maximum_number_of_maps]{0};

		bool isForwardStrand[maximum_number_of_maps]{0};

		//get hash values
		minhashfunc(sequence, hashValues, isForwardStrand);

		// insert
		for (int map = 0; map < minparams.maps; ++map) {
			Key_t key = hashValues[map] & key_mask;
			Value_t value(readnum);

			if (!minhashTables[map]->add(key, value, readnum)) {
				throw std::runtime_error(("error adding key to map. key " + std::to_string(key) + " " + std::to_string(value) + " " + std::to_string(readnum)));
			}
		}
	}

	std::int64_t getNumberOfCandidates(const std::string& sequence) const noexcept{
		Handle handle;
		return getNumberOfCandidates(sequence, handle);
	}

	std::int64_t getNumberOfCandidates(const std::string& sequence, Handle& handle) const noexcept{
		static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
		// we do not consider reads which are shorter than k
		if(sequence.size() < unsigned(minparams.k))
			return 0;

		std::uint64_t hashValues[maximum_number_of_maps]{0};

		bool isForwardStrand[maximum_number_of_maps]{0};
		//TIMERSTARTCPU(minhashfunc);
		minhashfunc(sequence, hashValues, isForwardStrand);
		//TIMERSTOPCPU(minhashfunc);


        handle.allUniqueResults.clear();
        handle.tmp.clear();

        for(int map = 0; map < minparams.maps; ++map) {
            Key_t key = hashValues[map] & key_mask;

			//TIMERSTARTCPU(get_ranged);
            auto entries_range = minhashTables[map]->get_ranged(key);
            std::size_t n_entries = std::distance(entries_range.first, entries_range.second);
			//TIMERSTOPCPU(get_ranged);

            if(map == 0){
				//TIMERSTARTCPU(reserve);
                handle.tmp.reserve(minparams.maps * n_entries);
                handle.allUniqueResults.reserve(minparams.maps * n_entries);
				//TIMERSTOPCPU(reserve);
            }

            //TIMERSTARTCPU(resizebeforeunion);
            handle.tmp.resize(handle.allUniqueResults.size() + n_entries);
			//TIMERSTOPCPU(resizebeforeunion);

			//TIMERSTARTCPU(setunion);
            auto union_end = std::set_union(entries_range.first,
                                                entries_range.second,
                                                handle.allUniqueResults.begin(),
                                                handle.allUniqueResults.end(),
                                                handle.tmp.begin());
			//TIMERSTOPCPU(setunion);

			//TIMERSTARTCPU(resizeafterunion);
			handle.tmp.resize(std::distance(handle.tmp.begin(), union_end));
			//TIMERSTOPCPU(resizeafterunion);

			//TIMERSTARTCPU(swap);
			std::swap(handle.tmp, handle.allUniqueResults);
			//TIMERSTOPCPU(swap);
        }

        assert(handle.allUniqueResults.size() <= std::numeric_limits<std::int64_t>::max());

		return std::int64_t(handle.allUniqueResults.size());

	}

    std::int64_t getNumberOfCandidatesUpperBound(const std::string& sequence) const noexcept{
		static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
		// we do not consider reads which are shorter than k
		if(sequence.size() < unsigned(minparams.k))
			return 0;

		std::uint64_t hashValues[maximum_number_of_maps]{0};

		bool isForwardStrand[maximum_number_of_maps]{0};
		//TIMERSTARTCPU(minhashfunc);
		minhashfunc(sequence, hashValues, isForwardStrand);
		//TIMERSTOPCPU(minhashfunc);

        std::size_t result = 0;

        for(int map = 0; map < minparams.maps; ++map) {
            Key_t key = hashValues[map] & key_mask;

			//TIMERSTARTCPU(get_ranged);
            const auto entries_range = minhashTables[map]->get_ranged(key);
            const std::size_t n_entries = std::distance(entries_range.first, entries_range.second);
            result += n_entries;
			//TIMERSTOPCPU(get_ranged);
        }

        assert(result >= std::size_t(minparams.maps));
        result -= minparams.maps; //remove self from each map result

		return std::int64_t(result);

	}

	std::vector<Result_t> getCandidates(const std::string& sequence,
										std::uint64_t max_number_candidates) const noexcept{
		static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
		// we do not consider reads which are shorter than k
		if(sequence.size() < unsigned(minparams.k))
			return {};

		std::uint64_t hashValues[maximum_number_of_maps]{0};

		bool isForwardStrand[maximum_number_of_maps]{0};
		//TIMERSTARTCPU(minhashfunc);
		minhashfunc(sequence, hashValues, isForwardStrand);
		//TIMERSTOPCPU(minhashfunc);


        std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;
        //TIMERSTARTCPU(getcandrest);
        for(int map = 0; map < minparams.maps && allUniqueResults.size() < max_number_candidates; ++map) {
            Key_t key = hashValues[map] & key_mask;

            auto entries_range = minhashTables[map]->get_ranged(key);
            std::size_t n_entries = std::distance(entries_range.first, entries_range.second);

            if(map == 0){
                //allUniqueResults.reserve(minparams.maps * entries.size());
                tmp.reserve(std::min(max_number_candidates, minparams.maps * n_entries));
                allUniqueResults.reserve(std::min(max_number_candidates, minparams.maps * n_entries));
            }

            tmp.resize(allUniqueResults.size() + n_entries);
            auto union_end = set_union_n_or_empty(entries_range.first,
                                                entries_range.second,
                                                allUniqueResults.begin(),
                                                allUniqueResults.end(),
                                                max_number_candidates,
                                                tmp.begin());
            if(tmp.begin() == union_end){
                return {};
            }else{
                tmp.resize(std::distance(tmp.begin(), union_end));
                std::swap(tmp, allUniqueResults);
            }
        }

        /*std::vector<Value_t> result2(allUniqueResults.size());
		auto result2end = getCandidates(result2.begin(), result2.end(), sequence, max_number_candidates);
		result2.resize(std::distance(result2.begin(), result2end));

		assert(result2.size() == allUniqueResults.size());
		assert(result2 == allUniqueResults);

		std::cout << result2.size() << " " << allUniqueResults.size() << " " << (result2 == allUniqueResults) << std::endl;*/

		return allUniqueResults;

	}

	template<class Iter>
	Iter getCandidates(Iter begin, Iter end, const std::string& sequence,
										std::uint64_t max_number_candidates) const noexcept{
		static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");

		const std::size_t totalresultrangesize = std::distance(begin, end);

		// we do not consider reads which are shorter than k
		if(sequence.size() < unsigned(minparams.k))
			return begin;

		std::uint64_t hashValues[maximum_number_of_maps]{0};

		bool isForwardStrand[maximum_number_of_maps]{0};
		//TIMERSTARTCPU(minhashfunc);
		minhashfunc(sequence, hashValues, isForwardStrand);
		//TIMERSTOPCPU(minhashfunc);

		//Iter curBegin = begin;
		Iter curEnd = begin;

        //std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;
        //TIMERSTARTCPU(getcandrest);
        for(int map = 0; map < minparams.maps && std::uint64_t(std::distance(begin, curEnd)) < max_number_candidates; ++map) {
            Key_t key = hashValues[map] & key_mask;

            auto entries_range = minhashTables[map]->get_ranged(key);
            std::size_t n_entries = std::distance(entries_range.first, entries_range.second);

            if(map == 0){
                tmp.reserve(std::min(max_number_candidates, minparams.maps * n_entries));
                //allUniqueResults.reserve(std::min(max_number_candidates, minparams.maps * n_entries));
            }

            tmp.resize(std::distance(begin, curEnd) + n_entries);
            auto union_end = set_union_n_or_empty(entries_range.first,
                                                entries_range.second,
                                                begin,
                                                curEnd,
                                                max_number_candidates,
                                                tmp.begin());
            if(tmp.begin() == union_end){
                return begin;
            }else{
                tmp.resize(std::distance(tmp.begin(), union_end));
				if(tmp.size() > totalresultrangesize)
					std::cout << tmp.size() << " > " << totalresultrangesize << std::endl;
				assert(tmp.size() <= totalresultrangesize);
				curEnd = std::swap_ranges(tmp.begin(), tmp.end(), begin);
            }
        }

		return curEnd;

	}

    std::tuple<std::vector<Result_t>,
        std::chrono::duration<double>,
        std::chrono::duration<double>,
        std::chrono::duration<double>,
        std::chrono::duration<double>,
        std::chrono::duration<double>,
        std::chrono::duration<double>> getCandidatesTimed(const std::string& sequence,
										std::uint64_t max_number_candidates) const noexcept{
		static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
		// we do not consider reads which are shorter than k
		if(sequence.size() < unsigned(minparams.k))
			return {};

        std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;
        std::chrono::duration<double> getTime{0};
        std::chrono::duration<double> mergeTime{0};
        std::chrono::duration<double> uniqueTime{0};
        std::chrono::duration<double> hashTime{0};
        std::chrono::duration<double> resizeTime{0};
        std::chrono::duration<double> totalTime{0};

        tpc = std::chrono::system_clock::now();

		std::uint64_t hashValues[maximum_number_of_maps]{0};

		bool isForwardStrand[maximum_number_of_maps]{0};
		//TIMERSTARTCPU(minhashfunc);
        tpa = std::chrono::system_clock::now();
		minhashfunc(sequence, hashValues, isForwardStrand);
        tpb = std::chrono::system_clock::now();
        hashTime += tpb - tpa;

		//TIMERSTOPCPU(minhashfunc);
		std::vector<Value_t> allUniqueResults;
		std::vector<Value_t> tmp;
		//TIMERSTARTCPU(getcandrest);
		for(int map = 0; map < minparams.maps && allUniqueResults.size() <= max_number_candidates; ++map) {
			Key_t key = hashValues[map] & key_mask;

            tpa = std::chrono::system_clock::now();
			std::vector<Value_t> entries = minhashTables[map]->get(key);
            tpb = std::chrono::system_clock::now();
            getTime += tpb - tpa;
			if(map == 0){
				//allUniqueResults.reserve(minparams.maps * entries.size());
				tmp.reserve(minparams.maps * entries.size());
                allUniqueResults.reserve(minparams.maps * entries.size());
			}

			if(!Map_t::resultsAreSorted){
				std::sort(entries.begin(), entries.end());
			}

            tpa = std::chrono::system_clock::now();

			tmp.resize(allUniqueResults.size() + entries.size());

            tpb = std::chrono::system_clock::now();
            resizeTime += tpb - tpa;

            tpa = std::chrono::system_clock::now();

			std::merge(entries.begin(), entries.end(), allUniqueResults.begin(), allUniqueResults.end(), tmp.begin());

            tpb = std::chrono::system_clock::now();
            mergeTime += tpb - tpa;

			std::swap(tmp, allUniqueResults);

            tpa = std::chrono::system_clock::now();

			auto uniqueEnd = std::unique(allUniqueResults.begin(), allUniqueResults.end());

            tpb = std::chrono::system_clock::now();
            uniqueTime += tpb - tpa;

            tpa = std::chrono::system_clock::now();

			allUniqueResults.resize(std::distance(allUniqueResults.begin(), uniqueEnd));

            tpb = std::chrono::system_clock::now();
            resizeTime += tpb - tpa;
		}
        tpd = std::chrono::system_clock::now();
        totalTime += tpd - tpc;
		//TIMERSTOPCPU(getcandrest);
		return {allUniqueResults, hashTime, getTime, mergeTime, uniqueTime, resizeTime, totalTime};
	}

    std::vector<Result_t> getCandidatesMergeUnique(const std::string& sequence,
                                        std::uint64_t max_number_candidates) const noexcept{
        static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
        // we do not consider reads which are shorter than k
        if(sequence.size() < unsigned(minparams.k))
            return {};

        std::uint64_t hashValues[maximum_number_of_maps]{0};

        bool isForwardStrand[maximum_number_of_maps]{0};
        //TIMERSTARTCPU(minhashfunc);
        minhashfunc(sequence, hashValues, isForwardStrand);
        //TIMERSTOPCPU(minhashfunc);

#if 0
        std::array<std::vector<Value_t>, maximum_number_of_maps> hashmapResults;
        std::size_t totalSumResults = 0;


        for(int map = 0; map < minparams.maps; ++map) {
            Key_t key = hashValues[map] & key_mask;

            hashmapResults[map] = minhashTables[map]->get(key);
            totalSumResults += hashmapResults[map].size();

            if(!Map_t::resultsAreSorted){
                std::sort(hashmapResults[map].begin(), hashmapResults[map].end());
            }
        }

        std::sort(hashmapResults.begin(),
                hashmapResults.begin() + minparams.maps,
                [](const auto& l, const auto& r){
                    return l.size() > r.size();
                });

        std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;

        tmp.reserve(std::min(hashmapResults[0].size() * minparams.maps, max_number_candidates));
        allUniqueResults.reserve(std::min(hashmapResults[0].size() * minparams.maps, max_number_candidates));

        for(int map = 0; map < minparams.maps && allUniqueResults.size() < max_number_candidates; ++map){
            const auto& entries = hashmapResults[map];

            tmp.resize(allUniqueResults.size() + entries.size());
            auto enditer = set_union_n(entries.begin(),
                                        entries.end(),
                                        allUniqueResults.begin(),
                                        allUniqueResults.end(),
                                        max_number_candidates,
                                        tmp.begin());

            tmp.resize(std::distance(tmp.begin(), enditer));
            std::swap(tmp, allUniqueResults);
        }

#else
        std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;

        for(int map = 0; map < minparams.maps && allUniqueResults.size() < max_number_candidates; ++map) {
            Key_t key = hashValues[map] & key_mask;

            std::vector<Value_t> entries = minhashTables[map]->get(key);
            if(map == 0){
                //allUniqueResults.reserve(minparams.maps * entries.size());
                tmp.reserve(minparams.maps * entries.size());
                allUniqueResults.reserve(minparams.maps * entries.size());
            }

            if(!Map_t::resultsAreSorted){
                std::sort(entries.begin(), entries.end());
            }

            tmp.resize(allUniqueResults.size() + entries.size());

            auto enditer = set_union_n(entries.begin(),
                                        entries.end(),
                                        allUniqueResults.begin(),
                                        allUniqueResults.end(),
                                        max_number_candidates,
                                        tmp.begin());

            tmp.resize(std::distance(tmp.begin(), enditer));
            std::swap(tmp, allUniqueResults);
        }
#endif
        allUniqueResults.resize(std::min(allUniqueResults.size(), max_number_candidates));

        //TIMERSTOPCPU(getcandrest);
        return allUniqueResults;
    }



	void resize(std::uint64_t nReads_){
		if(nReads_ == 0) throw std::runtime_error("Minhasher::init cannnot be called with argument 0");
		if(nReads_-1 > max_read_num)
			throw std::runtime_error("Minhasher::init: Minhasher is configured for only" + std::to_string(max_read_num) + " reads, not " + std::to_string(nReads_) + "!!!");

		nReads = nReads_;

		for (std::size_t i = 0; i < minhashTables.size(); ++i){
			auto& table = minhashTables[i];
			table->resize(nReads_);
		}
	}

	void transform(){

		for (std::size_t i = 0; i < minhashTables.size(); ++i){
			std::cout << "Transforming table " << i << std::endl;
			auto& table = minhashTables[i];
			table->transform();
		}
	}



private:
	void minhashfunc(const std::string& sequence, std::uint64_t* minhashSignature, bool* isForwardStrand) const noexcept{
        std::uint64_t kmerHashValues[maximum_number_of_maps]{0};

		std::uint64_t fhVal = 0;
        std::uint64_t rhVal = 0;
		bool isForward = false;
		// calc hash values of first canonical kmer
		NTMC64(sequence.c_str(), minparams.k, minparams.maps, minhashSignature, fhVal, rhVal, isForward);

		for (int j = 0; j < minparams.maps; ++j) {
			isForwardStrand[j] = isForward;
		}

		//calc hash values of remaining canonical kmers
		for (size_t i = 0; i < sequence.size() - minparams.k; ++i) {
			NTMC64(fhVal, rhVal, sequence[i], sequence[i + minparams.k], minparams.k, minparams.maps, kmerHashValues, isForward);

			for (int j = 0; j < minparams.maps; ++j) {
				if (minhashSignature[j] > kmerHashValues[j]){
					minhashSignature[j] = kmerHashValues[j];
					isForwardStrand[j] = isForward;
				}
			}
		}
	}
};

}

#endif
