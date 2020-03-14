#ifndef CARE_MINHASHER_HPP
#define CARE_MINHASHER_HPP

#include "options.hpp"
#include "hpc_helpers.cuh"
#include "util.hpp"

#include <config.hpp>
#include <memorymanagement.hpp>

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
#include <iterator>
#include <unordered_map>

//#ifdef __NVCC__
//#include <gpu/nvtxtimelinemarkers.hpp>
//#endif



namespace care{

    namespace minhasherdetail{

		/*
		 * hash map to map keys to indices using linear probing
		 */
		template<class Key_t, class Index_t>
		struct KeyIndexMap{
            using Pair_t = std::pair<Key_t, Index_t>;
			Pair_t EmptySlot{0, std::numeric_limits<Index_t>::max()};

			std::uint64_t murmur_hash_3_uint64_t(std::uint64_t x) const{
				x ^= x >> 33;
				x *= 0xff51afd7ed558ccd;
				x ^= x >> 33;
				x *= 0xc4ceb9fe1a85ec53;
				x ^= x >> 33;

				return x;
			}

            std::uint32_t mueller_hash_uint32_t(std::uint32_t x) const{
                x = ((x >> 16) ^ x) * 0x45d9f3b;
                x = ((x >> 16) ^ x) * 0x45d9f3b;
                x = ((x >> 16) ^ x);

				return x;
			}

            std::uint32_t murmur_integer_finalizer_hash_uint32_t(std::uint32_t x) const{
                x ^= x >> 16;
                x *= 0x85ebca6b;
                x ^= x >> 13;
                x *= 0xc2b2ae35;
                x ^= x >> 16;

				return x;
			}

            std::uint64_t maxProbes{0};
            std::uint64_t size;
			std::vector<Pair_t> keyToIndexMap;           

			KeyIndexMap() : KeyIndexMap(0){}
			KeyIndexMap(std::uint64_t size) : size(size){
				if(size == std::numeric_limits<Index_t>::max())
					throw std::runtime_error("KeyIndexMap: too many keys!");

				keyToIndexMap.resize(size, KeyIndexMap::EmptySlot);
			}

            KeyIndexMap(const KeyIndexMap&) = default;
            KeyIndexMap(KeyIndexMap&&) = default;

            KeyIndexMap& operator=(const KeyIndexMap& rhs){
                maxProbes = rhs.maxProbes;
                size = rhs.size;
                keyToIndexMap = rhs.keyToIndexMap;
                
                return *this;
            }

            KeyIndexMap& operator=(KeyIndexMap&& rhs) noexcept{
                maxProbes = std::move(rhs.maxProbes);
                size = std::move(rhs.size);
                keyToIndexMap = std::move(rhs.keyToIndexMap);

                return *this;
            }

            bool operator==(const KeyIndexMap& rhs) const{
                if(maxProbes != rhs.maxProbes){
                    return false;
                }
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
                std::uint64_t probes = 0;
                std::uint64_t pos = murmur_hash_3_uint64_t(key) % size;
                while(keyToIndexMap[pos] != KeyIndexMap::EmptySlot){
                    pos++;
                    //wrap-around
                    if(pos == size){
                        pos = 0;
                    }
                    probes++;
                }
                keyToIndexMap[pos].first = key;
                keyToIndexMap[pos].second = value;
                //std::cerr << "probes insert: " << probes << "\n";

                maxProbes = std::max(maxProbes, probes);
            }

            Index_t get(Key_t key) const noexcept{
                std::uint64_t probes = 0;
                std::uint64_t pos = murmur_hash_3_uint64_t(key) % size;
                while(keyToIndexMap[pos].first != key){
                    pos++;
                    //wrap-around
                    if(pos == size){
                        pos = 0;
                    }
                    probes++;
                    if(maxProbes < probes){
                        return std::numeric_limits<Index_t>::max();
                    }
                }
                //std::cerr << "probes get: " << probes << "\n";
                return keyToIndexMap[pos].second;
            }

            template<int N>
            std::array<Index_t, N> getN(const Key_t* keys) const noexcept{

                std::array<std::uint64_t, N> pos;
                for(int i = 0; i < N; i++){
                    pos[i] = murmur_hash_3_uint64_t(keys[i]) % size;
                    __builtin_prefetch(&keyToIndexMap[pos], 0, 1);
                }
                std::array<std::uint64_t, N> probes{0};
                std::array<Index_t, N> result;

                for(int i = 0; i < N; i++){
                    while(keyToIndexMap[pos[i]].first != keys[i]){
                        pos[i]++;
                        //wrap-around
                        if(pos[i] == size){
                            pos[i] = 0;
                        }
                        probes[i]++;
                        if(maxProbes < probes[i]){
                            result[i] = std::numeric_limits<Index_t>::max();
                            break;
                        }
                    }
                    result[i] = keyToIndexMap[pos[i]].second;
                }

                return result;
            }

            std::size_t numBytes() const{
                return keyToIndexMap.size() * sizeof(Pair_t);
            }

            std::size_t allocationSizeInBytes() const{
                return keyToIndexMap.capacity() * sizeof(Pair_t);
            }

            static std::size_t getRequiredSizeInBytes(std::uint64_t elements){
                return elements * sizeof(Pair_t);
            }

			void clear() noexcept{
				keyToIndexMap.clear();
                maxProbes = 0;
			}

			void destroy() noexcept{
				clear();
				keyToIndexMap.shrink_to_fit();
			}

            void writeToStream(std::ofstream& outstream) const{
                outstream.write(reinterpret_cast<const char*>(&maxProbes), sizeof(std::uint64_t));
                outstream.write(reinterpret_cast<const char*>(&size), sizeof(std::uint64_t));
                outstream.write(reinterpret_cast<const char*>(keyToIndexMap.data()), keyToIndexMap.size() * sizeof(Pair_t));
            }

            void readFromStream(std::ifstream& instream){
                instream.read(reinterpret_cast<char*>(&maxProbes), sizeof(std::uint64_t));

                instream.read(reinterpret_cast<char*>(&size), sizeof(std::uint64_t));
                keyToIndexMap.resize(size);
                instream.read(reinterpret_cast<char*>(keyToIndexMap.data()), keyToIndexMap.size() * sizeof(Pair_t));
                //std::cerr << "keyToIndexMap.size = " << size << ", bytes = " << (keyToIndexMap.size() * sizeof(Pair_t)) << '\n';
            }
		};


        /*
		 * hash map to map keys to indices using linear probing
		 */
		template<class Key_t, class Index_t>
		struct KeyToIndexLengthPairMap{
            using IndexLengthPair_t = std::pair<Index_t, BucketSize>;
            using Pair_t = std::pair<Key_t, IndexLengthPair_t>;
			Pair_t EmptySlot{0, IndexLengthPair_t{std::numeric_limits<Index_t>::max(), std::numeric_limits<BucketSize>::max()}};

			std::uint64_t murmur_hash_3_uint64_t(std::uint64_t x) const{
				x ^= x >> 33;
				x *= 0xff51afd7ed558ccd;
				x ^= x >> 33;
				x *= 0xc4ceb9fe1a85ec53;
				x ^= x >> 33;

				return x;
			}

            std::uint32_t mueller_hash_uint32_t(std::uint32_t x) const{
                x = ((x >> 16) ^ x) * 0x45d9f3b;
                x = ((x >> 16) ^ x) * 0x45d9f3b;
                x = ((x >> 16) ^ x);

				return x;
			}

            std::uint32_t murmur_integer_finalizer_hash_uint32_t(std::uint32_t x) const{
                x ^= x >> 16;
                x *= 0x85ebca6b;
                x ^= x >> 13;
                x *= 0xc2b2ae35;
                x ^= x >> 16;

				return x;
			}

            std::uint64_t maxProbes{0};
            std::uint64_t size;
			std::vector<Pair_t> keyToIndexMap;           

			KeyToIndexLengthPairMap() : KeyToIndexLengthPairMap(0){}
			KeyToIndexLengthPairMap(std::uint64_t size) : size(size){
				if(size >= std::numeric_limits<Index_t>::max())
					throw std::runtime_error("KeyToIndexLengthPairMap: too many keys!");

				keyToIndexMap.resize(size, KeyToIndexLengthPairMap::EmptySlot);
			}

            KeyToIndexLengthPairMap(const KeyToIndexLengthPairMap&) = default;
            KeyToIndexLengthPairMap(KeyToIndexLengthPairMap&&) = default;

            KeyToIndexLengthPairMap& operator=(const KeyToIndexLengthPairMap& rhs){
                maxProbes = rhs.maxProbes;
                size = rhs.size;
                keyToIndexMap = rhs.keyToIndexMap;
                
                return *this;
            }

            KeyToIndexLengthPairMap& operator=(KeyToIndexLengthPairMap&& rhs) noexcept{
                maxProbes = std::move(rhs.maxProbes);
                size = std::move(rhs.size);
                keyToIndexMap = std::move(rhs.keyToIndexMap);

                return *this;
            }

            bool operator==(const KeyToIndexLengthPairMap& rhs) const{
                if(maxProbes != rhs.maxProbes){
                    return false;
                }
                if(size != rhs.size)
                    return false;
                if(keyToIndexMap != rhs.keyToIndexMap)
                    return false;
                return true;
            }

            bool operator!=(const KeyToIndexLengthPairMap& rhs) const{
                return !(*this == rhs);
            }

            void insert(Key_t key, Index_t index, BucketSize length) noexcept{
                std::uint64_t probes = 0;
                std::uint64_t pos = murmur_hash_3_uint64_t(key) % size;
                while(keyToIndexMap[pos] != KeyToIndexLengthPairMap::EmptySlot){
                    pos++;
                    //wrap-around
                    if(pos == size){
                        pos = 0;
                    }
                    probes++;
                }
                keyToIndexMap[pos].first = key;
                keyToIndexMap[pos].second.first = index;
                keyToIndexMap[pos].second.second = length;
                //std::cerr << "probes insert: " << probes << "\n";

                maxProbes = std::max(maxProbes, probes);
            }

            IndexLengthPair_t get(Key_t key) const noexcept{
                std::uint64_t probes = 0;
                std::uint64_t pos = murmur_hash_3_uint64_t(key) % size;
                while(keyToIndexMap[pos].first != key){
                    pos++;
                    //wrap-around
                    if(pos == size){
                        pos = 0;
                    }
                    probes++;
                    if(maxProbes < probes){
                        return {std::numeric_limits<Index_t>::max(), std::numeric_limits<BucketSize>::max()};
                    }
                }
                //std::cerr << "probes get: " << probes << "\n";
                return keyToIndexMap[pos].second;
            }

            template<int N>
            std::array<IndexLengthPair_t, N> getN(const Key_t* keys) const noexcept{

                std::array<std::uint64_t, N> pos;
                for(int i = 0; i < N; i++){
                    pos[i] = murmur_hash_3_uint64_t(keys[i]) % size;
                    __builtin_prefetch(&keyToIndexMap[pos], 0, 1);
                }
                std::array<std::uint64_t, N> probes{0};
                std::array<IndexLengthPair_t, N> result;

                for(int i = 0; i < N; i++){
                    while(keyToIndexMap[pos[i]].first != keys[i]){
                        pos[i]++;
                        //wrap-around
                        if(pos[i] == size){
                            pos[i] = 0;
                        }
                        probes[i]++;
                        if(maxProbes < probes[i]){
                            result[i].first = std::numeric_limits<Index_t>::max();
                            result[i].second = std::numeric_limits<BucketSize>::max();
                            break;
                        }
                    }
                    result[i] = keyToIndexMap[pos[i]].second;
                }

                return result;
            }

            std::size_t numBytes() const{
                return keyToIndexMap.size() * sizeof(Pair_t);
            }

            std::size_t allocationSizeInBytes() const{
                return keyToIndexMap.capacity() * sizeof(Pair_t);
            }

            static std::size_t getRequiredSizeInBytes(std::uint64_t elements){
                return elements * sizeof(Pair_t);
            }

			void clear() noexcept{
				keyToIndexMap.clear();
                maxProbes = 0;
			}

			void destroy() noexcept{
				clear();
				keyToIndexMap.shrink_to_fit();
			}

            void writeToStream(std::ofstream& outstream) const{
                outstream.write(reinterpret_cast<const char*>(&maxProbes), sizeof(std::uint64_t));
                outstream.write(reinterpret_cast<const char*>(&size), sizeof(std::uint64_t));
                outstream.write(reinterpret_cast<const char*>(keyToIndexMap.data()), keyToIndexMap.size() * sizeof(Pair_t));
            }

            void readFromStream(std::ifstream& instream){
                instream.read(reinterpret_cast<char*>(&maxProbes), sizeof(std::uint64_t));

                instream.read(reinterpret_cast<char*>(&size), sizeof(std::uint64_t));
                keyToIndexMap.resize(size);
                instream.read(reinterpret_cast<char*>(keyToIndexMap.data()), keyToIndexMap.size() * sizeof(Pair_t));
                //std::cerr << "keyToIndexMap.size = " << size << ", bytes = " << (keyToIndexMap.size() * sizeof(Pair_t)) << '\n';
            }
		};




		template<class key_t, class value_t, class index_t>
		struct KeyValueMapFixedSize{
			using Key_t = key_t;
			using Value_t = value_t;
			using Index_t = index_t;

            using Count_t = std::uint16_t;

			static constexpr bool resultsAreSorted = true;

			Index_t size;
			Index_t nKeys;
			Index_t nValues;
			bool noMoreWrites;
            bool canUseGpu = false;
			std::vector<Key_t> keys;
			std::vector<Value_t> values;
            std::vector<Count_t> counts;
			std::vector<Index_t> countsPrefixSum;
            std::vector<Key_t> keysWithoutValues;

			double load = 0.8;
			KeyIndexMap<Key_t, Index_t> keyIndexMap;
            KeyToIndexLengthPairMap<Key_t, Index_t> keyToIndexLengthPairMap;

            KeyValueMapFixedSize() : KeyValueMapFixedSize(0){
			}

			KeyValueMapFixedSize(Index_t size_)
                : size(size_), nKeys(size_), nValues(size_), noMoreWrites(false){
				keys.resize(size);
				values.resize(size);
			}

            KeyValueMapFixedSize(const KeyValueMapFixedSize& other){
                *this = other;
            }

            KeyValueMapFixedSize(KeyValueMapFixedSize&& other) noexcept{
                *this = std::move(other);
            }

            KeyValueMapFixedSize& operator=(const KeyValueMapFixedSize& other) noexcept{
                size = other.size;
                nKeys = other.nKeys;
                nValues = other.nValues;
                noMoreWrites = other.noMoreWrites;
                keys = other.keys;
                values = other.values;
                counts = other.counts;
                countsPrefixSum = other.countsPrefixSum;
                keysWithoutValues = other.keysWithoutValues;
                keyIndexMap = other.keyIndexMap;
                keyToIndexLengthPairMap = other.keyToIndexLengthPairMap;
                return *this;
            }

            KeyValueMapFixedSize& operator=(KeyValueMapFixedSize&& other) noexcept{
                size = other.size;
                nKeys = other.nKeys;
                nValues = other.nValues;
                noMoreWrites = other.noMoreWrites;
                keys = std::move(other.keys);
                values = std::move(other.values);
                counts = std::move(other.counts);
                countsPrefixSum = std::move(other.countsPrefixSum);
                keysWithoutValues = std::move(other.keysWithoutValues);
                keyIndexMap = std::move(other.keyIndexMap);
                keyToIndexLengthPairMap = std::move(other.keyToIndexLengthPairMap);
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
                if(counts != rhs.counts)
                    return false;
                if(countsPrefixSum != rhs.countsPrefixSum)
                    return false;
                if(keysWithoutValues != rhs.keysWithoutValues){
                    return false;
                }
                if(keyIndexMap != rhs.keyIndexMap){
                    return false;
                }
                if(keyToIndexLengthPairMap != rhs.keyToIndexLengthPairMap){
                    return false;
                }
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

                assert(nKeys == keys.size() || nKeys <= keyIndexMap.keyToIndexMap.size());
                assert(nValues == values.size());

                //outstream.write(reinterpret_cast<const char*>(keys.data()), sizeof(Key_t) * nKeys);
                outstream.write(reinterpret_cast<const char*>(values.data()), sizeof(Value_t) * nValues);

                std::size_t nCounts = countsPrefixSum.size();
                outstream.write(reinterpret_cast<const char*>(&nCounts), sizeof(std::size_t));
                outstream.write(reinterpret_cast<const char*>(countsPrefixSum.data()), sizeof(Index_t) * nCounts);

                std::size_t emptyKeys = keysWithoutValues.size();
                outstream.write(reinterpret_cast<const char*>(&emptyKeys), sizeof(std::size_t));
                outstream.write(reinterpret_cast<const char*>(keysWithoutValues.data()), sizeof(Key_t) * emptyKeys);

                std::size_t elements = counts.size();
                outstream.write(reinterpret_cast<const char*>(&elements), sizeof(std::size_t));
                outstream.write(reinterpret_cast<const char*>(counts.data()), sizeof(Count_t) * elements);

                keyIndexMap.writeToStream(outstream);

                keyToIndexLengthPairMap.writeToStream(outstream);
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

                keys = std::vector<Key_t>();
                //keys.resize(nKeys);
                //values.resize(nValues);
		        values = std::vector<Value_t>();
		        values.resize(nValues);

                //instream.read(reinterpret_cast<char*>(keys.data()), sizeof(Key_t) * nKeys);
                instream.read(reinterpret_cast<char*>(values.data()), sizeof(Value_t) * nValues);

                std::size_t nCountsPrefixSum = 0;
                instream.read(reinterpret_cast<char*>(&nCountsPrefixSum), sizeof(std::size_t));
                countsPrefixSum.resize(nCountsPrefixSum);
                instream.read(reinterpret_cast<char*>(countsPrefixSum.data()), sizeof(Index_t) * nCountsPrefixSum);

                std::size_t emptyKeys = 0;
                instream.read(reinterpret_cast<char*>(&emptyKeys), sizeof(std::size_t));
                keysWithoutValues.resize(emptyKeys);
                instream.read(reinterpret_cast<char*>(keysWithoutValues.data()), sizeof(Key_t) * emptyKeys);

                std::size_t elements = 0;
                instream.read(reinterpret_cast<char*>(&elements), sizeof(std::size_t));
                counts.resize(elements);
                instream.read(reinterpret_cast<char*>(counts.data()), sizeof(Count_t) * elements);

                keyIndexMap.readFromStream(instream);

                keyToIndexLengthPairMap.readFromStream(instream);
            }

            std::size_t numBytes() const{
                return keys.size() * sizeof(Key_t)
                    + values.size() * sizeof(Value_t)
                    + counts.size() * sizeof(Count_t)
                    + countsPrefixSum.size() * sizeof(Index_t)
                    + keysWithoutValues.size() * sizeof(Key_t)
                    + keyIndexMap.numBytes()
                    + keyToIndexLengthPairMap.numBytes();
            }

            std::size_t allocationSizeInBytes() const{
                return keys.capacity() * sizeof(Key_t)
                    + values.capacity() * sizeof(Value_t)
                    + counts.capacity() * sizeof(Count_t)
                    + countsPrefixSum.capacity() * sizeof(Index_t)
                    + keysWithoutValues.capacity() * sizeof(Key_t)
                    + keyIndexMap.allocationSizeInBytes()
                    + keyToIndexLengthPairMap.allocationSizeInBytes();
            }

            static std::size_t getRequiredSizeInBytesBeforeCompaction(std::uint64_t elements){
                return elements * sizeof(Key_t)
                    + elements * sizeof(Value_t)
                    + 4 * 1024;
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
                counts.clear();
				countsPrefixSum.clear();
                keysWithoutValues.clear();
				keyIndexMap.clear();
                keyToIndexLengthPairMap.clear();
			}

			void destroy() noexcept{
				clear();
				keys.shrink_to_fit();
				values.shrink_to_fit();
                counts.shrink_to_fit();
				countsPrefixSum.shrink_to_fit();
                keysWithoutValues.shrink_to_fit();
				keyIndexMap.shrink_to_fit();
                keyToIndexLengthPairMap.shrink_to_fit();
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

			std::vector<Value_t> get(Key_t key) const noexcept{
                assert(noMoreWrites);

                const auto pair = get_ranged(key);
                return {pair.first, pair.second};                
			}

			std::pair<const Value_t*, const Value_t*> get_ranged(Key_t key) const noexcept{
                assert(noMoreWrites);

				// auto range = std::equal_range(keys.begin(), keys.end(), key);
				// if(range.first == keys.end()) return {};
                //
				// Index_t index = std::distance(keys.begin(), range.first);

                // auto lb = std::lower_bound(keys.begin(), keys.end(), key);
                // if(lb == keys.end() || *lb != key) {
                //     return {};
                // }
                // const Index_t index = std::distance(keys.begin(), lb);

                /*
                //nvtx::push_range("check empty key", 6);
                auto emptyKeyIter = std::lower_bound(keysWithoutValues.begin(), keysWithoutValues.end(), key);
                //nvtx::pop_range("check empty key");

                if(!(emptyKeyIter != keysWithoutValues.end() && *emptyKeyIter == key)){
                    //nvtx::push_range("fetch index",3);
                    const Index_t index = keyIndexMap.get(key);
                    //nvtx::pop_range("fetch index");

				    //if(index != std::numeric_limits<Index_t>::max()){
				        return {&values[countsPrefixSum[index]], &values[countsPrefixSum[index+1]]};
                    //}else{
                    //    return {};
                    //}
                }else{
                    return {}; //key has no values
                }
                */

               //nvtx::push_range("fetch index",3);

                // const Index_t index = keyIndexMap.get(key);

                // if(size_t(index) >=  countsPrefixSum.size() || size_t(index+1) >= countsPrefixSum.size()) {
                //     std::cerr << "\ninvalid index returned by keyIndexMap: key = " << key << ", returned index = " << index << "cPS.size() = " << countsPrefixSum.size() << "\n";
                // }else{
                //     if(size_t(countsPrefixSum[index]) > values.size() || size_t(countsPrefixSum[index+1]) > values.size()){
                //         std::cerr << "\ninvalid prefix sum at index " << index << " or " << (index+1) << ". cPS = " << countsPrefixSum[index] << " " << countsPrefixSum[index+1] << " values.size() = " << values.size() << "\n";
                //         assert(false);
                //     }
                // }

                // return {&values[countsPrefixSum[index]], &values[countsPrefixSum[index+1]]};
                

                const auto indexLengthPair = keyToIndexLengthPairMap.get(key);                

                if(size_t(indexLengthPair.first) > values.size() || size_t(indexLengthPair.first + indexLengthPair.second) > values.size()){
                    std::cerr << "\n invalid indexLengthPair . cPS = " << indexLengthPair.first 
                            << ", length " << indexLengthPair.second << " values.size() = " << values.size() << "\n";
                    assert(false);
                }
                
                return {&values[indexLengthPair.first], &values[indexLengthPair.first + indexLengthPair.second]};
                //nvtx::pop_range("fetch index");
			}

		};
    }

struct Minhasher {

    using Index_t = read_number; //read id type
    using Value_t = Index_t; //Value type for hashmap
    using Result_t = Index_t; // Return value for minhash query
    using Map_t = minhasherdetail::KeyValueMapFixedSize<kmer_type, Value_t, Index_t>; //internal map type

    static constexpr int bits_key = sizeof(kmer_type) * 8;
	static constexpr std::uint64_t key_mask = (std::uint64_t(1) << (bits_key - 1)) | ((std::uint64_t(1) << (bits_key - 1)) - 1);
    static constexpr std::uint64_t max_read_num = std::numeric_limits<Index_t>::max();
    static constexpr int maximum_kmer_length = max_k<kmer_type>::value;

    struct Handle{
        using Range_t = std::pair<const Value_t*, const Value_t*>;
        std::vector<Range_t> ranges;
		std::vector<Value_t> allUniqueResults;
        SetUnionHandle<Value_t> suHandle;

        

        std::vector<Value_t> contiguousDataOfRanges;
        std::vector<Range_t> multiranges;
		std::vector<Value_t> multiallUniqueResults;
        std::vector<std::uint64_t> multiminhashSignatures;
        std::vector<int> numResultsPerSequence;
        std::vector<int> numResultsPerSequencePrefixSum;

        std::vector<Value_t>& result() noexcept{
            return allUniqueResults;
        }

        std::vector<Value_t>& multiresults() noexcept{
            return multiallUniqueResults;
        }

        // int numResultsOfSequence(int i) const{
        //     assert(i < int(numResultsPerSequence.size()));
        //     return numResultsPerSequence[i];
        // }
	};

	// the actual maps
	std::vector<std::unique_ptr<Map_t>> minhashTables;
	MinhashOptions minparams;
    read_number nReads;
    bool canUseGpu = false;
    bool allowUVM = false;

    Minhasher();

    Minhasher(const MinhashOptions& parameters);

    Minhasher(const Minhasher&) = delete;
    Minhasher& operator=(const Minhasher&) = delete;

    Minhasher(Minhasher&& rhs);
    Minhasher& operator=(Minhasher&& rhs);

    bool operator==(const Minhasher& rhs) const;

    bool operator!=(const Minhasher& rhs) const;

    std::size_t numBytes() const;

    MemoryUsage getMemoryInfo() const;

    void saveToFile(const std::string& filename) const;

    void loadFromFile(const std::string& filename);

	void init(std::uint64_t nReads);
    void initMap(int map);
    void moveassignMap(int mapId, Map_t&& newMap);

	void clear();

	void destroy();

    std::map<int, std::int64_t> getBinSizeHistogramOfMap(const Minhasher::Map_t& table) const;
    std::map<int, std::int64_t> getBinSizeHistogramOfMap(int tableId) const;
    std::vector<std::map<int, std::int64_t>> getBinSizeHistogramsOfMaps() const;

    void insertSequenceIntoExternalTables(const std::string& sequence, 
                                            read_number readnum,                                                     
                                            const std::vector<int>& tableIds,
                                            std::vector<Map_t>& tables,
                                            const std::vector<int>& hashIds) const;

    void insertSequenceIntoExternalTables(const std::uint64_t* hashValues, 
                                            int numHashValues,
                                            read_number readnum,                                                     
                                            const std::vector<int>& tableIds,
                                            std::vector<Minhasher::Map_t>& tables) const;

    void insertSequence(const std::string& sequence, read_number readnum, std::vector<int> mapIds);

	void insertSequence(const std::string& sequence, read_number readnum);


    std::pair<const Value_t*, const Value_t*> queryMap(int mapid,
                                                        Map_t::Key_t key) const noexcept;



    void calculateMinhashSignatures(
            Minhasher::Handle& handle,
            const std::vector<std::string>& sequences) const;     

    void calculateMinhashSignatures(
            Minhasher::Handle& handle,
            const char* sequences,
            int numSequences,
            const int* sequenceLengths,
            int sequencesPitch) const;

    void queryPrecalculatedSignatures(Minhasher::Handle& handle, int numSequences) const;   

    void makeUniqueQueryResults(Minhasher::Handle& handle, int numSequences) const;                                      


    std::vector<Result_t> getCandidates(
        const std::string& sequence,
        int num_hits,
        std::uint64_t max_number_candidates) const noexcept;

    std::vector<Result_t> getCandidates_any_map(const std::string& sequence,
                                        std::uint64_t max_number_candidates) const noexcept;

    void getCandidates_any_map(
            Minhasher::Handle& handle,
            const std::string& sequence,
            std::uint64_t) const noexcept;

    void getCandidates_any_map(
            Minhasher::Handle& handle,
            const char* sequence,
            int sequenceLength,
            std::uint64_t) const noexcept;

    /*
        This version of getCandidates returns only read ids which are found in at least num_hits maps
    */
    std::vector<Result_t> getCandidates_some_maps2(const std::string& sequence,
                                        int num_hits,
                                        std::uint64_t max_number_candidates) const noexcept;

    std::vector<Result_t> getCandidates_some_maps(const std::string& sequence,
                                        int num_hits,
                                        std::uint64_t max_number_candidates) const noexcept;

    /*
        This version of getCandidates returns only read ids which are found in all maps
    */
    std::vector<Result_t> getCandidates_all_maps(const std::string& sequence,
                                        std::uint64_t max_number_candidates) const noexcept;

// #############################

/*
    Query number of candidates
*/

    std::int64_t getNumberOfCandidates(const std::string& sequence,
                                        int num_hits) const noexcept;

    std::int64_t getNumberOfCandidatesUpperBound(const std::string& sequence) const noexcept;

//###################################################

    int getResultsPerMapThreshold() const;

	void resize(std::uint64_t nReads_);


private:

	std::array<std::uint64_t, maximum_number_of_maps>
    minhashfunc(const std::string& sequence) const noexcept;

    std::array<std::uint64_t, maximum_number_of_maps> 
    minhashfunc(const char* sequence, int sequenceLength) const noexcept;

    // std::array<std::uint64_t, maximum_number_of_maps>
    // minhashfunc_other(const std::string& sequence) const noexcept;

    // std::array<std::uint64_t, maximum_number_of_maps> 
    // minhashfunc_other(const char* sequence, int sequenceLength) const noexcept;

    void insertIntoMap(int map, std::uint64_t hashValue, read_number readNumber);
    void insertIntoExternalTable(Minhasher::Map_t& table, std::uint64_t hashValue, read_number readnum) const;
};



int calculateResultsPerMapThreshold(int coverage);




}

#endif
