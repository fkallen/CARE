#ifndef CARE_MINHASHER_HPP
#define CARE_MINHASHER_HPP

#include "hpc_helpers.cuh"
#include "util.hpp"

#include <config.hpp>
#include <memorymanagement.hpp>
#include <options.hpp>
#include <readstorage.hpp>

#include <gpu/gpuhashtable.cuh>


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

                assert(size == keyToIndexMap.size());
                
                return *this;
            }

            KeyIndexMap& operator=(KeyIndexMap&& rhs) noexcept{
                maxProbes = std::move(rhs.maxProbes);
                size = std::move(rhs.size);
                keyToIndexMap = std::move(rhs.keyToIndexMap);

                rhs.size = 0;

                assert(size == keyToIndexMap.size());

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
                size = 0;
                maxProbes = 0;
			}

			void destroy() noexcept{
				clear();
				keyToIndexMap.shrink_to_fit();
			}

            void writeToStream(std::ostream& outstream) const{
                outstream.write(reinterpret_cast<const char*>(&maxProbes), sizeof(std::uint64_t));
                outstream.write(reinterpret_cast<const char*>(&size), sizeof(std::uint64_t));
                outstream.write(reinterpret_cast<const char*>(keyToIndexMap.data()), keyToIndexMap.size() * sizeof(Pair_t));
            }

            void readFromStream(std::istream& instream){
                instream.read(reinterpret_cast<char*>(&maxProbes), sizeof(std::uint64_t));

                instream.read(reinterpret_cast<char*>(&size), sizeof(std::uint64_t));
                keyToIndexMap.resize(size);
                instream.read(reinterpret_cast<char*>(keyToIndexMap.data()), keyToIndexMap.size() * sizeof(Pair_t));
                //std::cerr << "keyToIndexMap.size = " << size << ", bytes = " << (keyToIndexMap.size() * sizeof(Pair_t)) << '\n';
            }
		};


        /*
		 * hash map to map keys to std::pair<Index_t, BucketSize>
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

                assert(size == keyToIndexMap.size());
                
                return *this;
            }

            KeyToIndexLengthPairMap& operator=(KeyToIndexLengthPairMap&& rhs) noexcept{
                maxProbes = std::move(rhs.maxProbes);
                size = std::move(rhs.size);
                keyToIndexMap = std::move(rhs.keyToIndexMap);

                rhs.size = 0;

                assert(size == keyToIndexMap.size());

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
                size = 0;
			}

			void destroy() noexcept{
				clear();
				keyToIndexMap.shrink_to_fit();
			}

            void writeToStream(std::ostream& outstream) const{
                outstream.write(reinterpret_cast<const char*>(&maxProbes), sizeof(std::uint64_t));
                outstream.write(reinterpret_cast<const char*>(&size), sizeof(std::uint64_t));
                outstream.write(reinterpret_cast<const char*>(keyToIndexMap.data()), keyToIndexMap.size() * sizeof(Pair_t));
            }

            void readFromStream(std::istream& instream){
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
                keyToIndexLengthPairMap = std::move(other.keyToIndexLengthPairMap);

                other.size = 0;
                other.nKeys = 0;
                other.nValues = 0;

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
                if(keyToIndexLengthPairMap != rhs.keyToIndexLengthPairMap){
                    return false;
                }
                return true;
            }

            bool operator!=(const KeyValueMapFixedSize& rhs) const{
                return !(*this == rhs);
            }

            void writeToStream(std::ostream& outstream) const{
                bool resultsAreSorted_towrite = resultsAreSorted;
                outstream.write(reinterpret_cast<const char*>(&resultsAreSorted_towrite), sizeof(bool));
                outstream.write(reinterpret_cast<const char*>(&size), sizeof(Index_t));
                outstream.write(reinterpret_cast<const char*>(&nKeys), sizeof(Index_t));
                outstream.write(reinterpret_cast<const char*>(&nValues), sizeof(Index_t));
                outstream.write(reinterpret_cast<const char*>(&noMoreWrites), sizeof(bool));
                outstream.write(reinterpret_cast<const char*>(&canUseGpu), sizeof(bool));

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

                keyToIndexLengthPairMap.writeToStream(outstream);
            }

            void readFromStream(std::istream& instream){
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

                keyToIndexLengthPairMap.readFromStream(instream);
            }

            std::size_t numBytes() const{
                return keys.size() * sizeof(Key_t)
                    + values.size() * sizeof(Value_t)
                    + counts.size() * sizeof(Count_t)
                    + countsPrefixSum.size() * sizeof(Index_t)
                    + keysWithoutValues.size() * sizeof(Key_t)
                    + keyToIndexLengthPairMap.numBytes();
            }

            std::size_t allocationSizeInBytes() const{
                return keys.capacity() * sizeof(Key_t)
                    + values.capacity() * sizeof(Value_t)
                    + counts.capacity() * sizeof(Count_t)
                    + countsPrefixSum.capacity() * sizeof(Index_t)
                    + keysWithoutValues.capacity() * sizeof(Key_t)
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
                keyToIndexLengthPairMap.clear();
			}

			void destroy() noexcept{
				clear();
				keys.shrink_to_fit();
				values.shrink_to_fit();
                counts.shrink_to_fit();
				countsPrefixSum.shrink_to_fit();
                keysWithoutValues.shrink_to_fit();
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

    using Key_t = kmer_type;

    using Index_t = read_number; //read id type
    using Value_t = Index_t; //Value type for hashmap
    using Result_t = Index_t; // Return value for minhash query
    //using Map_t = minhasherdetail::KeyValueMapFixedSize<kmer_type, Value_t, Index_t>; //internal map type
    using Map_t = care::gpu::CpuReadOnlyMultiValueHashTable<kmer_type, read_number>;

    using Range_t = std::pair<const Value_t*, const Value_t*>;

    static constexpr int bits_key = sizeof(kmer_type) * 8;
	static constexpr std::uint64_t key_mask = (std::uint64_t(1) << (bits_key - 1)) | ((std::uint64_t(1) << (bits_key - 1)) - 1);
    static constexpr std::uint64_t max_read_num = std::numeric_limits<Index_t>::max();
    static constexpr int maximum_kmer_length = max_k<kmer_type>::value;

    struct Handle{        
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

    int kmerSize;
    int resultsPerMapThreshold;
    read_number nReads;
	std::vector<std::unique_ptr<Map_t>> minhashTables;

    // Minhasher();

    // Minhasher(const MinhashOptions& parameters);

    Minhasher() : Minhasher(16, 50){

    }

    Minhasher(int kmerSize, int resultsPerMapThreshold)
        : kmerSize(kmerSize), resultsPerMapThreshold(resultsPerMapThreshold){

    }

    Minhasher(const Minhasher&) = delete;
    Minhasher& operator=(const Minhasher&) = delete;

    Minhasher(Minhasher&& rhs);
    Minhasher& operator=(Minhasher&& rhs);

    bool operator==(const Minhasher& rhs) const;

    bool operator!=(const Minhasher& rhs) const;

    void construct(
        const FileOptions& fileOptions,
        const RuntimeOptions& runtimeOptions,
        const MemoryOptions& memoryOptions,
        std::uint64_t nReads,
        const CorrectionOptions& correctionOptions,
        care::cpu::ContiguousReadStorage& readStorage
    );

    int getNumberOfMaps() const{
        return minhashTables.size();
    }

    int getKmerSize() const{
        return kmerSize;
    }

    int getNumResultsPerMapThreshold() const{
        return resultsPerMapThreshold;
    }

    MemoryUsage getMemoryInfo() const;

    void writeToStream(std::ostream& os) const;

    void loadFromStream(std::ifstream& is);

	void init(std::uint64_t nReads);
    void initMap(int map);
    void moveassignMap(int mapId, Map_t&& newMap);

	void clear();

	void destroy();

    std::pair<const Value_t*, const Value_t*> queryMap(int mapid,
                                                        Key_t key) const noexcept;



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

    void queryPrecalculatedSignatures(
        const std::uint64_t* signatures, //maximum_number_of_maps signatures per sequence
        Minhasher::Range_t* ranges, //maximum_number_of_maps signatures per sequence
        int* totalNumResultsInRanges, 
        int numSequences) const;

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

    //std::int64_t getNumberOfCandidates(const std::string& sequence,
    //                                    int num_hits) const noexcept;
    //
    //std::int64_t getNumberOfCandidatesUpperBound(const std::string& sequence) const noexcept;

//###################################################

    int getResultsPerMapThreshold() const;

	void resize(std::uint64_t nReads_);


private:

    std::array<std::uint64_t, maximum_number_of_maps>
    minhashfunc(const std::string& sequence) const noexcept{
        return minhashfunc(sequence, getNumberOfMaps());
    }

    std::array<std::uint64_t, maximum_number_of_maps> 
    minhashfunc(const char* sequence, int sequenceLength) const noexcept{
        return minhashfunc(sequence, sequenceLength, getNumberOfMaps());
    }


	std::array<std::uint64_t, maximum_number_of_maps>
    minhashfunc(const std::string& sequence, int numHashfuncs) const noexcept;

    std::array<std::uint64_t, maximum_number_of_maps> 
    minhashfunc(const char* sequence, int sequenceLength, int numHashfuncs) const noexcept;

    void insertIntoMap(int map, std::uint64_t hashValue, read_number readNumber);
    void insertIntoExternalTable(Minhasher::Map_t& table, std::uint64_t hashValue, read_number readnum) const;
};



int calculateResultsPerMapThreshold(int coverage);




}

#endif
