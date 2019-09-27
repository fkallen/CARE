#ifndef CARE_MINHASHER_HPP
#define CARE_MINHASHER_HPP

#include "options.hpp"
#include "hpc_helpers.cuh"
#include "util.hpp"

#include <config.hpp>

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



namespace care{

    namespace minhasherdetail{
        template<class T> struct max_k;
        template<> struct max_k<std::uint8_t>{static constexpr int value = 4;};
        template<> struct max_k<std::uint16_t>{static constexpr int value = 8;};
        template<> struct max_k<std::uint32_t>{static constexpr int value = 16;};
        template<> struct max_k<std::uint64_t>{static constexpr int value = 32;};

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

			std::vector<Pair_t> keyToIndexMap;
			std::uint64_t size;

            std::size_t numBytes() const{
                return keyToIndexMap.size() * sizeof(Pair_t);
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
                //std::uint32_t probes = 1;
                //std::uint32_t pos = murmur_integer_finalizer_hash_uint32_t(key) % size;
				while(keyToIndexMap[pos] != KeyIndexMap::EmptySlot){
					pos = (pos + 1) % size;
					probes++;
				}
				keyToIndexMap[pos].first = key;
				keyToIndexMap[pos].second = value;
                //std::cerr << "probes insert: " << probes << "\n";
			}

			Index_t get(Key_t key) const noexcept{
                std::uint64_t probes = 1;
				std::uint64_t pos = murmur_hash_3_uint64_t(key) % size;
				while(keyToIndexMap[pos].first != key){
					pos = (pos + 1) % size;
					probes++;
				}
                //std::cerr << "probes get: " << probes << "\n";
				return keyToIndexMap[pos].second;
			}

			void clear() noexcept{
				keyToIndexMap.clear();
			}

			void destroy() noexcept{
				clear();
				keyToIndexMap.shrink_to_fit();
			}

            void writeToStream(std::ofstream& outstream) const{
                outstream.write(reinterpret_cast<const char*>(&size), sizeof(std::uint64_t));
                outstream.write(reinterpret_cast<const char*>(keyToIndexMap.data()), keyToIndexMap.size() * sizeof(Pair_t));
            }

            void readFromStream(std::ifstream& instream){
                instream.read(reinterpret_cast<char*>(&size), sizeof(std::uint64_t));
                keyToIndexMap.resize(size);
                instream.read(reinterpret_cast<char*>(keyToIndexMap.data()), keyToIndexMap.size() * sizeof(Pair_t));
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
            std::vector<Key_t> keysWithoutValues;
            std::vector<int> deviceIds;

			double load = 0.8;
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
                keysWithoutValues = other.keysWithoutValues;
                keyIndexMap = other.keyIndexMap;
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
                keysWithoutValues = std::move(other.keysWithoutValues);
                keyIndexMap = std::move(other.keyIndexMap);
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
                if(keysWithoutValues != rhs.keysWithoutValues){
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

                keyIndexMap.writeToStream(outstream);
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

                keyIndexMap.readFromStream(instream);
            }

            std::size_t numBytes() const{
                return keys.size() * sizeof(Key_t)
                    + values.size() * sizeof(Value_t)
                    + countsPrefixSum.size() * sizeof(Index_t)
                    + keysWithoutValues.size() * sizeof(Key_t)
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
                keysWithoutValues.clear();
				keyIndexMap.clear();
			}

			void destroy() noexcept{
				clear();
				keys.shrink_to_fit();
				values.shrink_to_fit();
				countsPrefixSum.shrink_to_fit();
                keysWithoutValues.shrink_to_fit();
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

			std::vector<Value_t> get(Key_t key) const noexcept{
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

                auto emptyKeyIter = std::lower_bound(keysWithoutValues.begin(), keysWithoutValues.end(), key);
                if(!(emptyKeyIter != keysWithoutValues.end() && *emptyKeyIter == key)){
                    const Index_t index = keyIndexMap.get(key);

				    return {&values[countsPrefixSum[index]], &values[countsPrefixSum[index+1]]};
                }else{
                    return {}; //key has no values
                }
                
			}

            Index_t prepare_get_ranged(Key_t key) const noexcept{
                assert(noMoreWrites);
                Index_t index = keyIndexMap.get(key);
                __builtin_prefetch(countsPrefixSum.data() + index, 0, 0);
                __builtin_prefetch(countsPrefixSum.data() + index + 1, 0, 0);

				return index;
			}

            std::pair<const Value_t*, const Value_t*> execute_get_ranged(Index_t preparedIndex) const noexcept{

				return {&values[countsPrefixSum[preparedIndex]], &values[countsPrefixSum[preparedIndex+1]]};
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

                auto emptyKeyIter = std::lower_bound(keysWithoutValues.begin(), keysWithoutValues.end(), key);
                if(!(emptyKeyIter != keysWithoutValues.end() && *emptyKeyIter == key)){
                    const Index_t index = keyIndexMap.get(key);

				    return {&values[countsPrefixSum[index]], &values[countsPrefixSum[index+1]]};
                }else{
                    return {}; //key has no values
                }
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
    static constexpr int maximum_kmer_length = minhasherdetail::max_k<kmer_type>::value;

    struct Handle{
		std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;
	};

	// the actual maps
	std::vector<std::unique_ptr<Map_t>> minhashTables;
	MinhashOptions minparams;
    read_number nReads;
    bool canUseGpu = false;
    std::vector<int> deviceIds;
    bool allowUVM = false;

    Minhasher();

    Minhasher(const MinhashOptions& parameters);

	Minhasher(const MinhashOptions& parameters, const std::vector<int>& deviceIds);

    Minhasher(const Minhasher&) = delete;
    Minhasher& operator=(const Minhasher&) = delete;

    Minhasher(Minhasher&& rhs);
    Minhasher& operator=(Minhasher&& rhs);

    bool operator==(const Minhasher& rhs) const;

    bool operator!=(const Minhasher& rhs) const;

    std::size_t numBytes() const;



    void saveToFile(const std::string& filename) const;

    void loadFromFile(const std::string& filename);

	void init(std::uint64_t nReads);
    void initMap(int map);

	void clear();

	void destroy();

    void insertSequence(const std::string& sequence, read_number readnum, std::vector<int> mapIds);

	void insertSequence(const std::string& sequence, read_number readnum);


    std::pair<const Value_t*, const Value_t*> queryMap(int mapid,
                                                        Map_t::Key_t key) const noexcept;


    std::vector<Result_t> getCandidates(const std::string& sequence,
                                        int num_hits,
                                        std::uint64_t max_number_candidates) const noexcept;

    std::vector<Result_t> getCandidates_any_map(const std::string& sequence,
                                        std::uint64_t max_number_candidates) const noexcept;

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
	void minhashfunc(const std::string& sequence, std::uint64_t* minhashSignature, bool* isForwardStrand) const noexcept;

    void minhashfunc_other(const std::string& sequence, std::uint64_t* minhashSignature, bool* isForwardStrand) const noexcept;

    void insertTupleIntoMap(int map, const std::uint64_t* hashValues, read_number readNumber);
};



int calculateResultsPerMapThreshold(int coverage);




}

#endif
