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

#ifdef __NVCC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/iterator/constant_iterator.h>

#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#endif

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
			std::vector<Key_t> keys;
			std::vector<Value_t> values;
			std::vector<Index_t> countsPrefixSum;

			double load = 0.5;
			KeyIndexMap<Key_t, Index_t> keyIndexMap;

            std::size_t numBytes() const{
                return keys.size() * sizeof(Key_t)
                    + keys.size() * sizeof(Key_t)
                    + values.size() * sizeof(Value_t)
                    + countsPrefixSum.size() * sizeof(Index_t)
                    + keyIndexMap.numBytes();
            }

			KeyValueMapFixedSize() : KeyValueMapFixedSize(0){
			}

			KeyValueMapFixedSize(Index_t size_) : size(size_), nKeys(size_), nValues(size_), noMoreWrites(false){
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

			void transform(){
				if(noMoreWrites) return;
				noMoreWrites = true;
				if(size == 0) return;

		#ifndef __NVCC__
		//#if 1

				std::vector<Index_t> indices(size);
				std::iota(indices.begin(), indices.end(), Index_t(0));

				//sort indices by key. if keys are equal, sort by value
				std::sort(indices.begin(), indices.end(), [&](auto a, auto b)->bool{
					if(keys[a] == keys[b]){
						return values[a] < values[b];
					}
					return keys[a] < keys[b];
				});

				std::vector<Value_t> sortedValues(size);
				for(Index_t i = 0; i < size; i++){
					sortedValues[i] = values[indices[i]];
				}

				std::swap(sortedValues, values);

				std::sort(keys.begin(), keys.end());

				std::vector<Index_t> counts(size, 0);

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

				keys.resize(unique_end);

				//make prefix sum of counts
				countsPrefixSum.resize(unique_end+1);
				countsPrefixSum[0] = 0;
				for(Index_t i = 0; i < unique_end; i++)
					countsPrefixSum[i+1] = countsPrefixSum[i] + counts[i];

				nKeys = unique_end;
		#else

				//sort values and keys by keys on gpu. equal keys are sorted by value
				thrust::device_vector<Key_t> d_keys(size);
				thrust::device_vector<Value_t> d_vals(size);
				thrust::device_vector<Index_t> d_indices(size);

				thrust::copy(keys.begin(), keys.end(), d_keys.begin());
				thrust::copy(values.begin(), values.end(), d_vals.begin());
				thrust::sequence(d_indices.begin(), d_indices.end(), Index_t(0));

				thrust::device_ptr<Key_t> keysptr = d_keys.data();
				thrust::device_ptr<Value_t> valuesptr = d_vals.data();

				thrust::sort(d_indices.begin(),
							d_indices.end(),
							[=] __device__ (const Index_t &lhs, const Index_t &rhs) {
								if(keysptr[lhs] == keysptr[rhs]){
									return valuesptr[lhs] < valuesptr[rhs];
								}
								return keysptr[lhs] < keysptr[rhs];
				});

				thrust::copy(thrust::make_permutation_iterator(d_vals.begin(), d_indices.begin()),
							thrust::make_permutation_iterator(d_vals.begin(), d_indices.end()),
							values.begin());

				thrust::copy(thrust::make_permutation_iterator(d_keys.begin(), d_indices.begin()),
							thrust::make_permutation_iterator(d_keys.begin(), d_indices.end()),
							keys.begin());

				thrust::copy(keys.begin(), keys.end(), d_keys.begin());

				nKeys = thrust::inner_product(d_keys.begin(), d_keys.end() - 1,
								d_keys.begin() + 1,
								Index_t(1),
								thrust::plus<Key_t>(),
								thrust::not_equal_to<Key_t>());

				keys.resize(nKeys);
				keys.shrink_to_fit();

				// resize histogram storage
				thrust::device_vector<Key_t> histogram_values(nKeys);
				thrust::device_vector<Index_t> histogram_counts(nKeys);
				thrust::device_vector<Index_t> histogram_counts_prefixsum(nKeys+1, 0); //inclusive with leading zero

				// compact find the end of each bin of values
				thrust::reduce_by_key(d_keys.begin(), d_keys.end(),
								thrust::constant_iterator<Index_t>(1),
								histogram_values.begin(),
								histogram_counts.begin());

				thrust::inclusive_scan(histogram_counts.begin(), histogram_counts.end(), histogram_counts_prefixsum.begin() + 1);

				countsPrefixSum.resize(nKeys+1);

				thrust::copy(histogram_values.begin(), histogram_values.end(), keys.begin());
				thrust::copy(histogram_counts_prefixsum.begin(), histogram_counts_prefixsum.end(), countsPrefixSum.begin());
		#endif

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

    std::size_t numBytes() const{
        //return minhashTables[0]->numBytes() * minhashTables.size();
        std::size_t result = 0;
        for(const auto& m : minhashTables)
            result += m->numBytes();
        return result;
    }

	Minhasher() : Minhasher(MinhashOptions{2,16}){}

	Minhasher(const MinhashOptions& parameters)
		: minparams(parameters), nReads(0)
	{
		if(maximum_number_of_maps < minparams.maps)
			throw std::runtime_error("Minhasher: Maximum number of maps is "
									+ std::to_string(maximum_number_of_maps) + "!");

		if(maximum_kmer_length < minparams.k){
			throw std::runtime_error("Minhasher is configured for maximum kmer length of "
									+ std::to_string(maximum_kmer_length) + "!");
		}
	}

	void init(std::uint64_t nReads_){
		if(nReads_ == 0) throw std::runtime_error("Minhasher::init cannnot be called with argument 0");
		if(nReads_-1 > max_read_num)
			throw std::runtime_error("Minhasher::init: Minhasher is configured for only" + std::to_string(max_read_num) + " reads, not " + std::to_string(nReads_) + "!!!");

		nReads = nReads_;

		minhashTables.resize(minparams.maps);

		for (int i = 0; i < minparams.maps; ++i) {
			minhashTables[i].reset();
			minhashTables[i].reset(new Map_t(nReads));
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

#if 1
		std::vector<Value_t> allUniqueResults;
		std::vector<Value_t> tmp;
		//TIMERSTARTCPU(getcandrest);
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
			std::merge(entries.begin(), entries.end(), allUniqueResults.begin(), allUniqueResults.end(), tmp.begin());
			std::swap(tmp, allUniqueResults);
			auto uniqueEnd = std::unique(allUniqueResults.begin(), allUniqueResults.end());
			allUniqueResults.resize(std::distance(allUniqueResults.begin(), uniqueEnd));
		}

        allUniqueResults.resize(std::min(allUniqueResults.size(), max_number_candidates));

		//TIMERSTOPCPU(getcandrest);
		return allUniqueResults;
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

        allUniqueResults.resize(std::min(allUniqueResults.size(), max_number_candidates));

        //TIMERSTOPCPU(getcandrest);
        return allUniqueResults;
#endif
      
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





	void transform(){
		for (auto& table : minhashTables) {
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
