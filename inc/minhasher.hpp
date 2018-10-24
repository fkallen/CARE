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

#ifdef __NVCC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/iterator/constant_iterator.h>

#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>

#include "thrust_custom_allocators.hpp"

#include <cub/cub.cuh>
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
            bool canUseGpu = false;
			std::vector<Key_t> keys;
			std::vector<Value_t> values;
			std::vector<Index_t> countsPrefixSum;

			double load = 0.5;
			KeyIndexMap<Key_t, Index_t> keyIndexMap;

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

                instream.read(reinterpret_cast<char*>(&size), sizeof(Index_t));
                instream.read(reinterpret_cast<char*>(&nKeys), sizeof(Index_t));
                instream.read(reinterpret_cast<char*>(&nValues), sizeof(Index_t));
                instream.read(reinterpret_cast<char*>(&noMoreWrites), sizeof(bool));
                instream.read(reinterpret_cast<char*>(&canUseGpu), sizeof(bool));

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

			KeyValueMapFixedSize() : KeyValueMapFixedSize(0, false){
			}

			KeyValueMapFixedSize(Index_t size_, bool canUseGpu) : size(size_), nKeys(size_), nValues(size_), noMoreWrites(false), canUseGpu(canUseGpu){
				keys.resize(size);
				values.resize(size);
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
				noMoreWrites = true;
				if(size == 0) return;

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

                auto cpu_transformation = [&](){
                    std::vector<Index_t> indices(size);
					TIMERSTARTCPU(iota);
    				std::iota(indices.begin(), indices.end(), Index_t(0));
					TIMERSTOPCPU(iota);

					TIMERSTARTCPU(sortindices);
    				//sort indices by key. if keys are equal, sort by value
    				std::sort(indices.begin(), indices.end(), [&](auto a, auto b)->bool{
    					if(keys[a] == keys[b]){
    						return values[a] < values[b];
    					}
    					return keys[a] < keys[b];
    				});

					TIMERSTOPCPU(sortindices);

					TIMERSTARTCPU(sortedvalues);
    				std::vector<Value_t> sortedValues(size);
    				for(Index_t i = 0; i < size; i++){
    					sortedValues[i] = values[indices[i]];
    				}
    				TIMERSTOPCPU(sortedvalues);

    				std::swap(sortedValues, values);

					TIMERSTARTCPU(sortkeys);
    				std::sort(keys.begin(), keys.end());
					TIMERSTOPCPU(sortkeys);

    				std::vector<Index_t> counts(size, 0);

					TIMERSTARTCPU(unique);
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
					TIMERSTOPCPU(unique);
    				keys.resize(unique_end);

					TIMERSTARTCPU(prefixsum);
    				//make prefix sum of counts
    				countsPrefixSum.resize(unique_end+1);
    				countsPrefixSum[0] = 0;
    				for(Index_t i = 0; i < unique_end; i++)
    					countsPrefixSum[i+1] = countsPrefixSum[i] + counts[i];
					TIMERSTOPCPU(prefixsum);
    				nKeys = unique_end;
                };


#ifdef __NVCC__
    		    if(!canUseGpu){
#endif
    				cpu_transformation();
#ifdef __NVCC__
                }else{

					auto cuda_transformation = [&](bool managed){
						bool success = false;

						auto deviceAlloc = [&](void** ptr, std::size_t bytes){
							return managed ? cudaMallocManaged(ptr, bytes) : cudaMalloc(ptr, bytes);
						};

						auto deviceFree = [](void* ptr){
							return cudaFree(ptr);
						};

						Key_t* d_keys = nullptr;
						Value_t* d_vals = nullptr;
						Index_t* d_indices = nullptr;
						//histogram storage
						Key_t* d_histogram_values = nullptr;
						Index_t* d_histogram_counts = nullptr;
						Index_t* d_histogram_counts_prefixsum = nullptr;

						auto errorhandler = [&](){
							if(managed){
								std::cerr << "cuda transformation with managed memory failed." << std::endl;
							}else{
								std::cerr << "cuda transformation failed." << std::endl;
							}

							deviceFree(d_keys);
							deviceFree(d_vals);
							deviceFree(d_indices);
							deviceFree(d_histogram_values);
							deviceFree(d_histogram_counts);
							deviceFree(d_histogram_counts_prefixsum);

							cudaGetLastError();

							success = false;
						};

						DefaultDeviceAllocator defaultDeviceAllocator;
						ManagedDeviceAllocator managedDeviceAllocator;

						auto defaultAllocatorPolicy = thrust::cuda::par(defaultDeviceAllocator);
						auto managedAllocatorPolicy = thrust::cuda::par(managedDeviceAllocator);

						try{
							cudaError_t status;
							status = deviceAlloc((void**)&d_keys, sizeof(Key_t) * size);
							if(status != cudaSuccess)
								throw TransformException("Bad alloc", __LINE__, sizeof(Key_t) * size);
							status = deviceAlloc((void**)&d_vals, sizeof(Value_t) * size);
							if(status != cudaSuccess)
								throw TransformException("Bad alloc", __LINE__, sizeof(Value_t) * size);
							status = deviceAlloc((void**)&d_indices, sizeof(Index_t) * size);
							if(status != cudaSuccess)
								throw TransformException("Bad alloc", __LINE__, sizeof(Index_t) * size);

							status = cudaMemcpy(d_keys, keys.data(), sizeof(Key_t) * size, H2D);
							if(status != cudaSuccess)
								throw TransformException("Bad cudamemcpy", __LINE__, sizeof(Key_t) * size);
							status = cudaMemcpy(d_vals, values.data(), sizeof(Value_t) * size, H2D);
							if(status != cudaSuccess)
								throw TransformException("Bad cudamemcpy", __LINE__, sizeof(Value_t) * size);

							thrust::device_ptr<Key_t> d_keys_ptr = thrust::device_pointer_cast(d_keys);
							thrust::device_ptr<Value_t> d_vals_ptr = thrust::device_pointer_cast(d_vals);
							thrust::device_ptr<Index_t> d_indices_ptr = thrust::device_pointer_cast(d_indices);

							try{
								if(managed){
									thrust::sequence(managedAllocatorPolicy, d_indices_ptr, d_indices_ptr + size, Index_t(0));
								}else{
									thrust::sequence(defaultAllocatorPolicy, d_indices_ptr, d_indices_ptr + size, Index_t(0));
								}
							}catch(const thrust::system_error& e){
								std::cerr << e.what() << std::endl;
								throw TransformException("", __LINE__);
							}

							//sort indices by key. equal keys are sorted by value
							try{
								TIMERSTARTCPU(thrustsort);
								if(managed){
									thrust::sort(managedAllocatorPolicy,
													d_indices,
													d_indices + size,
													[=] __device__ (const Index_t &lhs, const Index_t &rhs) {
														if(d_keys[lhs] == d_keys[rhs]){
															return d_vals[lhs] < d_vals[rhs];
														}
														return d_keys[lhs] < d_keys[rhs];
									});
								}else{

									thrust::sort(defaultAllocatorPolicy,
													d_indices,
													d_indices + size,
													[=] __device__ (const Index_t &lhs, const Index_t &rhs) {
														if(d_keys[lhs] == d_keys[rhs]){
															return d_vals[lhs] < d_vals[rhs];
														}
														return d_keys[lhs] < d_keys[rhs];
									});
								}
								TIMERSTOPCPU(thrustsort);
							}catch(const thrust::system_error& e){
								std::cerr << e.what() << std::endl;
								throw TransformException("", __LINE__);
							}



							try{
								//sort values by indices and copy to host
								Value_t* d_vals_tmp;
								status = deviceAlloc((void**)&d_vals_tmp, sizeof(Value_t) * size);
								if(status != cudaSuccess)
									throw TransformException("Bad alloc", __LINE__, sizeof(Value_t) * size);
								thrust::device_ptr<Value_t> d_vals_tmp_ptr = thrust::device_pointer_cast(d_vals_tmp);

								thrust::copy(thrust::make_permutation_iterator(d_vals_ptr, d_indices_ptr),
											thrust::make_permutation_iterator(d_vals_ptr, d_indices_ptr + size),
											d_vals_tmp_ptr);

								thrust::copy(d_vals_tmp_ptr,
											d_vals_tmp_ptr + size,
											values.begin());

								deviceFree(d_vals_tmp);
								deviceFree(d_vals);
								d_vals = nullptr;

								//sort keys by indices
								Key_t* d_keys_tmp;
								status = deviceAlloc((void**)&d_keys_tmp, sizeof(Key_t) * size);
								if(status != cudaSuccess)
									throw TransformException("Bad alloc", __LINE__, sizeof(Key_t) * size);
								thrust::device_ptr<Key_t> d_keys_tmp_ptr = thrust::device_pointer_cast(d_keys_tmp);

								thrust::copy(thrust::make_permutation_iterator(d_keys_ptr, d_indices_ptr),
											thrust::make_permutation_iterator(d_keys_ptr, d_indices_ptr + size),
											d_keys_tmp_ptr);

								std::swap(d_keys_tmp_ptr, d_keys_ptr);
								std::swap(d_keys_tmp, d_keys);

								deviceFree(d_keys_tmp);



								/*if(managed){
									thrust::copy(managedAllocatorPolicy,
													thrust::make_permutation_iterator(d_vals_ptr, d_indices_ptr),
													thrust::make_permutation_iterator(d_vals_ptr, d_indices_ptr + size),
													values.data());


									thrust::copy(managedAllocatorPolicy,
												thrust::make_permutation_iterator(d_keys_ptr, d_indices_ptr),
												thrust::make_permutation_iterator(d_keys_ptr, d_indices_ptr + size),
												keys.data());
								}else{
									thrust::copy(defaultAllocatorPolicy,
													thrust::make_permutation_iterator(d_vals_ptr, d_indices_ptr),
													thrust::make_permutation_iterator(d_vals_ptr, d_indices_ptr + size),
													values.data());

									thrust::copy(defaultAllocatorPolicy,
												thrust::make_permutation_iterator(d_keys_ptr, d_indices_ptr),
												thrust::make_permutation_iterator(d_keys_ptr, d_indices_ptr + size),
												keys.data());
								}*/
							}catch(const thrust::system_error& e){
								std::cerr << e.what() << std::endl;
								throw TransformException("", __LINE__);
							}
							//copy sorted keys back to gpu
							//status = cudaMemcpy(d_keys, keys.data(), sizeof(Key_t) * size, H2D);
							//if(status != cudaSuccess)
							//	throw TransformException("Bad cudamemcpy", __LINE__);

							//deviceFree(d_vals);
							deviceFree(d_indices);
							//d_vals = nullptr;
							d_indices = nullptr;
							try{
								if(managed){
									nKeys = thrust::inner_product(managedAllocatorPolicy,
																d_keys, d_keys + size - 1,
																d_keys + 1,
																Index_t(1),
																thrust::plus<Key_t>(),
																thrust::not_equal_to<Key_t>());
								}else{
									nKeys = thrust::inner_product(defaultAllocatorPolicy,
																d_keys, d_keys + size - 1,
																d_keys + 1,
																Index_t(1),
																thrust::plus<Key_t>(),
																thrust::not_equal_to<Key_t>());
								}
							}catch(const thrust::system_error& e){
								std::cerr << e.what() << std::endl;
								throw TransformException("", __LINE__);
							}

							keys.resize(nKeys);
							keys.shrink_to_fit();

							status = deviceAlloc((void**)&d_histogram_values, sizeof(Key_t) * nKeys);
							if(status != cudaSuccess)
								throw TransformException("Bad alloc", __LINE__, sizeof(Key_t) * nKeys);
							status = deviceAlloc((void**)&d_histogram_counts, sizeof(Index_t) * nKeys);
							if(status != cudaSuccess)
								throw TransformException("Bad alloc", __LINE__, sizeof(Index_t) * nKeys);
							status = deviceAlloc((void**)&d_histogram_counts_prefixsum, sizeof(Index_t) * (nKeys+1));
							if(status != cudaSuccess)
								throw TransformException("Bad alloc", __LINE__, sizeof(Index_t) * (nKeys+1));

							try{
								if(managed){
									thrust::fill(managedAllocatorPolicy, d_histogram_counts_prefixsum, d_histogram_counts_prefixsum + (nKeys+1), Index_t(0));
								}else{
									thrust::fill(defaultAllocatorPolicy, d_histogram_counts_prefixsum, d_histogram_counts_prefixsum + (nKeys+1), Index_t(0));
								}
							}catch(const thrust::system_error& e){
								std::cerr << e.what() << std::endl;
								throw TransformException("", __LINE__);
							}
							// compact find the end of each bin of values
							try{
								if(managed){
									thrust::reduce_by_key(managedAllocatorPolicy,
														d_keys, d_keys + size,
															thrust::constant_iterator<Index_t>(1),
															d_histogram_values,
															d_histogram_counts);

									thrust::inclusive_scan(managedAllocatorPolicy, d_histogram_counts, d_histogram_counts + nKeys, d_histogram_counts_prefixsum + 1);
								}else{
									thrust::reduce_by_key(defaultAllocatorPolicy,
													d_keys, d_keys + size,
														thrust::constant_iterator<Index_t>(1),
														d_histogram_values,
														d_histogram_counts);

									thrust::inclusive_scan(defaultAllocatorPolicy, d_histogram_counts, d_histogram_counts + nKeys, d_histogram_counts_prefixsum + 1);
								}
							}catch(const thrust::system_error& e){
								std::cerr << e.what() << std::endl;
								throw TransformException("", __LINE__);
							}
							countsPrefixSum.resize(nKeys+1);

							status = cudaMemcpy(keys.data(), d_histogram_values, sizeof(Key_t) * nKeys, D2H);
							if(status != cudaSuccess)
								throw TransformException("Bad cudamemcpy", __LINE__, sizeof(Key_t) * nKeys);
							status = cudaMemcpy(countsPrefixSum.data(), d_histogram_counts_prefixsum, sizeof(Index_t) * (nKeys+1), D2H);
							if(status != cudaSuccess)
								throw TransformException("Bad cudamemcpy", __LINE__, sizeof(Index_t) * (nKeys+1));

							deviceFree(d_keys); CUERR;
							deviceFree(d_histogram_values); CUERR;
							deviceFree(d_histogram_counts); CUERR;
							deviceFree(d_histogram_counts_prefixsum); CUERR;

							success = true;

						}catch(const std::bad_alloc& e){
							std::cerr << e.what() << std::endl;
							errorhandler();
						}catch(const thrust::system_error& e){
							std::cerr << e.what() << std::endl;
							errorhandler();
						}catch(const TransformException& e){
							std::cerr << e.what() << std::endl;
							std::cerr << "Line: " << e.getLine() << std::endl;
							std::cerr << "Value: " << e.getValue() << std::endl;
							errorhandler();
						}catch(...){
							errorhandler();
						}
						std::cout << "Transformation done" << std::endl;
						return success;
					};

					int nGpus = 0;
					cudaGetDeviceCount(&nGpus); CUERR;
					if(nGpus > 0){
						bool success = cuda_transformation(false);
						if(!success){
							cudaDeviceProp prop;
							cudaGetDeviceProperties(&prop, 0); CUERR;

							bool isCapableOfUsingManagedMemory = prop.concurrentManagedAccess == 1;
							if(isCapableOfUsingManagedMemory){
								std::cout << "Falling back to cuda managed memory transformation" << std::endl;
								success = cuda_transformation(true);
							}
							if(!success){
								std::cout << "Falling back to cpu transformation" << std::endl;
								cpu_transformation();
							}
						}
					}else{
						cpu_transformation();
					}
				}

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
    bool canUseGpu = false;

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

	Minhasher() : Minhasher(MinhashOptions{2,16}, false){}

	Minhasher(const MinhashOptions& parameters, bool canUseGpu)
		: minparams(parameters), nReads(0), canUseGpu(canUseGpu)
	{
		if(maximum_number_of_maps < minparams.maps)
			throw std::runtime_error("Minhasher: Maximum number of maps is "
									+ std::to_string(maximum_number_of_maps) + "!");

		if(maximum_kmer_length < minparams.k){
			throw std::runtime_error("Minhasher is configured for maximum kmer length of "
									+ std::to_string(maximum_kmer_length) + "!");
		}
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
        assert(canUseGpu == canUseGpu_loaded);

        minhashTables.resize(minparams.maps);

		for (int i = 0; i < minparams.maps; ++i) {
			minhashTables[i].reset();
			minhashTables[i].reset(new Map_t(nReads, canUseGpu));
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
			minhashTables[i].reset(new Map_t(nReads, canUseGpu));
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
