#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#ifdef __NVCC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/iterator/constant_iterator.h>

#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#endif

//store size keys
template<class key_t>
struct KVMapFixed{
	std::uint64_t size;
	std::uint64_t nKeys;
	std::uint64_t nValues;
	bool noMoreWrites;
	std::unique_ptr<key_t[]> keys;
	std::unique_ptr<std::uint64_t[]> values;
	std::unique_ptr<std::uint64_t[]> countsPrefixSum;

	KVMapFixed() : KVMapFixed(0){
	}

	KVMapFixed(std::uint64_t size_) : size(size_), nKeys(size_), nValues(0), noMoreWrites(false){
		keys = std::make_unique<key_t[]>(size);
	}

	void print(std::ostream& stream){
		stream << size << " " << nKeys << " " << nValues << " " << noMoreWrites << '\n';
		stream << "keys:\n";
		for(std::uint64_t i = 0; i < nKeys; i++)
			stream << keys[i] << '\n';
		stream << "values:\n";
		for(std::uint64_t i = 0; i < nValues; i++)
			stream << values[i] << '\n';
		stream << "countsPrefixSum:\n";
		for(std::uint64_t i = 0; nValues != 0 && i < nValues + 1; i++)
			stream << countsPrefixSum[i] << '\n';
	}

	void clear(){
		size = 0;
		nKeys = 0;
		nValues = 0;
		noMoreWrites = false;
		keys.reset();
		values.reset();
		countsPrefixSum.reset();
	}

	void clearAndResize(std::uint64_t size_){
		clear();
		size = size_;
		nKeys = size_;
		keys = std::make_unique<key_t[]>(size);
	}

	bool add(key_t key, std::uint64_t index){
		if(index >= size){
			std::cout << "KVMapFixed: want to set index " << index << " but size is " << size << '\n';
			return false;
		}

		if(!noMoreWrites){
			keys[index] = key;
			return true;
		}else{
			std::cout << "KVMapFixed: want to set " << index << " but no more writes allowed\n";
			return false;
		}
	}

	std::vector<std::uint64_t> get(key_t key){
		if(!noMoreWrites){
			std::cout << "KVMapFixed: want to get key " << key << " but writes are still allowed. Need to call freeze() beforehand\n";
			return {};
		}
		std::uint64_t* first = keys.get();
		std::uint64_t* last = first + nKeys;

		auto range = std::equal_range(first, last, key);
		if(range.first == last) return {};

		std::uint64_t index = range.first - keys.get();
		return {&values[countsPrefixSum[index]], &values[countsPrefixSum[index+1]]};
	}

	void freeze(){
		if(noMoreWrites) return;
		noMoreWrites = true;
		if(size == 0) return;

		values = std::make_unique<std::uint64_t[]>(size);
		nValues = size;

#ifndef __NVCC__
//#if 1
		//sort indices and keys by keys
		std::iota(values.get(), values.get() + size, std::uint64_t(0));

		std::sort(values.get(), values.get() + size, [&](auto a, auto b)->bool{
			return keys[a] < keys[b];
		});

		std::sort(keys.get(), keys.get() + size);

		std::unique_ptr<std::uint64_t[]> counts = std::make_unique<std::uint64_t[]>(size);
		std::memset(counts.get(), 0, sizeof(std::uint64_t) * size);

		//make keys unique and count frequency of each key
		std::uint64_t unique_end = 1;
		counts[0]++;
		std::uint64_t prev = keys[0];
		for(std::uint64_t i = 1; i < size; i++){
			std::uint64_t cur = keys[i];
			if(cur == prev){
				counts[unique_end-1]++;
			}else{
				keys[unique_end] = cur;
				counts[unique_end]++;
				unique_end++;
			}
			prev = cur;
		}

		//make prefix sum of counts
		countsPrefixSum = std::make_unique<std::uint64_t[]>(unique_end+1);		
		countsPrefixSum[0] = 0;
		for(std::uint64_t i = 0; i < unique_end; i++)
			countsPrefixSum[i+1] = countsPrefixSum[i] + counts[i];

		nKeys = unique_end;
#else
		//sort indices and keys by keys on gpu
		thrust::device_vector<std::uint64_t> d_keys(size);		
		thrust::device_vector<std::uint64_t> d_vals(size);

		thrust::copy(keys.get(), keys.get() + size, d_keys.begin());
		thrust::sequence(d_vals.begin(), d_vals.end());

		thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

		thrust::copy(d_vals.begin(), d_vals.end(), values.get());

		nKeys = thrust::inner_product(d_keys.begin(), d_keys.end() - 1,
						d_keys.begin() + 1,
						std::uint64_t(1),
						thrust::plus<std::uint64_t>(),
						thrust::not_equal_to<std::uint64_t>());

		// resize histogram storage
		thrust::device_vector<std::uint64_t> histogram_values(nKeys);
		thrust::device_vector<std::uint64_t> histogram_counts(nKeys);
		thrust::device_vector<std::uint64_t> histogram_counts_prefixsum(nKeys+1, 0); //inclusive with leading zero
		  
		// compact find the end of each bin of values
		thrust::reduce_by_key(d_keys.begin(), d_keys.end(),
				        thrust::constant_iterator<std::uint64_t>(1),
				        histogram_values.begin(),
				        histogram_counts.begin());

		thrust::inclusive_scan(histogram_counts.begin(), histogram_counts.end(), histogram_counts_prefixsum.begin() + 1);

		countsPrefixSum = std::make_unique<std::uint64_t[]>(nKeys+1);

		thrust::copy(histogram_values.begin(), histogram_values.end(), keys.get());
		thrust::copy(histogram_counts_prefixsum.begin(), histogram_counts_prefixsum.end(), countsPrefixSum.get());
#endif
	}
};
