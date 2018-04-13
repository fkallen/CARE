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

namespace care{

//store size keys
template<class key_t, class value_t, class index_t>
struct KVMapFixed{
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

	KVMapFixed() : KVMapFixed(0){
	}

	KVMapFixed(Index_t size_) : size(size_), nKeys(size_), nValues(size_), noMoreWrites(false){
		keys.resize(size);
        values.resize(size);
	}

	void print(std::ostream& stream){
		stream << size << " " << nKeys << " " << nValues << " " << noMoreWrites << '\n';
		stream << "keys:\n";
		for(Index_t i = 0; i < nKeys; i++)
			stream << keys[i] << '\n';
		stream << "values:\n";
		for(Index_t i = 0; i < nValues; i++)
			stream << values[i] << '\n';
		stream << "countsPrefixSum:\n";
		for(Index_t i = 0; nValues != 0 && i < nValues + 1; i++)
			stream << countsPrefixSum[i] << '\n';
	}

	void clear(){
		size = 0;
		nKeys = 0;
		nValues = 0;
		noMoreWrites = false;
		keys.clear();
		values.clear();
		countsPrefixSum.clear();
	}

	void clearAndResize(Index_t size_){
		clear();
		size = size_;
		nKeys = size_;
        nValues = size_;
        keys.resize(size_);
        values.resize(size_);
	}

	bool add(Key_t key, Value_t value, Index_t index){
		if(index >= size){
			std::cout << "KVMapFixed: want to set index " << index << " but size is " << size << '\n';
			return false;
		}

		if(!noMoreWrites){
			keys[index] = key;
            values[index] = value;
			return true;
		}else{
			std::cout << "KVMapFixed: want to set " << index << " but no more writes allowed\n";
			return false;
		}
	}

	std::vector<Value_t> get(Key_t key){
		if(!noMoreWrites){
			std::cout << "KVMapFixed: want to get key " << key << " but writes are still allowed. Need to call freeze() beforehand\n";
			return {};
		}

		auto range = std::equal_range(keys.begin(), keys.end(), key);
		if(range.first == keys.end()) return {};

		Index_t index = std::distance(keys.begin(), range.first);
		return {&values[countsPrefixSum[index]], &values[countsPrefixSum[index+1]]};
	}

	void freeze(){
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
	}
};

}
