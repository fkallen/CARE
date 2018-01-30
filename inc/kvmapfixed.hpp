#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

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

	bool set(std::uint64_t index, key_t key){
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

	std::vector<std::uint64_t> getByKey(key_t key){
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

		//sort indices by keys
		values = std::make_unique<std::uint64_t[]>(size);
		nValues = size;
		std::iota(values.get(), values.get() + size, std::uint64_t(0));
		std::sort(values.get(), values.get() + size, [&](auto a, auto b)->bool{
			return keys[a] < keys[b];
		});

		//sort keys
		std::sort(keys.get(), keys.get() + size);

		std::unique_ptr<std::uint64_t[]> counts = std::make_unique<std::uint64_t[]>(size);
		std::memset(counts.get(), 0, sizeof(std::uint64_t) * size);

		//make keys unique and count frequency of each key
		std::uint64_t unique_end = 1;
		counts[0]++;
		std::uint64_t prev = 0;
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
		counts.reset();

		//shrink keys array
		std::unique_ptr<std::uint64_t[]> tmp = std::make_unique<std::uint64_t[]>(unique_end);
		std::memcpy(tmp.get(), keys.get(), sizeof(std::uint64_t) * unique_end);
		keys = std::move(tmp);

		nKeys = unique_end;
	}
};
