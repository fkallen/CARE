#ifndef GANJA_OPEN_ADDRESSING_MULTI_HASH_MAP_CUH
#define GANJA_OPEN_ADDRESSING_MULTI_HASH_MAP_CUH

#include <omp.h>
#include <atomic>
#include <algorithm>
#include <vector>
#include "data_types.cuh"
#include "atomic_helpers.cuh"

#include <fstream>
#include <memory>
#include <cassert>
#include <iostream>
#include <cstring>

template <
    typename index_t,
    index_t bits_key,
    index_t bits_val,
    typename funct_t,
    typename probe_t>
struct OpenAddressingMultiHashMap {

    typedef KeyValuePair_t<index_t,bits_key,bits_val> entry_t;
    typedef std::atomic<entry_t> atomic_t;

    struct internal_elem{
	index_t count;
	index_t key = entry_t::get_empty_payload();
	index_t* values;
    };

    struct compact_map{
	std::unique_ptr<internal_elem[]> data;
	std::unique_ptr<index_t[]> valuebuffer;
	index_t nData = 0;
	index_t nValuebuffer = 0;
	internal_elem* dataptr;
	index_t* valueptr;
    };

    static constexpr index_t bits_for_key = bits_key;
    static constexpr index_t bits_for_val = bits_val;

    static_assert(sizeof(std::atomic<index_t>) == sizeof(index_t),"");

    std::unique_ptr<atomic_t[]> data;
    funct_t hash_func;
    probe_t prob_func;
    index_t capacity;
    index_t probe_length;
    std::atomic<index_t> size = {0};

    compact_map transformedData;
    bool isTransformer = false;

    OpenAddressingMultiHashMap(){
	data.reset();
    }

    OpenAddressingMultiHashMap(const OpenAddressingMultiHashMap& other):
				hash_func(other.hash_func),
				prob_func(other.prob_func),
				capacity(other.capacity),
				probe_length(other.capacity)
{

	data.reset(new atomic_t[capacity]);

	//std::copy ( other.data , other.data + other.capacity, data );

    }


    OpenAddressingMultiHashMap(
        const index_t capacity_,
        const funct_t hash_func_,
        const probe_t prob_func_) : hash_func(hash_func_),
                                    prob_func(prob_func_),
				    capacity(capacity_),
                                    probe_length(capacity_)
 {

	data.reset(new atomic_t[capacity]);

        clear();
    }

    ~OpenAddressingMultiHashMap() {
    }

    void clear(){
        #pragma omp parallel for
        for (index_t index = 0; index < capacity; index++)
            data[index].store(entry_t::get_empty());
    }

    OpenAddressingMultiHashMap& operator=(const OpenAddressingMultiHashMap& other){

	probe_length = other.probe_length;
	hash_func = other.hash_func;
	prob_func = other.prob_func;

	if(other.capacity > capacity){
		capacity = other.capacity;
		data.reset();
		data.reset(new atomic_t[capacity]);
	}
	//std::copy ( other.data , other.data + other.capacity, data );

	return *this;
    }

    std::vector<index_t> get(
        const index_t& key) const {

        if(!isTransformer){
		return get_nontransformed(key);
	}else{
		return get_transformed(key);
	}
    }

    std::vector<index_t> get_nontransformed(
        const index_t& key) const {
	//std::cout << "get_nontransformed\n";

        std::vector<index_t> result;
	if(!isTransformer){
		index_t index = hash_func(key) % capacity;

		if (key >= entry_t::mask_key)
		    return result;

		for (index_t iters = 0; iters < probe_length; ++iters) {

		    entry_t probed = data[index].load(std::memory_order_relaxed);

		    if (probed == entry_t::get_empty())
		        return result;

		    if (probed.get_key() == key)
		        result.push_back(probed.get_val());

		    index = prob_func(index, iters, key) % capacity;
		}
	}

        return result;
    }

    std::vector<index_t> get_transformed(
        const index_t& key) const {
	//std::cout << "get_transformed\n";
#if 0
	if(isTransformer){
		index_t index = hash_func(key) % transformedData.nData;

		if (key >= entry_t::mask_key)
		    return {};

		for (index_t iters = 0; iters < transformedData.nData; ++iters) {

		    const auto& probed = transformedData.data[index];

		    if (probed.key == key){
			std::vector<index_t> result {probed.values, probed.values + probed.count};
		        return result;
		    }

		    index = prob_func(index, iters, key) % transformedData.nData;
		}
	}
#else


	if(isTransformer){
		//search key in sorted buckets(sorted by key)

		if (key >= entry_t::mask_key)
		    return {};

		internal_elem tmp;
		tmp.key = key;

		bool exists = std::binary_search(transformedData.data.get(), 
					    transformedData.data.get() + transformedData.nData, 
					    tmp,
						[](auto a, auto b){
							return a.key < b.key;
						});

		assert(exists);

		if(exists){
			auto range = std::equal_range(transformedData.data.get(), 
						      transformedData.data.get() + transformedData.nData,
						      tmp,
							[](auto a, auto b){
								return a.key < b.key;
							});
			const auto& probed = *range.first;
			std::vector<index_t> result {probed.values, probed.values + probed.count};
			return result;
		}else{
			return {};
		}
	}

#endif

        return {};
    }

    bool add(
        const index_t key,
        const index_t val) {

	if(!isTransformer){

		if (key >= entry_t::mask_key || val >= entry_t::mask_val)
		    return false;

		entry_t nil = entry_t::get_empty(), entry; entry.set_pair(key, val);
		index_t index = hash_func(key) % capacity;

		for (index_t iters = 0; iters < probe_length; ++iters) {

		    const entry_t pair = data[index].load(std::memory_order_relaxed);

		    if (pair == nil) {
		        std::atomic_compare_exchange_strong(&data[index], &nil, entry);

		        if (nil == entry_t::get_empty()){
			    //probingDistance[index] = iters;
		            return ++size;
			}
		        else
		            nil = entry_t::get_empty();
		    }

		    index = prob_func(index, iters, key) % capacity;
		}
	}

        return false;
    }

    void saveToFile(std::string filename) const{
	std::ofstream out(filename, std::ios::binary);
	out.write(reinterpret_cast<const char*>(&capacity), sizeof(index_t));
	out.write(reinterpret_cast<const char*>(&probe_length), sizeof(index_t));
	out.write(reinterpret_cast<const char*>(&size), sizeof(std::atomic<index_t>));
	out.write(reinterpret_cast<const char*>(&transformedData.nData), sizeof(index_t));
	out.write(reinterpret_cast<const char*>(&transformedData.nValuebuffer), sizeof(index_t));
	out.write(reinterpret_cast<const char*>(&isTransformer), sizeof(bool));
	for(index_t i = 0; i < capacity; i++){
		out.write(reinterpret_cast<const char*>(&data[i]), sizeof(atomic_t));
	}
	for(index_t i = 0; i < transformedData.nData; i++){
		out.write(reinterpret_cast<const char*>(&(transformedData.data[i].count)), sizeof(index_t));
		out.write(reinterpret_cast<const char*>(&(transformedData.data[i].key)), sizeof(index_t));
	}
	for(index_t i = 0; i < transformedData.nValuebuffer; i++){
		out.write(reinterpret_cast<const char*>(&(transformedData.valuebuffer[i])), sizeof(index_t));
	}
    }

    bool loadFromFile(std::string filename){
	std::ifstream in(filename, std::ios::binary);
	if(in){
		in.read(reinterpret_cast<char*>(&capacity), sizeof(index_t));
		in.read(reinterpret_cast<char*>(&probe_length), sizeof(index_t));
		in.read(reinterpret_cast<char*>(&size), sizeof(std::atomic<index_t>));
		in.read(reinterpret_cast<char*>(&transformedData.nData), sizeof(index_t));
		in.read(reinterpret_cast<char*>(&transformedData.nValuebuffer), sizeof(index_t));
		in.read(reinterpret_cast<char*>(&isTransformer), sizeof(bool));

		data.reset();
		data.reset(new atomic_t[capacity]);
		transformedData.data.reset(new internal_elem[transformedData.nData]);
		transformedData.valuebuffer.reset(new index_t[transformedData.nValuebuffer]);

		index_t* ptr = transformedData.valuebuffer.get();

		for(index_t i = 0; i < capacity; i++){
			in.read(reinterpret_cast<char*>(&data[i]), sizeof(atomic_t));
		}
		for(index_t i = 0; i < transformedData.nData; i++){
			in.read(reinterpret_cast<char*>(&(transformedData.data[i].count)), sizeof(index_t));
			in.read(reinterpret_cast<char*>(&(transformedData.data[i].key)), sizeof(index_t));
			transformedData.data[i].values = ptr;
			ptr += transformedData.data[i].count;
		}
		for(index_t i = 0; i < transformedData.nValuebuffer; i++){
			in.read(reinterpret_cast<char*>(&(transformedData.valuebuffer[i])), sizeof(index_t));
		}
		return true;
	}
	return false;
    }

	void transform(){
		if(isTransformer) return;

		std::unique_ptr<entry_t[]> copy = std::make_unique<entry_t[]>(capacity);
		std::memcpy((void*)&copy[0], (void*)&data[0], sizeof(entry_t) * capacity);

		entry_t* workingdata = (entry_t*)&copy[0];

		/*for(int i = 0; i < capacity; i++){
			std::cout << '(' << workingdata[i].get_key() << " , " << workingdata[i].get_val() << ')' << '\n';
		}
		std::cout <<'\n';*/
		TIMERSTARTCPU(partition_and_sort);
		//send all empty slots to the back
		entry_t* empty_begin = std::partition(&workingdata[0], &workingdata[capacity], [](auto a){return a != entry_t::get_empty();});

		index_t nNonEmptySlots = empty_begin - workingdata;
		//std::cout << "part done. nNonEmptySlots = " << nNonEmptySlots << '\n';

		//sort non empty slots by key
		std::sort((index_t*)&workingdata[0], (index_t*)empty_begin, [](index_t entry_a, index_t entry_b){
			return ((entry_t*)&entry_a)->get_key() < ((entry_t*)&entry_b)->get_key();
		});

		TIMERSTOPCPU(partition_and_sort);

		/*for(int i = 0; i < capacity; i++){
			std::cout << '(' << workingdata[i].get_key() << " , " << workingdata[i].get_val() << ')' << '\n';
		}*/

		

		transformedData.valuebuffer.reset(new index_t[nNonEmptySlots]);
		transformedData.valueptr = transformedData.valuebuffer.get();

		TIMERSTARTCPU(copynonemptyslots);

		std::memcpy(transformedData.valuebuffer.get(), (index_t*)workingdata, sizeof(index_t) * nNonEmptySlots);

		TIMERSTOPCPU(copynonemptyslots);

		/*std::cout << "transformeddata values\n";
		for(int i = 0; i < nNonEmptySlots; i++){
			std::cout << '(' << (((entry_t*)(&transformedData.valuebuffer[0])) + i)->get_key() << " , " << (((entry_t*)(&transformedData.valuebuffer[0])) + i)->get_val() << ')' << '\n';
		}*/

		index_t uniqueCount = 0;
		entry_t previousElem = entry_t::get_empty();

		TIMERSTARTCPU(countuniquekeys);
		for(index_t i = 0; i < nNonEmptySlots; i++){
			entry_t entry = *((entry_t*)&transformedData.valuebuffer[i]);

			if(entry.get_key() != previousElem.get_key())
				uniqueCount++;

			previousElem = entry;
		}
		TIMERSTOPCPU(countuniquekeys);

		/*TIMERSTARTCPU(stable_sort);
		//stable_sort non empty slots by bucket in new table
		std::stable_sort(transformedData.valuebuffer.get(), 
				 transformedData.valuebuffer.get() + nNonEmptySlots, 
				 [=](index_t entry_a, index_t entry_b){
				 	return ((entry_t*)&entry_a)->get_key() % uniqueCount < ((entry_t*)&entry_b)->get_key() % uniqueCount;
				 }
		);
		TIMERSTOPCPU(stable_sort);*/

		//std::cout << "unique keys " << uniqueCount << '\n';

		transformedData.data.reset(new internal_elem[uniqueCount]());
		transformedData.nData = uniqueCount;
		transformedData.nValuebuffer = nNonEmptySlots;
		transformedData.dataptr = transformedData.data.get();

		TIMERSTARTCPU(setbucketpointers);

		/*index_t previousKey = entry_t::get_empty_payload();
		index_t hv = previousKey % uniqueCount;
		for(index_t i = 0; i < nNonEmptySlots; i++){

			entry_t entry = *((entry_t*)&transformedData.valuebuffer[i]);
			index_t key = entry.get_key();

			if(previousKey != key){
				index_t index = hash_func(key) % transformedData.nData;
				index_t iters = 0;
				while(transformedData.data[index].key != entry_t::get_empty_payload()){
					index = prob_func(index, iters, key) % transformedData.nData;
					iters++;
				}

				transformedData.data[index].values = &transformedData.valuebuffer[i];
				transformedData.data[index].key = entry.get_key();
				transformedData.data[index].count++;

				hv = index;		
			}else{
				transformedData.data[hv].count++;
			}
			transformedData.valuebuffer[i] = entry.get_val();
			previousKey = entry.get_key();
		}*/

		index_t previousKey = ((entry_t*)&transformedData.valuebuffer[0])->get_key();
		index_t currentBucket = 0;
		transformedData.data[currentBucket].values = &transformedData.valuebuffer[0];
		transformedData.data[currentBucket].key = previousKey;
		transformedData.valuebuffer[0] = ((entry_t*)&transformedData.valuebuffer[0])->get_val();
		transformedData.data[currentBucket].count++;

		for(index_t i = 1; i < nNonEmptySlots; i++){

			entry_t entry = *((entry_t*)&transformedData.valuebuffer[i]);
			index_t key = entry.get_key();

			if(previousKey != key){
				currentBucket++;
				transformedData.data[currentBucket].values = &transformedData.valuebuffer[i];
				transformedData.data[currentBucket].key = key;		
			}
			transformedData.data[currentBucket].count++;
			transformedData.valuebuffer[i] = entry.get_val();
			previousKey = key;
		}

		TIMERSTOPCPU(setbucketpointers);

		/*TIMERSTARTCPU(sortbucketsbykey);
		std::sort(transformedData.data.get(), transformedData.data.get() + transformedData.nData, [](auto a, auto b){
			return a.key < b.key;
		});
		TIMERSTOPCPU(sortbucketsbykey);*/
		
		/*std::cout << "transformed : \n";
		for(int i = 0; i < uniqueCount; i++){
			internal_elem bucket = transformedData.data[i];
			std::cout << "bucket " << i << ", key " << bucket.key << ", count " << bucket.count << '\n';
			std::cout << "values\n";
			for(int k = 0; k < bucket.count; k++){
				std::cout << bucket.values[k] << '\n'; 
			}
		}*/
#if 1
		TIMERSTARTCPU(check);
		// check transformed data
		for(size_t i = 0; i < uniqueCount; i++){
			index_t key = transformedData.data[i].key;

			auto origValues = get_nontransformed(key);

			if(origValues.size() != transformedData.data[i].count)
				throw std::runtime_error("expected count: " + std::to_string(origValues.size())+ ", is :" + std::to_string(transformedData.data[i].count));

			std::sort(origValues.begin(), origValues.end());
			std::vector<index_t> newValues(transformedData.data[i].values, transformedData.data[i].values + transformedData.data[i].count);
			//for(index_t elem : newValues)
			//	elem = ((entry_t*)&elem)->get_val();
			std::sort(newValues.begin(), newValues.end());

			for(index_t j = 0; j < origValues.size(); j++){
				
				if(newValues[j] != origValues[j])
					throw std::runtime_error("expected value: " + std::to_string(origValues[j])+ ", is :" + std::to_string(newValues[j]));
			}
		}
		TIMERSTOPCPU(check);
#endif
		capacity = 0;
		isTransformer = true;
	}

};

#endif
