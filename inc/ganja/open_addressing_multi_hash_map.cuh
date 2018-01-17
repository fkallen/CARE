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

	struct compact_map_elem{
		index_t* values;
		index_t count;
	};

    struct compact_map{
	std::unique_ptr<compact_map_elem[]> elems;
	std::unique_ptr<index_t[]> keys;
	std::unique_ptr<index_t[]> valuebuffer;
	index_t nBuckets = 0;
	index_t nValues = 0;
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

	if(isTransformer){
		if (key >= entry_t::mask_key)
		    return {};

		const bool exists = std::binary_search(transformedData.keys.get(), 
						    transformedData.keys.get() + transformedData.nBuckets, 
						    key);

		assert(exists);

		//search key in sorted array. get index in array to find bucket number. get values in this bucket
		auto range = std::equal_range(transformedData.keys.get(), 
					    transformedData.keys.get() + transformedData.nBuckets, 
					    key);
		index_t index = range.first - transformedData.keys.get();
		return {transformedData.elems[index].values, transformedData.elems[index].values + transformedData.elems[index].count};
	}

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
	out.write(reinterpret_cast<const char*>(&transformedData.nBuckets), sizeof(index_t));
	out.write(reinterpret_cast<const char*>(&transformedData.nValues), sizeof(index_t));
	out.write(reinterpret_cast<const char*>(&isTransformer), sizeof(bool));
	for(index_t i = 0; i < capacity; i++){
		out.write(reinterpret_cast<const char*>(&data[i]), sizeof(atomic_t));
	}
	for(index_t i = 0; i < transformedData.nBuckets; i++){
		out.write(reinterpret_cast<const char*>(&(transformedData.elems[i])), sizeof(compact_map_elem));
	}
	for(index_t i = 0; i < transformedData.nBuckets; i++){
		out.write(reinterpret_cast<const char*>(&(transformedData.keys[i])), sizeof(index_t));
	}
	for(index_t i = 0; i < transformedData.nValues; i++){
		out.write(reinterpret_cast<const char*>(&(transformedData.valuebuffer[i])), sizeof(index_t));
	}
    }

    bool loadFromFile(std::string filename){
	std::ifstream in(filename, std::ios::binary);
	if(in){
		in.read(reinterpret_cast<char*>(&capacity), sizeof(index_t));
		in.read(reinterpret_cast<char*>(&probe_length), sizeof(index_t));
		in.read(reinterpret_cast<char*>(&size), sizeof(std::atomic<index_t>));
		in.read(reinterpret_cast<char*>(&transformedData.nBuckets), sizeof(index_t));
		in.read(reinterpret_cast<char*>(&transformedData.nValues), sizeof(index_t));
		in.read(reinterpret_cast<char*>(&isTransformer), sizeof(bool));

		data.reset();
		transformedData.elems.reset();
		transformedData.keys.reset();
		transformedData.valuebuffer.reset();

		data.reset(new atomic_t[capacity]);
		transformedData.elems.reset(new compact_map_elem[transformedData.nBuckets]);
		transformedData.keys.reset(new index_t[transformedData.nBuckets]);
		transformedData.valuebuffer.reset(new index_t[transformedData.nValues]);

		for(index_t i = 0; i < capacity; i++){
			in.read(reinterpret_cast<char*>(&data[i]), sizeof(atomic_t));
		}
		for(index_t i = 0; i < transformedData.nBuckets; i++){
			in.read(reinterpret_cast<char*>(&(transformedData.elems[i])), sizeof(compact_map_elem));
		}
		for(index_t i = 0; i < transformedData.nBuckets; i++){
			in.read(reinterpret_cast<char*>(&(transformedData.keys[i])), sizeof(index_t));
		}

		index_t* ptr = transformedData.valuebuffer.get();
		for(index_t i = 0; i < transformedData.nBuckets; i++){
			transformedData.elems[i].values = ptr;
			ptr += transformedData.elems[i].count;
		}

		for(index_t i = 0; i < transformedData.nValues; i++){
			in.read(reinterpret_cast<char*>(&(transformedData.valuebuffer[i])), sizeof(index_t));
		}
		return true;
	}
	return false;
    }

	void transform(){
		if(isTransformer) return;
		if(size==0) return;

		//if SAFE_TRANSFORMATION, transformed map uses a copy of the key-value pairs. this allows to check transformed map for correctness
		//if !SAFE_TRANSFORMATION, transformed map uses the key-value pairs directly. less memory consumption. no correctness check
		constexpr bool SAFE_TRANSFORMATION = false;

		std::unique_ptr<entry_t[]> copy;

		entry_t* workingdata;

		if(SAFE_TRANSFORMATION){
			copy = std::make_unique<entry_t[]>(capacity);
			std::memcpy((void*)&copy[0], (void*)&data[0], sizeof(entry_t) * capacity);
			workingdata = (entry_t*)&copy[0];
		}else{
			atomic_t* oldptr = data.release();
			index_t* ptr = reinterpret_cast<index_t*>(oldptr);
			transformedData.valuebuffer.reset(ptr);
			workingdata = (entry_t*)&transformedData.valuebuffer[0];
		}

		//send all empty slots to the back
		//sort non empty slots by key
		TIMERSTARTCPU(partition_and_sort);
		
		entry_t* empty_begin = std::partition(&workingdata[0], 
						      &workingdata[capacity], 
						      [](auto a){return a != entry_t::get_empty();});

		index_t nNonEmptySlots = empty_begin - workingdata;

		std::sort((index_t*)&workingdata[0], 
			  (index_t*)empty_begin, [](index_t entry_a, index_t entry_b){
			  return ((entry_t*)&entry_a)->get_key() < ((entry_t*)&entry_b)->get_key();
		});

		TIMERSTOPCPU(partition_and_sort);

		if(SAFE_TRANSFORMATION){
			transformedData.valuebuffer = std::make_unique<index_t[]>(nNonEmptySlots);
		}
		transformedData.valueptr = transformedData.valuebuffer.get();

		if(SAFE_TRANSFORMATION){
			TIMERSTARTCPU(copynonemptyslots);
			std::memcpy(transformedData.valuebuffer.get(), (index_t*)workingdata, sizeof(index_t) * nNonEmptySlots);
			TIMERSTOPCPU(copynonemptyslots);
		}

		//count number of unique keys
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


		transformedData.elems = std::make_unique<compact_map_elem[]>(uniqueCount);
		transformedData.keys = std::make_unique<index_t[]>(uniqueCount);
		transformedData.nBuckets = uniqueCount;
		transformedData.nValues = nNonEmptySlots;
		std::memset(transformedData.elems.get(), 0, sizeof(compact_map_elem) * uniqueCount);

		//let buckets point to the correct values

		TIMERSTARTCPU(setbucketpointers);

		index_t previousKey = ((entry_t*)&transformedData.valuebuffer[0])->get_key();
		index_t currentBucket = 0;
		//add first value to first bucket
		transformedData.elems[currentBucket].values = &transformedData.valuebuffer[0];
		transformedData.keys[currentBucket] = previousKey;
		transformedData.elems[currentBucket].count++;
		transformedData.valuebuffer[0] = ((entry_t*)&transformedData.valuebuffer[0])->get_val();
		//add remaining values to buckets
		for(index_t i = 1; i < nNonEmptySlots; i++){

			entry_t entry = *((entry_t*)&transformedData.valuebuffer[i]);
			index_t key = entry.get_key();

			if(previousKey != key){
				currentBucket++;
				transformedData.elems[currentBucket].values = &transformedData.valuebuffer[i];
				transformedData.keys[currentBucket] = key;		
			}
			transformedData.elems[currentBucket].count++;
			transformedData.valuebuffer[i] = entry.get_val();
			previousKey = key;
		}

		TIMERSTOPCPU(setbucketpointers);

		if(SAFE_TRANSFORMATION){
			TIMERSTARTCPU(check);
			// check transformed data
			for(size_t i = 0; i < uniqueCount; i++){
				index_t key = transformedData.keys[i];

				auto origValues = get_nontransformed(key);

				if(origValues.size() != transformedData.elems[i].count)
					throw std::runtime_error("expected count: " + std::to_string(origValues.size())+ ", is :" + std::to_string(transformedData.elems[i].count));

				std::sort(origValues.begin(), origValues.end());
				std::vector<index_t> newValues(transformedData.elems[i].values, transformedData.elems[i].values + transformedData.elems[i].count);
				std::sort(newValues.begin(), newValues.end());

				for(index_t j = 0; j < origValues.size(); j++){
				
					if(newValues[j] != origValues[j])
						throw std::runtime_error("expected value: " + std::to_string(origValues[j])+ ", is :" + std::to_string(newValues[j]));
				}
			}
			TIMERSTOPCPU(check);
		}

		capacity = 0;
		isTransformer = true;
	}

};

#endif
