#ifndef CARE_SORTBYGENERATEDKEYS_HPP
#define CARE_SORTBYGENERATEDKEYS_HPP

#include <hpc_helpers.cuh>

#include <cstdint>
#include <memory>
#include <numeric>
#include <algorithm>
#include <iostream>

#include <thrust/sort.h>

/*
    KeyType KeyGenerator::operator()(IndexType i)  returns i-th key

    bool KeyComparator::operator()(KeyType l, KeyType r)
*/

template<class IndexType, class ValueType, class KeyGenerator, class KeyComparator>
bool sortValuesByGeneratedKeys1(
    std::size_t memoryLimitBytes,
    ValueType* values,
    IndexType numValues,
    KeyGenerator keyGenerator,
    KeyComparator keyComparator
){
    using KeyType = decltype(keyGenerator(IndexType{0}));

    std::cerr << " sortValuesByGeneratedKeys1 \n";

    std::size_t sizeOfKeys = SDIV(sizeof(KeyType) * numValues, sizeof(std::size_t)) * sizeof(std::size_t);
    std::size_t sizeOfIndices = SDIV(sizeof(IndexType) * numValues, sizeof(std::size_t)) * sizeof(std::size_t);
    std::size_t sizeOfValues = SDIV(sizeof(ValueType) * numValues, sizeof(std::size_t)) * sizeof(std::size_t);

    std::size_t requiredBytes = std::max(sizeOfValues, sizeOfKeys) + std::size_t(sizeOfIndices) + sizeof(std::size_t);

    if(requiredBytes >= memoryLimitBytes){
        std::cerr << sizeOfValues << " " <<  sizeOfKeys << " " <<  sizeOfIndices << " " << memoryLimitBytes << "\n";
        return false;
    }

    auto buffer = std::make_unique<char[]>(sizeOfIndices + std::max(sizeOfValues, sizeOfKeys));
    IndexType* const indices = (IndexType*)(buffer.get());
    KeyType* const keys = (KeyType*)(((char*)indices) + sizeOfIndices);
    ValueType* const newValues = (ValueType*)(((char*)indices) + sizeOfIndices);

    helpers::CpuTimer timer1("extractkeys");

    for(IndexType i = 0; i < numValues; i++){
        keys[i] = keyGenerator(i);
    }

    timer1.stop();
    timer1.print();

    helpers::CpuTimer timer2("sort indices");

    std::iota(indices, indices + numValues, IndexType(0));

    std::sort(
        indices, indices + numValues,
        [&](const auto& l, const auto& r){
            return keyComparator(keys[l], keys[r]);
        }
    );

    timer2.stop();
    timer2.print();

    //keys are no longer used. their memory is reused by newValues

    helpers::CpuTimer timer3("permute");
    for(IndexType i = 0; i < numValues; i++){
        newValues[i] = values[indices[i]];
    }
    std::copy_n(newValues, numValues, values);
    //permute(offsetsBegin, indices.data(), indices.size());

    timer3.stop();
    timer3.print();

    return true;
}




/*
    KeyType KeyGenerator::operator()(IndexType i)  returns i-th key

    bool KeyComparator::operator()(KeyType l, KeyType r)
*/

template<class IndexType, class ValueType, class KeyGenerator, class KeyComparator>
bool sortValuesByGeneratedKeys2(
    std::size_t memoryLimitBytes,
    ValueType* values,
    IndexType numValues,
    KeyGenerator keyGenerator,
    KeyComparator keyComparator
){
    using KeyType = decltype(keyGenerator(IndexType{0}));

    std::cerr << " sortValuesByGeneratedKeys2 \n";

    std::size_t sizeOfKeys = SDIV(sizeof(KeyType) * numValues, sizeof(std::size_t)) * sizeof(std::size_t);
    std::size_t sizeOfValues = SDIV(sizeof(ValueType) * numValues, sizeof(std::size_t)) * sizeof(std::size_t);

    std::size_t requiredBytes = 2 * sizeOfKeys + sizeOfValues;

    if(requiredBytes >= memoryLimitBytes){
        std::cerr << sizeOfValues << " " <<  sizeOfKeys << " " << memoryLimitBytes << "\n";
        return false;
    }

    auto buffer = std::make_unique<char[]>(sizeOfKeys);
    KeyType* const keys = (KeyType*)(buffer.get());

    helpers::CpuTimer timer1("extractkeys");

    for(IndexType i = 0; i < numValues; i++){
        keys[i] = keyGenerator(i);
    }

    timer1.stop();
    timer1.print();

    helpers::CpuTimer timer2("sort by key");

    thrust::sort_by_key(keys, keys + numValues, values);

    timer2.stop();
    timer2.print();

    return true;
}


/*
    Sorts the values of key-value pairs by key. Keys are generated via functor
*/
template<class IndexType, class ValueType, class KeyGenerator, class KeyComparator>
bool sortValuesByGeneratedKeys(
    std::size_t memoryLimitBytes,
    ValueType* values,
    IndexType numValues,
    KeyGenerator keyGenerator,
    KeyComparator keyComparator
){

    bool success = false;

    try{
        success = sortValuesByGeneratedKeys2<IndexType>(memoryLimitBytes, values, numValues, keyGenerator, keyComparator);
    } catch (...){
        std::cerr << "Fallback\n";
    }

    if(success) return true;

    try{
        success = sortValuesByGeneratedKeys1<IndexType>(memoryLimitBytes, values, numValues, keyGenerator, keyComparator);
    } catch (...){
        std::cerr << "Fallback\n";
    }

    return success;
}




#endif