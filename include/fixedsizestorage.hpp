#ifndef CARE_FIXEDSIZESTORAGE_HPP
#define CARE_FIXEDSIZESTORAGE_HPP

#include <algorithm>
#include <memory>
#include <cstdint>
#include <util.hpp>

#include <hpc_helpers.cuh>
#include <sortbygeneratedkeys.hpp>

namespace care{

    template<class T>
    struct FixedSizeStorage{
        std::size_t memoryLimitBytes = 0;
        std::unique_ptr<std::size_t[]> rawData = nullptr;

        // offsets and elements share the same memory in rawData
        // elements grow from left to right
        // offsets grow from right to left

        std::uint8_t* elementsBegin = nullptr;
        std::uint8_t* elementsEnd = nullptr;

        std::size_t* offsetsBegin = nullptr;
        std::size_t* offsetsEnd = nullptr; 
        
        std::int64_t numStoredElementsInMemory = 0;


        FixedSizeStorage() = default;
        FixedSizeStorage(const FixedSizeStorage&) = delete;
        FixedSizeStorage& operator=(const FixedSizeStorage&) = delete;

        FixedSizeStorage(FixedSizeStorage&& rhs){
            memoryLimitBytes = std::exchange(rhs.memoryLimitBytes, 0);

            rawData = std::exchange(rhs.rawData, nullptr);
            elementsBegin = std::exchange(rhs.elementsBegin, nullptr);
            elementsEnd = std::exchange(rhs.elementsEnd, nullptr);
            offsetsBegin = std::exchange(rhs.offsetsBegin, nullptr);
            offsetsEnd = std::exchange(rhs.offsetsEnd, nullptr);

            numStoredElementsInMemory = std::exchange(rhs.numStoredElementsInMemory, 0);
        }

        FixedSizeStorage& operator=(FixedSizeStorage&& rhs){
            memoryLimitBytes = std::exchange(rhs.memoryLimitBytes, 0);
            
            rawData = std::exchange(rhs.rawData, nullptr);
            elementsBegin = std::exchange(rhs.elementsBegin, nullptr);
            elementsEnd = std::exchange(rhs.elementsEnd, nullptr);
            offsetsBegin = std::exchange(rhs.offsetsBegin, nullptr);
            offsetsEnd = std::exchange(rhs.offsetsEnd, nullptr);

            numStoredElementsInMemory = std::exchange(rhs.numStoredElementsInMemory, 0);

            return *this;
        }

        FixedSizeStorage(std::size_t memoryLimitBytes_)
                : memoryLimitBytes(memoryLimitBytes_){

            const std::size_t numSizeTs = memoryLimitBytes / sizeof(std::size_t);

            rawData = std::make_unique<std::size_t[]>(numSizeTs);

            elementsBegin = reinterpret_cast<std::uint8_t*>(rawData.get());
            elementsEnd = elementsBegin;
            offsetsEnd = rawData.get() + numSizeTs;
            offsetsBegin = offsetsEnd;
        }

        std::int64_t getNumStoredElements() const{
            return numStoredElementsInMemory;
        }

        const std::uint8_t* getElementsData() const{
            return elementsBegin;
        }

        const std::size_t* getElementOffsets() const{
            return offsetsBegin;
        }

        std::size_t getNumOccupiedRawElementBytes() const{
            return std::distance(elementsBegin, elementsEnd);
        }

        std::size_t getNumOccupiedRawOffsetBytes() const{
            return std::distance(offsetsEnd, offsetsBegin);
        }

        std::size_t getSizeInBytes() const{
            return memoryLimitBytes;
        }

        void clear(){
            const std::size_t numSizeTs = memoryLimitBytes / sizeof(std::size_t);

            numStoredElementsInMemory = 0;
            elementsBegin = reinterpret_cast<std::uint8_t*>(rawData.get());
            elementsEnd = elementsBegin;
            offsetsEnd = rawData.get() + numSizeTs;
            offsetsBegin = offsetsEnd;
        }

        void destroy(){
            memoryLimitBytes = 0;
            rawData = nullptr;
            elementsBegin = nullptr;
            elementsEnd = nullptr;
            offsetsEnd = nullptr;
            offsetsBegin = nullptr;
            numStoredElementsInMemory = 0;
        }

        // Serializer::operator()(element, begin, end)
        // serialized element fits into range [begin, end): copy serialized elements into range starting at begin. 
        //   return pointer to position after last written byte
        // serialized element does not fit into range: do not modify range. return nullptr
        template<class Serializer>
        bool insert(const T& element, Serializer serialize){
            std::size_t* newOffsetsBegin = offsetsBegin - 1;

            //check that new offset does not reach into element buffer
            if(((void*)elementsEnd) > ((void*)newOffsetsBegin)){
                return false;
            }

            std::uint8_t* const newDataPtr = serialize(element, elementsEnd, (std::uint8_t*)newOffsetsBegin);

            if(newDataPtr != nullptr){
                offsetsBegin = newOffsetsBegin;
                *offsetsBegin = std::distance(elementsBegin, elementsEnd);

                elementsEnd = newDataPtr;
                numStoredElementsInMemory++;
                return true;
            }else{
                return false;
            }
        }

        template<class Serializer>
        bool insert(const T* element, Serializer serialize){
            std::size_t* newOffsetsBegin = offsetsBegin - 1;

            //check that new offset does not reach into element buffer
            if(((void*)elementsEnd) > ((void*)newOffsetsBegin)){
                return false;
            }

            std::uint8_t* const newDataPtr = serialize(*element, elementsEnd, (std::uint8_t*)newOffsetsBegin);

            if(newDataPtr != nullptr){
                offsetsBegin = newOffsetsBegin;
                *offsetsBegin = std::distance(elementsBegin, elementsEnd);

                elementsEnd = newDataPtr;
                numStoredElementsInMemory++;
                return true;
            }else{
                return false;
            }
        }

        bool insert(const FixedSizeStorage<T>& other){
            std::size_t otherBytes = other.getNumOccupiedRawElementBytes() + other.getNumOccupiedRawOffsetBytes();

            std::size_t myFreeBytes = std::distance((const char*)elementsEnd, (const char*)offsetsBegin);
            myFreeBytes -= sizeof(std::size_t); //make sure that offsets can be stored at size_t boundary

            if(myFreeBytes >= otherBytes){
                numStoredElementsInMemory += other.getNumStoredElements();
                elementsEnd = std::copy(
                    other.elementsBegin,
                    other.elementsEnd,
                    elementsEnd
                );

                std::size_t* newOffsetsBegin = offsetsBegin - other.getNumStoredElements();

                assert(((void*)elementsEnd) <= ((void*)newOffsetsBegin));

                std::copy(
                    other.offsetsBegin,
                    other.offsetsEnd,
                    newOffsetsBegin
                );

                offsetsBegin = newOffsetsBegin;

                return true;
            }else{
                return false;
            }
        }



        template<class ExtractKey, class KeyComparator>
        void sort(std::size_t memoryForSortingInBytes, ExtractKey extractKey, KeyComparator keyComparator){
            using Key = decltype(extractKey(nullptr));

            if(getNumStoredElements() == 0) return;
#if 1
            bool sortValuesSuccess = false;

            auto keyGenerator = [&](auto i){
                const std::uint8_t* ptr = elementsBegin + offsetsBegin[i];
                const Key key = extractKey(ptr);
                return key;
            };

            try{

                if(std::size_t(getNumStoredElements()) <= std::size_t(std::numeric_limits<std::uint32_t>::max())){
                    sortValuesSuccess = sortValuesByGeneratedKeys<std::uint32_t>(
                        memoryForSortingInBytes,
                        offsetsBegin,
                        getNumStoredElements(),
                        keyGenerator,
                        keyComparator
                    );
                }else{
                    sortValuesSuccess = sortValuesByGeneratedKeys<std::uint64_t>(
                        memoryForSortingInBytes,
                        offsetsBegin,
                        getNumStoredElements(),
                        keyGenerator,
                        keyComparator
                    );
                }

            } catch (...){
                std::cerr << "Final fallback\n";
            }

            if(sortValuesSuccess) return;
#endif
            auto offsetcomparator = [&](std::size_t elementOffset1, std::size_t elementOffset2){
                const std::uint8_t* ptr1 = elementsBegin + elementOffset1;
                const std::uint8_t* ptr2 = elementsBegin + elementOffset2;
                const Key key1 = extractKey(ptr1);
                const Key key2 = extractKey(ptr2);

                return keyComparator(key1, key2);
            };

            std::sort(offsetsBegin, offsetsEnd, offsetcomparator);
        }

        //Func::operator()(const std::uint8_t* ptr) ptr points to first byte of object
        template<class Func>
        void forEachPointer(Func&& consume){
            for(std::int64_t i = 0; i < getNumStoredElements(); i++){
                const std::size_t offset = getElementOffsets()[i];
                const std::uint8_t* ptr = getElementsData() + offset;

                consume(ptr);
            }
        }
        
    };

}





#endif