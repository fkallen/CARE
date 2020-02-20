#ifndef CARE_FIXEDSIZESTORAGE_HPP
#define CARE_FIXEDSIZESTORAGE_HPP

#include <algorithm>
#include <memory>
#include <cstdint>

namespace care{


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

        std::size_t getNumOccupiedRawBytes() const{
            return std::distance(elementsBegin, elementsEnd);
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
        template<class T, class Serializer>
        bool insert(T&& element, Serializer serialize){
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

        // Ptrcomparator::operator()(ptr1, ptr2)
        // compare serialized element pointed to by ptr1 with serialized element pointer to by ptr2
        template<class Ptrcomparator>
        void sort(Ptrcomparator&& ptrcomparator){
            auto offsetcomparator = [&](std::size_t elementOffset1, std::size_t elementOffset2){
                return ptrcomparator(elementsBegin + elementOffset1, elementsBegin + elementOffset2);
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