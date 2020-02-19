#ifndef CARE_FIXEDSIZESTORAGE_HPP
#define CARE_FIXEDSIZESTORAGE_HPP

#include <algorithm>
#include <memory>
#include <cstdint>

namespace care{


    struct FixedSizeStorage{
        std::size_t memoryLimitBytes = 0;
        std::unique_ptr<std::uint8_t[]> rawData = nullptr;
        std::size_t rawDataBytes = 0;
        std::uint8_t* currentDataPtr = nullptr;

        std::unique_ptr<std::size_t[]> elementOffsets = nullptr;
        std::int64_t maxElementsInMemory = 0;    
        
        std::int64_t numStoredElementsInMemory = 0;


        FixedSizeStorage() = default;
        FixedSizeStorage(const FixedSizeStorage&) = delete;
        FixedSizeStorage& operator=(const FixedSizeStorage&) = delete;

        FixedSizeStorage(FixedSizeStorage&& rhs){
            rawData = std::exchange(rhs.rawData, nullptr);
            rawDataBytes = std::exchange(rhs.rawDataBytes, 0);
            currentDataPtr = std::exchange(rhs.currentDataPtr, nullptr);
            elementOffsets = std::exchange(rhs.elementOffsets, nullptr);
            maxElementsInMemory = std::exchange(rhs.maxElementsInMemory, 0);
            numStoredElementsInMemory = std::exchange(rhs.numStoredElementsInMemory, 0);
        }

        FixedSizeStorage& operator=(FixedSizeStorage&& rhs){
            rawData = std::exchange(rhs.rawData, nullptr);
            rawDataBytes = std::exchange(rhs.rawDataBytes, 0);
            currentDataPtr = std::exchange(rhs.currentDataPtr, nullptr);
            elementOffsets = std::exchange(rhs.elementOffsets, nullptr);
            maxElementsInMemory = std::exchange(rhs.maxElementsInMemory, 0);
            numStoredElementsInMemory = std::exchange(rhs.numStoredElementsInMemory, 0);

            return *this;
        }

        FixedSizeStorage(std::size_t memoryLimitBytes_)
                : memoryLimitBytes(memoryLimitBytes_){

            maxElementsInMemory = 1024 * 1024 * 128;
            const std::size_t memoryForOffsets = maxElementsInMemory * sizeof(std::size_t);

            std::size_t limit = memoryLimitBytes;
            if(memoryForOffsets < limit){
                elementOffsets = std::make_unique<std::size_t[]>(maxElementsInMemory);
                limit -= memoryForOffsets;

                if(limit > 0){
                    rawData = std::make_unique<std::uint8_t[]>(limit);
                    rawDataBytes = limit;
                    currentDataPtr = rawData.get();
                }else{
                    elementOffsets = nullptr;
                    rawData = nullptr;
                    rawDataBytes = 0;
                    currentDataPtr = rawData.get();
                }
            }else{
                elementOffsets = nullptr;
                rawData = nullptr;
                rawDataBytes = 0;
                currentDataPtr = rawData.get();
            }
        }

        std::int64_t getNumStoredElements() const{
            return numStoredElementsInMemory;
        }

        std::size_t getNumRawBytes() const{
            return rawDataBytes;
        }

        const std::uint8_t* getRawData() const{
            return rawData.get();
        }

        const std::size_t* getElementOffsets() const{
            return elementOffsets.get();
        }

        std::size_t getNumOccupiedRawBytes() const{
            return std::distance(rawData.get(), currentDataPtr);
        }

        void destroy(){
            rawData = nullptr;
            rawDataBytes = 0;
            currentDataPtr = nullptr;
            elementOffsets = nullptr;
            maxElementsInMemory = 0;
        }

        template<class T, class Serializer>
        bool insert(T&& element, Serializer serialize){
            if(getNumStoredElements() >= maxElementsInMemory){
                return false;
            }

            std::uint8_t* const newDataPtr = serialize(element, currentDataPtr, rawData.get() + rawDataBytes);
            if(newDataPtr != nullptr){
                elementOffsets[numStoredElementsInMemory] = std::distance(rawData.get(), currentDataPtr);

                currentDataPtr = newDataPtr;
                numStoredElementsInMemory++;
                return true;
            }else{
                return false;
            }
        }

        template<class Ptrcomparator>
        void sort(Ptrcomparator&& ptrcomparator){
            auto offsetcomparator = [&](std::size_t elementOffset1, std::size_t elementOffset2){
                return ptrcomparator(getRawData() + elementOffset1, getRawData() + elementOffset2);
            };

            std::sort(elementOffsets.get(), elementOffsets.get() + getNumStoredElements(), offsetcomparator);
        }
    };


}





#endif