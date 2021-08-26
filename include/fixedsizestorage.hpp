#ifndef CARE_FIXEDSIZESTORAGE_HPP
#define CARE_FIXEDSIZESTORAGE_HPP

#include <algorithm>
#include <memory>
#include <cstdint>
#include <util.hpp>

#include <hpc_helpers.cuh>
#include <sortbygeneratedkeys.hpp>

#include <mmapbuffer.hpp>

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

        void saveToStream(std::ostream& stream) {
            const std::size_t numSizeTs = memoryLimitBytes / sizeof(std::size_t);

            stream.write(reinterpret_cast<const char*>(&memoryLimitBytes), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&numSizeTs), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(rawData.get()), sizeof(std::size_t) * numSizeTs);

            std::size_t distanceEB = std::distance((char*)rawData.get(), (char*)elementsBegin);
            std::size_t distanceEE = std::distance((char*)rawData.get(), (char*)elementsEnd);
            std::size_t distanceOB = std::distance((char*)rawData.get(), (char*)offsetsBegin);
            std::size_t distanceOE = std::distance((char*)rawData.get(), (char*)offsetsEnd);

            stream.write(reinterpret_cast<const char*>(&distanceEB), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&distanceEE), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&distanceOB), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&distanceOE), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&numStoredElementsInMemory), sizeof(std::int64_t));

            stream.flush();
        }

        void loadFromStream(std::istream& stream){
            std::size_t numSizeTs = 0;
            stream.read(reinterpret_cast<char*>(&memoryLimitBytes), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&numSizeTs), sizeof(std::size_t));

            rawData = nullptr;
            rawData = std::make_unique<std::size_t[]>(numSizeTs);

            stream.read(reinterpret_cast<char*>(rawData.get()), sizeof(std::size_t) * numSizeTs);

            std::size_t distanceEB = 0;
            std::size_t distanceEE = 0;
            std::size_t distanceOB = 0;
            std::size_t distanceOE = 0;

            stream.read(reinterpret_cast<char*>(&distanceEB), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&distanceEE), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&distanceOB), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&distanceOE), sizeof(std::size_t));

            elementsBegin = (std::uint8_t*)(((char*)rawData.get()) + distanceEB);
            elementsEnd = (std::uint8_t*)(((char*)rawData.get()) + distanceEE);
            offsetsBegin = (std::size_t*)(((char*)rawData.get()) + distanceOB);
            offsetsEnd = (std::size_t*)(((char*)rawData.get()) + distanceOE);

            stream.read(reinterpret_cast<char*>(&numStoredElementsInMemory), sizeof(std::int64_t));
        }


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

        void compact(){
            //no-op
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
            return sizeof(std::size_t) * std::distance(offsetsBegin, offsetsEnd);
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

            // std::cerr << "FixedSizeStorage::sort(" << memoryForSortingInBytes << ")\n";
            // std::cerr << "getSizeInBytes(): " << getSizeInBytes() << "\n";
            // std::cerr << "getNumOccupiedRawElementBytes(): " << getNumOccupiedRawElementBytes() << "\n";
            // std::cerr << "getNumOccupiedRawOffsetBytes(): " << getNumOccupiedRawOffsetBytes() << "\n";

            if(getNumStoredElements() == 0) return;
#if 1
            bool sortValuesSuccess = false;

            auto keyGenerator = [&](auto i){
                const std::uint8_t* ptr = elementsBegin + offsetsBegin[i];
                const Key key = extractKey(ptr);
                return key;
            };

            try{
                //std::cerr << "getNumStoredElements() = " << getNumStoredElements() << "\n";
                if(std::size_t(getNumStoredElements()) <= std::size_t(std::numeric_limits<std::uint32_t>::max())){
                    //std::cerr << "sortValuesByGeneratedKeys<std::uint32_t>\n";
                    sortValuesSuccess = sortValuesByGeneratedKeys<std::uint32_t>(
                        memoryForSortingInBytes,
                        offsetsBegin,
                        getNumStoredElements(),
                        keyGenerator,
                        keyComparator
                    );
                }else{
                    //std::cerr << "sortValuesByGeneratedKeys<std::uint64_t>\n";
                    sortValuesSuccess = sortValuesByGeneratedKeys<std::uint64_t>(
                        memoryForSortingInBytes,
                        offsetsBegin,
                        getNumStoredElements(),
                        keyGenerator,
                        keyComparator
                    );
                }

                // for(std::size_t i = 1; i < getNumStoredElements(); i++){
                //     const std::uint8_t* ptrl = elementsBegin + offsetsBegin[i-1];
                //     const Key keyl = extractKey(ptrl);

                //     const std::uint8_t* ptrr = elementsBegin + offsetsBegin[i];
                //     const Key keyr = extractKey(ptrr);

                //     if(keyl > keyr){
                //         std::cerr << "Error, results not sorted. i = " << i << ", keyl = " << keyl << ", keyr = " << keyr << "\n";
                //         assert(false);
                //     }
                // }

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

    #if 0

    template<class T>
    struct FixedSizeStorageMMap{
        std::size_t memoryLimitBytes = 0;
        MMapBuffer rawData{};

        // offsets and elements share the same memory in rawData
        // elements grow from left to right
        // offsets grow from right to left

        std::uint8_t* elementsBegin = nullptr;
        std::uint8_t* elementsEnd = nullptr;

        std::size_t* offsetsBegin = nullptr;
        std::size_t* offsetsEnd = nullptr; 
        
        std::int64_t numStoredElementsInMemory = 0;

        void saveToStream(std::ostream& stream) {
            const std::size_t numSizeTs = memoryLimitBytes / sizeof(std::size_t);

            stream.write(reinterpret_cast<const char*>(&memoryLimitBytes), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&numSizeTs), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(rawData.get()), sizeof(std::size_t) * numSizeTs);

            std::size_t distanceEB = std::distance((char*)rawData.get(), (char*)elementsBegin);
            std::size_t distanceEE = std::distance((char*)rawData.get(), (char*)elementsEnd);
            std::size_t distanceOB = std::distance((char*)rawData.get(), (char*)offsetsBegin);
            std::size_t distanceOE = std::distance((char*)rawData.get(), (char*)offsetsEnd);

            stream.write(reinterpret_cast<const char*>(&distanceEB), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&distanceEE), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&distanceOB), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&distanceOE), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&numStoredElementsInMemory), sizeof(std::int64_t));

            stream.flush();
        }

        void loadFromStream(std::istream& stream){
            std::size_t numSizeTs = 0;
            stream.read(reinterpret_cast<char*>(&memoryLimitBytes), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&numSizeTs), sizeof(std::size_t));

            rawData.resize(sizeof(std::size_t) * numSizeTs);

            stream.read(reinterpret_cast<char*>(rawData.get()), sizeof(std::size_t) * numSizeTs);

            std::size_t distanceEB = 0;
            std::size_t distanceEE = 0;
            std::size_t distanceOB = 0;
            std::size_t distanceOE = 0;

            stream.read(reinterpret_cast<char*>(&distanceEB), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&distanceEE), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&distanceOB), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&distanceOE), sizeof(std::size_t));

            elementsBegin = (std::uint8_t*)(((char*)rawData.get()) + distanceEB);
            elementsEnd = (std::uint8_t*)(((char*)rawData.get()) + distanceEE);
            offsetsBegin = (std::size_t*)(((char*)rawData.get()) + distanceOB);
            offsetsEnd = (std::size_t*)(((char*)rawData.get()) + distanceOE);

            stream.read(reinterpret_cast<char*>(&numStoredElementsInMemory), sizeof(std::int64_t));
        }


        FixedSizeStorageMMap() = default;
        FixedSizeStorageMMap(const FixedSizeStorageMMap&) = delete;
        FixedSizeStorageMMap& operator=(const FixedSizeStorageMMap&) = delete;

        FixedSizeStorageMMap(FixedSizeStorageMMap&& rhs){
            memoryLimitBytes = std::exchange(rhs.memoryLimitBytes, 0);

            rawData = std::move(rhs.rawData);
            elementsBegin = std::exchange(rhs.elementsBegin, nullptr);
            elementsEnd = std::exchange(rhs.elementsEnd, nullptr);
            offsetsBegin = std::exchange(rhs.offsetsBegin, nullptr);
            offsetsEnd = std::exchange(rhs.offsetsEnd, nullptr);

            numStoredElementsInMemory = std::exchange(rhs.numStoredElementsInMemory, 0);
        }

        FixedSizeStorageMMap& operator=(FixedSizeStorageMMap&& rhs){
            memoryLimitBytes = std::exchange(rhs.memoryLimitBytes, 0);
            
            rawData = std::move(rhs.rawData);
            elementsBegin = std::exchange(rhs.elementsBegin, nullptr);
            elementsEnd = std::exchange(rhs.elementsEnd, nullptr);
            offsetsBegin = std::exchange(rhs.offsetsBegin, nullptr);
            offsetsEnd = std::exchange(rhs.offsetsEnd, nullptr);

            numStoredElementsInMemory = std::exchange(rhs.numStoredElementsInMemory, 0);

            return *this;
        }

        FixedSizeStorageMMap(std::size_t memoryLimitBytes_)
                : memoryLimitBytes(memoryLimitBytes_){

            const std::size_t numSizeTs = memoryLimitBytes / sizeof(std::size_t);

            rawData.resize(sizeof(std::size_t) * numSizeTs);

            elementsBegin = reinterpret_cast<std::uint8_t*>(rawData.get());
            elementsEnd = elementsBegin;
            offsetsEnd = ((std::size_t*)rawData.get()) + numSizeTs;
            offsetsBegin = offsetsEnd;
        }

        void compact(){
            //move offsets to the end of elements, and release the unoccupied memory

            const std::size_t numOffsets = getNumStoredElements();
            const std::size_t pointeroffset = ((std::size_t)elementsEnd) % sizeof(std::size_t);
            //align new pointer to std::size_t boundary
            std::size_t* newOffsetsBegin = (std::size_t*)(((char*)elementsEnd) + sizeof(std::size_t) - pointeroffset);
            if(newOffsetsBegin < offsetsBegin){
                //ok, can proceed

                offsetsEnd = std::copy_n(offsetsBegin, numOffsets, newOffsetsBegin);
                offsetsBegin = newOffsetsBegin;

                const std::size_t remainingBytes = getNumOccupiedRawElementBytes() + getNumOccupiedRawOffsetBytes();

                rawData.resize(remainingBytes);
                rawData.shrink_to_fit();
                memoryLimitBytes = remainingBytes;

                elementsBegin = reinterpret_cast<std::uint8_t*>(rawData.get());
                elementsEnd = elementsBegin + getNumOccupiedRawElementBytes();
            }
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
            return sizeof(std::size_t) * std::distance(offsetsBegin, offsetsEnd);
        }

        std::size_t getSizeInBytes() const{
            return memoryLimitBytes;
        }

        void clear(){
            const std::size_t numSizeTs = memoryLimitBytes / sizeof(std::size_t);
            rawData.clear();

            numStoredElementsInMemory = 0;
            elementsBegin = reinterpret_cast<std::uint8_t*>(rawData.get());
            elementsEnd = elementsBegin;
            offsetsEnd = ((std::size_t*)rawData.get()) + numSizeTs;
            offsetsBegin = offsetsEnd;
        }

        void destroy(){
            memoryLimitBytes = 0;
            rawData.destroy();
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

        bool insert(const FixedSizeStorageMMap<T>& other){
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

            std::cerr << "FixedSizeStorage::sort(" << memoryForSortingInBytes << ")\n";
            std::cerr << "getSizeInBytes(): " << getSizeInBytes() << "\n";
            std::cerr << "getNumOccupiedRawElementBytes(): " << getNumOccupiedRawElementBytes() << "\n";
            std::cerr << "getNumOccupiedRawOffsetBytes(): " << getNumOccupiedRawOffsetBytes() << "\n";

            if(getNumStoredElements() == 0) return;
#if 1
            bool sortValuesSuccess = false;

            auto keyGenerator = [&](auto i){
                const std::uint8_t* ptr = elementsBegin + offsetsBegin[i];
                const Key key = extractKey(ptr);
                return key;
            };

            try{
                //std::cerr << "getNumStoredElements() = " << getNumStoredElements() << "\n";
                if(std::size_t(getNumStoredElements()) <= std::size_t(std::numeric_limits<std::uint32_t>::max())){
                    //std::cerr << "sortValuesByGeneratedKeys<std::uint32_t>\n";
                    sortValuesSuccess = sortValuesByGeneratedKeys<std::uint32_t>(
                        memoryForSortingInBytes,
                        offsetsBegin,
                        getNumStoredElements(),
                        keyGenerator,
                        keyComparator
                    );
                }else{
                    //std::cerr << "sortValuesByGeneratedKeys<std::uint64_t>\n";
                    sortValuesSuccess = sortValuesByGeneratedKeys<std::uint64_t>(
                        memoryForSortingInBytes,
                        offsetsBegin,
                        getNumStoredElements(),
                        keyGenerator,
                        keyComparator
                    );
                }

                // for(std::size_t i = 1; i < getNumStoredElements(); i++){
                //     const std::uint8_t* ptrl = elementsBegin + offsetsBegin[i-1];
                //     const Key keyl = extractKey(ptrl);

                //     const std::uint8_t* ptrr = elementsBegin + offsetsBegin[i];
                //     const Key keyr = extractKey(ptrr);

                //     if(keyl > keyr){
                //         std::cerr << "Error, results not sorted. i = " << i << ", keyl = " << keyl << ", keyr = " << keyr << "\n";
                //         assert(false);
                //     }
                // }

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

    #endif
}





#endif