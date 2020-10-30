#ifndef CARE_MEMORY_FILE
#define CARE_MEMORY_FILE

#include <filesort.hpp>
#include <filehelpers.hpp>
#include <fixedsizestorage.hpp>
#include <memorymanagement.hpp>


#include <vector>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>
#include <functional>
#include <memory>



namespace care{


template<class T>
struct MemoryFileFixedSize{

    using ValueType = T;

    struct Twrapper{

        Twrapper() = default;
        Twrapper(const Twrapper&) = default;
        Twrapper(Twrapper&&) = default;
        Twrapper& operator=(const Twrapper&) = default;
        Twrapper& operator=(Twrapper&&) = default;

        Twrapper(const T& rhs){
            data = rhs;
        }
        Twrapper(T&& rhs){
            data = std::move(rhs);
        }
        Twrapper& operator=(const T& rhs){
            data = rhs;
            return *this;
        }
        Twrapper& operator=(T&& rhs){
            data = std::move(rhs);
            return *this;
        }

        T data;

        friend std::ostream& operator<<(std::ostream& s, const Twrapper& i){
            i.data.writeToBinaryStream(s);
            return s;
        }

        friend std::istream& operator>>(std::istream& s, Twrapper& i){
            i.data.readFromBinaryStream(s);
            return s;
        }

        std::uint8_t* copyToContiguousMemory(std::uint8_t* begin, std::uint8_t* end) const{
            return data.copyToContiguousMemory(begin, end);
        }

        void copyFromContiguousMemory(const std::uint8_t* begin){
            data.copyFromContiguousMemory(begin);
        }
    };

    struct Reader{
        Reader() = default;
        Reader(
            const std::uint8_t* rawData_, 
            const std::size_t* elementOffsets_, 
            std::int64_t numElementsInMemory_, 
            std::string filename)
                : 
                currentIsMemory(true),
                rawData(rawData_),
                elementOffsets(elementOffsets_),
                numElementsInMemory(numElementsInMemory_),
                fileinputstream(std::ifstream(filename, std::ios::binary)),
                fileiterator(std::istream_iterator<Twrapper>(fileinputstream)),
                fileend(std::istream_iterator<Twrapper>()){
        }

        Reader(const Reader& rhs) = delete;
        Reader(Reader&& rhs) = default;

        Reader& operator=(const Reader& rhs) = delete;
        Reader& operator=(Reader&& rhs) = default;

        ~Reader(){}

        bool hasNext() const{
            return (elementIndexInMemory != numElementsInMemory) || (fileiterator != fileend);
        }

        const T* next(){
            assert(hasNext());

            if(elementIndexInMemory != numElementsInMemory){
                const std::uint8_t* const ptr = rawData + elementOffsets[elementIndexInMemory];
                currentMemoryElement.copyFromContiguousMemory(ptr);

                ++elementIndexInMemory;
                return &currentMemoryElement;
            }else{
                currentFileElement = std::move(*fileiterator);
                ++fileiterator;
                currentIsMemory = false;

                return &(currentFileElement.data);
            }
        }

        const T* current() const{
            if(currentIsMemory){
                return &currentMemoryElement;
            }else{
                return &(currentFileElement.data);
            }
        }

        bool currentIsMemory;

        const std::uint8_t* rawData;
        const std::size_t* elementOffsets;

        std::int64_t elementIndexInMemory = 0;
        std::int64_t numElementsInMemory;
        T currentMemoryElement;

        Twrapper currentFileElement;
        std::ifstream fileinputstream;
        std::istream_iterator<Twrapper> fileiterator;
        std::istream_iterator<Twrapper> fileend;
    };

    MemoryFileFixedSize() = default;

    MemoryFileFixedSize(std::size_t memoryLimitBytes, std::string file)
        : 
          memoryStorage(memoryLimitBytes),
          filename(file),
          outputstream(filename, std::ios::binary){

        assert(outputstream.good());
    }

    ~MemoryFileFixedSize(){
        if(filename != ""){
            //std::cerr << "Deleting file " << filename << " of MemoryFileFixedSize\n";
            filehelpers::deleteFiles({filename});
        }
    }

    MemoryFileFixedSize(const MemoryFileFixedSize&) = delete;
    MemoryFileFixedSize& operator=(const MemoryFileFixedSize&) = delete;

    MemoryFileFixedSize(MemoryFileFixedSize&& rhs){
        *this = std::move(rhs);
    }
    MemoryFileFixedSize& operator=(MemoryFileFixedSize&& rhs){
        isUsingFile = std::exchange(rhs.isUsingFile, false);
        memoryStorage = std::exchange(rhs.memoryStorage, FixedSizeStorage<T>{0});
        numStoredElementsInFile = std::exchange(rhs.numStoredElementsInFile, 0);
        filename = std::exchange(rhs.filename, "");
        outputstream = std::exchange(rhs.outputstream, std::ofstream{});

        return *this;
    }

    Reader makeReader() const{
        return Reader{memoryStorage.getElementsData(), memoryStorage.getElementOffsets(), memoryStorage.getNumStoredElements(), filename};
    }

    void flush(){
        outputstream.flush();
    }

    bool storeElement(T&& element){
        T tmp(std::move(element));

        if(!isUsingFile){
            bool success = storeInMemory(tmp);
            if(!success){
                isUsingFile = true;
                success = storeInFile(std::move(tmp));
            }
            return success;
        }else{
            return storeInFile(std::move(tmp));
        }
    }

    bool storeElement(const T* element){

        if(!isUsingFile){
            bool success = storeInMemory(element);
            if(!success){
                isUsingFile = true;
                success = storeInFile(element);
            }
            return success;
        }else{
            return storeInFile(element);
        }
    }

    std::int64_t getNumElementsInMemory() const{
        return memoryStorage.getNumStoredElements();
    }

    std::int64_t getNumElementsInFile() const{
        return numStoredElementsInFile;
    }

    std::int64_t getNumElements() const{
        return getNumElementsInMemory() + getNumElementsInFile();
    }

    template<class Ptrcomparator, class TComparator>
    void sort(const std::string& tempdir, std::size_t memoryForSortingInBytes, Ptrcomparator&& ptrcomparator, TComparator&& elementcomparator){
    //void sort(const std::string& tempdir){
        std::cerr << "Sorting memory file:";
        std::cerr << " elements in memory = " << getNumElementsInMemory();
        std::cerr << " elements in file = " << getNumElementsInFile();
        std::cerr << '\n';

        // auto offsetcomparator = [&](std::size_t elementOffset1, std::size_t elementOffset2){
        //     return ptrcomparator(memoryStorage.getElementsData() + elementOffset1, memoryStorage.getElementsData() + elementOffset2);
        // };

        bool success = false;

        try{
            if(getNumElementsInFile() == 0){
                memoryStorage.sort(ptrcomparator);
                success = true;
            }
        }catch(...){

        }

        if(success){
            return;
        }



        //if all elements from memory and file fit into memoryForSortingInBytes bytes, create new fixed size storage with that size
        //insert all elements into that, then sort it in memory.
        try{

            auto sdiv = [](auto x, auto y){
                return ((x+y-1) / y);
            };

            outputstream.flush();
            std::size_t filesizeBytes = filehelpers::getSizeOfFileBytes(filename);

            std::size_t requiredMemForAllElements = memoryStorage.getNumOccupiedRawElementBytes();
            requiredMemForAllElements += filesizeBytes;
            requiredMemForAllElements += sizeof(std::size_t) * getNumElements();
            requiredMemForAllElements = sdiv(requiredMemForAllElements, sizeof(std::size_t)) * sizeof(std::size_t);

            std::cerr << "requiredMemForAllElements " << requiredMemForAllElements << ", memoryForSortingInBytes " << memoryForSortingInBytes << "\n";
            
            if(requiredMemForAllElements < memoryForSortingInBytes){
                FixedSizeStorage<T> newMemoryStorage(requiredMemForAllElements);

                success = newMemoryStorage.insert(memoryStorage);
                assert(success);

                Reader r = makeFileOnlyReader();

                auto serialize = [](const auto& element, auto beginptr, auto endptr){
                    return element.copyToContiguousMemory(beginptr, endptr);
                };

                while(r.hasNext()){
                    const T* element = r.next();
                    const bool inserted = newMemoryStorage.insert(*element, serialize);
                    assert(inserted);
                }

                newMemoryStorage.sort(ptrcomparator);

                std::swap(memoryStorage, newMemoryStorage);
                numStoredElementsInFile = 0;
                outputstream = std::ofstream(filename, std::ios::binary);
            }
        }catch(...){

        }

        if(success){
            return;
        }

        
        
        //append unsorted elements in memory to file
        const std::size_t memoryBytes = memoryStorage.getNumOccupiedRawElementBytes();
        outputstream.write(reinterpret_cast<const char*>(memoryStorage.getElementsData()), memoryBytes);
        outputstream.flush();


        std::size_t numElements = getNumElements();
        numStoredElementsInFile = numElements;
        
        memoryStorage.destroy();

        memoryForSortingInBytes += memoryStorage.getSizeInBytes();

        //perform mergesort on file
        auto wrapperptrcomparator = [&](const std::uint8_t* l, const std::uint8_t* r){
            return ptrcomparator(l, r);
        };

        auto wrappercomparator = [&](const auto& l, const auto& r){
            return elementcomparator(l.data, r.data);
        };

        assert(memoryForSortingInBytes > 0);

        filesort::fixedmemory::binKeySort<Twrapper>(tempdir,
                        {filename}, 
                        filename+"2",
                        memoryForSortingInBytes,
                        wrapperptrcomparator,
                        wrappercomparator);

        filehelpers::renameFileSameMount(filename+"2", filename);

        outputstream = std::ofstream(filename, std::ios::binary | std::ios::app);
    }

private:
    Reader makeFileOnlyReader() const{
        return Reader{nullptr, nullptr, 0, filename};
    }

    bool storeInMemory(const T& element){

        auto serialize = [](const auto& element, auto beginptr, auto endptr){
            return element.copyToContiguousMemory(beginptr, endptr);
        };

        return memoryStorage.insert(element, serialize);
    }

    bool storeInMemory(const T* element){

        auto serialize = [](const auto& element, auto beginptr, auto endptr){
            return element.copyToContiguousMemory(beginptr, endptr);
        };

        return memoryStorage.insert(element, serialize);
    }

    bool storeInFile(T&& element){
        outputstream << Twrapper{std::move(element)};
        bool result = bool(outputstream);
        if(result){
            numStoredElementsInFile++;
        }
        
        return result;
    }

    bool storeInFile(const T* element){
        element->writeToBinaryStream(outputstream);

        bool result = bool(outputstream);
        if(result){
            numStoredElementsInFile++;
        }
        
        return result;
    }

    bool isUsingFile = false;

    FixedSizeStorage<T> memoryStorage;
    std::int64_t numStoredElementsInFile = 0;

    std::string filename = "";    
    std::ofstream outputstream;
};


}

#endif