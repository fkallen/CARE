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

#if 0
/*
    T must have a function size_t heapsize() which returns the size 
    of memory allocated on the heap which is owned by an instance of T

    T must have bool writeToBinaryStream(std::ostream& s) const;
    T must have bool readFromBinaryStream(std::istream& s);
*/
template<class T>
struct MemoryFile{

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
    };

    struct Reader{
        Reader() = default;
        Reader(const std::vector<T>& vec, std::string filename)
                : 
                memoryiterator(vec.begin()),
                memoryend(vec.end()),
                fileinputstream(std::ifstream(filename, std::ios::binary)),
                fileiterator(std::istream_iterator<Twrapper>(fileinputstream)),
                fileend(std::istream_iterator<Twrapper>()){
        }

        bool hasNext() const{
            return (memoryiterator != memoryend) || (fileiterator != fileend);
        }

        const T* next(){
            assert(hasNext());

            if(memoryiterator != memoryend){
                const T* data = &(*memoryiterator);
                ++memoryiterator;
                return data;
            }else{
                currentFileElement = std::move(*fileiterator);
                ++fileiterator;

                return &(currentFileElement.data);
            }
        }

        Twrapper currentFileElement;
        typename std::vector<T>::const_iterator memoryiterator;
        typename std::vector<T>::const_iterator memoryend;
        std::ifstream fileinputstream;
        std::istream_iterator<Twrapper> fileiterator;
        std::istream_iterator<Twrapper> fileend;
    };

    MemoryFile() = default;

    MemoryFile(std::size_t memoryLimit, std::string file)
            : MemoryFile(memoryLimit, file, [](const T&){return 0;}){
    }

    template<class Func>
    MemoryFile(std::size_t memoryLimit, std::string file, Func&& heapUsageFunc)
        : maxMemoryOfVectorAndHeap(memoryLimit),
          filename(file),
          getHeapUsageOfElement(std::move(heapUsageFunc)){

        outputstream = std::ofstream(filename, std::ios::binary);
    }

    bool storeElement(T element){
        return store(std::move(element));
    }

    template<class Tcomparator>
    void sort(const std::string& tempdir, Tcomparator&& comparator){

        std::cerr << vector.size() << " elements stored in memory\n";

        //try to sort vector in memory
        bool vectorCouldBeSortedInMemory = false;
        if(onlyInMemory()){
            try{
                std::sort(vector.begin(), vector.end(), comparator);
                vectorCouldBeSortedInMemory = true;
            }catch(std::bad_alloc& e){
                ; //nothing
            }
        }

        if(vectorCouldBeSortedInMemory && onlyInMemory()){
            ; //done
            return;
        }

        //append unsorted vector to file

        for(auto&& element : vector){
            auto wrap = Twrapper{std::move(element)};
            outputstream << wrap;
        }

        outputstream.flush();

        vector = std::vector<T>(); //free memory of vector

        //perform mergesort on file
        auto wrappercomparator = [&](const auto& l, const auto& r){
            return comparator(l.data, r.data);
        };
        auto wrapperheapusage = [&](const auto& w){
            return getHeapUsageOfElement(w.data);
        };

        care::filesort::binKeySort<Twrapper>(tempdir,
                        {filename}, 
                        filename+"2",
                        wrappercomparator,
                        wrapperheapusage);

        filehelpers::renameFileSameMount(filename+"2", filename);

        outputstream = std::ofstream(filename, std::ios::binary | std::ios::app);
    }

    Reader makeReader() const{
        return Reader{vector, filename};
    }

    void flush(){
        outputstream.flush();
    }

    bool onlyInMemory() const{
        return !isUsingFile;
    }

private:

    bool store(T&& element){
        if(!isUsingFile){
            auto getAvailableMemoryInBytes = [](){
                const std::size_t availableMemoryInKB = getAvailableMemoryInKB();
                return availableMemoryInKB << 10;
            };
            
            auto getMemLimit = [&](){
                size_t availableMemory = getAvailableMemoryInBytes();

                constexpr std::size_t oneGB = std::size_t(1) << 30; 
                constexpr std::size_t safetybuffer = oneGB;

                if(availableMemory > safetybuffer){
                    availableMemory -= safetybuffer;
                }else{
                    availableMemory = 0;
                }
                if(availableMemory > oneGB){
                    //round down to next multiple of 1GB
                    availableMemory = (availableMemory / oneGB) * oneGB;
                }
                return availableMemory;
            };

            //size_t memLimit = getMemLimit();

            if(numStoredElements < 2 || numStoredElements % 65536 == 0){
                maxMemoryOfVectorAndHeap = getMemLimit();
            }
            

            //check if element could be saved in memory, disregaring vector growth
            if(vector.capacity() * sizeof(T) + usedHeapMemory + getHeapUsageOfElement(element) <= maxMemoryOfVectorAndHeap){
                if(vector.size() < vector.capacity()){
                    bool retval = true;

                    try{
                        retval = storeInMemory(std::move(element));
                    }catch(std::bad_alloc& e){
                        std::cerr << "switch to file storage after " << numStoredElements << " insertions.\n";
                        isUsingFile = true;
                        retval = storeInFile(std::move(element));
                    }

                    return retval;
                }else{ //size == capacity
                    
                    if(2 * vector.capacity() * sizeof(T) + usedHeapMemory + getHeapUsageOfElement(element) <= maxMemoryOfVectorAndHeap){
                        bool retval = true;

                        try{
                            retval = storeInMemory(std::move(element));
                        }catch(std::bad_alloc& e){
                            std::cerr << "switch to file storage after " << numStoredElements << " insertions.\n";
                            isUsingFile = true;
                            retval = storeInFile(std::move(element));
                        }

                        return retval;
                    }else{
                        std::cerr << "switch to file storage after " << numStoredElements << " insertions.\n";
                        isUsingFile = true;
                        return storeInFile(std::move(element));
                    }
                }
            }else{
                std::cerr << "switch to file storage after " << numStoredElements << " insertions.\n";
                isUsingFile = true;
                return storeInFile(std::move(element));
            }
        }else{
            return storeInFile(std::move(element));
        }
    }

    bool storeInMemory(T&& element){
        usedHeapMemory += getHeapUsageOfElement(element);
        vector.emplace_back(std::move(element));        
        numStoredElements++;

        return true;
    }

    bool storeInFile(T&& element){
        outputstream << Twrapper{std::move(element)};
        numStoredElements++;

        return bool(outputstream);
    }

    bool isUsingFile = false; //if at least 1 element has been written to file
    std::int64_t numStoredElements = 0; // number of stored elements
    std::size_t usedHeapMemory = 0;
    std::size_t maxMemoryOfVectorAndHeap = 0; // usedMemoryOfVector <= maxMemoryOfVector must always hold
    std::function<std::size_t(const T&)> getHeapUsageOfElement;
    std::vector<T> vector; //elements stored in memory
    std::ofstream outputstream;
    std::string filename = "";    
};



#endif













template<class T>
struct MemoryFileFixedSize{

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
                rawData(rawData_),
                elementOffsets(elementOffsets_),
                numElementsInMemory(numElementsInMemory_),
                fileinputstream(std::ifstream(filename, std::ios::binary)),
                fileiterator(std::istream_iterator<Twrapper>(fileinputstream)),
                fileend(std::istream_iterator<Twrapper>()){
        }

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

                return &(currentFileElement.data);
            }
        }

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

            outputstream.flush();
            std::size_t filesizeBytes = filehelpers::getSizeOfFileBytes(filename);

            std::size_t requiredMemForAllElements = memoryStorage.getNumOccupiedRawElementBytes();
            requiredMemForAllElements += filesizeBytes;
            requiredMemForAllElements += sizeof(std::size_t) * getNumElements();
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
                    newMemoryStorage.insert(*element, serialize);
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