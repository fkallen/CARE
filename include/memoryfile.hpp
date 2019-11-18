#ifndef CARE_MEMORY_FILE
#define CARE_MEMORY_FILE

#include <vector>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>
#include <functional>

#include <filesort.hpp>

/*
    T must have a function size_t heapsize() which returns the size 
    of memory allocated on the heap which is owned by an instance of T

    T must have bool writeToBinaryStream(std::ostream& s) const;
    T must have bool readFromBinaryStream(std::istream& s);
*/
template<class T>
struct MemoryFile{

    struct Twrapper{

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
                fileinputstream(std::ifstream(filename, std::ios::binary)){

            fileiterator = std::istream_iterator<Twrapper>(fileinputstream);
            fileend = std::istream_iterator<Twrapper>();
        }

        bool hasNext() const{
            return (memoryiterator != memoryend) || (fileiterator != fileend);
        }

        T next() const{
            assert(hasNext());

            if(memoryiterator != memoryend){
                T data = *memoryiterator;
                ++memoryiterator;
                return data;
            }else{
                Twrapper wrapper = *fileiterator;
                ++fileiterator;
                return wrapper.data;
            }
        }

        mutable typename std::vector<T>::const_iterator memoryiterator;
        typename std::vector<T>::const_iterator memoryend;
        mutable std::istream_iterator<Twrapper> fileiterator;
        std::istream_iterator<Twrapper> fileend;
        std::ifstream fileinputstream;
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

        outputstream =  std::ofstream(filename, std::ios::binary);
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

        renameFileSameMount(filename+"2", filename);

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
            //check if element could be saved in memory, disregaring vector growth
            if(vector.capacity() * sizeof(T) + usedHeapMemory + getHeapUsageOfElement(element) <= maxMemoryOfVectorAndHeap){
                if(vector.size() < vector.capacity()){
                    return storeInMemory(std::move(element));
                }else{ //size == capacity
                    
                    if(2 * vector.capacity() * sizeof(T) + usedHeapMemory + getHeapUsageOfElement(element) <= maxMemoryOfVectorAndHeap){
                        return storeInMemory(std::move(element));
                    }else{
                        isUsingFile = true;
                        return storeInFile(std::move(element));
                    }
                }
            }else{
                isUsingFile = true;
                return storeInFile(std::move(element));
            }
        }else{
            return storeInFile(std::move(element));
        }
    }

    bool storeInMemory(T&& element){
        vector.emplace_back(std::move(element));
        usedHeapMemory += getHeapUsageOfElement(element);
        numStoredElements++;

        return true;
    }

    bool storeInFile(T&& element){
        outputstream << Twrapper{element};
        numStoredElements++;

        return bool(outputstream);
    }

    bool isUsingFile = false; //if at least 1 element has been written to file
    std::int64_t numStoredElements = 0; // number of stored elements
    std::size_t usedHeapMemory = 0;
    std::size_t maxMemoryOfVectorAndHeap = 0; // usedMemoryOfVector <= maxMemoryOfVector must always hold
    std::function<std::size_t(T)> getHeapUsageOfElement;
    std::vector<T> vector; //elements stored in memory
    std::ofstream outputstream;
    std::string filename = "";    
};

#endif