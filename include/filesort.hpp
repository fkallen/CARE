#ifndef CARE_FILESORT_HPP
#define CARE_FILESORT_HPP

#include <filehelpers.hpp>
//#include <util.hpp>
#include <memorymanagement.hpp>
#include <fixedsizestorage.hpp>

#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <cstdio>
#include <chrono>
#include <numeric>
#include <functional>
#include <iterator>

namespace care{
namespace filesort{


template<class T, class Tcomparator>
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile, Tcomparator&& comparator);

template<class T>
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile);

namespace detail{

    template<class T, class Tcomparator>
    void
    binKeyMergeSortedChunksImpl(bool remove, 
                                const std::string& tempdir, 
                                const std::vector<std::string>& infilenames, 
                                const std::string& outfilename, 
                                Tcomparator&& comparator){
                                    
        //if no input file specified create empty outputfile and return
        if(infilenames.empty()){            
            std::ofstream outfile(outfilename);
            outfile.close();
            return;
        }
        //merge the temp files
        std::vector<std::string> filenames = infilenames;
        std::vector<std::string> newfilenames;

        int step = 0;
        while(filenames.size() > 2){
            const int numtempfiles = filenames.size();
            for(int i = 0; i < numtempfiles ; i += 2){
                //merge tempfile i with i+1
                if(i+1 < numtempfiles){

                    std::string outtmpname(tempdir+"/"+std::to_string(i) + "-" + std::to_string(step));

                    std::cerr << "merge " << filenames[i] << " + " << filenames[i+1] << " into " <<  outtmpname << "\n";

                    binKeyMergeTwoFiles<T>(filenames[i], filenames[i+1], outtmpname, comparator);

                    newfilenames.emplace_back(std::move(outtmpname));

                    if(step > 0 || remove){
                        filehelpers::removeFile(filenames[i]);
                        filehelpers::removeFile(filenames[i+1]);
                    }
                    
                }else{
                    newfilenames.emplace_back(filenames[i]);
                }
            }

            filenames = std::move(newfilenames);
            step++;
        }

        assert(filenames.size() > 0);

        if(filenames.size() == 1){
            filehelpers::renameFileSameMount(filenames[0], outfilename);
        }else{
            std::cerr << "merge " << filenames[0] << " + " << filenames[1] << " into " <<  outfilename << "\n";
            binKeyMergeTwoFiles<T>(filenames[0], filenames[1], outfilename, comparator);

            if(step > 0 || remove){
                filehelpers::removeFile(filenames[0]);
                filehelpers::removeFile(filenames[1]);
            }
        }
    }

} //namespace detail



//merge two sorted input files into sorted output file
template<class T, class Tcomparator>
void binKeyMergeTwoFiles(
                        const std::string& infile1, 
                        const std::string& infile2, 
                        const std::string& outfile, 
                        Tcomparator&& comparator){
    std::ifstream in1(infile1);
    std::ifstream in2(infile2);
    std::ofstream out(outfile);

    assert(in1);
    assert(in2);
    assert(out);

    std::merge(std::istream_iterator<T>(in1),
                std::istream_iterator<T>(),
                std::istream_iterator<T>(in2),
                std::istream_iterator<T>(),
                std::ostream_iterator<T>(out, ""),
                comparator);
}

template<class Index_t>
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile){
    binKeyMergeTwoFiles(infile1, infile2, outfile, std::less<Index_t>{});
}

//split input files into sorted chunks. returns filenames of sorted chunks
//each line in infile must begin with a number of type Index_t which was written in binary mode.
//infile is sorted by this number using comparator Comp
//sorted chunks will be named tempdir/temp_i where i is the chunk number
template<class T, class Tcomparator, class Theapsizefunc>
std::vector<std::string>
binKeySplitIntoSortedChunks(const std::vector<std::string>& infilenames, 
                            const std::string& tempdir, 
                            std::size_t buffersizeInBytes, 
                            Tcomparator&& comparator,
                            Theapsizefunc&& getHeapUsage){

    //const std::size_t availableMemory = buffersizeInBytes;

    std::size_t availableMemory = 0;

    auto getAvailableMemoryInBytes = [](){
        const std::size_t availableMemoryInKB = getAvailableMemoryInKB();
        return availableMemoryInKB << 10;
    };

    auto updateAvailableMemory = [&](){
        availableMemory = getAvailableMemoryInBytes();

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
    };

    updateAvailableMemory();

    std::size_t initialAvailableMemory = availableMemory;
   
    T item;
    bool itemProcessed = true;

    std::vector<T> buffer;

    std::size_t usedHeapMemory = 0;

    auto couldAddElementToBuffer = [&](const T& element){
        if(buffer.size() < 2 || buffer.size() % 65536 == 0){
            updateAvailableMemory();
            //std::cerr << buffer.size() << " " << availableMemory << '\n';

            usedHeapMemory = initialAvailableMemory - availableMemory;
        }

        if(buffer.capacity() * sizeof(T) + usedHeapMemory + getHeapUsage(element) <= availableMemory){
            if(buffer.size() < buffer.capacity()){
                return true;
            }else{ //size == capacity                
                if(2 * buffer.capacity() * sizeof(T) + usedHeapMemory + getHeapUsage(element) <= availableMemory){
                    return true;
                }else{
                    return false;
                }
            }
        }else{
            return false;
        }
    };

    std::int64_t numtempfiles = 0;
    std::vector<std::string> tempfilenames;

    //split input files into sorted temp files
    for(const auto& filename : infilenames){
        std::ifstream istream(filename);
        if(!istream){
            assert(false);
        }

        usedHeapMemory = 0;
        buffer.clear();

        //handle item which did not fit into buffer in previous iteration
        if(!itemProcessed){
            usedHeapMemory += getHeapUsage(item);
            buffer.emplace_back(std::move(item));
            itemProcessed = true;
        }

        while(bool(istream >> item)){
            usedHeapMemory += getHeapUsage(item);
            buffer.emplace_back(std::move(item));

            //TIMERSTARTCPU(readingbatch);
            while(bool(istream >> item)){
                if(couldAddElementToBuffer(item)){
                    usedHeapMemory += getHeapUsage(item);
                    buffer.emplace_back(std::move(item));
                }else{
                    itemProcessed = false;
                    break;
                }
            }
            //TIMERSTOPCPU(readingbatch);

            std::string tempfilename(tempdir+"/tmp_"+std::to_string(numtempfiles));
            std::ofstream sortedtempfile(tempfilename);

            //TIMERSTARTCPU(actualsort);
            std::cerr << "sort " << buffer.size() << " elements in memory \n";

            try{
                std::sort(buffer.begin(), buffer.end(), comparator);
            }catch(std::bad_alloc& e){
                filehelpers::removeFile(tempfilename);
                throw e;
            }

            //TIMERSTOPCPU(actualsort);

            //TIMERSTARTCPU(writingsortedbatch);
            std::cerr << "save " << buffer.size() << " elements to file " <<  tempfilename << '\n';
            std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<T>(sortedtempfile, ""));

            //TIMERSTOPCPU(writingsortedbatch); 

            //TIMERSTARTCPU(clear);
            //buffer.clear();
            buffer = std::vector<T>{};
            usedHeapMemory = 0;
            updateAvailableMemory();
            tempfilenames.emplace_back(std::move(tempfilename));
            numtempfiles++;
            //TIMERSTOPCPU(clear);
            
            //TIMERSTARTCPU(flushclose);
            sortedtempfile.flush();
            sortedtempfile.close();
            //TIMERSTOPCPU(flushclose);
        }
    }
    return tempfilenames;
}


template<class T, class Tcomparator, class Theapsizefunc>
std::vector<std::string>
binKeySplitIntoSortedChunks(const std::vector<std::string>& infilenames, 
                            const std::string& tempdir, 
                            Tcomparator&& comparator,
                            Theapsizefunc&& getHeapUsage){
                                
    std::size_t availableMemoryInKB = getAvailableMemoryInKB();
    std::size_t availableMemory = availableMemoryInKB << 10;

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

    std::cerr << "Available memory: " << availableMemory << "\n";

    assert(availableMemory > 0);

    constexpr int maxIters = 4;
    for(int i = 0; i < maxIters; i++){
        try{
            return binKeySplitIntoSortedChunks<T>(infilenames, 
                            tempdir, 
                            availableMemory, 
                            std::move(comparator),
                            std::move(getHeapUsage));
        }catch(...){
            availableMemory /= 2;
        }
    }

    throw std::runtime_error("Could not sort files");
    return {};
}



template<class T, class Tcomparator>
void
binKeyMergeSortedChunksAndDeleteChunks(const std::string& tempdir,
                                        const std::vector<std::string>& infilenames, 
                                        const std::string& outfilename, 
                                        Tcomparator&& comparator){
    detail::binKeyMergeSortedChunksImpl<T>(true, tempdir, infilenames, outfilename, comparator);
}

template<class T, class Tcomparator>
void
binKeyMergeSortedChunksAndDeleteChunks(const std::string& tempdir,
                                        const std::vector<std::string>& infilenames, 
                                        const std::string& outfilename){
    binKeyMergeSortedChunksAndDeleteChunks<T>(tempdir, infilenames, outfilename, std::less<T>{});
}

template<class T, class Tcomparator>
void
binKeyMergeSortedChunks(const std::string& tempdir,
                        const std::vector<std::string>& infilenames, 
                        const std::string& outfilename, 
                        Tcomparator&& comparator){
    detail::binKeyMergeSortedChunksImpl<T>(false, tempdir, infilenames, outfilename, comparator);
}

template<class T, class Tcomparator>
void
binKeyMergeSortedChunks(const std::string& tempdir,
                        const std::vector<std::string>& infilenames, 
                        const std::string& outfilename){
    binKeyMergeSortedChunks<T>(infilenames, tempdir, outfilename, std::less<T>{});
}


//sort infile to outfile
//infile is sorted by this number using comparator Comp
template<class T, class Tcomparator, class Theapsizefunc>
void binKeySort(const std::string& tempdir,
                const std::vector<std::string>& infilenames, 
                const std::string& outfilename,
                Tcomparator&& comparator,
                Theapsizefunc&& getHeapUsage){

    //TIMERSTARTCPU(split);
    auto tempfilenames = binKeySplitIntoSortedChunks<T>(infilenames, tempdir, comparator, getHeapUsage);
    //TIMERSTOPCPU(split);
    //TIMERSTARTCPU(merge);
    binKeyMergeSortedChunksAndDeleteChunks<T>(tempdir, tempfilenames, outfilename, comparator);
    //TIMERSTOPCPU(merge);
}

template<class T, class Tcomparator>
void binKeySort(const std::string& tempdir,
                const std::vector<std::string>& infilenames, 
                const std::string& outfilename,
                Tcomparator&& comparator){
    binKeySort(tempdir, infilenames, outfilename, comparator, [](const auto&){return 0;});
}






namespace fixedmemory{


template<class T, class Tcomparator>
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile, Tcomparator&& comparator);

template<class T>
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile);

namespace detail{

    template<class T, class Tcomparator>
    void
    binKeyMergeSortedChunksImpl(bool remove, 
                                const std::string& tempdir, 
                                const std::vector<std::string>& infilenames, 
                                const std::string& outfilename, 
                                Tcomparator&& comparator){
                                    
        //if no input file specified create empty outputfile and return
        if(infilenames.empty()){            
            std::ofstream outfile(outfilename);
            outfile.close();
            return;
        }
        //merge the temp files
        std::vector<std::string> filenames = infilenames;
        std::vector<std::string> newfilenames;

        int step = 0;
        while(filenames.size() > 2){
            const int numtempfiles = filenames.size();
            for(int i = 0; i < numtempfiles ; i += 2){
                //merge tempfile i with i+1
                if(i+1 < numtempfiles){

                    std::string outtmpname(tempdir+"/"+std::to_string(i) + "-" + std::to_string(step));

                    std::cerr << "merge " << filenames[i] << " + " << filenames[i+1] << " into " <<  outtmpname << "\n";

                    binKeyMergeTwoFiles<T>(filenames[i], filenames[i+1], outtmpname, comparator);

                    newfilenames.emplace_back(std::move(outtmpname));

                    if(step > 0 || remove){
                        filehelpers::removeFile(filenames[i]);
                        filehelpers::removeFile(filenames[i+1]);
                    }
                    
                }else{
                    newfilenames.emplace_back(filenames[i]);
                }
            }

            filenames = std::move(newfilenames);
            step++;
        }

        assert(filenames.size() > 0);

        if(filenames.size() == 1){
            filehelpers::renameFileSameMount(filenames[0], outfilename);
        }else{
            std::cerr << "merge " << filenames[0] << " + " << filenames[1] << " into " <<  outfilename << "\n";
            binKeyMergeTwoFiles<T>(filenames[0], filenames[1], outfilename, comparator);

            if(step > 0 || remove){
                filehelpers::removeFile(filenames[0]);
                filehelpers::removeFile(filenames[1]);
            }
        }
    }

} //namespace detail



//merge two sorted input files into sorted output file
template<class T, class Tcomparator>
void binKeyMergeTwoFiles(
                        const std::string& infile1, 
                        const std::string& infile2, 
                        const std::string& outfile, 
                        Tcomparator&& comparator){
    std::ifstream in1(infile1);
    std::ifstream in2(infile2);
    std::ofstream out(outfile);

    assert(in1);
    assert(in2);
    assert(out);

    std::merge(std::istream_iterator<T>(in1),
                std::istream_iterator<T>(),
                std::istream_iterator<T>(in2),
                std::istream_iterator<T>(),
                std::ostream_iterator<T>(out, ""),
                comparator);
}

template<class Index_t>
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile){
    binKeyMergeTwoFiles(infile1, infile2, outfile, std::less<Index_t>{});
}

//split input files into sorted chunks. returns filenames of sorted chunks
//each line in infile must begin with a number of type Index_t which was written in binary mode.
//infile is sorted by this number using comparator Comp
//sorted chunks will be named tempdir/temp_i where i is the chunk number
template<class T, class ExtractKey, class KeyComparator>
std::vector<std::string>
binKeySplitIntoSortedChunksImpl(const std::vector<std::string>& infilenames, 
                            const std::string& tempdir, 
                            std::size_t memoryLimit, 
                            ExtractKey extractKey, KeyComparator keyComparator){
    //using Key = decltype(extractKey(nullptr));

    FixedSizeStorage<T> memoryStorage(memoryLimit);

    auto serialize = [](const auto& element, auto beginptr, auto endptr){
        return element.copyToContiguousMemory(beginptr, endptr);
    };


    auto resetState = [&](){
        memoryStorage.clear();
    };

    T item;
    bool itemProcessed = true;

    auto tryAddElementToBuffer = [&](const T& element){
        return memoryStorage.insert(element, serialize);
    };

    std::int64_t numtempfiles = 0;
    std::vector<std::string> tempfilenames;

    //split input files into sorted temp files
    for(const auto& filename : infilenames){
        std::ifstream istream(filename);
        if(!istream){
            assert(false);
        }

        //handle item which did not fit into buffer in previous iteration //TODO move to end of while loop
        if(!itemProcessed){
            bool ok = tryAddElementToBuffer(item);
            assert(ok);
            itemProcessed = true;
        }

        while(bool(istream >> item)){
            bool ok = tryAddElementToBuffer(item);
            assert(ok);

            //TIMERSTARTCPU(readingbatch);
            while(bool(istream >> item)){
                ok = tryAddElementToBuffer(item);
                if(!ok){
                    itemProcessed = false;
                    break;
                }
            }
            //TIMERSTOPCPU(readingbatch);

            std::string tempfilename(tempdir+"/tmp_"+std::to_string(numtempfiles));
            std::ofstream sortedtempfile(tempfilename);

            //TIMERSTARTCPU(actualsort);
            std::cerr << "sort " << memoryStorage.getNumStoredElements() << " elements in memory \n";

            try{
                memoryStorage.sort(0, extractKey, keyComparator);
            }catch(std::bad_alloc& e){
                filehelpers::removeFile(tempfilename);
                throw e;
            }

            //TIMERSTOPCPU(actualsort);

            //TIMERSTARTCPU(writingsortedbatch);
            std::cerr << "save " << memoryStorage.getNumStoredElements() << " elements to file " <<  tempfilename << '\n';
            memoryStorage.forEachPointer([&](const auto ptr){
                T tmp;
                tmp.copyFromContiguousMemory(ptr);
                sortedtempfile << tmp;
            });

            //TIMERSTOPCPU(writingsortedbatch); 

            //TIMERSTARTCPU(clear);
            resetState();

            tempfilenames.emplace_back(std::move(tempfilename));
            numtempfiles++;
            //TIMERSTOPCPU(clear);
            
            //TIMERSTARTCPU(flushclose);
            sortedtempfile.flush();
            sortedtempfile.close();
            //TIMERSTOPCPU(flushclose);
        }
    }
    return tempfilenames;
}


template<class T, class ExtractKey, class KeyComparator>
std::vector<std::string>
binKeySplitIntoSortedChunks(const std::vector<std::string>& infilenames, 
                            const std::string& tempdir, 
                            std::size_t memoryLimit,
                            ExtractKey extractKey, KeyComparator keyComparator){

    constexpr int maxIters = 4;
    for(int i = 0; i < maxIters; i++){
        try{
            return binKeySplitIntoSortedChunksImpl<T>(infilenames, 
                            tempdir, 
                            memoryLimit, 
                            extractKey, keyComparator);
        }catch(...){
            memoryLimit /= 2;
        }
    }

    throw std::runtime_error("Could not sort files");
    return {};
}



template<class T, class Tcomparator>
void
binKeyMergeSortedChunksAndDeleteChunks(const std::string& tempdir,
                                        const std::vector<std::string>& infilenames, 
                                        const std::string& outfilename, 
                                        Tcomparator&& comparator){
    detail::binKeyMergeSortedChunksImpl<T>(true, tempdir, infilenames, outfilename, comparator);
}


template<class T, class Tcomparator>
void
binKeyMergeSortedChunks(const std::string& tempdir,
                        const std::vector<std::string>& infilenames, 
                        const std::string& outfilename, 
                        Tcomparator&& comparator){
    detail::binKeyMergeSortedChunksImpl<T>(false, tempdir, infilenames, outfilename, comparator);
}


//sort infile to outfile
//infile is sorted by this number using comparator Comp
template<class T, class ExtractKey, class KeyComparator, class Tcomparator>
void binKeySort(const std::string& tempdir,
                const std::vector<std::string>& infilenames, 
                const std::string& outfilename,
                std::size_t memoryLimit,
                ExtractKey extractKey, KeyComparator keyComparator,
                Tcomparator&& comparator){

    //TIMERSTARTCPU(split);
    auto tempfilenames = binKeySplitIntoSortedChunks<T>(infilenames, tempdir, memoryLimit, extractKey, keyComparator);
    //TIMERSTOPCPU(split);
    //TIMERSTARTCPU(merge);
    binKeyMergeSortedChunksAndDeleteChunks<T>(tempdir, tempfilenames, outfilename, comparator);
    //TIMERSTOPCPU(merge);
}






} //namespace fixedmemory


























































}    
}



#endif
