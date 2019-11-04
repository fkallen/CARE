#ifndef CARE_FILESORT_HPP
#define CARE_FILESORT_HPP

#include <filehelpers.hpp>
#include <util.hpp>

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

template<class Index_t, class Comp>
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile, Comp&& comparator);

template<class Index_t>
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile);

namespace detail{

    template<class Index_t>
    struct Data{
        Index_t first;
        std::string second;

        Data() = default;
        Data(Index_t f, std::string s) noexcept 
            : first(f), second(std::move(s)){}
        Data(const Data&) = default;
        Data(Data&&) = default;
        Data& operator=(const Data&) = default;
        Data& operator=(Data&&) = default;

        friend std::ostream& operator<< (std::ostream& stream, const Data& data){
            stream.write(reinterpret_cast<const char*>(&data.first), sizeof(Index_t));
            stream << data.second << "\n";

            return stream;
        }

        friend std::istream& operator>>(std::istream& stream, Data& data){
            stream.read(reinterpret_cast<char*>(&data.first), sizeof(Index_t));
            std::getline(stream, data.second);
            
            return stream;
        }
    };

    template<class Index_t, class Comp>
    void
    binKeyMergeSortedChunksImpl(bool remove, 
                                const std::string& tempdir, 
                                const std::vector<std::string>& infilenames, 
                                const std::string& outfilename, 
                                Comp&& comparator){
                                    
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

                    binKeyMergeTwoFiles<Index_t>(filenames[i], filenames[i+1], outtmpname, comparator);

                    newfilenames.emplace_back(std::move(outtmpname));

                    if(step > 0 || remove){
                        removeFile(filenames[i]);
                        removeFile(filenames[i+1]);
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
            renameFileSameMount(filenames[0], outfilename);
        }else{
            std::cerr << "merge " << filenames[0] << " + " << filenames[1] << " into " <<  outfilename << "\n";
            binKeyMergeTwoFiles<Index_t>(filenames[0], filenames[1], outfilename, comparator);

            if(step > 0 || remove){
                removeFile(filenames[0]);
                removeFile(filenames[1]);
            }
        }
    }

} //namespace detail

//sort multiple textfiles into single sorted output file.
//sort is numeric, ascending, using the number represented by the keyIndex-th token in each line
//keyIndex is 1-based.
__inline__
int gnuTxtNumericSort(
                    const std::string& tempdir, 
                    const std::vector<std::string>& filenames, 
                    const std::string& outfilename, 
                    int keyIndex = 1, 
                    int numThreads = 1){

    assert(std::all_of(filenames.begin(), filenames.end(), [&](const auto& s){return s != outfilename;}));
    assert(tempdir != "/tmp");

    std::stringstream commandbuilder;
    commandbuilder << "sort --parallel=" << numThreads 
                    << " -k" << keyIndex << "," << keyIndex 
                    << " -n"
                    << " -T " << tempdir
                    << " ";
    for(const auto& filename : filenames){
        commandbuilder << "\"" << filename << "\" ";
    }
    commandbuilder << " -o " << "\"" << outfilename << "\"";

    const std::string command = commandbuilder.str();

    std::cerr << "Running shell command: " << command << "\n";

    return std::system(command.c_str());
}

//merge multiple sorted textfiles into single sorted output file.
//sort is numeric, ascending, using the number represented by the keyIndex-th token in each line
//keyIndex is 1-based.
__inline__
int gnuTxtNumericMerge(
                    const std::string& tempdir,
                    const std::vector<std::string>& filenames, 
                    const std::string& outfilename, 
                    int keyIndex = 1, 
                    int numThreads = 1){

    assert(std::all_of(filenames.begin(), filenames.end(), [&](const auto& s){return s != outfilename;}));
    assert(tempdir != "/tmp");

    std::stringstream commandbuilder;
    commandbuilder << "sort --parallel=" << numThreads 
                    << " -k" << keyIndex << "," << keyIndex 
                    << " -n"
                    << " -m"
                    << " -T " << tempdir
                    << " ";
    for(const auto& filename : filenames){
        commandbuilder << "\"" << filename << "\" ";
    }
    commandbuilder << " -o " << "\"" << outfilename << "\"";

    const std::string command = commandbuilder.str();

    std::cerr << "Running shell command: " << command << "\n";
    
    return std::system(command.c_str());
}



//merge two sorted input files into sorted output file
//each line in input files must begin with a number of type Index_t which was written in binary mode.
//input files must be sorted in ascending order by this number
template<class Index_t, class Comp>
void binKeyMergeTwoFiles(
                        const std::string& infile1, 
                        const std::string& infile2, 
                        const std::string& outfile, 
                        Comp&& indexcomparator){
    std::ifstream in1(infile1);
    std::ifstream in2(infile2);
    std::ofstream out(outfile);

    assert(in1);
    assert(in2);
    assert(out);

    using Data_t = detail::Data<Index_t>;

    auto dataComparator = [&](const Data_t& l, const Data_t& r){
        return indexcomparator(l.first, r.first);
    };

    std::merge(std::istream_iterator<Data_t>(in1),
                std::istream_iterator<Data_t>(),
                std::istream_iterator<Data_t>(in2),
                std::istream_iterator<Data_t>(),
                std::ostream_iterator<Data_t>(out, ""),
                dataComparator);

    // detail::Data<Index_t> d1;
    // detail::Data<Index_t> d2;

    // int written = 0; // the element of which file was smaller
    // int numwritten = 0;

    // int numread = 0;

    // while(in1 && in2){

    //     if(written != 2){
    //         if(!detail::dataFromStream(in1, d1)){
    //             break;
    //         }
    //         numread++;
    //     }        
    //     if(written != 1){
    //         if(!detail::dataFromStream(in2, d2)){
    //             break;
    //         }
    //         numread++;
    //     }

    //     if(comparator(d1.first, d2.first)){
    //         detail::dataToStream(out, d1);
    //         written = 1;
    //         numwritten++;
    //     }else{
    //         detail::dataToStream(out, d2);
    //         written = 2;
    //         numwritten++;
    //     }
    // }

    // out.flush();

    // if(in1 && written == 2){
    //     detail::dataToStream(out, d1);
    //     numwritten++;
    // }

    // if(in2 && written == 1){
    //     detail::dataToStream(out, d2);
    //     numwritten++;
    // }

    // while(detail::dataFromStream(in1, d1)){
    //     numread++;

    //     if(written == 1 && comparator(d2.first, d1.first)){
    //         detail::dataToStream(out, d2);
    //         written = 0;
    //         numwritten++;
    //     }
    //     detail::dataToStream(out, d1);
    //     numwritten++;
    // }

    // while(detail::dataFromStream(in2, d2)){
    //     numread++;

    //     if(written == 2 && comparator(d1.first, d2.first)){
    //         detail::dataToStream(out, d1);
    //         written = 0;
    //         numwritten++;
    //     }
    //     detail::dataToStream(out, d2);
    //     numwritten++;
    // }

    // assert(numread == numwritten);
}

template<class Index_t>
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile){
    binKeyMergeTwoFiles(infile1, infile2, outfile, std::less<Index_t>{});
}

//split input files into sorted chunks. returns filenames of sorted chunks
//each line in infile must begin with a number of type Index_t which was written in binary mode.
//infile is sorted by this number using comparator Comp
//sorted chunks will be named tempdir/temp_i where i is the chunk number
template<class Index_t, class Comp>
std::vector<std::string>
binKeySplitIntoSortedChunks(const std::vector<std::string>& infilenames, 
                            const std::string& tempdir, 
                            Comp&& indexcomparator){


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

    std::cerr << "Available memory for sort: " << availableMemory / 2 << "\n";

    availableMemory /= 2;

    std::size_t availableGPUMemory = 0;
    #ifdef USE_THRUST    
    {
        std::size_t free, total;
        cudaMemGetInfo(&free, &total); CUERR;
        availableGPUMemory = free;

        constexpr std::size_t MB256 = std::size_t(1) << 28; 
        constexpr std::size_t gpusafetybuffer = MB256;

        if(availableGPUMemory > gpusafetybuffer){
            availableGPUMemory -= gpusafetybuffer;
        }else{
            availableGPUMemory = 0;
        }
        if(availableGPUMemory > MB256){
            //round down to next multiple of 256 MB
            availableGPUMemory = (availableGPUMemory / MB256) * MB256;
        }

        std::cerr << "Available gpu memory: " << availableGPUMemory << "\n";

        std::cerr << "Available gpu memory for sort: " << availableGPUMemory / 4 << "\n";

        availableGPUMemory /= 2;
    }
    #endif
    
    //constexpr int itemsPerTempFile = 100;

    using Data_t = detail::Data<Index_t>;

    auto dataComparator = [&](const Data_t& l, const Data_t& r){
        return indexcomparator(l.first, r.first);
    };

    Data_t item;
    constexpr auto dataSize = sizeof(Data_t);

    std::vector<Data_t> buffer;

    std::size_t stringmem = 0;

    auto couldAddElementToBuffer = [&](){
        bool result = true;
        #ifdef USE_THRUST
            result = result && (2 * numberBuffer.capacity() * dataSize) < availableGPUMemory;
        #endif
        //buffer.size() < itemsPerTempFile
        if(buffer.size() >= buffer.capacity()){
            result = result && (stringmem + 2 * buffer.capacity() * dataSize) < availableMemory;
        }
        return result;
    };

    std::int64_t numtempfiles = 0;
    std::vector<std::string> tempfilenames;

    //split input files into sorted temp files
    for(const auto& filename : infilenames){
        std::ifstream istream(filename);
        if(!istream){
            assert(false);
        }

        stringmem = 0;
        buffer.clear();

        while(bool(istream >> item)){
            stringmem += item.second.capacity();
            buffer.emplace_back(std::move(item));

            TIMERSTARTCPU(readingbatch);
            while(couldAddElementToBuffer()
                    && bool(istream >> item)){

                stringmem += item.second.capacity();
                
                buffer.emplace_back(std::move(item));
            }
            TIMERSTOPCPU(readingbatch);

            std::string tempfilename(tempdir+"/tmp_"+std::to_string(numtempfiles));
            std::ofstream sortedtempfile(tempfilename);

            TIMERSTARTCPU(actualsort);
            std::cerr << "sort " << buffer.size() << " elements in memory " <<  tempfilename << "\n";

            std::sort(buffer.begin(), buffer.end(), dataComparator);

            TIMERSTOPCPU(actualsort);

            TIMERSTARTCPU(writingsortedbatch);
            std::cerr << "save " << buffer.size() << " elements to file " <<  tempfilename << "\n";
            std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<Data_t>(sortedtempfile, ""));

            TIMERSTOPCPU(writingsortedbatch); 

            TIMERSTARTCPU(clear);
            buffer.clear();
            stringmem = 0;
            tempfilenames.emplace_back(std::move(tempfilename));
            numtempfiles++;
            TIMERSTOPCPU(clear);
            
            TIMERSTARTCPU(flushclose);
            sortedtempfile.flush();
            sortedtempfile.close();
            TIMERSTOPCPU(flushclose);
        }
    }
    return tempfilenames;
}



template<class Index_t, class Comp>
void
binKeyMergeSortedChunksAndDeleteChunks(const std::string& tempdir,
                                        const std::vector<std::string>& infilenames, 
                                        const std::string& outfilename, 
                                        Comp&& indexcomparator){
    detail::binKeyMergeSortedChunksImpl<Index_t>(true, tempdir, infilenames, outfilename, indexcomparator);
}

template<class Index_t, class Comp>
void
binKeyMergeSortedChunksAndDeleteChunks(const std::string& tempdir,
                                        const std::vector<std::string>& infilenames, 
                                        const std::string& outfilename){
    binKeyMergeSortedChunksAndDeleteChunks<Index_t>(tempdir, infilenames, outfilename, std::less<Index_t>{});
}

template<class Index_t, class Comp>
void
binKeyMergeSortedChunks(const std::string& tempdir,
                        const std::vector<std::string>& infilenames, 
                        const std::string& outfilename, 
                        Comp&& indexcomparator){
    detail::binKeyMergeSortedChunksImpl<Index_t>(false, tempdir, infilenames, outfilename, indexcomparator);
}

template<class Index_t, class Comp>
void
binKeyMergeSortedChunks(const std::string& tempdir,
                        const std::vector<std::string>& infilenames, 
                        const std::string& outfilename){
    binKeyMergeSortedChunks<Index_t>(infilenames, tempdir, outfilename, std::less<Index_t>{});
}


//sort infile to outfile
//each line in infile must begin with a number of type Index_t which was written in binary mode.
//infile is sorted by this number using comparator Comp
template<class Index_t, class Comp>
void binKeySort(const std::string& tempdir,
                const std::vector<std::string>& infilenames, 
                const std::string& outfilename,
                Comp&& indexcomparator){
    TIMERSTARTCPU(split);
    auto tempfilenames = binKeySplitIntoSortedChunks<Index_t>(infilenames, tempdir, indexcomparator);
    TIMERSTOPCPU(split);
    TIMERSTARTCPU(merge);
    binKeyMergeSortedChunksAndDeleteChunks<Index_t>(tempdir, tempfilenames, outfilename, indexcomparator);
    TIMERSTOPCPU(merge);
}

template<class Index_t>
void binKeySort(const std::string& tempdir,
                const std::vector<std::string>& infilenames, 
                const std::string& outfilename){

    binKeySort<Index_t>(tempdir, infilenames, outfilename, std::less<Index_t>{});
}


}    
}



#endif
