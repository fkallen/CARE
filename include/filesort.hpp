#ifndef CARE_FILESORT_HPP
#define CARE_FILESORT_HPP

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

namespace care{
namespace filesort{

template<class Index_t, class Comp>
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile, Comp&& comparator);

template<class Index_t>
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile);

namespace detail{

    template<class Index_t>
    using Data = std::pair<Index_t, std::string>;

    template<class Index_t>
    bool dataLessThan(const Data<Index_t>& l, const Data<Index_t>& r){
        return l.first < r.first;
    }

    template<class Index_t>
    bool dataFromStream(std::ifstream& stream, Index_t& number, std::string& line){

        bool result = bool(stream) && bool(stream.read(reinterpret_cast<char*>(&number), sizeof(Index_t)))
                        && bool(std::getline(stream, line));

        return result;
    }

    template<class Index_t>
    bool dataFromStream(std::ifstream& stream, Data<Index_t>& d){
        bool result = dataFromStream(stream, d.first, d.second);

        return result;
    }

    template<class Index_t>
    bool dataToStream(std::ofstream& stream, const Index_t& number, const std::string& line){
        stream.write(reinterpret_cast<const char*>(&number), sizeof(Index_t));
        stream << line << "\n";
        return bool(stream);
    }

    template<class Index_t>
    bool dataToStream (std::ofstream& stream, const Data<Index_t>& d){
        return dataToStream(stream, d.first, d.second);
    }

    template<class Index_t, class Comp>
    void
    binKeyMergeSortedChunksImpl(bool remove, const std::vector<std::string>& infilenames, const std::string& outfilename, Comp&& comparator){
        //merge the temp files
        std::vector<std::string> tempfilenames = infilenames;
        std::vector<std::string> newtempfilenames;

        int step = 0;
        while(tempfilenames.size() > 2){
            const int numtempfiles = tempfilenames.size();
            for(int i = 0; i < numtempfiles ; i += 2){
                //merge tempfile i with i+1
                if(i+1 < numtempfiles){
                    std::string outtmpname(tempfilenames[i]+"-"+std::to_string(step));

                    std::cerr << "merge " << tempfilenames[i] << " + " << tempfilenames[i+1] << " into " <<  outtmpname << "\n";

                    binKeyMergeTwoFiles<Index_t>(tempfilenames[i], tempfilenames[i+1], outtmpname, comparator);

                    newtempfilenames.emplace_back(std::move(outtmpname));

                    if(step > 0 || remove){
                        std::cerr << "delete " << tempfilenames[i] << "\n";
                        std::cerr << "delete " << tempfilenames[i+1] << "\n";

                        int res1 = std::remove(tempfilenames[i].c_str());
                        int res2 = std::remove(tempfilenames[i+1].c_str());
                        if(res1 != 0){
                            std::perror("remove");
                        }
                        if(res2 != 0){
                            std::perror("remove");
                        }
                        assert(res1 == 0);
                        assert(res2 == 0);
                    }
                    
                }else{
                    newtempfilenames.emplace_back(tempfilenames[i]);
                }
            }

            tempfilenames = std::move(newtempfilenames);
            step++;
        }

        assert(tempfilenames.size() > 0);

        if(tempfilenames.size() == 1){
            std::cerr << "Rename " << tempfilenames[0] << " to " << outfilename << "\n";
            int res = std::rename(tempfilenames[0].c_str(), outfilename.c_str());
            if(res != 0){
                std::perror("rename");
            }
            assert(res == 0);
        }else{
            std::cerr << "merge " << tempfilenames[0] << " + " << tempfilenames[1] << " into " <<  outfilename << "\n";
            binKeyMergeTwoFiles<Index_t>(tempfilenames[0], tempfilenames[1], outfilename, comparator);

            if(step > 0 || remove){
                std::cerr << "delete " << tempfilenames[0] << "\n";
                std::cerr << "delete " << tempfilenames[1] << "\n";

                int res1 = std::remove(tempfilenames[0].c_str());
                int res2 = std::remove(tempfilenames[1].c_str());
                if(res1 != 0){
                    std::perror("remove");
                }
                if(res2 != 0){
                    std::perror("remove");
                }
                assert(res1 == 0);
                assert(res2 == 0);
            }
        }
    }

} //namespace detail

//sort multiple textfiles into single sorted output file.
//sort is numeric, ascending, using the number represented by the keyIndex-th token in each line
//keyIndex is 1-based.
__inline__
int gnuTxtNumericSort(const std::vector<std::string>& filenames, 
                    const std::string& outfilename, 
                    int keyIndex = 1, 
                    int numThreads = 1){

    assert(std::all_of(filenames.begin(), filenames.end(), [&](const auto& s){return s != outfilename;}));

    std::stringstream commandbuilder;
    commandbuilder << "sort --parallel=" << numThreads <<" -k" << keyIndex << "," << keyIndex << " -n ";
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
int gnuTxtNumericMerge(const std::vector<std::string>& filenames, 
                    const std::string& outfilename, 
                    int keyIndex = 1, 
                    int numThreads = 1){

    assert(std::all_of(filenames.begin(), filenames.end(), [&](const auto& s){return s != outfilename;}));

    std::stringstream commandbuilder;
    commandbuilder << "sort --parallel=" << keyIndex <<" -k" << keyIndex << "," << keyIndex << " -n -m";
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
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile, Comp&& comparator){
    std::ifstream in1(infile1);
    std::ifstream in2(infile2);
    std::ofstream out(outfile);

    assert(in1);
    assert(in2);
    assert(out);

    detail::Data<Index_t> d1;
    detail::Data<Index_t> d2;

    int written = 0; // the element of which file was smaller
    int numwritten = 0;

    int numread = 0;

    while(in1 && in2){

        if(written != 2){
            if(!detail::dataFromStream(in1, d1)){
                break;
            }
            numread++;
        }        
        if(written != 1){
            if(!detail::dataFromStream(in2, d2)){
                break;
            }
            numread++;
        }

        if(comparator(d1.first, d2.first)){
            detail::dataToStream(out, d1);
            written = 1;
            numwritten++;
        }else{
            detail::dataToStream(out, d2);
            written = 2;
            numwritten++;
        }
    }

    out.flush();

    if(in1 && written == 2){
        detail::dataToStream(out, d1);
        numwritten++;
    }

    if(in2 && written == 1){
        detail::dataToStream(out, d2);
        numwritten++;
    }

    while(detail::dataFromStream(in1, d1)){
        numread++;

        if(written == 1 && comparator(d2.first, d1.first)){
            detail::dataToStream(out, d2);
            written = 0;
            numwritten++;
        }
        detail::dataToStream(out, d1);
        numwritten++;
    }

    while(detail::dataFromStream(in2, d2)){
        numread++;

        if(written == 2 && comparator(d1.first, d2.first)){
            detail::dataToStream(out, d1);
            written = 0;
            numwritten++;
        }
        detail::dataToStream(out, d2);
        numwritten++;
    }

    assert(numread == numwritten);
}

template<class Index_t>
void binKeyMergeTwoFiles(const std::string& infile1, const std::string& infile2, const std::string& outfile){
    binKeyMergeTwoFiles(infile1, infile2, outfile, std::less<Index_t>{});
}

//split input files into sorted chunks. returns filenames of sorted chunks
//each line in infile must begin with a number of type Index_t which was written in binary mode.
//infile is sorted by this number using comparator Comp
//sorted chunks will be named prefix_temp_i where i is the chunk number
template<class Index_t, class Comp>
std::vector<std::string>
binKeySplitIntoSortedChunks(const std::vector<std::string>& infilenames, const std::string& prefix, Comp&& comparator){

    auto getAvailableMemoryInKB = []() -> std::size_t {
        //https://stackoverflow.com/questions/349889/how-do-you-determine-the-amount-of-linux-system-ram-in-c
        std::string token;
        std::ifstream file("/proc/meminfo");
        assert(bool(file));
        while(file >> token) {
            if(token == "MemAvailable:") {
                std::size_t mem;
                if(file >> mem) {
                    return mem;
                } else {
                    return 0;       
                }
            }
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        return 0;
    };

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

    detail::Data<Index_t> item;
    constexpr auto dataSize = sizeof(detail::Data<Index_t>);

    std::vector<detail::Data<Index_t>> buffer;
    std::vector<Index_t> numberBuffer;
    std::vector<std::string> stringBuffer;

    std::size_t stringmem = 0;

    auto couldAddElementToBuffer = [&](){
        bool result = true;
        #ifdef USE_THRUST
            result = result && (2 * numberBuffer.capacity() * dataSize) < availableGPUMemory;
        #endif
        //buffer.size() < itemsPerTempFile
        if(numberBuffer.size() >= numberBuffer.capacity()){
            result = result && (stringmem + 2 * numberBuffer.capacity() * dataSize) < availableMemory;
        }
        return result;
    };

    int numtempfiles = 0;
    std::vector<std::string> tempfilenames;

    //split input files into sorted temp files
    for(const auto& filename : infilenames){
        std::ifstream istream(filename);
        if(!istream){
            assert(false);
        }

        stringmem = 0;
        buffer.clear(),
        numberBuffer.clear();
        stringBuffer.clear();

        while(detail::dataFromStream(istream, item)){
            stringmem += item.second.capacity();
            //buffer.emplace_back(std::move(item));
            numberBuffer.emplace_back(item.first);
            stringBuffer.emplace_back(std::move(item.second));

            while(couldAddElementToBuffer()
                    && detail::dataFromStream(istream, item)){

                stringmem += item.second.capacity();
                
                //buffer.emplace_back(std::move(item));
                numberBuffer.emplace_back(item.first);
                stringBuffer.emplace_back(std::move(item.second));
            }

            std::string tempfilename(prefix+"_tmp_"+std::to_string(numtempfiles));
            std::ofstream sortedtempfile(tempfilename);

            std::vector<int> indices(numberBuffer.size());

            #ifdef USE_THRUST
                std::cerr << "gpu sort " << buffer.size() << " elements into " <<  tempfilename << "\n";

                thrust::device_vector<int> d_indices = indices;
                thrust::device_vector<Index_t> d_numbers = numberBuffer;
                auto dnumbersPtr = thrust::raw_pointer_cast(d_numbers.data());
                thrust::sequence(d_indices.begin(), d_indices.end(), 0);
                thrust::sort(d_indices.begin(), d_indices.end(), [=] __device__ (auto l, auto r){
                    return dnumbersPtr[l] < dnumbersPtr[r];
                });
                //thrust::device_vector<Index_t> d_sortednumbers(d_numbers.size());            
                //thrust::copy(thrust::make_permutation_iterator(d_numbers.begin(), d_indices.begin()),
                //            thrust::make_permutation_iterator(d_numbers.begin(), d_indices.end()),
                //            d_sortednumbers.begin());
                thrust::copy(d_indices.begin(), d_indices.end(), indices.begin());
                //thrust::copy(d_sortednumbers.begin(), d_sortednumbers.end(), numberBuffer.begin());

                for(int i = 0; i < int(indices.size()); i++){
                    int position = indices[i];
                    detail::dataToStream(sortedtempfile, numberBuffer[position], stringBuffer[position]);
                }
            #else     
                std::cerr << "sort " << indices.size() << " elements into " <<  tempfilename << "\n";

                std::iota(indices.begin(), indices.end(), 0);
                ///std::sort(buffer.begin(), buffer.end());
                std::sort(indices.begin(), indices.end(), [&](auto l, auto r){
                    return numberBuffer[l] < numberBuffer[r];
                });
                for(auto i : indices){
                    detail::dataToStream(sortedtempfile, numberBuffer[i], stringBuffer[i]);
                }
            #endif       

            buffer.clear();
            stringmem = 0;
            tempfilenames.emplace_back(std::move(tempfilename));
            numtempfiles++;
        }
    }
    return tempfilenames;
}



template<class Index_t, class Comp>
void
binKeyMergeSortedChunksAndDeleteChunks(const std::vector<std::string>& infilenames, const std::string& outfilename, Comp&& comparator){
    detail::binKeyMergeSortedChunksImpl<Index_t>(true, infilenames, outfilename, comparator);
}

template<class Index_t, class Comp>
void
binKeyMergeSortedChunksAndDeleteChunks(const std::vector<std::string>& infilenames, const std::string& outfilename){
    binKeyMergeSortedChunksAndDeleteChunks<Index_t>(infilenames, outfilename, std::less<Index_t>{});
}

template<class Index_t, class Comp>
void
binKeyMergeSortedChunks(const std::vector<std::string>& infilenames, const std::string& outfilename, Comp&& comparator){
    detail::binKeyMergeSortedChunksImpl<Index_t>(false, infilenames, outfilename, comparator);
}

template<class Index_t, class Comp>
void
binKeyMergeSortedChunks(const std::vector<std::string>& infilenames, const std::string& outfilename){
    binKeyMergeSortedChunks<Index_t>(infilenames, outfilename, std::less<Index_t>{});
}


//sort infile to outfile
//each line in infile must begin with a number of type Index_t which was written in binary mode.
//infile is sorted by this number using comparator Comp
//the absolute filepath of each temporary file will begin with tmpprefix
template<class Index_t, class Comp>
void binKeySort(const std::vector<std::string>& infilenames, const std::string& outfilename, const std::string& tmpprefix, Comp&& comparator){

    auto tempfilenames = binKeySplitIntoSortedChunks<Index_t>(infilenames, tmpprefix, comparator);

    binKeyMergeSortedChunksAndDeleteChunks<Index_t>(tempfilenames, outfilename, comparator);
}

template<class Index_t>
void binKeySort(const std::vector<std::string>& infilenames, const std::string& outfilename, const std::string& tmpprefix){
    binKeySort<Index_t>(infilenames, outfilename, tmpprefix, std::less<Index_t>{});
}


}    
}



#endif