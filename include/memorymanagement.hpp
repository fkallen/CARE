#ifndef CARE_MEMORY_MANAGEMENT_HPP
#define CARE_MEMORY_MANAGEMENT_HPP

#include <unistd.h>
#include <sys/resource.h>

#include <string>
#include <cstdint>
#include <cassert>
#include <fstream>
#include <limits>
#include <map>


    struct MemoryUsage{
        std::size_t host = 0;
        std::map<int, std::size_t> device;
    };





    __inline__
    std::size_t getAvailableMemoryInKB_linux(){
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

    __inline__ 
    std::size_t getCurrentRSS_linux(){
        std::ifstream in("/proc/self/statm");
        std::size_t tmp, rss;
        in >> tmp >> rss;
        
        return rss * sysconf(_SC_PAGESIZE);
    }

    __inline__
    std::size_t getRSSLimit_linux(){
        rlimit rlim;
        int ret = getrlimit(RLIMIT_RSS, &rlim);
        if(ret != 0){
            std::perror("Could not get RSS limit!");
            return 0;
        }
        return rlim.rlim_cur;    
    }


    __inline__
    std::size_t getAvailableMemoryInKB(){
        //return getAvailableMemoryInKB_linux();

        return std::min(getAvailableMemoryInKB_linux(), (getRSSLimit_linux() - getCurrentRSS_linux()) / 1024);
    };



#endif