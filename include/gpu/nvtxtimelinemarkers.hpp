#ifndef NVVPTIMELINEMARKERS_HPP
#define NVVPTIMELINEMARKERS_HPP

#include <string>

namespace nvtx{
    void push_range(const std::string& name, int cid);
    void pop_range(const std::string& name);
    void pop_range();

    struct ScopedRange{
        ScopedRange() : ScopedRange("unnamed", 0){}
        ScopedRange(const std::string& name, int cid){
            push_range(name, cid);
        }
        ~ScopedRange(){
            pop_range();
        }
    };
}


#endif
