#ifndef NVVPTIMELINEMARKERS_HPP
#define NVVPTIMELINEMARKERS_HPP

namespace nvtx{
    void push_range(const std::string& name, int cid);
    void pop_range(const std::string& name);
    void pop_range();
}


#endif
