#ifndef CARE_CPUGPUPROXY_HPP
#define CARE_CPUGPUPROXY_HPP

#include <vector>

namespace care{

    std::vector<int> getUsableDeviceIds(std::vector<int> deviceIds);

}


#endif
