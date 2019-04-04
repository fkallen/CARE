#include <cpugpuproxy.hpp>

#include <config.hpp>

#include <hpc_helpers.cuh>

#include <algorithm>
#include <iostream>
#include <vector>

namespace care{

    std::vector<int> getUsableDeviceIds(std::vector<int> deviceIds){
        int nDevices;

        cudaGetDeviceCount(&nDevices); CUERR;

        std::vector<int> invalidIds;

        for(int id : deviceIds) {
            if(id >= nDevices) {
                invalidIds.emplace_back(id);
                std::cout << "Found invalid device Id: " << id << std::endl;
            }
        }

        if(invalidIds.size() > 0) {
            std::cout << "Available GPUs on your machine:" << std::endl;
            for(int j = 0; j < nDevices; j++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, j); CUERR;
                std::cout << "Id " << j << " : " << prop.name << std::endl;
            }

            for(int invalidid : invalidIds) {
                deviceIds.erase(std::find(deviceIds.begin(), deviceIds.end(), invalidid));
            }
        }

        return deviceIds;
    }

}
