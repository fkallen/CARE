#ifndef CARE_DISPATCH_CARE_EXTEND_GPU_CUH
#define CARE_DISPATCH_CARE_EXTEND_GPU_CUH

#include <options.hpp>

#include <vector>

namespace care{

    namespace extension{

        std::vector<int> getUsableDeviceIds(std::vector<int> deviceIds);

    }

    void performExtension(
        ProgramOptions programOptions
    );

}



#endif