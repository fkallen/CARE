#ifndef CARE_SINGLEGPUMINHASHERCONSTRUCTION_CUH
#define CARE_SINGLEGPUMINHASHERCONSTRUCTION_CUH

#ifdef CARE_HAS_WARPCORE

#include <gpu/gpureadstorage.cuh>

#include <gpu/singlegpuminhasher.cuh>
#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>

namespace care{
namespace gpu{
    

        std::unique_ptr<SingleGpuMinhasher>
        constructSingleGpuMinhasherFromGpuReadStorage(
            const ProgramOptions& programOptions,
            const GpuReadStorage& gpuReadStorage
        );

}
}

#endif

















#endif