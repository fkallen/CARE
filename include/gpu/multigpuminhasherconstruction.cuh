#ifndef CARE_MULTIGPUMINHASHERCONSTRUCTION_CUH
#define CARE_MULTIGPUMINHASHERCONSTRUCTION_CUH


#ifdef CARE_HAS_WARPCORE

#include <gpu/gpureadstorage.cuh>

#include <gpu/multigpuminhasher.cuh>
#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>

namespace care{
namespace gpu{
    

        std::unique_ptr<MultiGpuMinhasher>
        constructMultiGpuMinhasherFromGpuReadStorage(
            const ProgramOptions& programOptions,
            const GpuReadStorage& gpuReadStorage
        );

}
}

#endif














#endif