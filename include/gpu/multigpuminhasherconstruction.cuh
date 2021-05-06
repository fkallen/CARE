#ifndef CARE_MULTIGPUMINHASHERCONSTRUCTION_CUH
#define CARE_MULTIGPUMINHASHERCONSTRUCTION_CUH

#include <gpu/gpureadstorage.cuh>

#include <gpu/multigpuminhasher.cuh>
#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>

namespace care{
namespace gpu{
    

        std::unique_ptr<MultiGpuMinhasher>
        constructMultiGpuMinhasherFromGpuReadStorage(
            const CorrectionOptions& correctionOptions,
            const FileOptions& fileOptions,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const GpuReadStorage& gpuReadStorage
        );

}
}
















#endif