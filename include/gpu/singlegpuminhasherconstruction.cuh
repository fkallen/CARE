#ifndef CARE_SINGLEGPUMINHASHERCONSTRUCTION_CUH
#define CARE_SINGLEGPUMINHASHERCONSTRUCTION_CUH


#include <gpu/gpureadstorage.cuh>

#include <gpu/singlegpuminhasher.cuh>
#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>

namespace care{
namespace gpu{
    

        std::unique_ptr<SingleGpuMinhasher>
        constructSingleGpuMinhasherFromGpuReadStorage(
            const CorrectionOptions& correctionOptions,
            const FileOptions& fileOptions,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const GpuReadStorage& gpuReadStorage
        );

}
}

















#endif