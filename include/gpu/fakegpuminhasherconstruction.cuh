#ifndef CARE_FAKEGPUMINHASHERCONSTRUCTION_CUH
#define CARE_FAKEGPUMINHASHERCONSTRUCTION_CUH


#include <gpu/gpureadstorage.cuh>

#include <gpu/fakegpuminhasher.cuh>
#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>

namespace care{
namespace gpu{
    

        std::unique_ptr<FakeGpuMinhasher>
        constructFakeGpuMinhasherFromGpuReadStorage(
            const CorrectionOptions& correctionOptions,
            const FileOptions& fileOptions,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const GpuReadStorage& gpuReadStorage
        );

}
}

















#endif