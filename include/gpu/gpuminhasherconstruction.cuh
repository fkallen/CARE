#ifndef CARE_GPUMINHASHERCONSTRUCTION_CUH
#define CARE_GPUMINHASHERCONSTRUCTION_CUH


#include <gpu/gpuminhasher.cuh>
#include <gpu/gpureadstorage.cuh>

#include <options.hpp>

#include <memory>
#include <utility>

namespace care{
namespace gpu{

    enum class GpuMinhasherType{
        Fake,
        Single,
        Multi,
        None
    };

    std::pair<std::unique_ptr<GpuMinhasher>, GpuMinhasherType>
    constructGpuMinhasherFromGpuReadStorage(
        const FileOptions &fileOptions,
        const RuntimeOptions &runtimeOptions,
        const MemoryOptions& memoryOptions,
        const CorrectionOptions& correctionOptions,
        const SequenceFileProperties& totalInputFileProperties,
        const GpuReadStorage& gpuReadStorage,
        GpuMinhasherType requestedType = GpuMinhasherType::None
    );


}
}

















#endif