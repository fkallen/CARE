
#include <gpu/gpuminhasherconstruction.cuh>

#include <gpu/gpuminhasher.cuh>
#include <gpu/fakegpuminhasherconstruction.cuh>
#include <gpu/singlegpuminhasherconstruction.cuh>
#include <gpu/multigpuminhasherconstruction.cuh>
#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>
#include <utility>


namespace care{
namespace gpu{
    
    
        std::pair<std::unique_ptr<GpuMinhasher>, GpuMinhasherType>
        constructGpuMinhasherFromGpuReadStorage(
            const FileOptions& fileOptions,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const CorrectionOptions& correctionOptions,
            const GpuReadStorage& gpuReadStorage,
            GpuMinhasherType requestedType
        ){
            if(requestedType == GpuMinhasherType::Fake || runtimeOptions.warpcore == 0){
                return std::make_pair(
                    constructFakeGpuMinhasherFromGpuReadStorage(
                        correctionOptions,
                        fileOptions,
                        runtimeOptions,
                        memoryOptions,
                        gpuReadStorage
                    ),
                    GpuMinhasherType::Fake
                );
            #ifdef CARE_HAS_WARPCORE
            }else if(requestedType == GpuMinhasherType::Single){
                return std::make_pair(
                    constructSingleGpuMinhasherFromGpuReadStorage(
                        correctionOptions,
                        fileOptions,
                        runtimeOptions,
                        memoryOptions,
                        gpuReadStorage
                    ),
                    GpuMinhasherType::Single
                );
            }else if(requestedType == GpuMinhasherType::Multi){
                return std::make_pair(
                    constructMultiGpuMinhasherFromGpuReadStorage(
                        correctionOptions,
                        fileOptions,
                        runtimeOptions,
                        memoryOptions,
                        gpuReadStorage
                    ),
                    GpuMinhasherType::Multi
                );
            #endif
            }else{
                return std::make_pair(
                    constructFakeGpuMinhasherFromGpuReadStorage(
                        correctionOptions,
                        fileOptions,
                        runtimeOptions,
                        memoryOptions,
                        gpuReadStorage
                    ),
                    GpuMinhasherType::Fake
                );
            }
        }
    
    
}
}