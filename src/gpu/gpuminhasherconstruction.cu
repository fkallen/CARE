
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

        std::string to_string(GpuMinhasherType type){
            switch(type){
                case GpuMinhasherType::Fake: return "FakeGpu";
                case GpuMinhasherType::FakeSingleHash: return "FakeGpuSingleHash";
                case GpuMinhasherType::Single: return "SingleGpu";
                case GpuMinhasherType::SingleSingleHash: return "SingleGpuSingleHash";
                case GpuMinhasherType::Multi: return "MultiGpu";
                case GpuMinhasherType::MultiSingleHash: return "MultiGpuSingleHash";
                case GpuMinhasherType::None: return "None";
                default: return "Unknown";
            }
        }
    
    
        std::pair<std::unique_ptr<GpuMinhasher>, GpuMinhasherType>
        constructGpuMinhasherFromGpuReadStorage(
            const FileOptions& fileOptions,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const CorrectionOptions& correctionOptions,
            const GpuReadStorage& gpuReadStorage,
            GpuMinhasherType requestedType
        ){
            if(requestedType == GpuMinhasherType::Fake || runtimeOptions.warpcore == 0 || correctionOptions.singlehash){
                if(correctionOptions.singlehash){                    
                    return std::make_pair(
                        constructFakeGpuSingleHashMinhasherFromGpuReadStorage(
                            correctionOptions,
                            fileOptions,
                            runtimeOptions,
                            memoryOptions,
                            gpuReadStorage
                        ),
                        GpuMinhasherType::FakeSingleHash
                    );
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
            #ifdef CARE_HAS_WARPCORE
            }else if(requestedType == GpuMinhasherType::Single || runtimeOptions.deviceIds.size() < 2){
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