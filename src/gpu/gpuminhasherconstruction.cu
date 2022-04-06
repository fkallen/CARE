
#include <gpu/gpuminhasherconstruction.cuh>

#include <gpu/gpuminhasher.cuh>
#include <gpu/fakegpuminhasherconstruction.cuh>
#include <gpu/singlegpuminhasherconstruction.cuh>
//#include <gpu/multigpuminhasherconstruction.cuh>
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
            const ProgramOptions& programOptions,
            const GpuReadStorage& gpuReadStorage,
            GpuMinhasherType requestedType
        ){
            if(requestedType == GpuMinhasherType::Fake || programOptions.warpcore == 0 || programOptions.singlehash){
                if(programOptions.singlehash){
                    assert(false);
                    // return std::make_pair(
                    //     constructFakeGpuSingleHashMinhasherFromGpuReadStorage(
                    //         programOptions,
                    //         gpuReadStorage
                    //     ),
                    //     GpuMinhasherType::FakeSingleHash
                    // );
                }else{
                    return std::make_pair(
                        constructFakeGpuMinhasherFromGpuReadStorage(
                            programOptions,
                            gpuReadStorage
                        ),
                        GpuMinhasherType::Fake
                    );
                }
            #ifdef CARE_HAS_WARPCORE
            }else if(requestedType == GpuMinhasherType::Single || programOptions.deviceIds.size() < 2){
                return std::make_pair(
                    constructSingleGpuMinhasherFromGpuReadStorage(
                        programOptions,
                        gpuReadStorage
                    ),
                    GpuMinhasherType::Single
                );
            // }else if(requestedType == GpuMinhasherType::Multi){
            //     return std::make_pair(
            //         constructMultiGpuMinhasherFromGpuReadStorage(
            //             programOptions,
            //             gpuReadStorage
            //         ),
            //         GpuMinhasherType::Multi
            //     );
            #endif
            }else{
                return std::make_pair(
                    constructFakeGpuMinhasherFromGpuReadStorage(
                        programOptions,
                        gpuReadStorage
                    ),
                    GpuMinhasherType::Fake
                );
            }
        }
    
    
}
}