#if 0

#ifndef CARE_FAKEGPUMINHASHERCONSTRUCTION_CUH
#define CARE_FAKEGPUMINHASHERCONSTRUCTION_CUH


#include <gpu/gpureadstorage.cuh>

#include <gpu/fakegpuminhasher.cuh>
//#include <gpu/fakegpusinglehashminhasher.cuh>
#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>

namespace care{
namespace gpu{
    

    std::unique_ptr<FakeGpuMinhasher>
    constructFakeGpuMinhasherFromGpuReadStorage(
        const ProgramOptions& programOptions,
        const GpuReadStorage& gpuReadStorage
    );

    // std::unique_ptr<FakeGpuSingleHashMinhasher>
    // constructFakeGpuSingleHashMinhasherFromGpuReadStorage(
    //     const ProgramOptions& programOptions,
    //     const GpuReadStorage& gpuReadStorage
    // );

}
}

#endif
















#endif