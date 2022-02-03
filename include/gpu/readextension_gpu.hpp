#ifndef CARE_READEXTENSION_GPU_HPP
#define CARE_READEXTENSION_GPU_HPP


#include <config.hpp>
#include <options.hpp>
#include <extendedread.hpp>

#include <gpu/gpuminhasher.cuh>
#include <gpu/gpureadstorage.cuh>

#include <functional>

namespace care{
namespace gpu{

    using SubmitReadyExtensionResultsCallback = std::function<void(
        std::vector<ExtendedRead> extendedReads, 
        std::vector<EncodedExtendedRead> encodedExtendedReads,
        std::vector<read_number> idsOfNotExtendedReads
    )>;
    
    void extend_gpu(
        const ProgramOptions& programOptions,
        const GpuMinhasher& minhasher,
        const GpuReadStorage& gpuReadStorage,
        SubmitReadyExtensionResultsCallback submitReadyResults //needs to be thread-safe
    );



} //namespace gpu

} //namespace care






#endif
