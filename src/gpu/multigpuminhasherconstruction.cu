
#include <gpu/gpuminhasherconstruction.cuh>

#include <gpu/multigpuminhasher.cuh>
#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>
#include <utility>


namespace care{
namespace gpu{
    

    #ifdef CARE_HAS_WARPCORE
        std::unique_ptr<MultiGpuMinhasher>
        constructMultiGpuMinhasherFromGpuReadStorage(
            const CorrectionOptions& correctionOptions,
            const FileOptions& /*fileOptions*/,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& /*memoryOptions*/,
            const GpuReadStorage& gpuReadStorage
        ){
            
            auto gpuMinhasher = std::make_unique<MultiGpuMinhasher>(
                gpuReadStorage.getNumberOfReads(), 
                calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage),
                correctionOptions.kmerlength,
                runtimeOptions.deviceIds
            );

            gpuMinhasher->constructFromReadStorage(
                runtimeOptions,
                gpuReadStorage.getNumberOfReads(),
                gpuReadStorage,
                gpuReadStorage.getSequenceLengthUpperBound(),
                correctionOptions.numHashFunctions
            );

            return gpuMinhasher;
        }
    #endif
    
}
}