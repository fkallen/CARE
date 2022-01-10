
#include <gpu/gpuminhasherconstruction.cuh>

#include <gpu/singlegpuminhasher.cuh>
#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>
#include <utility>


namespace care{
namespace gpu{
    


    #ifdef CARE_HAS_WARPCORE
        std::unique_ptr<SingleGpuMinhasher>
        constructSingleGpuMinhasherFromGpuReadStorage(
            const ProgramOptions& programOptions,
            const GpuReadStorage& gpuReadStorage
        ){
            
            auto gpuMinhasher = std::make_unique<SingleGpuMinhasher>(
                gpuReadStorage.getNumberOfReads(), 
                calculateResultsPerMapThreshold(programOptions.estimatedCoverage), 
                programOptions.kmerlength
            );

            gpuMinhasher->constructFromReadStorage(
                programOptions,
                gpuReadStorage.getNumberOfReads(),
                gpuReadStorage,
                gpuReadStorage.getSequenceLengthUpperBound(),
                programOptions.numHashFunctions
            );

            return gpuMinhasher;
        }

    #endif
    
}
}