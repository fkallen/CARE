
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
            const ProgramOptions& programOptions,
            const GpuReadStorage& gpuReadStorage
        ){
            
            auto gpuMinhasher = std::make_unique<MultiGpuMinhasher>(
                gpuReadStorage.getNumberOfReads(), 
                calculateResultsPerMapThreshold(programOptions.estimatedCoverage),
                programOptions.kmerlength,
                programOptions.deviceIds
            );

            gpuMinhasher->constructFromReadStorage(
                programOptions,
                gpuReadStorage.getNumberOfReads(),
                gpuReadStorage,
                gpuReadStorage.getSequenceLengthUpperBound(),
                programOptions.numHashFunctions
            );

            if(programOptions.replicateGpuData){
                bool ok = gpuMinhasher->tryReplication();
                if(ok){
                    std::cerr << "Replicated hash tables to each gpu\n";
                }
            }

            return gpuMinhasher;
        }
    #endif
    
}
}