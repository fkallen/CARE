
#include <gpu/fakegpuminhasherconstruction.cuh>

#include <gpu/fakegpuminhasher.cuh>
#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>
#include <utility>


namespace care{
namespace gpu{
    

        std::unique_ptr<FakeGpuMinhasher>
        constructFakeGpuMinhasherFromGpuReadStorage(
            const CorrectionOptions& correctionOptions,
            const FileOptions& fileOptions,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const GpuReadStorage& gpuReadStorage
        ){
            auto gpuMinhasher = std::make_unique<FakeGpuMinhasher>(
                gpuReadStorage.getNumberOfReads(),
                calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage),
                correctionOptions.kmerlength
            );

            if(fileOptions.load_hashtables_from != ""){

                std::ifstream is(fileOptions.load_hashtables_from);
                assert((bool)is);
    
                const int loadedMaps = gpuMinhasher->loadFromStream(is, correctionOptions.numHashFunctions);
    
                std::cout << "Loaded " << loadedMaps << " hash tables from " << fileOptions.load_hashtables_from << std::endl;
            }else{
                gpuMinhasher->constructFromReadStorage(
                    fileOptions,
                    runtimeOptions,
                    memoryOptions,
                    gpuReadStorage.getNumberOfReads(), 
                    correctionOptions,
                    gpuReadStorage
                );
            }

            return gpuMinhasher;
        }    
    
}
}