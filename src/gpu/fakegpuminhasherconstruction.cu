
#include <gpu/fakegpuminhasherconstruction.cuh>

#include <gpu/fakegpuminhasher.cuh>
#include <gpu/fakegpusinglehashminhasher.cuh>

#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>
#include <utility>


namespace care{
namespace gpu{


        std::unique_ptr<FakeGpuMinhasher>
        constructFakeGpuMinhasherFromGpuReadStorage(
            const ProgramOptions& programOptions,
            const GpuReadStorage& gpuReadStorage
        ){
            float loadfactor = programOptions.hashtableLoadfactor;
            
            auto gpuMinhasher = std::make_unique<FakeGpuMinhasher>(
                gpuReadStorage.getNumberOfReads(),
                calculateResultsPerMapThreshold(programOptions.estimatedCoverage),
                programOptions.kmerlength,
                loadfactor
            );

            if(programOptions.load_hashtables_from != ""){

                std::ifstream is(programOptions.load_hashtables_from);
                assert((bool)is);
    
                const int loadedMaps = gpuMinhasher->loadFromStream(is, programOptions.numHashFunctions);
    
                std::cout << "Loaded " << loadedMaps << " hash tables from " << programOptions.load_hashtables_from << std::endl;
            }else{
                gpuMinhasher->constructFromReadStorage(
                    programOptions,
                    gpuReadStorage.getNumberOfReads(), 
                    gpuReadStorage
                );
            }

            return gpuMinhasher;
        }

        std::unique_ptr<FakeGpuSingleHashMinhasher>
        constructFakeGpuSingleHashMinhasherFromGpuReadStorage(
            const ProgramOptions& programOptions,
            const GpuReadStorage& gpuReadStorage
        ){
            float loadfactor = programOptions.hashtableLoadfactor;
            
            auto gpuMinhasher = std::make_unique<FakeGpuSingleHashMinhasher>(
                gpuReadStorage.getNumberOfReads(),
                255,
                programOptions.kmerlength,
                loadfactor
            );

            if(programOptions.load_hashtables_from != ""){

                std::ifstream is(programOptions.load_hashtables_from);
                assert((bool)is);

                const int loadedMaps = gpuMinhasher->loadFromStream(is, programOptions.numHashFunctions);

                std::cout << "Loaded " << loadedMaps << " hash tables from " << programOptions.load_hashtables_from << std::endl;
            }else{
                gpuMinhasher->constructFromReadStorage(
                    programOptions,
                    gpuReadStorage.getNumberOfReads(), 
                    gpuReadStorage
                );
            }

            return gpuMinhasher;
        }
    
}
}