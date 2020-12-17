
#include <gpu/gpuminhasherconstruction.cuh>

#include <gpu/gpuminhasher.cuh>
#include <gpu/fakegpuminhasher.cuh>
#include <gpu/singlegpuminhasher.cuh>
#include <gpu/multigpuminhasher.cuh>
#include <minhasher.hpp>

#include <options.hpp>

#include <memory>
#include <utility>


namespace care{
namespace gpu{
    

        std::unique_ptr<FakeGpuMinhasher>
        constructFakeGpuMinhasherFromGpuReadStorage(
            const CorrectionOptions& correctionOptions,
            const FileOptions& fileOptions,
            const SequenceFileProperties& totalInputFileProperties,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const DistributedReadStorage& gpuReadStorage
        ){
            auto gpuMinhasher = std::make_unique<FakeGpuMinhasher>(
                correctionOptions.kmerlength, 
                calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage)
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
                    totalInputFileProperties.nReads, 
                    correctionOptions,
                    gpuReadStorage
                );
            }

            return gpuMinhasher;
        }


        std::unique_ptr<SingleGpuMinhasher>
        constructSingleGpuMinhasherFromGpuReadStorage(
            const CorrectionOptions& correctionOptions,
            const FileOptions& /*fileOptions*/,
            const SequenceFileProperties& totalInputFileProperties,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& /*memoryOptions*/,
            const DistributedReadStorage& gpuReadStorage
        ){
            
            auto gpuMinhasher = std::make_unique<SingleGpuMinhasher>(
                totalInputFileProperties.nReads, 
                calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage), 
                correctionOptions.kmerlength
            );

            gpuMinhasher->constructFromReadStorage(
                runtimeOptions,
                totalInputFileProperties.nReads,
                gpuReadStorage,
                totalInputFileProperties.maxSequenceLength,
                correctionOptions.numHashFunctions
            );

            return gpuMinhasher;
        }

        std::unique_ptr<MultiGpuMinhasher>
        constructMultiGpuMinhasherFromGpuReadStorage(
            const CorrectionOptions& correctionOptions,
            const FileOptions& /*fileOptions*/,
            const SequenceFileProperties& totalInputFileProperties,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& /*memoryOptions*/,
            const DistributedReadStorage& gpuReadStorage
        ){
            
            auto gpuMinhasher = std::make_unique<MultiGpuMinhasher>(
                totalInputFileProperties.nReads, 
                calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage), 
                correctionOptions.kmerlength,
                std::vector<int>{0,0} //runtimeOptions.deviceIds
            );

            gpuMinhasher->constructFromReadStorage(
                runtimeOptions,
                totalInputFileProperties.nReads,
                gpuReadStorage,
                totalInputFileProperties.maxSequenceLength,
                correctionOptions.numHashFunctions
            );

            return gpuMinhasher;
        }

    
        std::pair<std::unique_ptr<GpuMinhasher>, GpuMinhasherType>
        constructGpuMinhasherFromGpuReadStorage(
            const FileOptions& fileOptions,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const CorrectionOptions& correctionOptions,
            const SequenceFileProperties& totalInputFileProperties,
            const DistributedReadStorage& gpuReadStorage,
            GpuMinhasherType requestedType
        ){
            if(requestedType == GpuMinhasherType::Fake){
                return std::make_pair(
                    constructFakeGpuMinhasherFromGpuReadStorage(
                        correctionOptions,
                        fileOptions,
                        totalInputFileProperties,
                        runtimeOptions,
                        memoryOptions,
                        gpuReadStorage
                    ),
                    GpuMinhasherType::Fake
                );
            }else if(requestedType == GpuMinhasherType::Single){
                return std::make_pair(
                    constructSingleGpuMinhasherFromGpuReadStorage(
                        correctionOptions,
                        fileOptions,
                        totalInputFileProperties,
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
                        totalInputFileProperties,
                        runtimeOptions,
                        memoryOptions,
                        gpuReadStorage
                    ),
                    GpuMinhasherType::Multi
                );
            }else{
                return std::make_pair(
                    constructFakeGpuMinhasherFromGpuReadStorage(
                        correctionOptions,
                        fileOptions,
                        totalInputFileProperties,
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