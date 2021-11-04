
#include <cpuminhasherconstruction.hpp>
#include <cpureadstorage.hpp>
#include <cpuminhasher.hpp>
#include <ordinaryminhasher.hpp>
#include <singlehashminhasher.hpp>

#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>
#include <utility>


namespace care{

        std::string to_string(CpuMinhasherType type){
            switch(type){
                case CpuMinhasherType::Ordinary: return "Ordinary";
                case CpuMinhasherType::OrdinarySingleHash: return "OrdinarySingleHash";
                case CpuMinhasherType::None: return "None";
                default: return "Unknown";
            }
        }

        std::unique_ptr<OrdinaryCpuMinhasher>
        constructOrdinaryCpuMinhasherFromCpuReadStorage(
            const CorrectionOptions& correctionOptions,
            const FileOptions& fileOptions,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const CpuReadStorage& cpuReadStorage
        ){
            float loadfactor = memoryOptions.hashtableLoadfactor;

            auto cpuMinhasher = std::make_unique<OrdinaryCpuMinhasher>(
                cpuReadStorage.getNumberOfReads(),
                calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage),
                correctionOptions.kmerlength,
                loadfactor
            );

            if(fileOptions.load_hashtables_from != ""){

                std::ifstream is(fileOptions.load_hashtables_from);
                assert((bool)is);
    
                const int loadedMaps = cpuMinhasher->loadFromStream(is, correctionOptions.numHashFunctions);
    
                std::cout << "Loaded " << loadedMaps << " hash tables from " << fileOptions.load_hashtables_from << std::endl;
            }else{
                cpuMinhasher->constructFromReadStorage(
                    fileOptions,
                    runtimeOptions,
                    memoryOptions,
                    cpuReadStorage.getNumberOfReads(), 
                    correctionOptions,
                    cpuReadStorage
                );
            }

            return cpuMinhasher;
        }

        std::unique_ptr<SingleHashCpuMinhasher>
        constructSingleHashCpuMinhasherFromCpuReadStorage(
            const CorrectionOptions& correctionOptions,
            const FileOptions& fileOptions,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const CpuReadStorage& cpuReadStorage
        ){
            float loadfactor = memoryOptions.hashtableLoadfactor;

            auto cpuMinhasher = std::make_unique<SingleHashCpuMinhasher>(
                cpuReadStorage.getNumberOfReads(),
                255,//calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage),
                correctionOptions.kmerlength,
                loadfactor
            );

            if(fileOptions.load_hashtables_from != ""){

                std::ifstream is(fileOptions.load_hashtables_from);
                assert((bool)is);
    
                const int loadedMaps = cpuMinhasher->loadFromStream(is, correctionOptions.numHashFunctions);
    
                std::cout << "Loaded " << loadedMaps << " hash tables from " << fileOptions.load_hashtables_from << std::endl;
            }else{
                cpuMinhasher->constructFromReadStorage(
                    fileOptions,
                    runtimeOptions,
                    memoryOptions,
                    cpuReadStorage.getNumberOfReads(), 
                    correctionOptions,
                    cpuReadStorage
                );
            }

            return cpuMinhasher;
        }


        std::pair<std::unique_ptr<CpuMinhasher>, CpuMinhasherType>
        constructCpuMinhasherFromCpuReadStorage(
            const FileOptions& fileOptions,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const CorrectionOptions& correctionOptions,
            const CpuReadStorage& cpuReadStorage,
            CpuMinhasherType requestedType
        ){
            if(requestedType == CpuMinhasherType::Ordinary){
                if(correctionOptions.singlehash){
                    return std::make_pair(
                        constructSingleHashCpuMinhasherFromCpuReadStorage(
                            correctionOptions,
                            fileOptions,
                            runtimeOptions,
                            memoryOptions,
                            cpuReadStorage
                        ),
                        CpuMinhasherType::OrdinarySingleHash
                    );
                }else{
                    return std::make_pair(
                        constructOrdinaryCpuMinhasherFromCpuReadStorage(
                            correctionOptions,
                            fileOptions,
                            runtimeOptions,
                            memoryOptions,
                            cpuReadStorage
                        ),
                        CpuMinhasherType::Ordinary
                    );
                }
            }else{
                return std::make_pair(
                    constructOrdinaryCpuMinhasherFromCpuReadStorage(
                        correctionOptions,
                        fileOptions,
                        runtimeOptions,
                        memoryOptions,
                        cpuReadStorage
                    ),
                    CpuMinhasherType::Ordinary
                );
            }
        }
    
    
}
