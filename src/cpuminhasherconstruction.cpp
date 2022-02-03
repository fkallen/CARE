
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
            const ProgramOptions& programOptions,
            const CpuReadStorage& cpuReadStorage
        ){
            float loadfactor = programOptions.hashtableLoadfactor;

            auto cpuMinhasher = std::make_unique<OrdinaryCpuMinhasher>(
                cpuReadStorage.getNumberOfReads(),
                calculateResultsPerMapThreshold(programOptions.estimatedCoverage),
                programOptions.kmerlength,
                loadfactor
            );

            if(programOptions.load_hashtables_from != ""){

                std::ifstream is(programOptions.load_hashtables_from);
                assert((bool)is);
    
                const int loadedMaps = cpuMinhasher->loadFromStream(is, programOptions.numHashFunctions);
    
                std::cout << "Loaded " << loadedMaps << " hash tables from " << programOptions.load_hashtables_from << std::endl;
            }else{
                cpuMinhasher->constructFromReadStorage(
                    programOptions,
                    cpuReadStorage.getNumberOfReads(), 
                    cpuReadStorage
                );
            }

            return cpuMinhasher;
        }

        std::unique_ptr<SingleHashCpuMinhasher>
        constructSingleHashCpuMinhasherFromCpuReadStorage(
            const ProgramOptions& programOptions,
            const CpuReadStorage& cpuReadStorage
        ){
            float loadfactor = programOptions.hashtableLoadfactor;

            auto cpuMinhasher = std::make_unique<SingleHashCpuMinhasher>(
                cpuReadStorage.getNumberOfReads(),
                255,//calculateResultsPerMapThreshold(programOptions.estimatedCoverage),
                programOptions.kmerlength,
                loadfactor
            );

            if(programOptions.load_hashtables_from != ""){

                std::ifstream is(programOptions.load_hashtables_from);
                assert((bool)is);
    
                const int loadedMaps = cpuMinhasher->loadFromStream(is, programOptions.numHashFunctions);
    
                std::cout << "Loaded " << loadedMaps << " hash tables from " << programOptions.load_hashtables_from << std::endl;
            }else{
                cpuMinhasher->constructFromReadStorage(
                    programOptions,
                    cpuReadStorage.getNumberOfReads(), 
                    cpuReadStorage
                );
            }

            return cpuMinhasher;
        }


        std::pair<std::unique_ptr<CpuMinhasher>, CpuMinhasherType>
        constructCpuMinhasherFromCpuReadStorage(
            const ProgramOptions& programOptions,
            const CpuReadStorage& cpuReadStorage,
            CpuMinhasherType requestedType
        ){
            if(requestedType == CpuMinhasherType::Ordinary){
                if(programOptions.singlehash){
                    return std::make_pair(
                        constructSingleHashCpuMinhasherFromCpuReadStorage(
                            programOptions,
                            cpuReadStorage
                        ),
                        CpuMinhasherType::OrdinarySingleHash
                    );
                }else{
                    return std::make_pair(
                        constructOrdinaryCpuMinhasherFromCpuReadStorage(
                            programOptions,
                            cpuReadStorage
                        ),
                        CpuMinhasherType::Ordinary
                    );
                }
            }else{
                return std::make_pair(
                    constructOrdinaryCpuMinhasherFromCpuReadStorage(
                        programOptions,
                        cpuReadStorage
                    ),
                    CpuMinhasherType::Ordinary
                );
            }
        }
    
    
}
