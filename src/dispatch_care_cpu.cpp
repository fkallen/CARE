#include <dispatch_care.hpp>

#include <config.hpp>
#include <options.hpp>

#include <readstorage.hpp>

#include <correct_cpu.hpp>

#include <build.hpp>

#include <minhasher.hpp>
#include <minhasher_transform.hpp>
//#include <candidatedistribution.hpp>

#include <sequence.hpp>

#include <vector>
#include <iostream>
#include <mutex>
#include <thread>
#include <memory>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;



namespace care{

    template<class minhasher_t,
             class readStorage_t>
    void printDataStructureMemoryUsage(const minhasher_t& minhasher, const readStorage_t& readStorage){
    	auto toGB = [](std::size_t bytes){
    			    double gb = bytes / 1024. / 1024. / 1024.0;
    			    return gb;
    		    };

    	std::cout << "reads take up " << toGB(readStorage.size()) << " GB." << std::endl;
    	std::cout << "hash maps take up " << toGB(minhasher.numBytes()) << " GB." << std::endl;
    }

    void printFileProperties(const std::string& filename, const SequenceFileProperties& props){
    	std::cout << "----------------------------------------" << std::endl;
    	std::cout << "File: " << filename << std::endl;
    	std::cout << "Reads: " << props.nReads << std::endl;
    	std::cout << "Minimum sequence length: " << props.minSequenceLength << std::endl;
    	std::cout << "Maximum sequence length: " << props.maxSequenceLength << std::endl;
    	std::cout << "----------------------------------------" << std::endl;
    }


    void performCorrection(MinhashOptions minhashOptions,
                            AlignmentOptions alignmentOptions,
                            CorrectionOptions correctionOptions,
                            RuntimeOptions runtimeOptions,
                            MemoryOptions memoryOptions,
                            FileOptions fileOptions,
                            GoodAlignmentProperties goodAlignmentProperties){

        std::cout << "Running CARE CPU" << std::endl;

        std::cout << "loading file and building data structures..." << std::endl;

        TIMERSTARTCPU(load_and_build);

        BuiltDataStructures dataStructures = buildAndSaveDataStructures(minhashOptions,
                                                                correctionOptions,
                                                                runtimeOptions,
                                                                memoryOptions,
                                                                fileOptions);

        TIMERSTOPCPU(load_and_build);

        auto& readStorage = dataStructures.builtReadStorage.data;
        auto& minhasher = dataStructures.builtMinhasher.data;
        auto& sequenceFileProperties = dataStructures.sequenceFileProperties;

        printDataStructureMemoryUsage(minhasher, readStorage);

        std::cout << "Running CARE CPU" << std::endl;

        cpu::correct_cpu(minhashOptions, alignmentOptions,
                    goodAlignmentProperties, correctionOptions,
                    runtimeOptions, fileOptions, sequenceFileProperties,
                    minhasher, readStorage);

        TIMERSTARTCPU(finalizing_files);

        TIMERSTOPCPU(finalizing_files);
    }

}
