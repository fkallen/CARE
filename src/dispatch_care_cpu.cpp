#include <dispatch_care.hpp>

#include <config.hpp>
#include <options.hpp>

#include <readstorage.hpp>

#include <correct_cpu.hpp>

#include <build.hpp>

#include <minhasher.hpp>
#include <minhasher_transform.hpp>
//#include <candidatedistribution.hpp>
#include <correctionresultprocessing.hpp>
#include <sequence.hpp>

#include <vector>
#include <iostream>
#include <mutex>
#include <thread>
#include <memory>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;



namespace care{

    std::vector<int> getUsableDeviceIds(std::vector<int> deviceIds){
        return std::vector<int>{};
    }

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


    void performCorrection(
                            CorrectionOptions correctionOptions,
                            RuntimeOptions runtimeOptions,
                            MemoryOptions memoryOptions,
                            FileOptions fileOptions,
                            GoodAlignmentProperties goodAlignmentProperties){

        std::cout << "Running CARE CPU" << std::endl;

        std::cout << "STEP 1: Database construction" << std::endl;

        TIMERSTARTCPU(STEP1);

        BuiltDataStructures dataStructures = buildAndSaveDataStructures2(
            correctionOptions,
            runtimeOptions,
            memoryOptions,
            fileOptions
        );

        TIMERSTOPCPU(STEP1);

        if(correctionOptions.autodetectKmerlength){
            correctionOptions.kmerlength = dataStructures.kmerlength;
        }

        auto& readStorage = dataStructures.builtReadStorage.data;
        auto& minhasher = dataStructures.builtMinhasher.data;
        auto& totalInputFileProperties = dataStructures.totalInputFileProperties;

        if(correctionOptions.mustUseAllHashfunctions && correctionOptions.numHashFunctions != minhasher.minparams.maps){
            std::cout << "Cannot use specified number of hash functions (" << correctionOptions.numHashFunctions <<")\n";
            std::cout << "Abort!\n";
            return;
        }

        printDataStructureMemoryUsage(minhasher, readStorage);

        std::cout << "STEP 2: Error correction" << std::endl;

        TIMERSTARTCPU(STEP2);

        auto partialResults = cpu::correct_cpu(
            goodAlignmentProperties, 
            correctionOptions,
            runtimeOptions, 
            fileOptions, 
            memoryOptions, 
            totalInputFileProperties,
            minhasher, 
            readStorage
        );

        TIMERSTOPCPU(STEP2);


        const std::size_t availableMemoryInBytes2 = getAvailableMemoryInKB() * 1024;
        std::size_t memoryForSorting = 0;

        if(availableMemoryInBytes2 > 1*(std::size_t(1) << 30)){
            memoryForSorting = availableMemoryInBytes2 - 1*(std::size_t(1) << 30);
        }

        std::cout << "STEP 3: Constructing output file(s)" << std::endl;

        TIMERSTARTCPU(STEP3);

        std::vector<FileFormat> formats;
        for(const auto& inputfile : fileOptions.inputfiles){
            formats.emplace_back(getFileFormat(inputfile));
        }
        std::vector<std::string> outputfiles;
        for(const auto& outputfilename : fileOptions.outputfilenames){
            outputfiles.emplace_back(fileOptions.outputdirectory + "/" + outputfilename);
        }
        constructOutputFileFromResults2(
            fileOptions.tempdirectory,
            fileOptions.inputfiles,            
            partialResults, 
            memoryForSorting,
            formats[0], 
            outputfiles, 
            false
        );

        TIMERSTOPCPU(STEP3);

        std::cout << "Construction of output file(s) finished." << std::endl;

    }

}
