#include <dispatch_care.hpp>

#include <config.hpp>
#include <options.hpp>
#include <readlibraryio.hpp>
#include <minhasher.hpp>
#include <build.hpp>
#include <gpu/distributedreadstorage.hpp>
#include <gpu/correct_gpu.hpp>
#include <correctionresultprocessing.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

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

        auto rsMemInfo = readStorage.getMemoryInfo();

        std::cout << "Reads occupy " << toGB(rsMemInfo.host) << " GB on host\n";
        for(const auto& pair : rsMemInfo.device){
            std::cout << "Reads occupy " << toGB(pair.second) << " GB on device " << pair.first << '\n';
        }

    	//std::cout << "reads take up " << toGB(readStorage.size()) << " GB." << std::endl;
    	std::cout << "hash maps take up " << toGB(minhasher.numBytes()) << " GB on host." << std::endl;
    }

    void performCorrection(
                            CorrectionOptions correctionOptions,
                            RuntimeOptions runtimeOptions,
                            MemoryOptions memoryOptions,
                            FileOptions fileOptions,
                            GoodAlignmentProperties goodAlignmentProperties){

        std::cout << "Running CARE GPU" << std::endl;

        if(runtimeOptions.deviceIds.size() == 0){
            std::cout << "No device ids found. Abort!" << std::endl;
            return;
        }

        std::cout << "loading file and building data structures..." << std::endl;

        TIMERSTARTCPU(load_and_build);

        gpu::BuiltGpuDataStructures dataStructuresgpu = gpu::buildAndSaveGpuDataStructures2(
            correctionOptions,
            runtimeOptions,
            memoryOptions,
            fileOptions
        );

        TIMERSTOPCPU(load_and_build);

        if(correctionOptions.autodetectKmerlength){
            correctionOptions.kmerlength = dataStructuresgpu.kmerlength;
        }

        auto& readStorage = dataStructuresgpu.builtReadStorage.data.readStorage;
        auto& minhasher = dataStructuresgpu.builtMinhasher.data;
        auto& totalInputFileProperties = dataStructuresgpu.totalInputFileProperties;

        if(correctionOptions.mustUseAllHashfunctions && correctionOptions.numHashFunctions != minhasher.minparams.maps){
            std::cout << "Cannot use specified number of hash functions (" << correctionOptions.numHashFunctions <<")\n";
            std::cout << "Abort!\n";
            return;
        }

        printDataStructureMemoryUsage(minhasher, readStorage);

        auto partialResults = gpu::correct_gpu(
            goodAlignmentProperties, 
            correctionOptions,
            runtimeOptions, 
            fileOptions, 
            memoryOptions,
            totalInputFileProperties,
            minhasher, 
            readStorage
        );

        //Merge corrected reads with input file to generate output file

        const std::size_t availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
        std::size_t memoryForSorting = 0;

        if(availableMemoryInBytes > 1*(std::size_t(1) << 30)){
            memoryForSorting = availableMemoryInBytes - 1*(std::size_t(1) << 30);
        }

        std::cout << "begin merging reads" << std::endl;

        TIMERSTARTCPU(merge);

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

        TIMERSTOPCPU(merge);

        std::cout << "end merging reads" << std::endl;

    }

}
