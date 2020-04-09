#include <dispatch_care.hpp>

#include <config.hpp>
#include <options.hpp>
#include <readlibraryio.hpp>
#include <minhasher.hpp>
#include <build.hpp>
#include <gpu/distributedreadstorage.hpp>
#include <gpu/correct_gpu.hpp>

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

        std::cout << "Running CARE GPU" << std::endl;

        if(runtimeOptions.deviceIds.size() == 0){
            std::cout << "No device ids found. Abort!" << std::endl;
            return;
        }

        std::cout << "loading file and building data structures..." << std::endl;

        TIMERSTARTCPU(set_up_datastructures);

        gpu::BuiltGpuDataStructures dataStructuresgpu = gpu::buildAndSaveGpuDataStructures(minhashOptions,
                                                                                    correctionOptions,
                                                                                    runtimeOptions,
                                                                                    memoryOptions,
                                                                                    fileOptions);

        TIMERSTOPCPU(set_up_datastructures);

        auto& readStorage = dataStructuresgpu.builtReadStorage.data.readStorage;
        auto& minhasher = dataStructuresgpu.builtMinhasher.data;
        auto& sequenceFileProperties = dataStructuresgpu.sequenceFileProperties;

        printDataStructureMemoryUsage(minhasher, readStorage);

        //gpu::correct_gpu(minhashOptions, alignmentOptions,
        gpu::correct_gpu(
            minhashOptions, 
            alignmentOptions,
            goodAlignmentProperties, 
            correctionOptions,
            runtimeOptions, 
            fileOptions, 
            memoryOptions,
            sequenceFileProperties,
            minhasher, 
            readStorage);

        TIMERSTARTCPU(finalizing_files);

        TIMERSTOPCPU(finalizing_files);
    }

}
