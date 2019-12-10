#include <dispatch_care.hpp>

#include <config.hpp>
#include <options.hpp>
#include <sequencefileio.hpp>
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

        auto memInfo = readStorage.getMemoryInfo();

        assert(memInfo.deviceIds.size() == memInfo.deviceSizeInBytes.size());

        std::cout << "Reads occupy " << toGB(memInfo.hostSizeInBytes) << " GB on host\n";
        for(size_t i = 0; i < memInfo.deviceIds.size(); i++){
            std::cout << "Reads occupy " << toGB(memInfo.deviceSizeInBytes[i]) << " GB on device " << memInfo.deviceIds[i] << '\n';
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

        TIMERSTARTCPU(candidateestimation);
        std::uint64_t maxCandidatesPerRead = runtimeOptions.max_candidates;

        //if(maxCandidatesPerRead == 0){
            // maxCandidatesPerRead = cpu::calculateMaxCandidatesPerReadThreshold(minhasher,
            //                                         readStorage,
            //                                         sequenceFileProperties.nReads / 10,
            //                                         correctionOptions.hits_per_candidate,
            //                                         runtimeOptions.threads
            //                                         //,"ncandidates.txt"
            //                                         );
            //assert(maxCandidatesPerRead != 0);
            //std::cout << "maxCandidates option not specified. Using estimation: " << maxCandidatesPerRead << std::endl;
        //}



        TIMERSTOPCPU(candidateestimation);

        printDataStructureMemoryUsage(minhasher, readStorage);

        gpu::correct_gpu(minhashOptions, alignmentOptions,
                            goodAlignmentProperties, correctionOptions,
                            runtimeOptions, fileOptions, sequenceFileProperties,
                            minhasher, readStorage,
                            maxCandidatesPerRead);

        TIMERSTARTCPU(finalizing_files);

        TIMERSTOPCPU(finalizing_files);
    }

}
