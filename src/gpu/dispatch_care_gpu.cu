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

    template<class readStorage_t>
    void saveReadStorageToFile(const readStorage_t& readStorage, const FileOptions& fileOptions){
    	if(fileOptions.save_binary_reads_to != "") {
    		readStorage.saveToFile(fileOptions.save_binary_reads_to);
    		std::cout << "Saved binary reads to file " << fileOptions.save_binary_reads_to << std::endl;
    	}
    }

    template<class minhasher_t>
    void saveMinhasherToFile(const minhasher_t& minhasher, const FileOptions& fileOptions){
    	if(fileOptions.save_hashtables_to != "") {
    		minhasher.saveToFile(fileOptions.save_hashtables_to);
    		std::cout << "Saved hash tables to file " << fileOptions.save_hashtables_to << std::endl;
    	}
    }

    template<class minhasher_t,
             class readStorage_t>
    void saveDataStructuresToFile(const minhasher_t& minhasher, const readStorage_t& readStorage, const FileOptions& fileOptions){
    	saveReadStorageToFile(readStorage, fileOptions);
    	saveMinhasherToFile(minhasher, fileOptions);
    }


    void performCorrection(MinhashOptions minhashOptions,
                            AlignmentOptions alignmentOptions,
                            CorrectionOptions correctionOptions,
                            RuntimeOptions runtimeOptions,
                            FileOptions fileOptions,
                            GoodAlignmentProperties goodAlignmentProperties){

        std::cout << "Running CARE GPU" << std::endl;

        if(runtimeOptions.deviceIds.size() == 0){
            std::cout << "No device ids found. Abort!" << std::endl;
            return;
        }

        filesys::create_directories(fileOptions.outputdirectory);

    	auto thread_id = std::this_thread::get_id();
    	std::string thread_id_string;
    	{
    		std::stringstream ss;
    		ss << thread_id;
    		thread_id_string = ss.str();
    	}

    	FileOptions iterFileOptions = fileOptions;

    	iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";

        std::cout << "loading file and building data structures..." << std::endl;

        TIMERSTARTCPU(load_and_build);

        gpu::BuiltGpuDataStructures dataStructuresgpu = gpu::buildGpuDataStructures(minhashOptions,
                                                                                    correctionOptions,
                                                                                    runtimeOptions,
                                                                                    fileOptions);

        TIMERSTOPCPU(load_and_build);

        auto& readStorage = dataStructuresgpu.builtReadStorage.data;
        auto& minhasher = dataStructuresgpu.builtMinhasher.data;
        auto& sequenceFileProperties = dataStructuresgpu.sequenceFileProperties;

        saveReadStorageToFile(readStorage, iterFileOptions);
        saveMinhasherToFile(minhasher, iterFileOptions);

        printFileProperties(fileOptions.inputfile, sequenceFileProperties);

        TIMERSTARTCPU(candidateestimation);
        std::uint64_t maxCandidatesPerRead = runtimeOptions.max_candidates;

        if(maxCandidatesPerRead == 0){
            // maxCandidatesPerRead = cpu::calculateMaxCandidatesPerReadThreshold(minhasher,
            //                                         readStorage,
            //                                         sequenceFileProperties.nReads / 10,
            //                                         correctionOptions.hits_per_candidate,
            //                                         runtimeOptions.threads
            //                                         //,"ncandidates.txt"
            //                                         );
            assert(maxCandidatesPerRead != 0);
            std::cout << "maxCandidates option not specified. Using estimation: " << maxCandidatesPerRead << std::endl;
        }



        TIMERSTOPCPU(candidateestimation);

        printDataStructureMemoryUsage(minhasher, readStorage);

        gpu::correct_gpu(minhashOptions, alignmentOptions,
                            goodAlignmentProperties, correctionOptions,
                            runtimeOptions, iterFileOptions, sequenceFileProperties,
                            minhasher, readStorage,
                            maxCandidatesPerRead);

        TIMERSTARTCPU(finalizing_files);

    	std::string toRename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
    	std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

        //rename feature file
        if(correctionOptions.extractFeatures){
            std::string tmpfeaturename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even_features";
            std::string outputfeaturename = fileOptions.outputfile + "_features";

    		std::rename(tmpfeaturename.c_str(), outputfeaturename.c_str());
        }

        TIMERSTOPCPU(finalizing_files);
    }

}
