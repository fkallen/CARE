#include <dispatch_care.hpp>
#include <gpu/gpuminhasher.cuh>

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

    std::vector<int> getUsableDeviceIds(std::vector<int> deviceIds){
        int nDevices;

        cudaGetDeviceCount(&nDevices); CUERR;

        std::vector<int> invalidIds;

        for(int id : deviceIds) {
            if(id >= nDevices) {
                invalidIds.emplace_back(id);
                std::cout << "Found invalid device Id: " << id << std::endl;
            }
        }

        if(invalidIds.size() > 0) {
            std::cout << "Available GPUs on your machine:" << std::endl;
            for(int j = 0; j < nDevices; j++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, j); CUERR;
                std::cout << "Id " << j << " : " << prop.name << std::endl;
            }

            for(int invalidid : invalidIds) {
                deviceIds.erase(std::find(deviceIds.begin(), deviceIds.end(), invalidid));
            }
        }

        return deviceIds;
    }

    

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

        std::cout << "STEP 1: Database construction" << std::endl;

        TIMERSTARTCPU(STEP1);

        gpu::BuiltGpuDataStructures dataStructuresgpu = gpu::buildAndSaveGpuDataStructures2(
            correctionOptions,
            runtimeOptions,
            memoryOptions,
            fileOptions
        );

        TIMERSTOPCPU(STEP1);

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



        //printDataStructureMemoryUsage(minhasher, readStorage);

        minhasher.destroy();

        TIMERSTARTCPU(build_newgpuminhasher);
        gpu::GpuMinhasher newGpuMinhasher(
            correctionOptions.kmerlength, 
            calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage)
        );

        if(fileOptions.load_hashtables_from != ""){

            std::ifstream is(fileOptions.load_hashtables_from);
            assert((bool)is);

            newGpuMinhasher.loadFromStream(is);

            std::cout << "Loaded hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }else{
            newGpuMinhasher.construct(
                fileOptions,
                runtimeOptions,
                memoryOptions,
                totalInputFileProperties.nReads, 
                correctionOptions,
                readStorage
            );

            if(correctionOptions.mustUseAllHashfunctions 
                && correctionOptions.numHashFunctions != newGpuMinhasher.getNumberOfMaps()){
                std::cout << "Cannot use specified number of hash functions (" 
                    << correctionOptions.numHashFunctions <<")\n";
                std::cout << "Abort!\n";
                return;
            }
        }

        if(fileOptions.save_hashtables_to != "") {
            std::cout << "Saving minhasher to file " << fileOptions.save_hashtables_to << std::endl;
            std::ofstream os(fileOptions.save_hashtables_to);
            assert((bool)os);

            newGpuMinhasher.writeToStream(os);

    		std::cout << "Saved minhasher" << std::endl;
        }



        TIMERSTOPCPU(build_newgpuminhasher);



        std::cout << "STEP 2: Error correction" << std::endl;

        TIMERSTARTCPU(STEP2);

        auto partialResults = gpu::correct_gpu(
            goodAlignmentProperties, 
            correctionOptions,
            runtimeOptions, 
            fileOptions, 
            memoryOptions,
            totalInputFileProperties,
            //minhasher, 
            newGpuMinhasher,
            readStorage
        );

        TIMERSTOPCPU(STEP2);

        //minhasher.destroy();
        newGpuMinhasher.destroy();
        readStorage.destroy();

        //Merge corrected reads with input file to generate output file

        const std::size_t availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
        std::size_t memoryForSorting = 0;

        if(availableMemoryInBytes > 1*(std::size_t(1) << 30)){
            memoryForSorting = availableMemoryInBytes - 1*(std::size_t(1) << 30);
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
