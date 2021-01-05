#include <dispatch_care.hpp>
#include <gpu/gpuminhasherconstruction.cuh>
#include <gpu/fakegpuminhasher.cuh>

#include <config.hpp>
#include <options.hpp>
#include <readlibraryio.hpp>

#include <gpu/distributedreadstorage.hpp>
#include <gpu/correct_gpu.hpp>
#include <correctionresultprocessing.hpp>

#include <gpu/multigpureadstorage.cuh>

#include <rangegenerator.hpp>

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

        //check gpu restrictions of remaining gpus
        invalidIds.clear();

        for(int id : deviceIds){
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, id);

            if(prop.major < 6){
                invalidIds.emplace_back(id);
                std::cerr << "Warning. Removing gpu id " << id << " because its not arch 6 or greater.\n";
            }

            if(prop.managedMemory != 1){
                invalidIds.emplace_back(id);
                std::cerr << "Warning. Removing gpu id " << id << " because it does not support managed memory. (may be required for hash table construction).\n";
            }
        }

        if(invalidIds.size() > 0) {
            for(int invalidid : invalidIds) {
                deviceIds.erase(std::find(deviceIds.begin(), deviceIds.end(), invalidid));
            }
        }

        return deviceIds;
    }

    

    template<class T>
    void printDataStructureMemoryUsage(const T& datastructure, const std::string& name){
    	auto toGB = [](std::size_t bytes){
    			    double gb = bytes / 1024. / 1024. / 1024.0;
    			    return gb;
    		    };

        auto memInfo = datastructure.getMemoryInfo();
        
        std::cout << name << " memory usage: " << toGB(memInfo.host) << " GB on host\n";
        for(const auto& pair : memInfo.device){
            std::cout << name << " memory usage: " << toGB(pair.second) << " GB on device " << pair.first << '\n';
        }
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

        cudaSetDevice(runtimeOptions.deviceIds[0]); CUERR;

        helpers::PeerAccessDebug peerAccess(runtimeOptions.deviceIds, true);
        peerAccess.enableAllPeerAccesses();

        std::uint64_t maximumNumberOfReads = fileOptions.nReads;
        int maximumSequenceLength = fileOptions.maximum_sequence_length;
        int minimumSequenceLength = fileOptions.minimum_sequence_length;
        bool scanned = false;

        if(fileOptions.load_binary_reads_from == ""){

            if(maximumNumberOfReads >= std::uint64_t(std::numeric_limits<read_number>::max())){
                std::cout << "Error. " << maximumNumberOfReads << " reads cannot be processed with the current config.hpp" << std::endl;
                std::exit(1);
            }

            if(maximumNumberOfReads == 0 || maximumSequenceLength == 0 || minimumSequenceLength == 0) {
                std::cout << "STEP 0: Determine input size" << std::endl;
                
                std::cout << "Scanning file(s) to get number of reads and min/max sequence length." << std::endl;

                maximumNumberOfReads = 0;
                maximumSequenceLength = 0;
                minimumSequenceLength = std::numeric_limits<int>::max();

                for(const auto& inputfile : fileOptions.inputfiles){
                    auto prop = getSequenceFileProperties(inputfile, runtimeOptions.showProgress);
                    maximumNumberOfReads += prop.nReads;
                    maximumSequenceLength = std::max(maximumSequenceLength, prop.maxSequenceLength);
                    minimumSequenceLength = std::min(minimumSequenceLength, prop.minSequenceLength);

                    std::cout << "----------------------------------------\n";
                    std::cout << "File: " << inputfile << "\n";
                    std::cout << "Reads: " << prop.nReads << "\n";
                    std::cout << "Minimum sequence length: " << prop.minSequenceLength << "\n";
                    std::cout << "Maximum sequence length: " << prop.maxSequenceLength << "\n";
                    std::cout << "----------------------------------------\n";

                    //result.inputFileProperties.emplace_back(prop);
                }

                scanned = true;
            }else{
                //std::cout << "Using the supplied max number of reads and min/max sequence length." << std::endl;
            }

            if(maximumNumberOfReads >= std::uint64_t(std::numeric_limits<read_number>::max())){
                std::cout << "Error. " << maximumNumberOfReads << " reads cannot be processed with the current config.hpp" << std::endl;
                std::exit(1);
            }
        }

        /*
            Step 1: 
            - load all reads from all input files into (gpu-)memory
            - construct minhash signatures of all reads and store them in hash tables
        */

        helpers::CpuTimer step1timer("STEP1");

        std::cout << "STEP 1: Database construction" << std::endl;

        helpers::CpuTimer buildReadStorageTimer("build_readstorage");

        gpu::DistributedReadStorage readStorage(
            runtimeOptions.deviceIds, 
            maximumNumberOfReads, 
            correctionOptions.useQualityScores, 
            minimumSequenceLength, 
            maximumSequenceLength
        );

        if(fileOptions.load_binary_reads_from != ""){

            readStorage.loadFromFile(fileOptions.load_binary_reads_from, runtimeOptions.deviceIds);

            if(correctionOptions.useQualityScores && !readStorage.canUseQualityScores())
                throw std::runtime_error("Quality scores are required but not present in preprocessed reads file!");
            if(!correctionOptions.useQualityScores && readStorage.canUseQualityScores())
                std::cerr << "Warning. The loaded preprocessed reads file contains quality scores, but program does not use them!\n";

            std::cout << "Loaded preprocessed reads from " << fileOptions.load_binary_reads_from << std::endl;

            readStorage.constructionIsComplete();
        }else{
            readStorage.construct(
                fileOptions.inputfiles,
                correctionOptions.useQualityScores,
                maximumNumberOfReads,
                minimumSequenceLength,
                maximumSequenceLength,
                runtimeOptions.threads,
                runtimeOptions.showProgress
            );
        }

        if(fileOptions.save_binary_reads_to != "") {
            std::cout << "Saving reads to file " << fileOptions.save_binary_reads_to << std::endl;
            helpers::CpuTimer timer("save_to_file");
            readStorage.saveToFile(fileOptions.save_binary_reads_to);
            timer.print();
    		std::cout << "Saved reads" << std::endl;
        }

        buildReadStorageTimer.print();
        
        SequenceFileProperties totalInputFileProperties;

        totalInputFileProperties.nReads = readStorage.getNumberOfReads();
        totalInputFileProperties.maxSequenceLength = readStorage.getStatistics().maximumSequenceLength;
        totalInputFileProperties.minSequenceLength = readStorage.getStatistics().minimumSequenceLength;

        if(!scanned){
            std::cout << "Determined the following read properties:\n";
            std::cout << "----------------------------------------\n";
            std::cout << "Total number of reads: " << totalInputFileProperties.nReads << "\n";
            std::cout << "Minimum sequence length: " << totalInputFileProperties.minSequenceLength << "\n";
            std::cout << "Maximum sequence length: " << totalInputFileProperties.maxSequenceLength << "\n";
            std::cout << "----------------------------------------\n";

            if(totalInputFileProperties.nReads >= std::uint64_t(std::numeric_limits<read_number>::max())){
                std::cout << "Error. " << totalInputFileProperties.nReads << " reads cannot be processed with the current config.hpp" << std::endl;
                std::exit(1);
            }
        }

        if(correctionOptions.autodetectKmerlength){
            const int maxlength = totalInputFileProperties.maxSequenceLength;

            auto getKmerSizeForHashing = [](int maximumReadLength){
                if(maximumReadLength < 160){
                    return 20;
                }else{
                    return 32;
                }
            };

            correctionOptions.kmerlength = getKmerSizeForHashing(maxlength);

            std::cout << "Will use k-mer length = " << correctionOptions.kmerlength << " for hashing.\n";
        }

        std::cout << "Reads with ambiguous bases: " << readStorage.getNumberOfReadsWithN() << std::endl;
        

        printDataStructureMemoryUsage(readStorage, "reads");


        helpers::CpuTimer buildMinhasherTimer("build_minhasher");

        auto minhasherAndType = gpu::constructGpuMinhasherFromGpuReadStorage(
            fileOptions,
            runtimeOptions,
            memoryOptions,
            correctionOptions,
            totalInputFileProperties,
            readStorage,
            gpu::GpuMinhasherType::Single
        );

        gpu::GpuMinhasher* const gpuMinhasher = minhasherAndType.first.get();

        buildMinhasherTimer.print();

        std::cout << "GpuMinhasher can use " << gpuMinhasher->getNumberOfMaps() << " maps\n";

        if(gpuMinhasher->getNumberOfMaps() <= 0){
            std::cout << "Cannot construct a single gpu hashtable. Abort!" << std::endl;
            return;
        }

        if(correctionOptions.mustUseAllHashfunctions 
            && correctionOptions.numHashFunctions != gpuMinhasher->getNumberOfMaps()){
            std::cout << "Cannot use specified number of hash functions (" 
                << correctionOptions.numHashFunctions <<")\n";
            std::cout << "Abort!\n";
            return;
        }

        if(minhasherAndType.second == gpu::GpuMinhasherType::Fake){

            gpu::FakeGpuMinhasher* fakeGpuMinhasher = dynamic_cast<gpu::FakeGpuMinhasher*>(gpuMinhasher);
            assert(fakeGpuMinhasher != nullptr);

            if(fileOptions.save_hashtables_to != "") {
                std::cout << "Saving minhasher to file " << fileOptions.save_hashtables_to << std::endl;
                std::ofstream os(fileOptions.save_hashtables_to);
                assert((bool)os);
                helpers::CpuTimer timer("save_to_file");
                fakeGpuMinhasher->writeToStream(os);
                timer.print();

                std::cout << "Saved minhasher" << std::endl;
            }

        }

        printDataStructureMemoryUsage(*gpuMinhasher, "hash tables");        

        step1timer.print();

#define USENEWRS

#ifdef USENEWRS
        readStorage.saveToFile("foootemp");
        readStorage.destroy();

        care::cpu::ContiguousReadStorage cpuReadStorage(
            totalInputFileProperties.nReads,
            correctionOptions.useQualityScores, 
            totalInputFileProperties.minSequenceLength, 
            totalInputFileProperties.maxSequenceLength
        );
            
        cpuReadStorage.loadFromFile("foootemp");
        std::cout << "Loaded cpu readstorage " << std::endl;

        std::vector<std::size_t> gpumemorylimits(runtimeOptions.deviceIds.size(), 0);
        for(int i = 0; i < int(runtimeOptions.deviceIds.size()); i++){
            std::size_t total = 0;
            cudaMemGetInfo(&gpumemorylimits[i], &total);

            std::size_t safety = 1 << 30;
            if(gpumemorylimits[i] > safety){
                gpumemorylimits[i] -= safety;
            }else{
                gpumemorylimits[i] = 0;
            }
        }

        helpers::CpuTimer cpugputimer("cpu->gpu readstorage");
        gpu::MultiGpuReadStorage gpuReadStorage(
            cpuReadStorage, 
            runtimeOptions.deviceIds, 
            gpumemorylimits
        );
        cpugputimer.print();

        std::cout << "constructed gpu readstorage " << std::endl;
#endif        



        std::cout << "STEP 2: Error correction" << std::endl;

        helpers::CpuTimer step2timer("STEP2");

        auto partialResults = gpu::correct_gpu(
            goodAlignmentProperties, 
            correctionOptions,
            runtimeOptions,
            fileOptions, 
            memoryOptions,
            totalInputFileProperties,
            *gpuMinhasher, 
#ifdef USENEWRS                                  
            gpuReadStorage
#else            
            readStorage
#endif            
        );

        step2timer.print();

        std::cout << "Correction throughput : ~" << (totalInputFileProperties.nReads / step2timer.elapsed()) << " reads/second.\n";

        gpuMinhasher->destroy();

        readStorage.destroy();
#ifdef USENEWRS        
        gpuReadStorage.destroy();
        cpuReadStorage.destroy();
#endif        

        //Merge corrected reads with input file to generate output file

        const std::size_t availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
        std::size_t memoryForSorting = 0;

        if(availableMemoryInBytes > 1*(std::size_t(1) << 30)){
            memoryForSorting = availableMemoryInBytes - 1*(std::size_t(1) << 30);
        }               

        std::cout << "STEP 3: Constructing output file(s)" << std::endl;

        helpers::CpuTimer step3timer("STEP3");

        std::vector<FileFormat> formats;
        for(const auto& inputfile : fileOptions.inputfiles){
            formats.emplace_back(getFileFormat(inputfile));
        }
        std::vector<std::string> outputfiles;
        for(const auto& outputfilename : fileOptions.outputfilenames){
            outputfiles.emplace_back(fileOptions.outputdirectory + "/" + outputfilename);
        }
        constructOutputFileFromCorrectionResults(
            fileOptions.tempdirectory,
            fileOptions.inputfiles, 
            partialResults, 
            memoryForSorting,
            formats[0],
            outputfiles, 
            false
        );

        step3timer.print();

        std::cout << "Construction of output file(s) finished." << std::endl;

    }

}
