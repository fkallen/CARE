#include <gpu/dispatch_care_correct_gpu.cuh>
#include <gpu/gpuminhasherconstruction.cuh>
#include <gpu/fakegpuminhasher.cuh>
#include <gpu/cudaerrorcheck.cuh>

#include <config.hpp>
#include <options.hpp>
#include <readlibraryio.hpp>
#include <memorymanagement.hpp>
#include <gpu/correct_gpu.hpp>
#include <correctionresultprocessing.hpp>
#include <classification.hpp>
#include <gpu/forest_gpu.cuh>

#include <gpu/multigpureadstorage.cuh>
#include <chunkedreadstorageconstruction.hpp>
#include <chunkedreadstorage.hpp>

#include <rangegenerator.hpp>


#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;

namespace care{

    namespace correction{

        std::vector<int> getUsableDeviceIds(std::vector<int> deviceIds){
            int nDevices;

            CUDACHECK(cudaGetDeviceCount(&nDevices));

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
                    CUDACHECK(cudaGetDeviceProperties(&prop, j));
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


    void loadPartialResultsAndConstructOutput(
        CorrectionOptions correctionOptions,
        RuntimeOptions runtimeOptions,
        MemoryOptions memoryOptions,
        FileOptions fileOptions,
        GoodAlignmentProperties goodAlignmentProperties,
        std::string filename
    ){
        std::cerr << "loadPartialResultsAndConstructOutput\n";

        MemoryFileFixedSize<EncodedTempCorrectedSequence> partialResults{0, fileOptions.tempdirectory + "/" + "MemoryFileFixedSizetmp"};

        std::ifstream is(filename);
        assert((bool)is);

        partialResults.loadFromStream(is);

        const std::size_t numTemp = partialResults.getNumElementsInMemory() + partialResults.getNumElementsInFile();
        const std::size_t numTempInMem = partialResults.getNumElementsInMemory();
        const std::size_t numTempInFile = partialResults.getNumElementsInFile();
    
        std::cerr << "Constructed " << numTemp << " corrections. "
            << numTempInMem << " corrections are stored in memory. "
            << numTempInFile << " corrections are stored in temporary file\n";

        //Merge corrected reads with input file to generate output file

        const std::size_t availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
        const auto partialResultMemUsage = partialResults.getMemoryInfo();

        std::cerr << "availableMemoryInBytes = " << availableMemoryInBytes << "\n";
        std::cerr << "memoryLimitOption = " << memoryOptions.memoryTotalLimit << "\n";
        std::cerr << "partialResultMemUsage = " << partialResultMemUsage.host << "\n";

        std::size_t memoryForSorting = std::min(
            availableMemoryInBytes,
            memoryOptions.memoryTotalLimit - partialResultMemUsage.host
        );

        if(memoryForSorting > 1*(std::size_t(1) << 30)){
            memoryForSorting = memoryForSorting - 1*(std::size_t(1) << 30);
        }
        std::cerr << "memoryForSorting = " << memoryForSorting << "\n";        

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
            false,
            runtimeOptions.showProgress
        );

        step3timer.print();

        //compareMaxRssToLimit(memoryOptions.memoryTotalLimit, "Error memorylimit after output construction");

        std::cout << "Construction of output file(s) finished." << std::endl;
    }

    void performCorrection(
        CorrectionOptions correctionOptions,
        RuntimeOptions runtimeOptions,
        MemoryOptions memoryOptions,
        FileOptions fileOptions,
        GoodAlignmentProperties goodAlignmentProperties
    ){

        std::cout << "Running CARE GPU" << std::endl;

        if(runtimeOptions.deviceIds.size() == 0){
            std::cout << "No device ids found. Abort!" << std::endl;
            return;
        }

        if(correctionOptions.correctionType == CorrectionType::Print 
            || correctionOptions.correctionTypeCands == CorrectionType::Print){

            std::cout << "CorrectionType Print is not supported in CARE GPU. Please use CARE CPU instead to print features. Abort!" << std::endl;
            return;
        }

        CUDACHECK(cudaSetDevice(runtimeOptions.deviceIds[0]));

        helpers::PeerAccessDebug peerAccess(runtimeOptions.deviceIds, true);
        peerAccess.enableAllPeerAccesses();

        //debug
        if(0){
            loadPartialResultsAndConstructOutput(
                correctionOptions,
                runtimeOptions,
                memoryOptions,
                fileOptions,
                goodAlignmentProperties,
                "partialresults1"
            );

            return;
        }

        
        /*
            Step 1: 
            - load all reads from all input files into (gpu-)memory
            - construct minhash signatures of all reads and store them in hash tables
        */

        helpers::CpuTimer step1timer("STEP1");

        std::cout << "STEP 1: Database construction" << std::endl;

        helpers::CpuTimer buildReadStorageTimer("build_readstorage");

        std::unique_ptr<ChunkedReadStorage> cpuReadStorage = constructChunkedReadStorageFromFiles(
            runtimeOptions,
            memoryOptions,
            fileOptions,
            correctionOptions.useQualityScores
        );

        buildReadStorageTimer.print();

        std::cout << "Determined the following read properties:\n";
        std::cout << "----------------------------------------\n";
        std::cout << "Total number of reads: " << cpuReadStorage->getNumberOfReads() << "\n";
        std::cout << "Minimum sequence length: " << cpuReadStorage->getSequenceLengthLowerBound() << "\n";
        std::cout << "Maximum sequence length: " << cpuReadStorage->getSequenceLengthUpperBound() << "\n";
        std::cout << "----------------------------------------\n";

        if(fileOptions.save_binary_reads_to != ""){
            std::cout << "Saving reads to file " << fileOptions.save_binary_reads_to << std::endl;
            helpers::CpuTimer timer("save_to_file");
            cpuReadStorage->saveToFile(fileOptions.save_binary_reads_to);
            timer.print();
            std::cout << "Saved reads" << std::endl;
        }

        if(correctionOptions.autodetectKmerlength){
            const int maxlength = cpuReadStorage->getSequenceLengthUpperBound();

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

        std::cout << "Reads with ambiguous bases: " << cpuReadStorage->getNumberOfReadsWithN() << std::endl;

        //compareMaxRssToLimit(memoryOptions.memoryTotalLimit, "Error memorylimit after cpureadstorage");

        std::vector<std::size_t> gpumemorylimits(runtimeOptions.deviceIds.size(), 0);

        // gpumemorylimits.resize(2);
        // std::fill(gpumemorylimits.begin(), gpumemorylimits.end(), 512000000);

        // std::vector<int> tempids2(gpumemorylimits.size(), 0);

        gpu::MultiGpuReadStorage gpuReadStorage(
            *cpuReadStorage, 
            runtimeOptions.deviceIds,
            //tempids2,
            gpumemorylimits,
            0
        );

        std::vector<gpu::GpuForest> anchorForests;
        std::vector<gpu::GpuForest> candidateForests;

        {
            ClfAgent clfAgent_(correctionOptions, fileOptions);

            for(int deviceId : runtimeOptions.deviceIds){
                cub::SwitchDevice sd{deviceId};
                if(correctionOptions.correctionType == CorrectionType::Forest){
                    anchorForests.emplace_back(*clfAgent_.classifier_anchor, deviceId);
                }

                if(correctionOptions.correctionTypeCands == CorrectionType::Forest){
                    candidateForests.emplace_back(*clfAgent_.classifier_cands, deviceId);
                }
            }

        }

        helpers::CpuTimer buildMinhasherTimer("build_minhasher");

        auto minhasherAndType = gpu::constructGpuMinhasherFromGpuReadStorage(
            fileOptions,
            runtimeOptions,
            memoryOptions,
            correctionOptions,
            gpuReadStorage,
            gpu::GpuMinhasherType::Multi
        );

        //compareMaxRssToLimit(memoryOptions.memoryTotalLimit, "Error memorylimit after gpuminhasher");

        gpu::GpuMinhasher* gpuMinhasher = minhasherAndType.first.get();

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

        //After minhasher is constructed, remaining gpu memory can be used to store reads

        std::fill(gpumemorylimits.begin(), gpumemorylimits.end(), 0);
        for(int i = 0; i < int(runtimeOptions.deviceIds.size()); i++){
            std::size_t total = 0;
            cudaMemGetInfo(&gpumemorylimits[i], &total);

            std::size_t safety = 1 << 30; //leave 1 GB for correction algorithm
            if(gpumemorylimits[i] > safety){
                gpumemorylimits[i] -= safety;
            }else{
                gpumemorylimits[i] = 0;
            }
        }

        std::size_t memoryLimitHost = memoryOptions.memoryTotalLimit 
            - cpuReadStorage->getMemoryInfo().host
            - gpuMinhasher->getMemoryInfo().host;

        // gpumemorylimits.resize(2);
        // std::fill(gpumemorylimits.begin(), gpumemorylimits.end(), 128000000);

        // std::vector<int> tempids(gpumemorylimits.size(), 0);

        helpers::CpuTimer cpugputimer("cpu->gpu reads");
        cpugputimer.start();
        gpuReadStorage.rebuild(
            *cpuReadStorage,
            runtimeOptions.deviceIds, 
            //tempids,
            gpumemorylimits,
            memoryLimitHost
        );
        cpugputimer.print();

        //compareMaxRssToLimit(memoryOptions.memoryTotalLimit, "Error memorylimit after gpureadstorage");

        //std::cout << "constructed gpu readstorage " << std::endl;

        printDataStructureMemoryUsage(gpuReadStorage, "reads");

        if(gpuReadStorage.isStandalone()){
            cpuReadStorage.reset();
        }



        std::cout << "STEP 2: Error correction" << std::endl;

        helpers::CpuTimer step2timer("STEP2");

        auto partialResults = gpu::correct_gpu(
            goodAlignmentProperties, 
            correctionOptions,
            runtimeOptions,
            fileOptions, 
            memoryOptions,
            *gpuMinhasher, 
            gpuReadStorage,
            anchorForests,
            candidateForests
        );

        step2timer.print();

        std::cout << "Correction throughput : ~" << (gpuReadStorage.getNumberOfReads() / step2timer.elapsed()) << " reads/second.\n";
        const std::size_t numTemp = partialResults.getNumElementsInMemory() + partialResults.getNumElementsInFile();
        const std::size_t numTempInMem = partialResults.getNumElementsInMemory();
        const std::size_t numTempInFile = partialResults.getNumElementsInFile();
    
        std::cerr << "Constructed " << numTemp << " corrections. "
            << numTempInMem << " corrections are stored in memory. "
            << numTempInFile << " corrections are stored in temporary file\n";

        //compareMaxRssToLimit(memoryOptions.memoryTotalLimit, "Error memorylimit after correction");


        minhasherAndType.first.reset();
        gpuMinhasher = nullptr;
        gpuReadStorage.destroy();
        cpuReadStorage.reset();

        // {
        //     std::cerr << "Saving partialresults\n";
        //     std::ofstream os("partialresults1");
        //     partialResults.saveToStream(os);
        // }

        //Merge corrected reads with input file to generate output file

        const std::size_t availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
        const auto partialResultMemUsage = partialResults.getMemoryInfo();

        // std::cerr << "availableMemoryInBytes = " << availableMemoryInBytes << "\n";
        // std::cerr << "memoryLimitOption = " << memoryOptions.memoryTotalLimit << "\n";
        // std::cerr << "partialResultMemUsage = " << partialResultMemUsage.host << "\n";

        std::size_t memoryForSorting = std::min(
            availableMemoryInBytes,
            memoryOptions.memoryTotalLimit - partialResultMemUsage.host
        );

        if(memoryForSorting > 1*(std::size_t(1) << 30)){
            memoryForSorting = memoryForSorting - 1*(std::size_t(1) << 30);
        }
        //std::cerr << "memoryForSorting = " << memoryForSorting << "\n";        

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
            false,
            runtimeOptions.showProgress
        );

        step3timer.print();

        //compareMaxRssToLimit(memoryOptions.memoryTotalLimit, "Error memorylimit after output construction");

        std::cout << "Construction of output file(s) finished." << std::endl;

    }


}
