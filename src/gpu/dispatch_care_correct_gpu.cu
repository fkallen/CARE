#include <gpu/dispatch_care_correct_gpu.cuh>
#include <gpu/gpuminhasherconstruction.cuh>
#include <gpu/fakegpuminhasher.cuh>
#include <gpu/cudaerrorcheck.cuh>

#include <config.hpp>
#include <options.hpp>
#include <readlibraryio.hpp>
#include <memorymanagement.hpp>
#include <correctedsequence.hpp>
#include <correctionresultoutput.hpp>
#include <gpu/correct_gpu.hpp>
#include <classification.hpp>
#include <gpu/forest_gpu.cuh>

#include <gpu/multigpureadstorage.cuh>
#include <chunkedreadstorageconstruction.hpp>
#include <chunkedreadstorage.hpp>

#include <rangegenerator.hpp>
#include <sortserializedresults.hpp>


#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <numeric>
#include <random>

#include <experimental/filesystem>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <gpu/rmm_utilities.cuh>

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




    void performCorrection(
        ProgramOptions programOptions
    ){

        std::cout << "Running CARE GPU" << std::endl;

        if(programOptions.deviceIds.size() == 0){
            std::cout << "No device ids found. Abort!" << std::endl;
            return;
        }

        if(programOptions.correctionType == CorrectionType::Print 
            || programOptions.correctionTypeCands == CorrectionType::Print){

            std::cout << "CorrectionType Print is not supported in CARE GPU. Please use CARE CPU instead to print features. Abort!" << std::endl;
            return;
        }

        CUDACHECK(cudaSetDevice(programOptions.deviceIds[0]));

        //debug buffer printf
        CUDACHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*1024*512));

        helpers::PeerAccessDebug peerAccess(programOptions.deviceIds, true);
        peerAccess.enableAllPeerAccesses();

        //set up memory pools for malloc_async
        for(auto id : programOptions.deviceIds){
            cudaMemPool_t defaultMemoryPool;
            CUDACHECK(cudaDeviceGetDefaultMemPool(&defaultMemoryPool, id));
            uint64_t threshold = UINT64_MAX;
            CUDACHECK(cudaMemPoolSetAttribute(defaultMemoryPool, cudaMemPoolAttrReleaseThreshold, &threshold));
        }

        //set up rmm resources
        std::vector<std::unique_ptr<MyRMMCudaAsyncResource>> rmmCudaAsyncResources;
        std::vector<std::unique_ptr<rmm::mr::logging_resource_adaptor<MyRMMCudaAsyncResource>>> rmmLoggingResources;
        std::vector<std::unique_ptr<std::ofstream>> logfilestreams;

        for(auto id : programOptions.deviceIds){
            cub::SwitchDevice sd(id);

            cudaMemPool_t defaultMemoryPool;
            CUDACHECK(cudaDeviceGetDefaultMemPool(&defaultMemoryPool, id));

            const bool autoflush = true;

            rmmCudaAsyncResources.push_back(std::make_unique<MyRMMCudaAsyncResource>(defaultMemoryPool));
            logfilestreams.push_back(std::make_unique<std::ofstream>("logging_device_0.txt"));
            rmmLoggingResources.push_back(std::make_unique<rmm::mr::logging_resource_adaptor<MyRMMCudaAsyncResource>>(rmmCudaAsyncResources.back().get(), *logfilestreams.back(), autoflush));

            rmm::mr::set_per_device_resource(rmm::cuda_device_id(id), rmmCudaAsyncResources.back().get());
            //rmm::mr::set_per_device_resource(rmm::cuda_device_id(id), rmmLoggingResources.back().get());
        }

        
        /*
            Step 1: 
            - load all reads from all input files into (gpu-)memory
            - construct minhash signatures of all reads and store them in hash tables
        */

        helpers::CpuTimer step1timer("STEP1");

        std::cout << "STEP 1: Database construction" << std::endl;

        helpers::CpuTimer buildReadStorageTimer("build_readstorage");

        const int numQualityBits = programOptions.qualityScoreBits;
        
        std::unique_ptr<ChunkedReadStorage> cpuReadStorage = constructChunkedReadStorageFromFiles(
            programOptions
        );

        buildReadStorageTimer.print();

        std::cout << "Determined the following read properties:\n";
        std::cout << "----------------------------------------\n";
        std::cout << "Total number of reads: " << cpuReadStorage->getNumberOfReads() << "\n";
        std::cout << "Minimum sequence length: " << cpuReadStorage->getSequenceLengthLowerBound() << "\n";
        std::cout << "Maximum sequence length: " << cpuReadStorage->getSequenceLengthUpperBound() << "\n";
        std::cout << "----------------------------------------\n";

        if(programOptions.save_binary_reads_to != ""){
            std::cout << "Saving reads to file " << programOptions.save_binary_reads_to << std::endl;
            helpers::CpuTimer timer("save_to_file");
            cpuReadStorage->saveToFile(programOptions.save_binary_reads_to);
            timer.print();
            std::cout << "Saved reads" << std::endl;
        }

        if(programOptions.autodetectKmerlength){
            const int maxlength = cpuReadStorage->getSequenceLengthUpperBound();

            auto getKmerSizeForHashing = [](int maximumReadLength){
                if(maximumReadLength < 160){
                    return 20;
                }else{
                    return 32;
                }
            };

            programOptions.kmerlength = getKmerSizeForHashing(maxlength);

            std::cout << "Will use k-mer length = " << programOptions.kmerlength << " for hashing.\n";
        }

        std::cout << "Reads with ambiguous bases: " << cpuReadStorage->getNumberOfReadsWithN() << std::endl;

        //compareMaxRssToLimit(programOptions.memoryTotalLimit, "Error memorylimit after cpureadstorage");

        std::vector<std::size_t> gpumemorylimits(programOptions.deviceIds.size(), 0);

        // gpumemorylimits.resize(2);
        // std::fill(gpumemorylimits.begin(), gpumemorylimits.end(), 512000000);

        // std::vector<int> tempids2(gpumemorylimits.size(), 0);


        gpu::MultiGpuReadStorage gpuReadStorage(
            *cpuReadStorage, 
            programOptions.deviceIds,
            //tempids2,
            gpumemorylimits,
            0,
            numQualityBits
        );

        std::vector<gpu::GpuForest> anchorForests;
        std::vector<gpu::GpuForest> candidateForests;

        {
            ClfAgent clfAgent_(programOptions);

            for(int deviceId : programOptions.deviceIds){
                cub::SwitchDevice sd{deviceId};
                if(programOptions.correctionType == CorrectionType::Forest){
                    anchorForests.emplace_back(*clfAgent_.classifier_anchor, deviceId);
                }

                if(programOptions.correctionTypeCands == CorrectionType::Forest){
                    candidateForests.emplace_back(*clfAgent_.classifier_cands, deviceId);
                }
            }

        }

        helpers::CpuTimer buildMinhasherTimer("build_minhasher");

        auto minhasherAndType = gpu::constructGpuMinhasherFromGpuReadStorage(
            programOptions,
            gpuReadStorage,
            gpu::GpuMinhasherType::Multi
        );

        //compareMaxRssToLimit(programOptions.memoryTotalLimit, "Error memorylimit after gpuminhasher");

        gpu::GpuMinhasher* gpuMinhasher = minhasherAndType.first.get();

        buildMinhasherTimer.print();

        std::cout << "Using minhasher type: " << to_string(minhasherAndType.second) << "\n";
        std::cout << "GpuMinhasher can use " << gpuMinhasher->getNumberOfMaps() << " maps\n";

        if(gpuMinhasher->getNumberOfMaps() <= 0){
            std::cout << "Cannot construct a single gpu hashtable. Abort!" << std::endl;
            return;
        }

        if(programOptions.mustUseAllHashfunctions 
            && programOptions.numHashFunctions != gpuMinhasher->getNumberOfMaps()){
            std::cout << "Cannot use specified number of hash functions (" 
                << programOptions.numHashFunctions <<")\n";
            std::cout << "Abort!\n";
            return;
        }

        if(minhasherAndType.second == gpu::GpuMinhasherType::Fake){

            gpu::FakeGpuMinhasher* fakeGpuMinhasher = dynamic_cast<gpu::FakeGpuMinhasher*>(gpuMinhasher);
            assert(fakeGpuMinhasher != nullptr);

            if(programOptions.save_hashtables_to != "") {
                std::cout << "Saving minhasher to file " << programOptions.save_hashtables_to << std::endl;
                std::ofstream os(programOptions.save_hashtables_to);
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

        //std::fill(gpumemorylimits.begin(), gpumemorylimits.end(), 2ull*1024ull*1024ull*1024ull);
        std::fill(gpumemorylimits.begin(), gpumemorylimits.end(), 0);
        for(int i = 0; i < int(programOptions.deviceIds.size()); i++){
            std::size_t total = 0;
            cudaMemGetInfo(&gpumemorylimits[i], &total);

            std::size_t safety = 1 << 30; //leave 1 GB for correction algorithm
            if(gpumemorylimits[i] > safety){
                gpumemorylimits[i] -= safety;
            }else{
                gpumemorylimits[i] = 0;
            }
        }

        std::size_t memoryLimitHost = programOptions.memoryTotalLimit 
            - cpuReadStorage->getMemoryInfo().host
            - gpuMinhasher->getMemoryInfo().host;

        // gpumemorylimits.resize(2);
        // std::fill(gpumemorylimits.begin(), gpumemorylimits.end(), 128000000);

        // std::vector<int> tempids(gpumemorylimits.size(), 0);

        helpers::CpuTimer cpugputimer("cpu->gpu reads");
        cpugputimer.start();
        gpuReadStorage.rebuild(
            *cpuReadStorage,
            programOptions.deviceIds, 
            //tempids,
            gpumemorylimits,
            memoryLimitHost,
            numQualityBits
        );
        cpugputimer.print();

        printDataStructureMemoryUsage(gpuReadStorage, "reads");

        //compareMaxRssToLimit(programOptions.memoryTotalLimit, "Error memorylimit after gpureadstorage");

        //std::cout << "constructed gpu readstorage " << std::endl;

        

        if(gpuReadStorage.isStandalone()){
            cpuReadStorage.reset();
        }


        // {
        //     auto rshandle = gpuReadStorage.makeHandle();
        //     helpers::SimpleAllocationPinnedHost<char> d_quality_data(128 * 10);
        //     helpers::SimpleAllocationPinnedHost<read_number> readIds(10);
        //     std::iota(readIds.begin(), readIds.end(), 0);

        //     gpuReadStorage.gatherQualities(
        //         rshandle,
        //         d_quality_data,
        //         128,
        //         AsyncConstBufferWrapper<read_number>(readIds.data()),
        //         readIds.data(),
        //         10,
        //         0
        //     );
        //     CUDACHECK(cudaDeviceSynchronize());

        //     for(int k = 0; k < 10; k++){
        //         for(int i = 0; i < 101; i++){
        //             std::cerr << d_quality_data[128 * k + i];
        //         }
        //         std::cerr << "\n";
        //     }
        //     std::exit(0);
        // }


        std::cout << "STEP 2: Error correction" << std::endl;

        helpers::CpuTimer step2timer("STEP2");

        auto partialResults = gpu::correct_gpu(
            programOptions,
            *gpuMinhasher, 
            gpuReadStorage,
            anchorForests,
            candidateForests
        );

        step2timer.print();

        std::cout << "Correction throughput : ~" << (gpuReadStorage.getNumberOfReads() / step2timer.elapsed()) << " reads/second.\n";
    
        std::cerr << "Constructed " << partialResults.size() << " corrections. ";
        std::cerr << "They occupy a total of " << (partialResults.dataBytes() + partialResults.offsetBytes()) << " bytes\n";

        //compareMaxRssToLimit(programOptions.memoryTotalLimit, "Error memorylimit after correction");


        minhasherAndType.first.reset();
        gpuMinhasher = nullptr;
        gpuReadStorage.destroy();
        cpuReadStorage.reset();


        //Merge corrected reads with input file to generate output file

        const std::size_t availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
        const auto partialResultMemUsage = partialResults.getMemoryInfo();

        // std::cerr << "availableMemoryInBytes = " << availableMemoryInBytes << "\n";
        // std::cerr << "memoryLimitOption = " << programOptions.memoryTotalLimit << "\n";
        // std::cerr << "partialResultMemUsage = " << partialResultMemUsage.host << "\n";

        std::size_t memoryForSorting = std::min(
            availableMemoryInBytes,
            programOptions.memoryTotalLimit - partialResultMemUsage.host
        );

        if(memoryForSorting > 1*(std::size_t(1) << 30)){
            memoryForSorting = memoryForSorting - 1*(std::size_t(1) << 30);
        }
        //std::cerr << "memoryForSorting = " << memoryForSorting << "\n";        

        std::cout << "STEP 3: Constructing output file(s)" << std::endl;

        helpers::CpuTimer step3timer("STEP3");

        helpers::CpuTimer sorttimer("sort_results_by_read_id");

        sortSerializedResultsByReadIdAscending<EncodedTempCorrectedSequence>(
            partialResults,
            memoryForSorting
        );

        sorttimer.print();

        std::vector<FileFormat> formats;
        for(const auto& inputfile : programOptions.inputfiles){
            formats.emplace_back(getFileFormat(inputfile));
        }
        std::vector<std::string> outputfiles;
        for(const auto& outputfilename : programOptions.outputfilenames){
            outputfiles.emplace_back(programOptions.outputdirectory + "/" + outputfilename);
        }
        constructOutputFileFromCorrectionResults(
            programOptions.inputfiles, 
            partialResults, 
            formats[0],
            outputfiles,
            programOptions.showProgress
        );

        step3timer.print();

        //compareMaxRssToLimit(programOptions.memoryTotalLimit, "Error memorylimit after output construction");

        std::cout << "Construction of output file(s) finished." << std::endl;

    }


}
