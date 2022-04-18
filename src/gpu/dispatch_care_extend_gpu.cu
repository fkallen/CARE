
#include <gpu/dispatch_care_extend_gpu.cuh>
#include <gpu/gpuminhasherconstruction.cuh>
//#include <gpu/fakegpuminhasher.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/multigpureadstorage.cuh>
#include <gpu/readextension_gpu.hpp>

#include <config.hpp>
#include <extensionagent.hpp>
#include <options.hpp>
#include <readlibraryio.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>
#include <chunkedreadstorageconstruction.hpp>
#include <chunkedreadstorage.hpp>


#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include <experimental/filesystem>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <gpu/rmm_utilities.cuh>

namespace filesys = std::experimental::filesystem;

namespace care{

    namespace extension{

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

    
    struct UsageStatistics {
        std::uint64_t reserved;
        std::uint64_t reservedHigh;
        std::uint64_t used;
        std::uint64_t usedHigh;
    };

    void getUsageStatistics(cudaMemPool_t memPool, UsageStatistics* statistics){
        cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, &statistics->reserved);
        cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &statistics->reservedHigh);
        cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, &statistics->used);
        cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &statistics->usedHigh);
    }

    void printUsageStatistics(cudaMemPool_t memPool){
        UsageStatistics stats;
        getUsageStatistics(memPool, &stats);
        std::cerr << "reserved: " << stats.reserved << ", reservedHigh: " << stats.reservedHigh << ", used: " << stats.used << ", usedHigh: " << stats.usedHigh << "\n";
    }


    void performExtension(
        ProgramOptions programOptions
    ){

        std::cout << "Running CARE EXTEND GPU" << std::endl;

        if(programOptions.deviceIds.size() == 0){
            std::cout << "No device ids found. Abort!" << std::endl;
            return;
        }

        CUDACHECK(cudaSetDevice(programOptions.deviceIds[0]));

        helpers::PeerAccessDebug peerAccess(programOptions.deviceIds, true);
        peerAccess.enableAllPeerAccesses();

        //Set up memory pool
        std::vector<std::unique_ptr<rmm::mr::cuda_async_memory_resource>> rmmCudaAsyncResources;
        for(auto id : programOptions.deviceIds){
            cub::SwitchDevice sd(id);

            auto resource = std::make_unique<rmm::mr::cuda_async_memory_resource>();
            rmm::mr::set_per_device_resource(rmm::cuda_device_id(id), resource.get());

            for(auto otherId : programOptions.deviceIds){
                if(otherId != id){
                    if(peerAccess.canAccessPeer(id, otherId)){
                        cudaMemAccessDesc accessDesc = {};
                        accessDesc.location.type = cudaMemLocationTypeDevice;
                        accessDesc.location.id = otherId;
                        accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

                        CUDACHECK(cudaMemPoolSetAccess(resource->pool_handle(), &accessDesc, 1));
                    }
                }
            }

            CUDACHECK(cudaDeviceSetMemPool(id, resource->pool_handle()));
            rmmCudaAsyncResources.push_back(std::move(resource));
        }

        helpers::CpuTimer step1timer("STEP1");

        std::cout << "STEP 1: Database construction" << std::endl;

        helpers::CpuTimer buildReadStorageTimer("build_readstorage");

        const int numQualityBits = programOptions.qualityScoreBits;
        
        std::unique_ptr<ChunkedReadStorage> cpuReadStorage = constructChunkedReadStorageFromFiles(
            programOptions
        );

        buildReadStorageTimer.print();

        //trim pools
        for(size_t i = 0; i < rmmCudaAsyncResources.size(); i++){
            cub::SwitchDevice sd(programOptions.deviceIds[i]);
            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaMemPoolTrimTo(rmmCudaAsyncResources[i]->pool_handle(), 0));
        }

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

        helpers::CpuTimer buildMinhasherTimer("build_minhasher");

        auto minhasherAndType = gpu::constructGpuMinhasherFromGpuReadStorage(
            programOptions,
            gpuReadStorage,
            gpu::GpuMinhasherType::Single
        );

        gpu::GpuMinhasher* gpuMinhasher = minhasherAndType.first.get();

        buildMinhasherTimer.print();

        //trim pools
        for(size_t i = 0; i < rmmCudaAsyncResources.size(); i++){
            cub::SwitchDevice sd(programOptions.deviceIds[i]);
            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaMemPoolTrimTo(rmmCudaAsyncResources[i]->pool_handle(), 0));
        }

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

        if(programOptions.save_hashtables_to != "" && gpuMinhasher->canWriteToStream()) {
            std::cout << "Saving minhasher to file " << programOptions.save_hashtables_to << std::endl;
            std::ofstream os(programOptions.save_hashtables_to);
            assert((bool)os);
            helpers::CpuTimer timer("save_to_file");
            gpuMinhasher->writeToStream(os);
            timer.print();

            std::cout << "Saved minhasher" << std::endl;
        }

        printDataStructureMemoryUsage(*gpuMinhasher, "hash tables");        

        step1timer.print();

        //After minhasher is constructed, remaining gpu memory can be used to store reads

        auto getGpuMemoryLimits = [&](){
            const std::size_t numGpus = programOptions.deviceIds.size();
            std::vector<std::size_t> limits(numGpus, 0);

            for(std::size_t i = 0; i < numGpus; i++){
                cub::SwitchDevice sd{programOptions.deviceIds[i]};
                
                std::size_t total = 0;
                cudaMemGetInfo(&limits[i], &total);

                std::size_t safety = 1 << 30; //leave 1 GB for algorithm
                if(limits[i] > safety){
                    limits[i] -= safety;
                }else{
                    limits[i] = 0;
                }
            }
            return limits;
        };

        gpumemorylimits = getGpuMemoryLimits();

        std::size_t memoryLimitHost = programOptions.memoryTotalLimit 
            - cpuReadStorage->getMemoryInfo().host
            - gpuMinhasher->getMemoryInfo().host;

        // gpumemorylimits.resize(2);
        // std::fill(gpumemorylimits.begin(), gpumemorylimits.end(), 512000000);

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

        gpumemorylimits = getGpuMemoryLimits();
        
        if(programOptions.replicateGpuData){
            if(gpuReadStorage.trySequenceReplication(gpumemorylimits)){
                std::cerr << "Replicated gpu read sequences to each gpu\n";
                gpumemorylimits = getGpuMemoryLimits();
            }
            
            if(gpuReadStorage.tryQualityReplication(gpumemorylimits)){
                std::cerr << "Replicated gpu read qualities to each gpu\n";
                gpumemorylimits = getGpuMemoryLimits();
            }
        }
        cpugputimer.print();

        std::cout << "constructed gpu readstorage " << std::endl;

        printDataStructureMemoryUsage(gpuReadStorage, "reads");

        if(gpuReadStorage.isStandalone()){
            cpuReadStorage.reset();
        }

        std::cout << "STEP 2: Read extension" << std::endl;

        helpers::CpuTimer step2timer("STEP2");

        ExtensionAgent<gpu::GpuMinhasher, gpu::GpuReadStorage> extensionAgent(
            programOptions,
            *gpuMinhasher, 
            gpuReadStorage
        );

        extensionAgent.run(
            &gpu::extend_gpu,
            [&](){
                step2timer.print();

                std::cout << "Extension throughput : ~" << (gpuReadStorage.getNumberOfReads() / step2timer.elapsed()) << " reads/second.\n"; //TODO: paired end? numreads / 2 ?

                minhasherAndType.first.reset();
                gpuMinhasher = nullptr;
                gpuReadStorage.destroy();
                cpuReadStorage.reset(); 

                //trim pools
                for(size_t i = 0; i < rmmCudaAsyncResources.size(); i++){
                    cub::SwitchDevice sd(programOptions.deviceIds[i]);
                    CUDACHECK(cudaDeviceSynchronize());
                    CUDACHECK(cudaMemPoolTrimTo(rmmCudaAsyncResources[i]->pool_handle(), 0));
                }
            }
        );

        std::cout << "Construction of output file(s) finished." << std::endl;

}

}
