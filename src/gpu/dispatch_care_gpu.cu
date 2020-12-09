#include <dispatch_care.hpp>
#include <gpu/gpuminhasher.cuh>

#include <gpu/singlegpuminhasher.cuh>

#include <config.hpp>
#include <options.hpp>
#include <readlibraryio.hpp>
#include <minhasher.hpp>
//#include <build.hpp>
#include <gpu/distributedreadstorage.hpp>
#include <gpu/correct_gpu.hpp>
#include <correctionresultprocessing.hpp>

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

//#define WARPMIN

#ifndef WARPMIN
        gpu::GpuMinhasher currentGpuMinhasher(
            correctionOptions.kmerlength, 
            calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage)
        );
#endif

#ifdef WARPMIN
        gpu::SingleGpuMinhasher sgpuMinhasher(totalInputFileProperties.nReads, calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage), correctionOptions.kmerlength);

        // int validNumHashFunctions = sgpuMinhasher.addHashfunctions(correctionOptions.numHashFunctions);

        // sgpuMinhasher.constructFromReadStorage(
        //     runtimeOptions,
        //     totalInputFileProperties.nReads,
        //     readStorage,
        //     totalInputFileProperties.maxSequenceLength,
        //     0,
        //     validNumHashFunctions //correctionOptions.numHashFunctions
        // );

        int validNumHashFunctions = sgpuMinhasher.constructFromReadStorage(
            runtimeOptions,
            totalInputFileProperties.nReads,
            readStorage,
            totalInputFileProperties.maxSequenceLength,
            correctionOptions.numHashFunctions
        );

        if(validNumHashFunctions == 0){
            std::cout << "Cannot construct a single gpu hashtable. Abort!" << std::endl;
            return;
        }

        std::cerr << "warpcore minhasher can use " << validNumHashFunctions << " maps\n";

        if(correctionOptions.mustUseAllHashfunctions 
            && correctionOptions.numHashFunctions != sgpuMinhasher.getNumberOfMaps()){
            std::cout << "Cannot use specified number of hash functions (" 
                << correctionOptions.numHashFunctions <<")\n";
            std::cout << "Abort!\n";
            return;
        }

        

#endif

#ifndef WARPMIN
        if(fileOptions.load_hashtables_from != ""){

            std::ifstream is(fileOptions.load_hashtables_from);
            assert((bool)is);

            const int loadedMaps = currentGpuMinhasher.loadFromStream(is, correctionOptions.numHashFunctions);

            std::cout << "Loaded " << loadedMaps << " hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }else{
            currentGpuMinhasher.construct(
                fileOptions,
                runtimeOptions,
                memoryOptions,
                totalInputFileProperties.nReads, 
                correctionOptions,
                readStorage
            );

            if(correctionOptions.mustUseAllHashfunctions 
                && correctionOptions.numHashFunctions != currentGpuMinhasher.getNumberOfMaps()){
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
            helpers::CpuTimer timer("save_to_file");
            currentGpuMinhasher.writeToStream(os);
            timer.print();

    		std::cout << "Saved minhasher" << std::endl;
        }

        printDataStructureMemoryUsage(currentGpuMinhasher, "hash tables");
#endif
        buildMinhasherTimer.print();

        step1timer.print();

#if 0        
        //compare minhashers

        {
            CudaStream stream;
            const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(totalInputFileProperties.maxSequenceLength);
            int batchsize = 1500;

            const int batches = SDIV(totalInputFileProperties.nReads, batchsize);

            gpu::GpuMinhasher::QueryHandle queryHandle1 = gpu::GpuMinhasher::makeQueryHandle();
            gpu::SingleGpuMinhasher::QueryHandle queryHandle2 = gpu::SingleGpuMinhasher::makeQueryHandle();

            helpers::SimpleAllocationPinnedHost<read_number> h_readIds(batchsize);
            helpers::SimpleAllocationDevice<read_number> d_readIds(batchsize);
            helpers::SimpleAllocationDevice<int> d_sequenceLengths(batchsize);
            helpers::SimpleAllocationDevice<unsigned int> d_encodedSequences(batchsize * encodedSequencePitchInInts);

            helpers::SimpleAllocationPinnedHost<read_number> h_similarReadIds(batchsize * 48 * calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage));
            helpers::SimpleAllocationPinnedHost<int> h_similarReadsPerSequence(batchsize);
            helpers::SimpleAllocationPinnedHost<int> h_similarReadsPerSequencePrefixSum(batchsize+1);

            helpers::SimpleAllocationPinnedHost<read_number> h_similarReadIds2(batchsize * 48 * calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage));
            helpers::SimpleAllocationPinnedHost<int> h_similarReadsPerSequence2(batchsize);
            helpers::SimpleAllocationPinnedHost<int> h_similarReadsPerSequencePrefixSum2(batchsize+1);

            helpers::SimpleAllocationDevice<read_number> d_similarReadIds(batchsize * 48 * calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage));
            helpers::SimpleAllocationDevice<int> d_similarReadsPerSequence(batchsize);
            helpers::SimpleAllocationDevice<int> d_similarReadsPerSequencePrefixSum(batchsize+1);

            helpers::SimpleAllocationDevice<read_number> d_similarReadIds2(batchsize * 48 * calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage));
            helpers::SimpleAllocationDevice<int> d_similarReadsPerSequence2(batchsize);
            helpers::SimpleAllocationDevice<int> d_similarReadsPerSequencePrefixSum2(batchsize+1);

            helpers::GpuTimer currentMinhasherTimer(stream, "current minhasher", 0);
            helpers::GpuTimer warpcoreMinhasherTimer(stream, "warpcore minhasher", 0);

            auto sequencehandle = readStorage.makeGatherHandleSequences();
            std::cerr << "Checking hashes\n";
            for(int batch = 0; batch < batches; batch++){

                const int bbegin = batch * batchsize;
                const int bend = std::min(int(totalInputFileProperties.nReads), (batch+1) * batchsize);
                const int bsize = bend - bbegin;

                for(int i = 0; i < bsize; i++){
                    h_readIds[i] = bbegin + i;
                }

                cudaMemcpyAsync(d_readIds, h_readIds, h_readIds.sizeInBytes(), H2D, stream); CUERR;

                readStorage.gatherSequenceDataToGpuBufferAsync(
                    nullptr,
                    sequencehandle,
                    d_encodedSequences,
                    encodedSequencePitchInInts,
                    h_readIds,
                    d_readIds,
                    bsize,
                    0,
                    stream
                );
            
                readStorage.gatherSequenceLengthsToGpuBufferAsync(
                    d_sequenceLengths,
                    0,
                    d_readIds,
                    bsize,
                    stream
                );               

                currentMinhasherTimer.start();

                ForLoopExecutor forLoopExecutor(nullptr, nullptr);
                helpers::CpuTimer oldTimer("oldTimer");
                nvtx::push_range("oldminhasher", 0);
                currentGpuMinhasher.getIdsOfSimilarReadsNormalExcludingSelfNew(
                    queryHandle1,
                    d_readIds,
                    h_readIds,
                    d_encodedSequences,
                    encodedSequencePitchInInts,
                    d_sequenceLengths,
                    bsize,
                    0, 
                    stream,
                    forLoopExecutor,
                    d_similarReadIds,
                    d_similarReadsPerSequence,
                    d_similarReadsPerSequencePrefixSum
                );
                nvtx::pop_range();

                cudaStreamSynchronize(stream); CUERR;
                oldTimer.stop();
                //oldTimer.print();
                currentMinhasherTimer.stop();

                warpcoreMinhasherTimer.start();

                helpers::CpuTimer newTimer("newTimer");
                nvtx::push_range("newminhasher", 0);
                #if 0
                sgpuMinhasher.queryExcludingSelf(
                    queryHandle2,
                    d_similarReadIds2,
                    d_similarReadsPerSequence2,
                    d_similarReadsPerSequencePrefixSum2,
                    d_encodedSequences,
                    bsize,
                    d_sequenceLengths,
                    encodedSequencePitchInInts,
                    d_readIds,
                    stream
                );
                #else
                sgpuMinhasher.getIdsOfSimilarReadsNormalExcludingSelfNew(
                    queryHandle2,
                    d_readIds,
                    h_readIds,
                    d_encodedSequences,
                    encodedSequencePitchInInts,
                    d_sequenceLengths,
                    bsize,
                    0, 
                    stream,
                    forLoopExecutor,
                    d_similarReadIds2,
                    d_similarReadsPerSequence2,
                    d_similarReadsPerSequencePrefixSum2
                );
                #endif
                nvtx::pop_range();            

                cudaStreamSynchronize(stream); CUERR;
                newTimer.stop();
                //newTimer.print();
                warpcoreMinhasherTimer.stop();

                cudaMemcpyAsync(h_similarReadIds, d_similarReadIds, d_similarReadIds.sizeInBytes(), D2H, stream); CUERR;
                cudaMemcpyAsync(h_similarReadsPerSequence, d_similarReadsPerSequence, d_similarReadsPerSequence.sizeInBytes(), D2H, stream); CUERR;
                cudaMemcpyAsync(h_similarReadsPerSequencePrefixSum, d_similarReadsPerSequencePrefixSum, d_similarReadsPerSequencePrefixSum.sizeInBytes(), D2H, stream); CUERR;

                cudaMemcpyAsync(h_similarReadIds2, d_similarReadIds2, d_similarReadIds2.sizeInBytes(), D2H, stream); CUERR;
                cudaMemcpyAsync(h_similarReadsPerSequence2, d_similarReadsPerSequence2, d_similarReadsPerSequence2.sizeInBytes(), D2H, stream); CUERR;
                cudaMemcpyAsync(h_similarReadsPerSequencePrefixSum2, d_similarReadsPerSequencePrefixSum2, d_similarReadsPerSequencePrefixSum2.sizeInBytes(), D2H, stream); CUERR;

                cudaDeviceSynchronize(); CUERR;

                for(int i = 0; i < bsize; i++){
                //for(int i = 0; i < std::min(10, bsize); i++){
                    // std::cerr << h_similarReadsPerSequence[i] << " , " << h_similarReadsPerSequence2[i] << "\n";
                    // for(int p = 0; p < h_similarReadsPerSequence[i]; p++){
                    //     std::cerr << h_similarReadIds[h_similarReadsPerSequencePrefixSum[i] + p] << ", ";
                    // }
                    // std::cerr << " AAA\n";

                    // for(int p = 0; p < h_similarReadsPerSequence2[i]; p++){
                    //     std::cerr << h_similarReadIds2[h_similarReadsPerSequencePrefixSum2[i] + p] << ", ";
                    // }
                    // std::cerr << " BBB\n";

                    if(!(h_similarReadsPerSequence[i] == h_similarReadsPerSequence2[i])){
                        std::cerr << "error batch " << batch << ", i = " << i << "\n";
                        std::cerr << h_similarReadsPerSequence[i] << " != " << h_similarReadsPerSequence2[i] << "\n";
                    }
                    assert(h_similarReadsPerSequence[i] == h_similarReadsPerSequence2[i]);

                    if(!(h_similarReadsPerSequencePrefixSum[i] == h_similarReadsPerSequencePrefixSum2[i])){
                        std::cerr << "error prefixsum batch " << batch << ", i = " << i << "\n";
                        std::cerr << h_similarReadsPerSequencePrefixSum[i] << " != " << h_similarReadsPerSequencePrefixSum2[i] << "\n";
                    }
                    assert(h_similarReadsPerSequencePrefixSum[i] == h_similarReadsPerSequencePrefixSum2[i]);
                    
                    for(int p = 0; p < h_similarReadsPerSequence[i]; p++){
                        auto old = h_similarReadIds[h_similarReadsPerSequencePrefixSum[i] + p];
                        auto notold = h_similarReadIds2[h_similarReadsPerSequencePrefixSum2[i] + p];

                        if(!(old == notold)){
                            std::cerr << "error batch " << batch << ", i = " << i << ", p = " << p << "\n";
                            std::cerr << old << " != " << notold << "\n";
                        }
                        assert(old == notold);
                    }
                }

                if(batch % (SDIV(batches,20)) == 0){
                    std::cerr << batch << "/ " << batches << "\n";
                }
            }

            std::cerr << "Checking hashes done\n";
            currentMinhasherTimer.print();
            warpcoreMinhasherTimer.print();

        }

#endif



//#ifdef WARPMIN
#if 0
        currentGpuMinhasher.destroy();
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
#ifndef WARPMIN
            currentGpuMinhasher,
#else 
            sgpuMinhasher,
#endif                        
            readStorage
        );

        step2timer.print();

        std::cout << "Correction throughput : ~" << (totalInputFileProperties.nReads / step2timer.elapsed()) << " reads/second.\n";

        #ifndef WARPMIN
            currentGpuMinhasher.destroy();
        #else 
            sgpuMinhasher.destroy();
        #endif 

        readStorage.destroy();

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
