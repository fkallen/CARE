#include <dispatch_care_correct_cpu.hpp>
#include <hpc_helpers.cuh>
#include <config.hpp>
#include <options.hpp>
#include <singlehashminhasher.hpp>
#include <correct_cpu.hpp>

#include <minhasherlimit.hpp>

#include <correctedsequence.hpp>
#include <correctionresultoutput.hpp>
#include <sequencehelpers.hpp>
#include <cpuminhasherconstruction.hpp>
#include <ordinaryminhasher.hpp>
#include <serializedobjectstorage.hpp>
#include <sortserializedresults.hpp>
#include <chunkedreadstorageconstruction.hpp>
#include <chunkedreadstorage.hpp>

#include <contiguousreadstorage.hpp>
#include <vector>
#include <iostream>
#include <mutex>
#include <thread>
#include <memory>

#include <experimental/filesystem>


namespace filesys = std::experimental::filesystem;



namespace care{

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
        GoodAlignmentProperties goodAlignmentProperties
    ){

        // {
        //     DoublePassMultiValueHashTable<int, char> table(0, 0.8f);
        //     std::vector<int> keys{0,1,0,1,2,2,3};
        //     std::vector<char> values{'0', '1', '0', '1', '2', '2', '3'};
        //     table.firstPassInsert(keys.data(), values.data(), keys.size());
        //     table.firstPassDone(1);
        //     table.secondPassInsert(keys.data(), values.data(), keys.size());
        //     table.secondPassDone();

        //     std::vector<int> uniquekeys{0,1,2,3};
        //     auto q0 = table.query(0);
        //     std::cerr << "0: ";
        //     for(int i = 0; i < q0.numValues; i++){
        //         std::cerr << q0.valuesBegin[i] << ", ";
        //     }
        //     std::cerr << "\n";

        //     auto q1 = table.query(1);
        //     std::cerr << "1: ";
        //     for(int i = 0; i < q1.numValues; i++){
        //         std::cerr << q1.valuesBegin[i] << ", ";
        //     }
        //     std::cerr << "\n";

        //     auto q2 = table.query(2);
        //     std::cerr << "2: ";
        //     for(int i = 0; i < q2.numValues; i++){
        //         std::cerr << q2.valuesBegin[i] << ", ";
        //     }
        //     std::cerr << "\n";

        //     auto q3 = table.query(3);
        //     std::cerr << "3: ";
        //     for(int i = 0; i < q3.numValues; i++){
        //         std::cerr << q3.valuesBegin[i] << ", ";
        //     }
        //     std::cerr << "\n";
        // }










        std::cout << "Running CARE CPU" << std::endl;

        helpers::CpuTimer step1Timer("STEP1");

        std::cout << "STEP 1: Database construction" << std::endl;

        helpers::CpuTimer buildReadStorageTimer("build_readstorage");

        const int numQualityBits = memoryOptions.qualityScoreBits;

        std::unique_ptr<ChunkedReadStorage> cpuReadStorage = constructChunkedReadStorageFromFiles(
            runtimeOptions,
            memoryOptions,
            fileOptions,
            correctionOptions.useQualityScores,
            numQualityBits
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

        printDataStructureMemoryUsage(*cpuReadStorage, "reads");

        //compareMaxRssToLimit(memoryOptions.memoryTotalLimit, "Error memorylimit after cpureadstorage");


        helpers::CpuTimer buildMinhasherTimer("build_minhasher");

        #if 1

        auto minhasherAndType = constructCpuMinhasherFromCpuReadStorage(
            fileOptions,
            runtimeOptions,
            memoryOptions,
            correctionOptions,
            *cpuReadStorage,
            CpuMinhasherType::Ordinary
        );

        //compareMaxRssToLimit(memoryOptions.memoryTotalLimit, "Error memorylimit after cpuminhasher");

        CpuMinhasher* cpuMinhasher = minhasherAndType.first.get();

        buildMinhasherTimer.print();

        std::cout << "CpuMinhasher can use " << cpuMinhasher->getNumberOfMaps() << " maps\n";

        if(cpuMinhasher->getNumberOfMaps() <= 0){
            std::cout << "Cannot construct a single cpu hashtable. Abort!" << std::endl;
            return;
        }

        if(correctionOptions.mustUseAllHashfunctions 
            && correctionOptions.numHashFunctions != cpuMinhasher->getNumberOfMaps()){
            std::cout << "Cannot use specified number of hash functions (" 
                << correctionOptions.numHashFunctions <<")\n";
            std::cout << "Abort!\n";
            return;
        }

        if(minhasherAndType.second == CpuMinhasherType::Ordinary){

            OrdinaryCpuMinhasher* ordinaryCpuMinhasher = dynamic_cast<OrdinaryCpuMinhasher*>(cpuMinhasher);
            assert(ordinaryCpuMinhasher != nullptr);

            if(fileOptions.save_hashtables_to != "") {
                std::cout << "Saving minhasher to file " << fileOptions.save_hashtables_to << std::endl;
                std::ofstream os(fileOptions.save_hashtables_to);
                assert((bool)os);
                helpers::CpuTimer timer("save_to_file");
                ordinaryCpuMinhasher->writeToStream(os);
                timer.print();

                std::cout << "Saved minhasher" << std::endl;
            }

        }

        printDataStructureMemoryUsage(*cpuMinhasher, "hash tables");

        step1Timer.print();

        std::cout << "STEP 2: Error correction" << std::endl;

        helpers::CpuTimer step2Timer("STEP2");

        auto partialResults = cpu::correct_cpu(
            goodAlignmentProperties, 
            correctionOptions,
            runtimeOptions, 
            fileOptions, 
            memoryOptions, 
            *cpuMinhasher, 
            *cpuReadStorage
        );

        step2Timer.print();

        std::cout << "Correction throughput : ~" << (cpuReadStorage->getNumberOfReads() / step2Timer.elapsed()) << " reads/second.\n";

        std::cerr << "Constructed " << partialResults.size() << " corrections. ";
        std::cerr << "They occupy a total of " << (partialResults.dataBytes() + partialResults.offsetBytes()) << " bytes\n";

        //compareMaxRssToLimit(memoryOptions.memoryTotalLimit, "Error memorylimit after correction");

        minhasherAndType.first.reset();
        cpuMinhasher = nullptr;        
        cpuReadStorage.reset();

        #else

        auto cpuMinhasher = std::make_unique<SingleHashCpuMinhasher>(
            cpuReadStorage->getNumberOfReads(),
            255,//calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage),
            correctionOptions.kmerlength,
            memoryOptions.hashtableLoadfactor
        );

        #if 1
        cpuMinhasher->constructFromReadStorage(
            fileOptions,
            runtimeOptions,
            memoryOptions,
            cpuReadStorage->getNumberOfReads(),
            correctionOptions,
            *cpuReadStorage
        );
        #else
        std::ifstream tablestreamA("tablestream.bin", std::ios::binary);
        cpuMinhasher->loadFromStream(tablestreamA);
        #endif


        //compareMaxRssToLimit(memoryOptions.memoryTotalLimit, "Error memorylimit after cpuminhasher");

        buildMinhasherTimer.print();

        std::cout << "CpuMinhasher can use " << cpuMinhasher->getNumberOfMaps() << " maps\n";

        if(cpuMinhasher->getNumberOfMaps() <= 0){
            std::cout << "Cannot construct a single cpu hashtable. Abort!" << std::endl;
            return;
        }

        if(correctionOptions.mustUseAllHashfunctions 
            && correctionOptions.numHashFunctions != cpuMinhasher->getNumberOfMaps()){
            std::cout << "Cannot use specified number of hash functions (" 
                << correctionOptions.numHashFunctions <<")\n";
            std::cout << "Abort!\n";
            return;
        }

        printDataStructureMemoryUsage(*cpuMinhasher, "hash tables");

        step1Timer.print();

        std::cout << "STEP 2: Error correction" << std::endl;

        helpers::CpuTimer step2Timer("STEP2");

        auto partialResults = cpu::correct_cpu(
            goodAlignmentProperties, 
            correctionOptions,
            runtimeOptions, 
            fileOptions, 
            memoryOptions, 
            *cpuMinhasher, 
            *cpuReadStorage
        );

        step2Timer.print();

        std::cout << "Correction throughput : ~" << (cpuReadStorage->getNumberOfReads() / step2Timer.elapsed()) << " reads/second.\n";

        std::cerr << "Constructed " << partialResults.size() << " corrections. ";
        std::cerr << "They occupy a total of " << (partialResults.dataBytes() + partialResults.offsetBytes()) << " bytes\n";

        //compareMaxRssToLimit(memoryOptions.memoryTotalLimit, "Error memorylimit after correction");

        cpuMinhasher = nullptr;        
        cpuReadStorage.reset();

        #endif

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

        helpers::CpuTimer step3Timer("STEP3");

        helpers::CpuTimer sorttimer("sort_results_by_read_id");

        sortSerializedResultsByReadIdAscending<EncodedTempCorrectedSequence>(
            partialResults,
            memoryForSorting
        );

        sorttimer.print();

        std::vector<FileFormat> formats;
        for(const auto& inputfile : fileOptions.inputfiles){
            formats.emplace_back(getFileFormat(inputfile));
        }
        std::vector<std::string> outputfiles;
        for(const auto& outputfilename : fileOptions.outputfilenames){
            outputfiles.emplace_back(fileOptions.outputdirectory + "/" + outputfilename);
        }
        constructOutputFileFromCorrectionResults(
            fileOptions.inputfiles, 
            partialResults, 
            formats[0],
            outputfiles,
            runtimeOptions.showProgress
        );

        step3Timer.print();

        //compareMaxRssToLimit(memoryOptions.memoryTotalLimit, "Error memorylimit after output construction");

        std::cout << "Construction of output file(s) finished." << std::endl;

    }

}
