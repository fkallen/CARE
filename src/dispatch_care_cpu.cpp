#include <dispatch_care.hpp>

#include <config.hpp>
#include <options.hpp>

#include <contiguousreadstorage.hpp>

#include <correct_cpu.hpp>

#include <minhasherlimit.hpp>

#include <correctionresultprocessing.hpp>
#include <sequencehelpers.hpp>
#include <cpuminhasherconstruction.hpp>
#include <ordinaryminhasher.hpp>


#include <readstorageconstruction2.hpp>
#include <chunkedreadstorage.hpp>

#include <vector>
#include <iostream>
#include <mutex>
#include <thread>
#include <memory>

#include <experimental/filesystem>


#include <dynamic2darray.hpp>

namespace filesys = std::experimental::filesystem;



namespace care{

    std::vector<int> getUsableDeviceIds(std::vector<int> deviceIds){
        return std::vector<int>{};
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

    void printFileProperties(const std::string& filename, const SequenceFileProperties& props){
    	std::cout << "----------------------------------------" << std::endl;
    	std::cout << "File: " << filename << std::endl;
    	std::cout << "Reads: " << props.nReads << std::endl;
    	std::cout << "Minimum sequence length: " << props.minSequenceLength << std::endl;
    	std::cout << "Maximum sequence length: " << props.maxSequenceLength << std::endl;
    	std::cout << "----------------------------------------" << std::endl;
    }


    void performCorrection(
                            CorrectionOptions correctionOptions,
                            RuntimeOptions runtimeOptions,
                            MemoryOptions memoryOptions,
                            FileOptions fileOptions,
                            GoodAlignmentProperties goodAlignmentProperties){

        std::unique_ptr<CpuReadStorage> cpuReadStorage = constructReadStorageFromFiles2(
            runtimeOptions,
            memoryOptions,
            fileOptions.inputfiles,
            correctionOptions.useQualityScores
        );



        std::cout << "Running CARE CPU" << std::endl;

        std::uint64_t maximumNumberOfReads = fileOptions.nReads;
        int maximumSequenceLength = fileOptions.maximum_sequence_length;
        int minimumSequenceLength = fileOptions.minimum_sequence_length;
        bool scanned = false;

        // if(fileOptions.load_binary_reads_from == ""){

        //     if(maximumNumberOfReads == 0 || maximumSequenceLength == 0 || minimumSequenceLength == 0) {
        //         std::cout << "STEP 0: Determine input size" << std::endl;
                
        //         std::cout << "Scanning file(s) to get number of reads and min/max sequence length." << std::endl;

        //         maximumNumberOfReads = 0;
        //         maximumSequenceLength = 0;
        //         minimumSequenceLength = std::numeric_limits<int>::max();

        //         for(const auto& inputfile : fileOptions.inputfiles){
        //             auto prop = getSequenceFileProperties(inputfile, runtimeOptions.showProgress);
        //             maximumNumberOfReads += prop.nReads;
        //             maximumSequenceLength = std::max(maximumSequenceLength, prop.maxSequenceLength);
        //             minimumSequenceLength = std::min(minimumSequenceLength, prop.minSequenceLength);

        //             std::cout << "----------------------------------------\n";
        //             std::cout << "File: " << inputfile << "\n";
        //             std::cout << "Reads: " << prop.nReads << "\n";
        //             std::cout << "Minimum sequence length: " << prop.minSequenceLength << "\n";
        //             std::cout << "Maximum sequence length: " << prop.maxSequenceLength << "\n";
        //             std::cout << "----------------------------------------\n";

        //             //result.inputFileProperties.emplace_back(prop);
        //         }

        //         scanned = true;
        //     }else{
        //         //std::cout << "Using the supplied max number of reads and min/max sequence length." << std::endl;
        //     }
        // }

        std::cout << "STEP 1: Database construction" << std::endl;


        helpers::CpuTimer step1Timer("STEP1");


        // helpers::CpuTimer buildReadStorageTimer("build_readstorage");

        // care::cpu::ContiguousReadStorage readStorage(
        //     maximumNumberOfReads, 
        //     correctionOptions.useQualityScores, 
        //     minimumSequenceLength, 
        //     maximumSequenceLength
        // );

        // if(fileOptions.load_binary_reads_from != ""){
            
        //     readStorage.loadFromFile(fileOptions.load_binary_reads_from);

        //     if(correctionOptions.useQualityScores && !readStorage.canUseQualityScores())
        //         throw std::runtime_error("Quality scores are required but not present in preprocessed reads file!");
        //     if(!correctionOptions.useQualityScores && readStorage.canUseQualityScores())
        //         std::cerr << "Warning. The loaded preprocessed reads file contains quality scores, but program does not use them!\n";

        //     std::cout << "Loaded preprocessed reads from " << fileOptions.load_binary_reads_from << std::endl;

        //     //readStorage.constructionIsComplete();
        // }else{
        //     readStorage.construct(
        //         fileOptions.inputfiles,
        //         correctionOptions.useQualityScores,
        //         maximumNumberOfReads,
        //         minimumSequenceLength,
        //         maximumSequenceLength,
        //         runtimeOptions.threads,
        //         runtimeOptions.showProgress
        //     );
        // }

        // buildReadStorageTimer.print();

        // if(fileOptions.save_binary_reads_to != "") {
        //     std::cout << "Saving reads to file " << fileOptions.save_binary_reads_to << std::endl;
        //     helpers::CpuTimer timer("save_to_file");
        //     readStorage.saveToFile(fileOptions.save_binary_reads_to);
        //     timer.print();
    	// 	std::cout << "Saved reads" << std::endl;
        // }

        // const int batchsize = 1024;
        // int pitchInElements = 10;
        // int qualPitch = 128;
        // std::vector<read_number> readIds(batchsize);
        // std::vector<unsigned int> data1(batchsize * pitchInElements);
        // std::vector<unsigned int> data2(batchsize * pitchInElements);
        // std::vector<int> lengths1(batchsize);
        // std::vector<int> lengths2(batchsize);
        // std::vector<char> qualities1(batchsize * qualPitch);
        // std::vector<char> qualities2(batchsize * qualPitch);
        // std::unique_ptr<bool[]> ambig1 = std::make_unique<bool[]>(batchsize);
        // std::unique_ptr<bool[]> ambig2 = std::make_unique<bool[]>(batchsize);

        // cpu::ContiguousReadStorage::GatherHandle gatherHandle{};

        // helpers::CpuTimer footimer("footimer");
        // helpers::CpuTimer normaltimer("normaltimer");



        // for(int i = 0; i < 30085710; i += batchsize){
        //     const int currentbatchsize = std::min(batchsize, 30085710 - i);
        //     std::iota(readIds.begin(), readIds.end(), i);

        //     std::fill(data1.begin(), data1.end(), 0);
        //     std::fill(data2.begin(), data2.end(), 0);
        //     std::fill(lengths1.begin(), lengths1.end(), 0);
        //     std::fill(lengths2.begin(), lengths2.end(), 0);
        //     std::fill(data2.begin(), data2.end(), 0);
        //     std::fill(qualities1.begin(), qualities2.end(), 0);
        //     std::fill(ambig1.get(), ambig1.get() + batchsize, 0);
        //     std::fill(ambig2.get(), ambig2.get() + batchsize, 0);

        //     footimer.start();
        //     fooStorage.gatherSequenceData(readIds.data(), currentbatchsize, data1.data(), pitchInElements);
        //     fooStorage.gatherSequenceLengths(readIds.data(), currentbatchsize, lengths1.data());
        //     fooStorage.gatherQualities(readIds.data(), currentbatchsize, qualities1.data(), qualPitch);
        //     fooStorage.areSequencesAmbiguous(ambig1.get(), readIds.data(), currentbatchsize);
        //     footimer.stop();
            
        //     normaltimer.start();
        //     readStorage.gatherSequenceData(gatherHandle, readIds.data(), currentbatchsize, data2.data(), pitchInElements);
        //     readStorage.gatherSequenceLengths(gatherHandle, readIds.data(), currentbatchsize, lengths2.data());
        //     readStorage.gatherSequenceQualities(gatherHandle, readIds.data(), currentbatchsize, qualities2.data(), qualPitch);
        //     for(int k = 0; k < currentbatchsize; k++){
        //         ambig2[k] = readStorage.readContainsN(readIds[k]);
        //     }
        //     normaltimer.stop();

        //     assert(0 == std::memcmp(ambig1.get(), ambig2.get(), sizeof(bool) * batchsize));
        //     assert(lengths1 == lengths2);
        //     assert(qualities1 == qualities2);

        //     for(int k = 0; k < currentbatchsize; k++){
        //         if(!ambig1[k] && !ambig2[k]){
        //             assert(0 == std::memcmp(data1.data() + pitchInElements * k, data2.data() + pitchInElements * k, pitchInElements * sizeof(unsigned int)));
        //         }
        //     }
        // }

        // footimer.print();
        // normaltimer.print();

        
        
        SequenceFileProperties totalInputFileProperties;

        // totalInputFileProperties.nReads = readStorage.getNumberOfReads();
        // totalInputFileProperties.maxSequenceLength = readStorage.getStatistics().maximumSequenceLength;
        // totalInputFileProperties.minSequenceLength = readStorage.getStatistics().minimumSequenceLength;

        totalInputFileProperties.nReads = cpuReadStorage->getNumberOfReads();
        totalInputFileProperties.maxSequenceLength = cpuReadStorage->getSequenceLengthUpperBound();
        totalInputFileProperties.minSequenceLength = cpuReadStorage->getSequenceLengthLowerBound();

        if(!scanned){
            std::cout << "Determined the following read properties:\n";
            std::cout << "----------------------------------------\n";
            std::cout << "Total number of reads: " << totalInputFileProperties.nReads << "\n";
            std::cout << "Minimum sequence length: " << totalInputFileProperties.minSequenceLength << "\n";
            std::cout << "Maximum sequence length: " << totalInputFileProperties.maxSequenceLength << "\n";
            std::cout << "----------------------------------------\n";
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

        std::cout << "Reads with ambiguous bases: " << cpuReadStorage->getNumberOfReadsWithN() << std::endl;        

        printDataStructureMemoryUsage(*cpuReadStorage, "reads");


        helpers::CpuTimer buildMinhasherTimer("build_minhasher");

        auto minhasherAndType = constructCpuMinhasherFromCpuReadStorage(
            fileOptions,
            runtimeOptions,
            memoryOptions,
            correctionOptions,
            totalInputFileProperties,
            *cpuReadStorage,
            CpuMinhasherType::Ordinary
        );

        CpuMinhasher* const cpuMinhasher = minhasherAndType.first.get();

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
            totalInputFileProperties,
            *cpuMinhasher, 
            //readStorage
            *cpuReadStorage
        );

        step2Timer.print();

        std::cout << "Correction throughput : ~" << (totalInputFileProperties.nReads / step2Timer.elapsed()) << " reads/second.\n";
        const std::size_t numTemp = partialResults.getNumElementsInMemory() + partialResults.getNumElementsInFile();
        const std::size_t numTempInMem = partialResults.getNumElementsInMemory();
        const std::size_t numTempInFile = partialResults.getNumElementsInFile();

        std::cerr << "Constructed " << numTemp << " corrections. "
            << numTempInMem << " corrections are stored in memory. "
            << numTempInFile << " corrections are stored in temporary file\n";

        cpuMinhasher->destroy();
        //readStorage.destroy();
        cpuReadStorage->destroy();

        const std::size_t availableMemoryInBytes2 = getAvailableMemoryInKB() * 1024;
        std::size_t memoryForSorting = 0;

        if(availableMemoryInBytes2 > 1*(std::size_t(1) << 30)){
            memoryForSorting = availableMemoryInBytes2 - 1*(std::size_t(1) << 30);
        }

        std::cout << "STEP 3: Constructing output file(s)" << std::endl;

        helpers::CpuTimer step3Timer("STEP3");

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

        step3Timer.print();

        std::cout << "Construction of output file(s) finished." << std::endl;

    }

}
