#include <dispatch_care.hpp>

#include <config.hpp>
#include <options.hpp>

#include <readstorage.hpp>

#include <correct_cpu.hpp>

#include <build.hpp>

#include <minhasher.hpp>
#include <minhasher_transform.hpp>
//#include <candidatedistribution.hpp>
#include <correctionresultprocessing.hpp>
#include <sequence.hpp>

#include <vector>
#include <iostream>
#include <mutex>
#include <thread>
#include <memory>

#include <experimental/filesystem>

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

        std::cout << "Running CARE CPU" << std::endl;

        std::uint64_t maximumNumberOfReads = fileOptions.nReads;
        int maximumSequenceLength = fileOptions.maximum_sequence_length;
        int minimumSequenceLength = fileOptions.minimum_sequence_length;
        bool scanned = false;

        if(fileOptions.load_binary_reads_from == ""){

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
        }

        std::cout << "STEP 1: Database construction" << std::endl;

        TIMERSTARTCPU(STEP1);

        TIMERSTARTCPU(build_readstorage);

        care::cpu::ContiguousReadStorage readStorageXXX(
            maximumNumberOfReads, 
            correctionOptions.useQualityScores, 
            minimumSequenceLength, 
            maximumSequenceLength
        );

        if(fileOptions.load_binary_reads_from != ""){

            TIMERSTARTCPU(load_from_file);
            readStorageXXX.loadFromFile(fileOptions.load_binary_reads_from);
            TIMERSTOPCPU(load_from_file);

            if(correctionOptions.useQualityScores && !readStorageXXX.canUseQualityScores())
                throw std::runtime_error("Quality scores are required but not present in preprocessed reads file!");
            if(!correctionOptions.useQualityScores && readStorageXXX.canUseQualityScores())
                std::cerr << "Warning. The loaded preprocessed reads file contains quality scores, but program does not use them!\n";

            std::cout << "Loaded preprocessed reads from " << fileOptions.load_binary_reads_from << std::endl;

            //readStorage.constructionIsComplete();
        }else{
            readStorageXXX.construct(
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
            TIMERSTARTCPU(save_to_file);
            readStorageXXX.saveToFile(fileOptions.save_binary_reads_to);
            TIMERSTOPCPU(save_to_file);
    		std::cout << "Saved reads" << std::endl;
        }

        TIMERSTOPCPU(build_readstorage);
        
        SequenceFileProperties totalInputFilePropertiesXXX;

        totalInputFilePropertiesXXX.nReads = readStorageXXX.getNumberOfReads();
        totalInputFilePropertiesXXX.maxSequenceLength = readStorageXXX.getStatistics().maximumSequenceLength;
        totalInputFilePropertiesXXX.minSequenceLength = readStorageXXX.getStatistics().minimumSequenceLength;

        if(!scanned){
            std::cout << "Determined the following read properties:\n";
            std::cout << "----------------------------------------\n";
            std::cout << "Total number of reads: " << totalInputFilePropertiesXXX.nReads << "\n";
            std::cout << "Minimum sequence length: " << totalInputFilePropertiesXXX.minSequenceLength << "\n";
            std::cout << "Maximum sequence length: " << totalInputFilePropertiesXXX.maxSequenceLength << "\n";
            std::cout << "----------------------------------------\n";
        }

        if(correctionOptions.autodetectKmerlength){
            const int maxlength = totalInputFilePropertiesXXX.maxSequenceLength;

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

        std::cout << "Reads with ambiguous bases: " << readStorageXXX.getNumberOfReadsWithN() << std::endl;
        

        printDataStructureMemoryUsage(readStorageXXX, "reads");





        // BuiltDataStructures dataStructures = buildAndSaveDataStructures2(
        //     correctionOptions,
        //     runtimeOptions,
        //     memoryOptions,
        //     fileOptions
        // );

        // BuiltDataStructure<Minhasher> aaa = build_minhasher(fileOptions,
        //                         			   runtimeOptions,
        //                                        memoryOptions,
        //                         			   totalInputFilePropertiesXXX.nReads,
        //                                        correctionOptions,
        //                         			   readStorageXXX);

        TIMERSTARTCPU(build_newgpuminhasher);
        Minhasher minhasher(
            correctionOptions.kmerlength, 
            calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage)
        );

        if(fileOptions.load_hashtables_from != ""){

            std::ifstream is(fileOptions.load_hashtables_from);
            assert((bool)is);

            minhasher.loadFromStream(is);

            std::cout << "Loaded hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }else{
            minhasher.construct(
                fileOptions,
                runtimeOptions,
                memoryOptions,
                totalInputFilePropertiesXXX.nReads, 
                correctionOptions,
                readStorageXXX
            );

            if(correctionOptions.mustUseAllHashfunctions 
                && correctionOptions.numHashFunctions != minhasher.getNumberOfMaps()){
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

            minhasher.writeToStream(os);

    		std::cout << "Saved minhasher" << std::endl;
        }

        printDataStructureMemoryUsage(minhasher, "hash tables");

        TIMERSTOPCPU(STEP1);

        // if(correctionOptions.autodetectKmerlength){
        //     correctionOptions.kmerlength = dataStructures.kmerlength;
        // }

        //auto& readStorage = dataStructures.builtReadStorage.data;
        //auto& minhasher = dataStructures.builtMinhasher.data;
        //auto& totalInputFileProperties = dataStructures.totalInputFileProperties;
        //auto& minhasher = aaa.data;
        //auto& totalInputFileProperties = totalInputFilePropertiesXXX;


        //printDataStructureMemoryUsage(minhasher, readStorageXXX);

        std::cout << "STEP 2: Error correction" << std::endl;

        TIMERSTARTCPU(STEP2);

        auto partialResults = cpu::correct_cpu(
            goodAlignmentProperties, 
            correctionOptions,
            runtimeOptions, 
            fileOptions, 
            memoryOptions, 
            totalInputFilePropertiesXXX,
            minhasher, 
            //readStorage
            readStorageXXX
        );

        TIMERSTOPCPU(STEP2);

        minhasher.destroy();
        //readStorage.destroy();
        readStorageXXX.destroy();

        const std::size_t availableMemoryInBytes2 = getAvailableMemoryInKB() * 1024;
        std::size_t memoryForSorting = 0;

        if(availableMemoryInBytes2 > 1*(std::size_t(1) << 30)){
            memoryForSorting = availableMemoryInBytes2 - 1*(std::size_t(1) << 30);
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
