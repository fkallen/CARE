#include <dispatch_care.hpp>
#include <hpc_helpers.cuh>
#include <config.hpp>
#include <options.hpp>

#include <readstorage.hpp>

#include <correct_cpu.hpp>

#include <minhasher.hpp>

#include <correctionresultprocessing.hpp>
#include <sequence.hpp>

#include <readextension_cpu.hpp>

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

        helpers::CpuTimer step1Timer("STEP1");

        helpers::CpuTimer buildReadStorageTimer("build_readstorage");


        care::cpu::ContiguousReadStorage readStorage(
            maximumNumberOfReads, 
            correctionOptions.useQualityScores, 
            minimumSequenceLength, 
            maximumSequenceLength
        );

        if(fileOptions.load_binary_reads_from != ""){

            readStorage.loadFromFile(fileOptions.load_binary_reads_from);

            if(correctionOptions.useQualityScores && !readStorage.canUseQualityScores())
                throw std::runtime_error("Quality scores are required but not present in preprocessed reads file!");
            if(!correctionOptions.useQualityScores && readStorage.canUseQualityScores())
                std::cerr << "Warning. The loaded preprocessed reads file contains quality scores, but program does not use them!\n";

            std::cout << "Loaded preprocessed reads from " << fileOptions.load_binary_reads_from << std::endl;

            //readStorage.constructionIsComplete();
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

        buildReadStorageTimer.print();

        if(fileOptions.save_binary_reads_to != "") {
            std::cout << "Saving reads to file " << fileOptions.save_binary_reads_to << std::endl;
            helpers::CpuTimer timer("save_to_file");
            readStorage.saveToFile(fileOptions.save_binary_reads_to);
            timer.print();
    		std::cout << "Saved reads" << std::endl;
        }

        
        
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
        Minhasher minhasher(
            correctionOptions.kmerlength, 
            calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage)
        );

        if(fileOptions.load_hashtables_from != ""){

            std::ifstream is(fileOptions.load_hashtables_from);
            assert((bool)is);

            const int loadedMaps = minhasher.loadFromStream(is, correctionOptions.numHashFunctions);

            std::cout << "Loaded " << loadedMaps << " hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }else{
            minhasher.construct(
                fileOptions,
                runtimeOptions,
                memoryOptions,
                totalInputFileProperties.nReads, 
                correctionOptions,
                readStorage
            );

            if(correctionOptions.mustUseAllHashfunctions 
                && correctionOptions.numHashFunctions != minhasher.getNumberOfMaps()){
                std::cout << "Cannot use specified number of hash functions (" 
                    << correctionOptions.numHashFunctions <<")\n";
                std::cout << "Abort!\n";
                return;
            }
        }

        buildMinhasherTimer.print();

        if(fileOptions.save_hashtables_to != "") {
            std::cout << "Saving minhasher to file " << fileOptions.save_hashtables_to << std::endl;
            std::ofstream os(fileOptions.save_hashtables_to);
            assert((bool)os);
            helpers::CpuTimer timer("save_to_file");
            minhasher.writeToStream(os);
            timer.print();
    		std::cout << "Saved minhasher" << std::endl;
        }



        printDataStructureMemoryUsage(minhasher, "hash tables");

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
            minhasher, 
            //readStorage
            readStorage
        );

        step2Timer.print();

        minhasher.destroy();
        readStorage.destroy();

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
            false
        );

        step3Timer.print();

        std::cout << "Construction of output file(s) finished." << std::endl;

    }


    void performExtension(
                            CorrectionOptions correctionOptions,
                            ExtensionOptions extensionOptions,
                            RuntimeOptions runtimeOptions,
                            MemoryOptions memoryOptions,
                            FileOptions fileOptions,
                            GoodAlignmentProperties goodAlignmentProperties){

        std::cout << "Running CARE EXTEND CPU" << std::endl;


        // {
        //     MemoryFileFixedSize<ExtendedRead> memfile{0, fileOptions.tempdirectory+"/tmpfile"};

        //     ExtendedRead er1;
        //     er1.readId1 = 0;
        //     er1.readId2 = 1;

        //     ExtendedRead er2;
        //     er2.readId1 = 2;
        //     er2.readId2 = 3;

        //     ExtendedRead er3;
        //     er3.readId1 = 4;
        //     er3.readId2 = 5;

        //     ExtendedRead er4;
        //     er4.readId1 = 6;
        //     er4.readId2 = 7;

        //     ExtendedRead er5;
        //     er5.readId1 = 8;
        //     er5.readId2 = 9;

        //     memfile.storeElement(std::move(er1));
        //     memfile.storeElement(std::move(er2));
        //     memfile.storeElement(std::move(er3));
        //     memfile.storeElement(std::move(er4));
        //     memfile.storeElement(std::move(er5));

        //     memfile.flush();
            
        //     if(true){
        //         auto ptrcomparator = [](const std::uint8_t* ptr1, const std::uint8_t* ptr2){
        //             read_number lid1, lid2;
        //             read_number rid1, rid2;
        //             std::memcpy(&lid1, ptr1, sizeof(read_number));
        //             std::memcpy(&lid2, ptr1, sizeof(read_number));
        //             std::memcpy(&rid1, ptr2, sizeof(read_number));
        //             std::memcpy(&rid2, ptr2, sizeof(read_number));

        //             std::cerr << "ptrcomparator: (" << lid1 << "," << lid2 << ") - (" << rid1 << "," << rid2 << ")\n";
                    
        //             if(lid1 < rid1) return true;
        //             if(lid1 > rid1) return false;
        //             if(lid2 < rid2) return true;
        //             return false;
        //         };

        //         auto elementcomparator = [](const auto& l, const auto& r){
        //             std::cerr << "elementcomp: (" << l.readId1 << "," << l.readId2 << ") - (" << r.readId1 << "," << r.readId2 << ")\n";
        //             if(l.readId1 < r.readId1) return true;
        //             if(l.readId1 > r.readId1) return false;
        //             if(l.readId2 < r.readId2) return true;
        //             return false;
        //         };

        //         TIMERSTARTCPU(sort_results_by_read_id);
        //         memfile.sort(fileOptions.tempdirectory, 10000000, ptrcomparator, elementcomparator);
        //         TIMERSTOPCPU(sort_results_by_read_id);
        //     }

            

        //     auto partialResultsReader = memfile.makeReader();

        //     std::cerr << "in mem: " << memfile.getNumElementsInMemory() << ", in file: " << memfile.getNumElementsInFile() << "\n";

        //     while(partialResultsReader.hasNext()){
        //         ExtendedRead extendedRead = *(partialResultsReader.next());
        //         std::cerr << extendedRead.readId1 << " " << extendedRead.readId2 << "\n";
        //     }
        // }

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

        helpers::CpuTimer step1Timer("STEP1");

        helpers::CpuTimer buildReadStorageTimer("build_readstorage");


        care::cpu::ContiguousReadStorage readStorage(
            maximumNumberOfReads, 
            correctionOptions.useQualityScores, 
            minimumSequenceLength, 
            maximumSequenceLength
        );

        if(fileOptions.load_binary_reads_from != ""){

            readStorage.loadFromFile(fileOptions.load_binary_reads_from);

            if(correctionOptions.useQualityScores && !readStorage.canUseQualityScores())
                throw std::runtime_error("Quality scores are required but not present in preprocessed reads file!");
            if(!correctionOptions.useQualityScores && readStorage.canUseQualityScores())
                std::cerr << "Warning. The loaded preprocessed reads file contains quality scores, but program does not use them!\n";

            std::cout << "Loaded preprocessed reads from " << fileOptions.load_binary_reads_from << std::endl;

            //readStorage.constructionIsComplete();
        }else{
            readStorage.constructPaired(
                fileOptions.inputfiles,
                correctionOptions.useQualityScores,
                maximumNumberOfReads,
                minimumSequenceLength,
                maximumSequenceLength,
                runtimeOptions.threads,
                runtimeOptions.showProgress
            );
        }

        buildReadStorageTimer.print();

        if(fileOptions.save_binary_reads_to != "") {
            std::cout << "Saving reads to file " << fileOptions.save_binary_reads_to << std::endl;
            helpers::CpuTimer timer("save_to_file");
            readStorage.saveToFile(fileOptions.save_binary_reads_to);
            timer.print();
    		std::cout << "Saved reads" << std::endl;
        }

        
        
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
        Minhasher minhasher(
            correctionOptions.kmerlength, 
            calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage)
        );

        if(fileOptions.load_hashtables_from != ""){

            std::ifstream is(fileOptions.load_hashtables_from);
            assert((bool)is);

            const int loadedMaps = minhasher.loadFromStream(is, correctionOptions.numHashFunctions);

            std::cout << "Loaded " << loadedMaps << " hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }else{
            minhasher.construct(
                fileOptions,
                runtimeOptions,
                memoryOptions,
                totalInputFileProperties.nReads, 
                correctionOptions,
                readStorage
            );

            if(correctionOptions.mustUseAllHashfunctions 
                && correctionOptions.numHashFunctions != minhasher.getNumberOfMaps()){
                std::cout << "Cannot use specified number of hash functions (" 
                    << correctionOptions.numHashFunctions <<")\n";
                std::cout << "Abort!\n";
                return;
            }
        }

        buildMinhasherTimer.print();

        if(fileOptions.save_hashtables_to != "") {
            std::cout << "Saving minhasher to file " << fileOptions.save_hashtables_to << std::endl;
            std::ofstream os(fileOptions.save_hashtables_to);
            assert((bool)os);
            helpers::CpuTimer timer("save_to_file");
            minhasher.writeToStream(os);
            timer.print();
    		std::cout << "Saved minhasher" << std::endl;
        }



        printDataStructureMemoryUsage(minhasher, "hash tables");

        step1Timer.print();

        std::cout << "STEP 2: Read extension" << std::endl;

        helpers::CpuTimer step2Timer("STEP2");

        auto partialResults = extend_cpu(
            goodAlignmentProperties, 
            correctionOptions,
            extensionOptions,
            runtimeOptions, 
            fileOptions, 
            memoryOptions, 
            totalInputFileProperties,
            minhasher, 
            readStorage
        );

        step2Timer.print();

        minhasher.destroy();
        readStorage.destroy();

        const std::size_t availableMemoryInBytes2 = getAvailableMemoryInKB() * 1024;
        std::size_t memoryForSorting = 0;

        if(availableMemoryInBytes2 > 1*(std::size_t(1) << 30)){
            memoryForSorting = availableMemoryInBytes2 - 1*(std::size_t(1) << 30);
        }

        (void)memoryForSorting;

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

        auto outputFormat = getFileFormat(fileOptions.inputfiles[0]);
        //no gz output
        if(outputFormat == FileFormat::FASTQGZ)
            outputFormat = FileFormat::FASTQ;
        if(outputFormat == FileFormat::FASTAGZ)
            outputFormat = FileFormat::FASTA;




//#define OUTPUTDEBUG
#ifdef OUTPUTDEBUG

        std::unique_ptr<SequenceFileWriter> writer = makeSequenceWriter(
            //fileOptions.outputdirectory + "/extensionresult.txt", 
            outputfiles[0],
            outputFormat
        );

#if 0
        std::sort(partialResults.begin(), partialResults.end(), 
            [](const auto& l, const auto& r){
                return l.readId1 < r.readId1;
            }
        );

        std::ofstream os(outputfiles[0]);

        for(const auto& extendedRead : partialResults){
            os << extendedRead.readId1 << ' ' << extendedRead.readId2 << ' ' 
                << extendedRead.reachedMate1 << ' ' << extendedRead.reachedMate2 << '\n';
            os << extendedRead.originalRead1 << '\n';
            os << extendedRead.originalRead2 << '\n';
            if(extendedRead.extendedRead1.length() > 0){
                os << extendedRead.extendedRead1 << '\n';
            }else{
                os << "----------\n";
            }
            if(extendedRead.extendedRead2.length() > 0){
                os << extendedRead.extendedRead2 << '\n';
            }else{
                os << "----------\n";
            }
        }
#else
        if(true){
            auto ptrcomparator = [](const std::uint8_t* ptr1, const std::uint8_t* ptr2){
                read_number lid1, lid2;
                read_number rid1, rid2;
                std::memcpy(&lid1, ptr1, sizeof(read_number));
                std::memcpy(&lid2, ptr1 + sizeof(read_number), sizeof(read_number));
                std::memcpy(&rid1, ptr2, sizeof(read_number));
                std::memcpy(&rid2, ptr2 + sizeof(read_number), sizeof(read_number));
                
                if(lid1 < rid1) return true;
                if(lid1 > rid1) return false;
                if(lid2 < rid2) return true;
                return false;
            };

            auto elementcomparator = [](const auto& l, const auto& r){
                if(l.readId1 < r.readId1) return true;
                if(l.readId1 > r.readId1) return false;
                if(l.readId2 < r.readId2) return true;
                return false;
            };

            TIMERSTARTCPU(sort_results_by_read_id);
            partialResults.sort(fileOptions.tempdirectory, memoryForSorting, ptrcomparator, elementcomparator);
            TIMERSTOPCPU(sort_results_by_read_id);
        }

        std::ofstream os(outputfiles[0]);

        std::int64_t count = 0;
        auto partialResultsReader = partialResults.makeReader();

        std::cerr << "in mem: " << partialResults.getNumElementsInMemory() << ", in file: " << partialResults.getNumElementsInFile() << "\n";

        while(partialResultsReader.hasNext()){
            // TempCorrectedSequence tcs = *(partialResultsReader.next());

            // Read read;
            // read.name = "" + std::to_string(count);
            // read.comment = "original read id " + std::to_string(tcs.readId);
            // read.sequence = std::move(tcs.sequence);
            // read.quality.resize(read.sequence.size());
            // std::fill(read.quality.begin(), read.quality.end(), 'F');

            // writer->writeRead(read.name, read.comment, read.sequence, read.quality);

            ExtendedRead extendedRead = *(partialResultsReader.next());

            os << extendedRead.readId1 << ' ' << extendedRead.readId2 << ' ' 
                << extendedRead.reachedMate1 << ' ' << extendedRead.reachedMate2 << '\n';
            os << extendedRead.originalRead1 << '\n';
            os << extendedRead.originalRead2 << '\n';
            if(extendedRead.extendedRead1.length() > 0){
                os << extendedRead.extendedRead1 << '\n';
            }else{
                os << "----------\n";
            }
            if(extendedRead.extendedRead2.length() > 0){
                os << extendedRead.extendedRead2 << '\n';
            }else{
                os << "----------\n";
            }

            count++;
        }
#endif
        // constructOutputFileFromResults2(
        //     fileOptions.tempdirectory,
        //     fileOptions.inputfiles,            
        //     partialResults, 
        //     memoryForSorting,
        //     formats[0], 
        //     outputfiles, 
        //     false
        // );
#else 

        constructOutputFileFromExtensionResults(
            fileOptions.tempdirectory,
            fileOptions.inputfiles,            
            partialResults, 
            memoryForSorting,
            outputFormat, 
            outputfiles, 
            false
        );


#endif 



        step3Timer.print();

        std::cout << "Construction of output file(s) finished." << std::endl;

    }

}
