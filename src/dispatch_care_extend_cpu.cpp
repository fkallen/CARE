#include <dispatch_care_extend_cpu.hpp>
#include <hpc_helpers.cuh>
#include <config.hpp>
#include <options.hpp>

#include <minhasherlimit.hpp>

#include <sequencehelpers.hpp>
#include <cpuminhasherconstruction.hpp>
#include <ordinaryminhasher.hpp>
#include <sortserializedresults.hpp>
#include <serializedobjectstorage.hpp>
#include <chunkedreadstorageconstruction.hpp>
#include <chunkedreadstorage.hpp>

#include <readextension_cpu.hpp>
#include <extendedread.hpp>
#include <extensionresultoutput.hpp>
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

    void performExtension(
        CorrectionOptions correctionOptions,
        ExtensionOptions extensionOptions,
        RuntimeOptions runtimeOptions,
        MemoryOptions memoryOptions,
        FileOptions fileOptions,
        GoodAlignmentProperties goodAlignmentProperties
    ){

        std::cout << "Running CARE EXTEND CPU" << std::endl;

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

        std::cout << "STEP 2: Read extension" << std::endl;

        helpers::CpuTimer step2timer("STEP2");

        if(extensionOptions.sortedOutput){

            const auto rsMemInfo = cpuReadStorage->getMemoryInfo();
            const auto mhMemInfo = cpuMinhasher->getMemoryInfo();

            std::size_t memoryAvailableBytesHost = memoryOptions.memoryTotalLimit;
            if(memoryAvailableBytesHost > rsMemInfo.host){
                memoryAvailableBytesHost -= rsMemInfo.host;
            }else{
                memoryAvailableBytesHost = 0;
            }
            if(memoryAvailableBytesHost > mhMemInfo.host){
                memoryAvailableBytesHost -= mhMemInfo.host;
            }else{
                memoryAvailableBytesHost = 0;
            }

            std::size_t availableMemoryInBytes = memoryAvailableBytesHost; //getAvailableMemoryInKB() * 1024;
            std::size_t memoryForPartialResultsInBytes = 0;

            if(availableMemoryInBytes > 3*(std::size_t(1) << 30)){
                memoryForPartialResultsInBytes = availableMemoryInBytes - 3*(std::size_t(1) << 30);
            }

            std::cerr << "Partial results may occupy " << (memoryForPartialResultsInBytes /1024. / 1024. / 1024.) 
                << " GB in memory. Remaining partial results will be stored in temp directory. \n";

            const std::size_t memoryLimitData = memoryForPartialResultsInBytes * 0.75;
            const std::size_t memoryLimitOffsets = memoryForPartialResultsInBytes * 0.25;

            SerializedObjectStorage partialResults(memoryLimitData, memoryLimitOffsets, fileOptions.tempdirectory + "/");
            std::vector<read_number> notExtendedIds;

            BackgroundThread outputThread(true);

            auto submitReadyResults = [&](
                std::vector<ExtendedRead> extendedReads, 
                std::vector<EncodedExtendedRead> encodedExtendedReads,
                std::vector<read_number> idsOfNotExtendedReads
            ){
                outputThread.enqueue(
                    [&, 
                        vec = std::move(extendedReads), 
                        encvec = std::move(encodedExtendedReads),
                        idsOfNotExtendedReads = std::move(idsOfNotExtendedReads)
                    ](){
                        notExtendedIds.insert(notExtendedIds.end(), idsOfNotExtendedReads.begin(), idsOfNotExtendedReads.end());

                        std::vector<std::uint8_t> tempbuffer(256);

                        for(const auto& er : encvec){
                            const std::size_t serializedSize = er.getSerializedNumBytes();
                            tempbuffer.resize(serializedSize);

                            auto end = er.copyToContiguousMemory(tempbuffer.data(), tempbuffer.data() + tempbuffer.size());
                            assert(end != nullptr);

                            partialResults.insert(tempbuffer.data(), end);
                        }
                    }
                );
            };

            extend_cpu(
                goodAlignmentProperties, 
                correctionOptions,
                extensionOptions,
                runtimeOptions, 
                fileOptions, 
                memoryOptions,
                *cpuMinhasher, 
                *cpuReadStorage,
                submitReadyResults
            );

            outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

            step2timer.print();

            std::cout << "Extension throughput : ~" << (cpuReadStorage->getNumberOfReads() / step2timer.elapsed()) << " reads/second.\n"; //TODO: paired end? numreads / 2 ?
        
            std::cerr << "Constructed " << partialResults.size() << " extensions. ";
            std::cerr << "They occupy a total of " << (partialResults.dataBytes() + partialResults.offsetBytes()) << " bytes\n";

            minhasherAndType.first.reset();
            cpuMinhasher = nullptr;        
            cpuReadStorage.reset();

            availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
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
            std::cerr << "memoryForSorting = " << memoryForSorting << "\n"; 

            std::cout << "STEP 3: Constructing output file(s)" << std::endl;

            helpers::CpuTimer step3Timer("STEP3");

            helpers::CpuTimer sorttimer("sort_results_by_read_id");

            sortSerializedResultsByReadIdAscending<EncodedExtendedRead>(
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

            auto outputFormat = getFileFormat(fileOptions.inputfiles[0]);
            //no gz output
            if(outputFormat == FileFormat::FASTQGZ)
                outputFormat = FileFormat::FASTQ;
            if(outputFormat == FileFormat::FASTAGZ)
                outputFormat = FileFormat::FASTA;

            const std::string extendedOutputfile = fileOptions.outputdirectory + "/" + fileOptions.extendedReadsOutputfilename;

            if(extensionOptions.outputRemainingReads){
                std::sort(notExtendedIds.begin(), notExtendedIds.end());

                constructOutputFileFromExtensionResults(
                    fileOptions.inputfiles,
                    partialResults, 
                    notExtendedIds,
                    outputFormat, 
                    extendedOutputfile,
                    outputfiles,
                    fileOptions.pairType
                );
            }else{
                constructOutputFileFromExtensionResults(
                    partialResults,
                    outputFormat, 
                    extendedOutputfile
                );
            }

            step3Timer.print();

            std::cout << "Construction of output file(s) finished." << std::endl;

        }else{
            BackgroundThread outputThread(true);

            auto outputFormat = getFileFormat(fileOptions.inputfiles[0]);
            //no gz output
            if(outputFormat == FileFormat::FASTQGZ)
                outputFormat = FileFormat::FASTQ;
            if(outputFormat == FileFormat::FASTAGZ)
                outputFormat = FileFormat::FASTA;

            const std::string extendedOutputfile = fileOptions.outputdirectory + "/" + fileOptions.extendedReadsOutputfilename;

            std::unique_ptr<SequenceFileWriter> writer = makeSequenceWriter(
                extendedOutputfile,
                outputFormat
            );

            std::vector<read_number> notExtendedIds;

            auto submitReadyResults = [&](
                std::vector<ExtendedRead> extendedReads, 
                std::vector<EncodedExtendedRead> encodedExtendedReads,
                std::vector<read_number> idsOfNotExtendedReads
            ){
                outputThread.enqueue(
                    [&, 
                        vec = std::move(extendedReads), 
                        encvec = std::move(encodedExtendedReads),
                        idsOfNotExtendedReads = std::move(idsOfNotExtendedReads)
                    ](){
                        notExtendedIds.insert(notExtendedIds.end(), idsOfNotExtendedReads.begin(), idsOfNotExtendedReads.end());

                        std::vector<std::uint8_t> tempbuffer(256);

                        for(const auto& er : vec){
                            writer->writeRead(makeOutputReadFromExtendedRead(er));
                        }
                    }
                );
            };

            extend_cpu(
                goodAlignmentProperties, 
                correctionOptions,
                extensionOptions,
                runtimeOptions, 
                fileOptions, 
                memoryOptions,
                *cpuMinhasher, 
                *cpuReadStorage,
                submitReadyResults
            );

            outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

            step2timer.print();

            std::cout << "Extension throughput : ~" << (cpuReadStorage->getNumberOfReads() / step2timer.elapsed()) << " reads/second.\n"; //TODO: paired end? numreads / 2 ?

            if(extensionOptions.outputRemainingReads){

                std::sort(notExtendedIds.begin(), notExtendedIds.end());

                std::vector<std::string> outputfiles;
                for(const auto& outputfilename : fileOptions.outputfilenames){
                    outputfiles.emplace_back(fileOptions.outputdirectory + "/" + outputfilename);
                }
            
                outputUnchangedReadPairs(
                    fileOptions.inputfiles,
                    notExtendedIds,
                    outputFormat,
                    outputfiles[0]
                );

            }

            std::cout << "Construction of output file(s) finished." << std::endl;
        }

    }
}
