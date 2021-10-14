#ifndef CARE_EXTENSION_AGENT_HPP
#define CARE_EXTENSION_AGENT_HPP

#include <config.hpp>
#include <options.hpp>
#include <threadpool.hpp>
#include <memorymanagement.hpp>
#include <serializedobjectstorage.hpp>
#include <extendedread.hpp>
#include <extensionresultoutput.hpp>
#include <sortserializedresults.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

namespace care{

    template<class MinhasherType, class ReadStorageType>
    struct ExtensionAgent{
        const GoodAlignmentProperties goodAlignmentProperties;
        const CorrectionOptions correctionOptions;
        const ExtensionOptions extensionOptions;
        const RuntimeOptions runtimeOptions;
        const FileOptions fileOptions;
        const MemoryOptions memoryOptions;
        const MinhasherType* minhasher;
        const ReadStorageType* readStorage;

        std::unique_ptr<SerializedObjectStorage> partialResults;
        BackgroundThread outputThread;
        std::vector<read_number> notExtendedIds;
        std::unique_ptr<SequenceFileWriter> writer;

        ExtensionAgent(
            const GoodAlignmentProperties& goodAlignmentProperties_,
            const CorrectionOptions& correctionOptions_,
            const ExtensionOptions& extensionOptions_,
            const RuntimeOptions& runtimeOptions_,
            const FileOptions& fileOptions_,
            const MemoryOptions& memoryOptions_,
            const MinhasherType& minhasher_,
            const ReadStorageType& readStorage_
        ) : 
            goodAlignmentProperties(goodAlignmentProperties_),
            correctionOptions(correctionOptions_),
            extensionOptions(extensionOptions_),
            runtimeOptions(runtimeOptions_),
            fileOptions(fileOptions_),
            memoryOptions(memoryOptions_),
            minhasher(&minhasher_),
            readStorage(&readStorage_),
            outputThread(false)
        {
            if(extensionOptions.sortedOutput){

                const auto rsMemInfo = readStorage->getMemoryInfo();
                const auto mhMemInfo = minhasher->getMemoryInfo();

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

                partialResults = std::make_unique<SerializedObjectStorage>(memoryLimitData, memoryLimitOffsets, fileOptions.tempdirectory + "/");
            }else{
                auto outputFormat = getFileFormat(fileOptions.inputfiles[0]);
                //no gz output
                if(outputFormat == FileFormat::FASTQGZ)
                    outputFormat = FileFormat::FASTQ;
                if(outputFormat == FileFormat::FASTAGZ)
                    outputFormat = FileFormat::FASTA;

                const std::string extendedOutputfile = fileOptions.outputdirectory + "/" + fileOptions.extendedReadsOutputfilename;

                writer = makeSequenceWriter(
                    extendedOutputfile,
                    outputFormat
                );
            }
        }

        template<class ExtensionEntryFunction, class Callback>
        void run(ExtensionEntryFunction doExtend, Callback callbackAfterExtenderFinished){
            outputThread.setMaximumQueueSize(runtimeOptions.threads);

            outputThread.start();

            if(extensionOptions.sortedOutput){
                doExtend(
                    goodAlignmentProperties, 
                    correctionOptions,
                    extensionOptions,
                    runtimeOptions, 
                    fileOptions, 
                    memoryOptions,
                    *minhasher, 
                    *readStorage,
                    [&](auto a, auto b, auto c){ submitReadyResultsForSorted(std::move(a), std::move(b), std::move(c)); }
                );

                outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

                std::cerr << "Constructed " << partialResults->size() << " extensions. ";
                std::cerr << "They occupy a total of " << (partialResults->dataBytes() + partialResults->offsetBytes()) << " bytes\n";

                callbackAfterExtenderFinished();

                const std::size_t availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
                const auto partialResultMemUsage = partialResults->getMemoryInfo();

                std::size_t memoryForSorting = std::min(
                    availableMemoryInBytes,
                    memoryOptions.memoryTotalLimit - partialResultMemUsage.host
                );

                if(memoryForSorting > 1*(std::size_t(1) << 30)){
                    memoryForSorting = memoryForSorting - 1*(std::size_t(1) << 30);
                }
                std::cerr << "memoryForSorting = " << memoryForSorting << "\n"; 

                std::cout << "STEP 3: Constructing output file(s)" << std::endl;
                helpers::CpuTimer step3timer("STEP 3");

                helpers::CpuTimer sorttimer("sort_results_by_read_id");

                sortSerializedResultsByReadIdAscending<EncodedExtendedRead>(
                    *partialResults,
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
                        *partialResults, 
                        notExtendedIds,
                        outputFormat, 
                        extendedOutputfile,
                        outputfiles,
                        fileOptions.pairType
                    );
                }else{
                    constructOutputFileFromExtensionResults(
                        *partialResults,
                        outputFormat, 
                        extendedOutputfile
                    );
                }

                step3timer.print();
            }else{

                auto outputFormat = getFileFormat(fileOptions.inputfiles[0]);
                //no gz output
                if(outputFormat == FileFormat::FASTQGZ)
                    outputFormat = FileFormat::FASTQ;
                if(outputFormat == FileFormat::FASTAGZ)
                    outputFormat = FileFormat::FASTA;

                const std::string extendedOutputfile = fileOptions.outputdirectory + "/" + fileOptions.extendedReadsOutputfilename;

                doExtend(
                    goodAlignmentProperties, 
                    correctionOptions,
                    extensionOptions,
                    runtimeOptions, 
                    fileOptions, 
                    memoryOptions,
                    *minhasher, 
                    *readStorage,
                    [&](auto a, auto b, auto c){ submitReadyResultsForUnsorted(std::move(a), std::move(b), std::move(c)); }
                );

                outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

                callbackAfterExtenderFinished();

                std::cout << "STEP 3: Constructing output file(s)" << std::endl;
                helpers::CpuTimer step3timer("STEP 3");

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

                step3timer.print();
            }
        }

        void submitReadyResultsForSorted(
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

                        partialResults->insert(tempbuffer.data(), end);
                    }
                }
            );
        }

        void submitReadyResultsForUnsorted(
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

                    for(const auto& er : vec){
                        writer->writeRead(makeOutputReadFromExtendedRead(er));
                    }
                }
            );
        }

    };


} // namespace care

#endif