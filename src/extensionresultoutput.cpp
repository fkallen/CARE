#include <extensionresultoutput.hpp>
#include <extendedread.hpp>

#include <hpc_helpers.cuh>
#include <readlibraryio.hpp>
#include <concurrencyhelpers.hpp>
#include <serializedobjectstorage.hpp>

#include <string>
#include <vector>
#include <cassert>
#include <sstream>
#include <future>
#include <optional>

namespace care{

    //convert extended read to struct which can be written to file
    Read makeOutputReadFromExtendedRead(const ExtendedRead& extendedRead){
        constexpr unsigned char foundMateStatus = static_cast<unsigned char>(ExtendedReadStatus::FoundMate);
        constexpr unsigned char repeatedStatus = static_cast<unsigned char>(ExtendedReadStatus::Repeated);
        unsigned char status = static_cast<unsigned char>(extendedRead.status);
        const bool foundMate = (status & foundMateStatus) == foundMateStatus;
        const bool repeated = (status & repeatedStatus) == repeatedStatus;

        std::stringstream sstream;
        sstream << extendedRead.readId;
        sstream << ' ' << (foundMate ? "reached:1" : "reached:0");
        sstream << ' ' << (extendedRead.mergedFromReadsWithoutMate ? "m:1" : "m:0");
        sstream << ' ' << (repeated ? "a:1" : "a:0");
        sstream << ' ';
        sstream << "lens:" << extendedRead.read1begin << ',' << extendedRead.read1end << ',' << extendedRead.read2begin << ',' << extendedRead.read2end;
        // if(extendedRead.status == ExtendedReadStatus::LengthAbort){
        //     sstream << "exceeded_length";
        // }else if(extendedRead.status == ExtendedReadStatus::CandidateAbort){
        //     sstream << "0_candidates";
        // }else if(extendedRead.status == ExtendedReadStatus::MSANoExtension){
        //     sstream << "msa_stop";
        // }

        Read res;
        res.header = sstream.str();
        res.sequence = extendedRead.getSequence();
        if(extendedRead.getQuality().size() != res.sequence.size()){
            res.quality.resize(res.sequence.length());
            std::fill(res.quality.begin(), res.quality.end(), 'F');
        }else{
            res.quality = extendedRead.getQuality();
        }

        return res;
    }



void writeExtensionResultsToFile(
    SerializedObjectStorage& partialResults, 
    FileFormat outputFormat,
    const std::string& outputfile
){

    std::unique_ptr<SequenceFileWriter> writer = makeSequenceWriter(
        //fileOptions.outputdirectory + "/extensionresult.txt", 
        outputfile,
        outputFormat
    );

    std::map<ExtendedReadStatus, std::int64_t> statusHistogram;

    std::int64_t pairsFound = 0;
    std::int64_t pairsRepeated = 0;

    const int expectedNumber = partialResults.size();
    int actualNumber = 0;

    for(std::size_t itemnumber = 0; itemnumber < partialResults.size(); itemnumber++){
        const std::uint8_t* serializedPtr = partialResults.getPointer(itemnumber);

        ExtendedRead extendedRead;

        #if 0
        extendedRead.copyFromContiguousMemory(serializedPtr);
        #else
        EncodedExtendedRead encext;
        encext.copyFromContiguousMemory(serializedPtr);
        extendedRead.decode(encext);
        #endif

        constexpr unsigned char foundMateStatus = static_cast<unsigned char>(ExtendedReadStatus::FoundMate);
        constexpr unsigned char repeatedStatus = static_cast<unsigned char>(ExtendedReadStatus::Repeated);
        unsigned char status = static_cast<unsigned char>(extendedRead.status);
        const bool foundMate = (status & foundMateStatus) == foundMateStatus;
        const bool repeated = (status & repeatedStatus) == repeatedStatus;

        Read res = makeOutputReadFromExtendedRead(extendedRead);

        writer->writeRead(res.header, res.sequence, res.quality);

        //statusHistogram[extendedRead.status]++;
        if(foundMate){
            pairsFound++;
        }

        if(repeated){
            pairsRepeated++;
        }

        actualNumber++;
    }

    if(actualNumber != expectedNumber){
        std::cerr << "Error actualNumber " << actualNumber << ", expectedNumber " << expectedNumber << "\n";
    }

    std::cerr << "Found mate: " << pairsFound << ", repeated: " << pairsRepeated << "\n";

    //assert(actualNumber == expectedNumber);

    // for(const auto& pair : statusHistogram){
    //     switch(pair.first){
    //         case ExtendedReadStatus::FoundMate: std::cout << "Found Mate: " << pair.second << "\n"; break;
    //         case ExtendedReadStatus::LengthAbort: 
    //         case ExtendedReadStatus::CandidateAbort: 
    //         case ExtendedReadStatus::MSANoExtension: 
    //         default: break;
    //     }
    // }
}


void outputUnchangedReadPairs(
    const std::vector<std::string>& originalReadFiles,
    const std::vector<read_number>& idsToOutput,
    FileFormat outputFormat,
    const std::string& outputfile
){

    assert(idsToOutput.size() % 2 == 0);

    helpers::CpuTimer mergetimer("Writing remaining reads to file");

    PairedInputReader pairedInputReader(originalReadFiles);
    std::unique_ptr<SequenceFileWriter> extendedReadWriter = makeSequenceWriter(outputfile, outputFormat);

    for(auto it = idsToOutput.begin(); it != idsToOutput.end(); it += 2){
        const read_number id1 = *it;
        const read_number id2 = *(it + 1);

        assert(id1 + 1 == id2);

        int readerstatus = pairedInputReader.next();

        while(readerstatus >= 0 && pairedInputReader.getCurrent1().globalReadId < id1){
            readerstatus = pairedInputReader.next();
        }

        if(readerstatus >= 0){
            assert(pairedInputReader.getCurrent1().globalReadId == id1);

            extendedReadWriter->writeRead(pairedInputReader.getCurrent1().read);
            extendedReadWriter->writeRead(pairedInputReader.getCurrent2().read);
        }else{
            std::cout << "Error occurred when outputting unchanged original reads. The output file may be incomplete!\n";
            break;
        }
    }

    mergetimer.print();
}



void constructOutputFileFromExtensionResults(
    SerializedObjectStorage& partialResults,
    FileFormat outputFormat,
    const std::string& extendedOutputfile
){

    writeExtensionResultsToFile(
        partialResults,
        outputFormat,
        extendedOutputfile
    );
}

void constructOutputFileFromExtensionResults(
    const std::vector<std::string>& originalReadFiles,
    SerializedObjectStorage& partialResults,
    const std::vector<read_number>& idsOfNotExtended,
    FileFormat outputFormat,
    const std::string& extendedOutputfile,
    const std::vector<std::string>& outputfiles,
    SequencePairType pairmode
){

    auto future1 = std::async(
        std::launch::async,
        writeExtensionResultsToFile,
        std::ref(partialResults),
        //FileFormat::FASTA, 
        outputFormat,
        extendedOutputfile
    );

    assert(std::is_sorted(idsOfNotExtended.begin(), idsOfNotExtended.end()));

    //const FileFormat remainingReadsFileFormat = getFileFormat(originalReadFiles[0]);

    if(pairmode == SequencePairType::PairedEnd){

        auto future2 = std::async(
            std::launch::async,
            outputUnchangedReadPairs,
            std::ref(originalReadFiles),
            std::ref(idsOfNotExtended),
            outputFormat,
            outputfiles[0]
        );
        future2.wait();

    }else{
        throw std::runtime_error("constructOutputFileFromExtensionResults called for single end pairmode");
    }

    future1.wait();
}




}