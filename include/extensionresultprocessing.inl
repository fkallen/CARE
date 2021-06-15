#include <extensionresultprocessing.hpp>
#include <programoutputprocessing.hpp>

#include <hpc_helpers.cuh>
#include <memoryfile.hpp>
#include <readlibraryio.hpp>

#include <string>
#include <vector>
#include <cassert>
#include <sstream>

namespace care{


void writeExtensionResultsToFile(
    const std::string& tempdir,
    MemoryFileFixedSize<ExtendedRead>& partialResults, 
    std::size_t memoryForSorting,
    FileFormat outputFormat,
    const std::string& outputfile,
    bool isSorted
){

    if(!isSorted){

        auto elementcomparator = [](const auto& l, const auto& r){
            return l.getReadId() < r.getReadId();
        };

        auto extractKey = [](const std::uint8_t* ptr){
            using ValueType = typename MemoryFileFixedSize<ExtendedRead>::ValueType;

            const read_number id = ValueType::parseReadId(ptr);
            
            return id;
        };

        auto keyComparator = std::less<read_number>{};

        helpers::CpuTimer timer("sort_results_by_read_id");

        bool fastSuccess = false; //partialResults.template trySortByKeyFast<read_number>(extractKey, keyComparator, memoryForSorting);

        if(!fastSuccess){            
            partialResults.sort(tempdir, memoryForSorting, extractKey, keyComparator, elementcomparator);
        }else{
            std::cerr << "fast sort worked!\n";
        }

        timer.print();
    }

    auto partialResultsReader = partialResults.makeReader();

    std::unique_ptr<SequenceFileWriter> writer = makeSequenceWriter(
        //fileOptions.outputdirectory + "/extensionresult.txt", 
        outputfile,
        outputFormat
    );

    std::cerr << "in mem: " << partialResults.getNumElementsInMemory() << ", in file: " << partialResults.getNumElementsInFile() << "\n";

    std::map<ExtendedReadStatus, std::int64_t> statusHistogram;

    const int expectedNumber = partialResults.getNumElementsInMemory() + partialResults.getNumElementsInFile();
    int actualNumber = 0;

    while(partialResultsReader.hasNext()){        

        ExtendedRead extendedRead = *(partialResultsReader.next());

        std::stringstream sstream;
        sstream << extendedRead.readId;
        sstream << ' ' << (extendedRead.status == ExtendedReadStatus::FoundMate ? "reached:1" : "reached:0");
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
        res.sequence = std::move(extendedRead.extendedSequence);
        if(extendedRead.qualityScores.size() != res.sequence.size()){
            res.quality.resize(res.sequence.length());
            std::fill(res.quality.begin(), res.quality.end(), 'F');
        }else{
            res.quality = std::move(extendedRead.qualityScores);
        }

        writer->writeRead(res.header, res.sequence, res.quality);

        statusHistogram[extendedRead.status]++;

        actualNumber++;
    }

    if(actualNumber != expectedNumber){
        std::cerr << "Error actualNumber " << actualNumber << ", expectedNumber " << expectedNumber << "\n";
    }

    //assert(actualNumber == expectedNumber);

    for(const auto& pair : statusHistogram){
        switch(pair.first){
            case ExtendedReadStatus::FoundMate: std::cout << "Found Mate: " << pair.second << "\n"; break;
            // case ExtendedReadStatus::LengthAbort: std::cout << "Too long: " << pair.second << "\n"; break;
            // case ExtendedReadStatus::CandidateAbort: std::cout << "Empty candidate list: " << pair.second << "\n"; break;
            // case ExtendedReadStatus::MSANoExtension: std::cout << "Did not grow: " << pair.second << "\n"; break;
            default: break;
        }
    }
}


std::optional<Read> combineExtendedReadWithOriginalRead(
    std::vector<ExtendedRead>& tmpresults, 
    const ReadWithId& readWithId
){
    if(tmpresults.size() == 0){
        return std::nullopt;
    }

    bool extended = readWithId.read.sequence.length() < tmpresults[0].extendedSequence.length();

    if(extended){
        Read extendedReadOut{};

        extendedReadOut.sequence = std::move(tmpresults[0].extendedSequence);
        extendedReadOut.quality = std::move(tmpresults[0].qualityScores);
        if(extendedReadOut.quality.size() != extendedReadOut.sequence.size()){
            extendedReadOut.quality.resize(extendedReadOut.sequence.size());
            std::fill(extendedReadOut.quality.begin(), extendedReadOut.quality.end(), 'A');
        }
        
        std::stringstream sstream;
        sstream << readWithId.globalReadId << ' ';
        sstream << (tmpresults[0].status == ExtendedReadStatus::FoundMate ? "reached:1" : "reached:0");
        sstream << ' ';
        sstream << "lens:" << tmpresults[0].read1begin << ',' << tmpresults[0].read1end << ',' << tmpresults[0].read2begin << ',' << tmpresults[0].read2end;

        extendedReadOut.header = sstream.str();

        return std::make_optional(std::move(extendedReadOut));
    }else{
        return std::nullopt;
    }
}


void constructOutputFileFromExtensionResults(
    const std::string& tempdir,
    const std::vector<std::string>& originalReadFiles,
    MemoryFileFixedSize<ExtendedRead>& partialResults, 
    std::size_t memoryForSorting,
    FileFormat outputFormat,
    const std::string& extendedOutputfile,
    const std::vector<std::string>& outputfiles,
    SequencePairType pairmode,
    bool isSorted,
    bool outputToSingleFile
){

    if(outputToSingleFile){                      
        writeExtensionResultsToFile(
            tempdir, 
            partialResults, 
            memoryForSorting, 
            outputFormat,
            extendedOutputfile, 
            isSorted
        );
    }else{
        // {
        //     std::map<ExtendedReadStatus, std::int64_t> statusHistogram2;
        //     auto partialResultsReader = partialResults.makeReader();

        //     while(partialResultsReader.hasNext()){
        //         ExtendedRead er = *(partialResultsReader.next());
        //         statusHistogram2[er.status]++;

        //         if(er.status == ExtendedReadStatus::MSANoExtension){
        //             //std::cerr << er.readId << "\n";
        //         }
        //     }

        //     std::cerr << "should be:\n";
        //     for(const auto& pair : statusHistogram2){
        //         switch(pair.first){
        //             case ExtendedReadStatus::FoundMate: std::cerr << "Found Mate: " << pair.second << "\n"; break;
        //             case ExtendedReadStatus::LengthAbort: std::cerr << "Too long: " << pair.second << "\n"; break;
        //             case ExtendedReadStatus::CandidateAbort: std::cerr << "Empty candidate list: " << pair.second << "\n"; break;
        //             case ExtendedReadStatus::MSANoExtension: std::cerr << "Did not grow: " << pair.second << "\n"; break;
        //         }
        //     }
        // }

        auto origIdResultIdLessThan = [&](read_number origId, read_number resultId){
            //return origId < (resultId / 2);
            //return origId < resultId;
            if(pairmode == SequencePairType::PairedEnd){
                return (origId / 2) < (resultId / 2);
            }else{
                return origId < resultId;
            }
        };

        std::map<ExtendedReadStatus, std::int64_t> statusHistogram;

        auto combine = [&](std::vector<ExtendedRead>& tmpresults, const ReadWithId& readWithId, ReadWithId* mate){
            //statusHistogram[tmpresults[0].status]++;

            return combineExtendedReadWithOriginalRead(tmpresults, readWithId);
        };

        mergeExtensionResultsWithOriginalReads_multithreaded<ExtendedRead>(
            tempdir,
            originalReadFiles,
            partialResults, 
            memoryForSorting,
            outputFormat,
            extendedOutputfile,
            outputfiles,
            pairmode,
            isSorted,
            combine,
            origIdResultIdLessThan
        );

        for(const auto& pair : statusHistogram){
            switch(pair.first){
                case ExtendedReadStatus::FoundMate: std::cout << "Found Mate: " << pair.second << "\n"; break;
                case ExtendedReadStatus::LengthAbort: std::cout << "Too long: " << pair.second << "\n"; break;
                case ExtendedReadStatus::CandidateAbort: std::cout << "Empty candidate list: " << pair.second << "\n"; break;
                case ExtendedReadStatus::MSANoExtension: std::cout << "Did not grow: " << pair.second << "\n"; break;
            }
        }

        // std::vector<std::string> firstOriginalReadFile{originalReadFiles.front()};

        // auto origIdResultIdLessThan = [](read_number origId, read_number resultId){
        //     return origId < (resultId / 2);
        // };

        // std::map<ExtendedReadStatus, std::int64_t> statusHistogram;

        // auto combine = [&](std::vector<ExtendedRead>& tmpresults, ReadWithId& readWithId){
        //     statusHistogram[tmpresults[0].status]++;

        //     combineExtendedReadWithOriginalRead(tmpresults, readWithId);
        // };

        // mergeResultsWithOriginalReads_multithreaded<ExtendedRead>(
        //     tempdir,
        //     firstOriginalReadFile,
        //     partialResults, 
        //     memoryForSorting,
        //     outputFormat,
        //     outputfiles,
        //     isSorted,
        //     combine,
        //     origIdResultIdLessThan
        // );

        // for(const auto& pair : statusHistogram){
        //     switch(pair.first){
        //         case ExtendedReadStatus::FoundMate: std::cout << "Found Mate: " << pair.second << "\n"; break;
        //         case ExtendedReadStatus::LengthAbort: std::cout << "Too long: " << pair.second << "\n"; break;
        //         case ExtendedReadStatus::CandidateAbort: std::cout << "Empty candidate list: " << pair.second << "\n"; break;
        //         case ExtendedReadStatus::MSANoExtension: std::cout << "Did not grow: " << pair.second << "\n"; break;
        //     }
        // }
    }

}


}