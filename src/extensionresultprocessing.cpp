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
        auto ptrcomparator = [](const std::uint8_t* ptr1, const std::uint8_t* ptr2){
            read_number lid1;
            read_number rid1;
            std::memcpy(&lid1, ptr1, sizeof(read_number));
            std::memcpy(&rid1, ptr2, sizeof(read_number));
            
            return lid1 < rid1;
        };

        auto elementcomparator = [](const auto& l, const auto& r){
            return l.readId < r.readId;
        };

        helpers::CpuTimer timer("sort_results_by_read_id");
        partialResults.sort(tempdir, memoryForSorting, ptrcomparator, elementcomparator);
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

    while(partialResultsReader.hasNext()){

        ExtendedRead extendedRead = *(partialResultsReader.next());

        std::stringstream sstream;
        sstream << extendedRead.readId;
        sstream << ' ' << (extendedRead.status == ExtendedReadStatus::FoundMate ? "reachedmate:1" : "reachedmate:0");
        sstream << ' ';
        if(extendedRead.status == ExtendedReadStatus::LengthAbort){
            sstream << "exceeded_length";
        }else if(extendedRead.status == ExtendedReadStatus::CandidateAbort){
            sstream << "0_candidates";
        }else if(extendedRead.status == ExtendedReadStatus::MSANoExtension){
            sstream << "msa_stop";
        }

        Read res;
        res.header = sstream.str();
        res.sequence = std::move(extendedRead.extendedSequence);
        res.quality.resize(res.sequence.length());
        std::fill(res.quality.begin(), res.quality.end(), 'F');

        writer->writeRead(res.header, res.sequence, res.quality);

        statusHistogram[extendedRead.status]++;
    }

    for(const auto& pair : statusHistogram){
        switch(pair.first){
            case ExtendedReadStatus::FoundMate: std::cout << "Found Mate: " << pair.second << "\n"; break;
            case ExtendedReadStatus::LengthAbort: std::cout << "Too long: " << pair.second << "\n"; break;
            case ExtendedReadStatus::CandidateAbort: std::cout << "Empty candidate list: " << pair.second << "\n"; break;
            case ExtendedReadStatus::MSANoExtension: std::cout << "Did not grow: " << pair.second << "\n"; break;
        }
    }
}


bool combineExtendedReadWithOriginalRead(
    std::vector<ExtendedRead>& tmpresults, 
    ReadWithId& readWithId,
    std::string& extendedSequence
){
    if(tmpresults.size() == 0){
        std::cerr << "read id " << readWithId.globalReadId << " no tmpresults!\n";
    }
    assert(tmpresults.size() > 0);

    bool extended = readWithId.read.sequence.length() < tmpresults[0].extendedSequence.length();
    //readWithId.read.sequence = std::move(tmpresults[0].extendedSequence);
    extendedSequence = std::move(tmpresults[0].extendedSequence);

    return extended;
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

        auto combine = [&](std::vector<ExtendedRead>& tmpresults, ReadWithId& readWithId, ReadWithId* mate, std::string& extendedSequence){
            statusHistogram[tmpresults[0].status]++;

            return combineExtendedReadWithOriginalRead(tmpresults, readWithId, extendedSequence);
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