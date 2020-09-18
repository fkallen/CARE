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


void constructOutputFileFromExtensionResults_impl(
    const std::string& tempdir,
    const std::vector<std::string>& originalReadFiles,
    MemoryFileFixedSize<ExtendedRead>& partialResults, 
    std::size_t memoryForSorting,
    FileFormat outputFormat,
    const std::vector<std::string>& outputfiles,
    bool isSorted
){
    assert(outputfiles.size() > 0);

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
        outputfiles[0],
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


void combineExtendedReadWithOriginalRead(
    std::vector<ExtendedRead>& tmpresults, 
    ReadWithId& readWithId
){
    if(tmpresults.size() == 0){
        std::cerr << "read id " << readWithId.globalReadId << " no tmpresults!\n";
    }
    assert(tmpresults.size() > 0);

    readWithId.read.sequence = std::move(tmpresults[0].extendedSequence);
}


void constructOutputFileFromExtensionResults(
    const std::string& tempdir,
    const std::vector<std::string>& originalReadFiles,
    MemoryFileFixedSize<ExtendedRead>& partialResults, 
    std::size_t memoryForSorting,
    FileFormat outputFormat,
    const std::vector<std::string>& outputfiles,
    bool isSorted
){
                        
    // constructOutputFileFromExtensionResults_impl(
    //     tempdir, 
    //     originalReadFiles, 
    //     partialResults, 
    //     memoryForSorting, 
    //     outputFormat,
    //     outputfiles, 
    //     isSorted
    // );

    std::vector<std::string> firstOriginalReadFile{originalReadFiles.front()};

    auto origIdResultIdLessThan = [](read_number origId, read_number resultId){
        return origId < (resultId / 2);
    };

    mergeResultsWithOriginalReads_multithreaded<ExtendedRead>(
        tempdir,
        firstOriginalReadFile,
        partialResults, 
        memoryForSorting,
        outputFormat,
        outputfiles,
        isSorted,
        combineExtendedReadWithOriginalRead,
        origIdResultIdLessThan
    );
}


}