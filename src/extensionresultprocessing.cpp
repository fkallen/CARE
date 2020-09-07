#include <extensionresultprocessing.hpp>

#include <memoryfile.hpp>
#include <readlibraryio.hpp>

#include <string>
#include <vector>
#include <cassert>

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

        TIMERSTARTCPU(sort_results_by_read_id);
        partialResults.sort(tempdir, memoryForSorting, ptrcomparator, elementcomparator);
        TIMERSTOPCPU(sort_results_by_read_id);
    }

    auto partialResultsReader = partialResults.makeReader();

    std::unique_ptr<SequenceFileWriter> writer = makeSequenceWriter(
        //fileOptions.outputdirectory + "/extensionresult.txt", 
        outputfiles[0],
        outputFormat
    );

    std::cerr << "in mem: " << partialResults.getNumElementsInMemory() << ", in file: " << partialResults.getNumElementsInFile() << "\n";

    while(partialResultsReader.hasNext()){

        ExtendedRead extendedRead = *(partialResultsReader.next());

        Read res;
        res.name = std::to_string(extendedRead.readId);
        res.comment = extendedRead.reachedMate ? "reachedmate:0" : "reachedmate:1";
        res.sequence = std::move(extendedRead.extendedSequence);
        res.quality.resize(res.sequence.length());
        std::fill(res.quality.begin(), res.quality.end(), 'F');

        writer->writeRead(res.name, res.comment, res.sequence, res.quality);
    }
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
                        
    constructOutputFileFromExtensionResults_impl(
        tempdir, 
        originalReadFiles, 
        partialResults, 
        memoryForSorting, 
        outputFormat,
        outputfiles, 
        isSorted
    );
}


}