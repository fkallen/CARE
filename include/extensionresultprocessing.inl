#include <extensionresultprocessing.hpp>

#include <hpc_helpers.cuh>
#include <memoryfile.hpp>
#include <readlibraryio.hpp>
#include <concurrencyhelpers.hpp>

#include <string>
#include <vector>
#include <cassert>
#include <sstream>
#include <future>

namespace care{

/*
    Write reads which were extended to extendedOutputfile.
    Write reads which did not grow to outputfiles
*/
template<class ResultType, class MemoryFile_t, class Combiner, class ReadIdComparator>
void mergeExtensionResultsWithOriginalReads_multithreaded(
    const std::string& tempdir,
    const std::vector<std::string>& originalReadFiles,
    MemoryFile_t& partialResults, 
    std::size_t memoryForSorting,
    FileFormat outputFormat,
    const std::string& extendedOutputfile,
    const std::vector<std::string>& outputfiles,
    SequencePairType pairmode,
    bool isSorted,
    Combiner combineResultsWithRead, /* combineResultsWithRead(std::vector<ResultType>& in, ReadWithId& in_out) */
    ReadIdComparator origIdResultIdLessThan
){

    assert(outputfiles.size() == 1 || originalReadFiles.size() == outputfiles.size());

    if(partialResults.getNumElements() == 0){
        if(outputfiles.size() == 1){
            std::ofstream outstream(outputfiles[0], std::ios::binary);
            if(!bool(outstream)){
                throw std::runtime_error("Cannot open output file " + outputfiles[0]);
            }

            for(const auto& ifname : originalReadFiles){
                std::ifstream instream(ifname, std::ios::binary);
                outstream << instream.rdbuf();
            }
        }else{
            const int numFiles = outputfiles.size();
            for(int i = 0; i < numFiles; i++){
                std::ofstream outstream(outputfiles[i], std::ios::binary);
                if(!bool(outstream)){
                    throw std::runtime_error("Cannot open output file " + outputfiles[0]);
                }

                std::ifstream instream(originalReadFiles[i], std::ios::binary);
                outstream << instream.rdbuf();
            }
        }

        return;
    }



    if(!isSorted){

        auto elementcomparator = [](const auto& l, const auto& r){
            return l.getReadId() < r.getReadId();
        };

        auto extractKey = [](const std::uint8_t* ptr){
            using ValueType = typename MemoryFile_t::ValueType;

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

    helpers::CpuTimer mergetimer("merging");

    struct ResultTypeBatch{
        int validItems = 0;
        int processedItems = 0;
        std::vector<ResultType> items;
    };

    std::array<ResultTypeBatch, 4> tcsBatches;

    SimpleSingleProducerSingleConsumerQueue<ResultTypeBatch*> freeTcsBatches;
    SimpleSingleProducerSingleConsumerQueue<ResultTypeBatch*> unprocessedTcsBatches;

    std::atomic<bool> noMoreTcsBatches{false};

    for(auto& batch : tcsBatches){
        freeTcsBatches.push(&batch);
    }

    constexpr int decoder_maxbatchsize = 100000;

    auto decoderFuture = std::async(std::launch::async,
        [&](){
            

            auto partialResultsReader = partialResults.makeReader();

            // std::chrono::time_point<std::chrono::system_clock> abegin, aend;
            // std::chrono::duration<double> adelta{0};

            // TIMERSTARTCPU(tcsparsing);

            while(partialResultsReader.hasNext()){
                ResultTypeBatch* batch = freeTcsBatches.pop();

                //abegin = std::chrono::system_clock::now();

                batch->items.resize(decoder_maxbatchsize);

                int batchsize = 0;
                while(batchsize < decoder_maxbatchsize && partialResultsReader.hasNext()){
                    batch->items[batchsize] = *(partialResultsReader.next());
                    batchsize++;
                }

                // aend = std::chrono::system_clock::now();
                // adelta += aend - abegin;

                batch->processedItems = 0;
                batch->validItems = batchsize;

                unprocessedTcsBatches.push(batch);
            }

            //std::cout << "# elapsed time ("<< "tcsparsing without queues" <<"): " << adelta.count()  << " s" << std::endl;

            // TIMERSTOPCPU(tcsparsing);

            noMoreTcsBatches = true;
        }
    );

    struct ReadBatch{
        int validItems = 0;
        int processedItems = 0;
        std::vector<ReadWithId> items;
        std::vector<ReadWithId> items2;
        std::vector<std::optional<Read>> extendedReads;
    };

    std::array<ReadBatch, 4> readBatches;

    // free -> unprocessed input -> unprocessed output -> free -> ...
    SimpleSingleProducerSingleConsumerQueue<ReadBatch*> freeReadBatches;
    SimpleSingleProducerSingleConsumerQueue<ReadBatch*> unprocessedInputreadBatches;
    SimpleSingleProducerSingleConsumerQueue<ReadBatch*> unprocessedOutputreadBatches;

    std::atomic<bool> noMoreInputreadBatches{false};

    constexpr int inputreader_maxbatchsize = 200000;

    for(auto& batch : readBatches){
        freeReadBatches.push(&batch);
    }

    auto makeBatchesFromMultiInputReader = [&](){
        MultiInputReader multiInputReader(originalReadFiles);

        while(multiInputReader.next() >= 0){

            ReadBatch* batch = freeReadBatches.pop();

            batch->items.resize(inputreader_maxbatchsize);
            //batch->items2.resize(inputreader_maxbatchsize);
            batch->extendedReads.resize(inputreader_maxbatchsize);

            for(auto& optional : batch->extendedReads){
                optional.reset();
            }

            std::swap(batch->items[0], multiInputReader.getCurrent()); //process element from outer loop next() call
            int batchsize = 1;

            while(batchsize < inputreader_maxbatchsize && multiInputReader.next() >= 0){
                std::swap(batch->items[batchsize], multiInputReader.getCurrent());
                batchsize++;
            }

            batch->processedItems = 0;
            batch->validItems = batchsize;

            unprocessedInputreadBatches.push(batch);                
        }

        noMoreInputreadBatches = true;
    };

    auto makeBatchesFromPairedInputReader = [&](){
        PairedInputReader pairedInputReader(originalReadFiles);

        while(pairedInputReader.next() >= 0){

            ReadBatch* batch = freeReadBatches.pop();

            batch->items.resize(inputreader_maxbatchsize);
            batch->items2.resize(inputreader_maxbatchsize);
            batch->extendedReads.resize(inputreader_maxbatchsize);

            for(auto& optional : batch->extendedReads){
                optional.reset();
            }

            std::swap(batch->items[0], pairedInputReader.getCurrent1()); //process element from outer loop next() call
            std::swap(batch->items2[0], pairedInputReader.getCurrent2()); //process element from outer loop next() call
            int batchsize = 1;

            while(batchsize < inputreader_maxbatchsize && pairedInputReader.next() >= 0){
                std::swap(batch->items[batchsize], pairedInputReader.getCurrent1());
                std::swap(batch->items2[batchsize], pairedInputReader.getCurrent2());
                batchsize++;
            }

            batch->processedItems = 0;
            batch->validItems = batchsize;

            unprocessedInputreadBatches.push(batch);                
        }

        noMoreInputreadBatches = true;
    };

    auto inputReaderFuture = std::async(std::launch::async,
        [&](){
            if(pairmode == SequencePairType::PairedEnd){
                makeBatchesFromPairedInputReader();
            }else{
                makeBatchesFromMultiInputReader();
            }                
        }
    );

    std::atomic<bool> noMoreOutputreadBatches{false};

    auto outputWriterFuture = std::async(std::launch::async,
        [&](){
            //no gz output
            auto format = outputFormat;

            if(format == FileFormat::FASTQGZ)
                format = FileFormat::FASTQ;
            if(format == FileFormat::FASTAGZ)
                format = FileFormat::FASTA;

            //auto extendedformat = FileFormat::FASTA; //extended reads are stored as fasta. No qualities available for gap.

            std::unique_ptr<SequenceFileWriter> extendedReadWriter = makeSequenceWriter(extendedOutputfile, /*format*/ FileFormat::FASTA);
            std::unique_ptr<SequenceFileWriter> extendedOrigReadWriter = makeSequenceWriter(extendedOutputfile + "_origs", format);

            std::vector<std::unique_ptr<SequenceFileWriter>> writerVector;

            for(const auto& outputfile : outputfiles){
                writerVector.emplace_back(makeSequenceWriter(outputfile, format));
            }

            ReadBatch* outputBatch = unprocessedOutputreadBatches.pop();

            while(outputBatch != nullptr){                
                
                int processed = outputBatch->processedItems;
                const int valid = outputBatch->validItems;
                while(processed < valid){
                    const bool extended = outputBatch->extendedReads[processed].has_value(); 

                    if(extended){
                        const auto& readWithId1 = outputBatch->items[processed];
                        const Read& read = *outputBatch->extendedReads[processed];
                        //extendedReadWriter->writeRead(str);
                        extendedReadWriter->writeRead(read.header, read.sequence, read.quality);

                        #if 1                         

                        if(pairmode == SequencePairType::SingleEnd){
                            extendedOrigReadWriter->writeRead(readWithId1.read);
                        }

                        #endif
                    }else{
                        if(pairmode == SequencePairType::PairedEnd){                                    
                            const auto& readWithId1 = outputBatch->items[processed];
                            const auto& readWithId2 = outputBatch->items2[processed];
                            writerVector[readWithId1.fileId]->writeRead(readWithId1.read);
                            writerVector[readWithId2.fileId]->writeRead(readWithId2.read);
                        }else{
                            const auto& readWithId1 = outputBatch->items[processed];
                            writerVector[readWithId1.fileId]->writeRead(readWithId1.read);
                        }
                    }
                    processed++;
                }

                freeReadBatches.push(outputBatch);     

                outputBatch = unprocessedOutputreadBatches.popOrDefault(
                    [&](){
                        return !noMoreOutputreadBatches;
                    },
                    nullptr
                );            
            }
        }
    );

    ResultTypeBatch* tcsBatch = unprocessedTcsBatches.popOrDefault(
        [&](){
            return !noMoreTcsBatches;  //its possible that there are no corrections. In that case, return nullptr
        },
        nullptr
    ); 

    ReadBatch* inputBatch = unprocessedInputreadBatches.pop();

    assert(!(inputBatch == nullptr && tcsBatch != nullptr)); //there must be at least one batch of input reads

    std::vector<ResultType> buffer;

    while(!(inputBatch == nullptr && tcsBatch == nullptr)){

        //modify reads in inputbatch, applying corrections.
        //then place inputbatch in outputqueue

        if(tcsBatch == nullptr){
            //all correction results are processed
            //copy remaining input reads to output file
            
            ; //nothing to do
        }else{

            auto last1 = inputBatch->items.begin() + inputBatch->validItems;
            auto last2 = tcsBatch->items.begin() + tcsBatch->validItems;

            auto first1 = inputBatch->items.begin()+ inputBatch->processedItems;
            auto first2 = tcsBatch->items.begin()+ tcsBatch->processedItems;   

            while(first1 != last1) {
                if(first2 == last2){
                    //all results are processed
                    //copy remaining input reads to output file

                    ; //nothing to do
                    break;
                }
                //assert(first2->readId >= first1->globalReadId);
                {

                    ReadWithId& readWithId = *first1;
                    
                    buffer.clear();
                    while(first2 != last2 && !(origIdResultIdLessThan(readWithId.globalReadId, first2->readId))){
                    //while(first2 != last2 && !(first1->globalReadId < first2->readId)){
                        buffer.push_back(*first2);
                        ++first2;
                        tcsBatch->processedItems++;

                        if(first2 == last2){                                
                            freeTcsBatches.push(tcsBatch);

                            tcsBatch = unprocessedTcsBatches.popOrDefault(
                                [&](){
                                    return !noMoreTcsBatches;  //its possible that there are no more results to process. In that case, return nullptr
                                },
                                nullptr
                            );

                            if(tcsBatch != nullptr){
                                //new batch could be fetched. update begin and end accordingly
                                last2 = tcsBatch->items.begin() + tcsBatch->validItems;
                                first2 = tcsBatch->items.begin()+ tcsBatch->processedItems;
                            }
                        }
                    }

                    const auto dist = std::distance(inputBatch->items.begin()+ inputBatch->processedItems, first1);

                    ReadWithId* matePtr = nullptr;
                    if(pairmode == SequencePairType::PairedEnd){                            
                        matePtr = inputBatch->items2.data() + inputBatch->processedItems + dist;
                    }

                    inputBatch->extendedReads[inputBatch->processedItems + dist] 
                        = combineResultsWithRead(buffer, readWithId, matePtr);

                    ++first1;
                }
            }
        }

        assert(inputBatch != nullptr);

        unprocessedOutputreadBatches.push(inputBatch);

        inputBatch = unprocessedInputreadBatches.popOrDefault(
            [&](){
                return !noMoreInputreadBatches;
            },
            nullptr
        );  
        
        assert(!(inputBatch == nullptr && tcsBatch != nullptr));

    }

    noMoreOutputreadBatches = true;

    decoderFuture.wait();
    inputReaderFuture.wait();
    outputWriterFuture.wait();

    mergetimer.print();
}



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
        sstream << ' ' << (extendedRead.mergedFromReadsWithoutMate ? "m:1" : "m:0");
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
        sstream << ' ' << (tmpresults[0].mergedFromReadsWithoutMate ? "m:1" : "m:0");
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
            FileFormat::FASTA, //outputFormat,
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
    }

}


}