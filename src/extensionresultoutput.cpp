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
        res.sequence = std::move(extendedRead.extendedSequence);
        if(extendedRead.qualityScores.size() != res.sequence.size()){
            res.quality.resize(res.sequence.length());
            std::fill(res.quality.begin(), res.quality.end(), 'F');
        }else{
            res.quality = std::move(extendedRead.qualityScores);
        }

        return res;
    }

/*
    Write reads which were extended to extendedOutputfile.
    Write reads which did not grow to outputfiles
*/

template<class ResultType, class Combiner, class ReadIdComparator>
void mergeSerializedExtensionResultsWithOriginalReads_multithreaded(
    const std::vector<std::string>& originalReadFiles,
    SerializedObjectStorage& partialResults, 
    FileFormat outputFormat,
    const std::string& extendedOutputfile,
    const std::vector<std::string>& outputfiles,
    SequencePairType pairmode,
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
                if(bool(instream) && instream.rdbuf()->in_avail() > 0){
                    outstream << instream.rdbuf();
                }
            }
        }else{
            const int numFiles = outputfiles.size();
            for(int i = 0; i < numFiles; i++){
                std::ofstream outstream(outputfiles[i], std::ios::binary);
                if(!bool(outstream)){
                    throw std::runtime_error("Cannot open output file " + outputfiles[0]);
                }

                std::ifstream instream(originalReadFiles[i], std::ios::binary);
                if(bool(instream) && instream.rdbuf()->in_avail() > 0){
                    outstream << instream.rdbuf();
                }
            }
        }

        return;
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
            
            // std::chrono::time_point<std::chrono::system_clock> abegin, aend;
            // std::chrono::duration<double> adelta{0};

            // TIMERSTARTCPU(tcsparsing);

            read_number previousId = 0;
            std::size_t itemnumber = 0;

            while(itemnumber < partialResults.size()){
                ResultTypeBatch* batch = freeTcsBatches.pop();

                //abegin = std::chrono::system_clock::now();

                batch->items.resize(decoder_maxbatchsize);

                int batchsize = 0;
                while(batchsize < decoder_maxbatchsize && itemnumber < partialResults.size()){
                    const std::uint8_t* serializedPtr = partialResults.getPointer(itemnumber);

                    ExtendedRead extendedRead;

                    #if 0
                    extendedRead.copyFromContiguousMemory(serializedPtr);
                    #else
                    EncodedExtendedRead encext;
                    encext.copyFromContiguousMemory(serializedPtr);
                    extendedRead.decode(encext);
                    #endif

                    batch->items[batchsize] = std::move(extendedRead);

                    if(batch->items[batchsize].readId < previousId){
                        std::cerr << "Error, results not sorted. itemnumber = " << itemnumber << ", previousId = " << previousId << ", currentId = " << batch->items[batchsize].readId << "\n";
                        assert(false);
                    }
                    batchsize++;
                    itemnumber++;
                    previousId = batch->items[batchsize].readId;
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

            std::unique_ptr<SequenceFileWriter> extendedReadWriter = makeSequenceWriter(extendedOutputfile, format);
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















std::optional<Read> combineExtendedReadWithOriginalRead(
    std::vector<ExtendedRead>& tmpresults, 
    const ReadWithId& readWithId
){
    if(tmpresults.size() == 0){
        return std::nullopt;
    }

    bool extended = readWithId.read.sequence.length() < tmpresults[0].extendedSequence.length();

    if(extended){
        const auto& extendedRead = tmpresults[0];

        return std::make_optional(makeOutputReadFromExtendedRead(extendedRead));
    }else{
        return std::nullopt;
    }
}






void constructOutputFileFromExtensionResults(
    const std::vector<std::string>& originalReadFiles,
    SerializedObjectStorage& partialResults, 
    FileFormat outputFormat,
    const std::string& extendedOutputfile,
    const std::vector<std::string>& outputfiles,
    SequencePairType pairmode,
    bool outputToSingleFile
){

    if(outputToSingleFile){                      
        writeExtensionResultsToFile(
            partialResults, 
            //FileFormat::FASTA, 
            outputFormat,
            extendedOutputfile
        );
    }else{

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

        auto combine = [&](std::vector<ExtendedRead>& tmpresults, const ReadWithId& readWithId, ReadWithId* /*mate*/){
            //statusHistogram[tmpresults[0].status]++;

            return combineExtendedReadWithOriginalRead(tmpresults, readWithId);
        };

        mergeSerializedExtensionResultsWithOriginalReads_multithreaded<ExtendedRead>(
            originalReadFiles,
            partialResults, 
            outputFormat,
            extendedOutputfile,
            outputfiles,
            pairmode,
            combine,
            origIdResultIdLessThan
        );

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