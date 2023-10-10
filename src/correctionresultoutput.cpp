#include <correctionresultoutput.hpp>
#include <correctedsequence.hpp>

#include <config.hpp>

#include <hpc_helpers.cuh>
#include <serializedobjectstorage.hpp>
#include <readlibraryio.hpp>
#include <threadpool.hpp>
#include <concurrencyhelpers.hpp>
#include <options.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <queue>

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <future>

namespace care{


template<class ResultType, class Combiner, class ProgressFunction>
void mergeSerializedResultsWithOriginalReads_multithreaded(
    const std::vector<std::string>& originalReadFiles,
    SerializedObjectStorage& partialResults, 
    FileFormat outputFormat,
    const std::vector<std::string>& outputfiles,
    Combiner combineResultsWithRead, /* combineResultsWithRead(std::vector<ResultType>& in, ReadWithId& in_out) */
    ProgressFunction addProgress,
    bool outputCorrectionQualityLabels,
    SequencePairType pairType
){
    assert(outputfiles.size() == 1 || originalReadFiles.size() == outputfiles.size());

    if(partialResults.getNumElements() == 0){
        if(outputfiles.size() == 1){
            if(pairType == SequencePairType::SingleEnd 
                || (pairType == SequencePairType::PairedEnd && originalReadFiles.size() == 1)){
                auto filewriter = makeSequenceWriter(outputfiles[0], outputFormat);

                MultiInputReader reader(originalReadFiles);
                while(reader.next() >= 0){
                    ReadWithId& readWithId = reader.getCurrent();
                    filewriter->writeRead(readWithId.read);
                }
            }else{
                assert(pairType == SequencePairType::PairedEnd);
                assert(originalReadFiles.size() == 2);
                auto filewriter = makeSequenceWriter(outputfiles[0], outputFormat);

                //merged output
                forEachReadInPairedFiles(originalReadFiles[0], originalReadFiles[1], 
                    [&](auto /*readnumber*/, const auto& read){
                        filewriter->writeRead(read);
                    }
                );
            }
        }else{
            const int numFiles = outputfiles.size();
            for(int i = 0; i < numFiles; i++){
                auto filewriter = makeSequenceWriter(outputfiles[i], outputFormat);

                forEachReadInFile(originalReadFiles[i], 
                    [&](auto /*readnumber*/, const auto& read){
                        filewriter->writeRead(read);
                    }
                );
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

    std::array<ResultTypeBatch, 6> tcsBatches;

    SimpleConcurrentQueue<ResultTypeBatch*> freeTcsBatches;
    SimpleConcurrentQueue<ResultTypeBatch*> unprocessedTcsBatches;

    for(auto& batch : tcsBatches){
        freeTcsBatches.push(&batch);
    }

    constexpr std::size_t decoder_maxbatchsize = 100000;
    constexpr int numDecoderThreads = 3;

    std::atomic<std::size_t> nextBatchIdToSubmit{0};
    std::atomic<std::size_t> globalBatchId{0};
    std::atomic<bool> decoderError{false};
    std::size_t numDecoderBatches = SDIV(partialResults.size(), decoder_maxbatchsize);

    auto decoderFunc = [&](){
        try{
            #ifndef NDEBUG
            read_number previousId = 0;
            #endif

            std::size_t batchId = globalBatchId.fetch_add(1);
            while(batchId < numDecoderBatches && !decoderError){
                ResultTypeBatch* batch = freeTcsBatches.pop();
                batch->items.resize(decoder_maxbatchsize);

                std::size_t begin = batchId * decoder_maxbatchsize;
                std::size_t end = std::min((batchId+1) * decoder_maxbatchsize, partialResults.size());
                std::size_t num = end - begin;

                for(std::size_t i = 0; i < num; i++){
                    const std::uint8_t* serializedPtr = partialResults.getPointer(begin + i);
                    batch->items[i].decodeFromEncodedSerializedPtr(serializedPtr);

                    #ifndef NDEBUG
                    if(batch->items[i].readId < previousId){
                        std::cerr << "Error, results not sorted. itemnumber = " << begin+i << ", previousId = " << previousId 
                            << ", currentId = " << batch->items[i].readId << "\n";
                        assert(false);
                    }
                    previousId = batch->items[i].readId;
                    #endif
                }
                batch->processedItems = 0;
                batch->validItems = num;

                if(numDecoderThreads > 1){
                    //ensure correct order of batches with multiple decoders
                    while(batchId != nextBatchIdToSubmit && !decoderError){}
                }

                unprocessedTcsBatches.push(batch);
                nextBatchIdToSubmit++;


                if(batchId == numDecoderBatches - 1){
                    //the thread that processes the last batch notifies that all decoders are done
                    unprocessedTcsBatches.push(nullptr);
                }

                batchId = globalBatchId.fetch_add(1);
            }
        }catch(...){
            std::cout<< "decoder exception\n";
            decoderError = true;
            globalBatchId = numDecoderBatches;
            unprocessedTcsBatches.push(nullptr);
        }
    };


    std::vector<std::future<void>> decoderFutures;
    for(int i = 0; i < numDecoderThreads; i++){
        decoderFutures.emplace_back(std::async(std::launch::async, decoderFunc));
    }


    struct ReadBatch{
        int validItems = 0;
        int processedItems = 0;
        std::vector<ReadWithId> items;
    };

    std::array<ReadBatch, 4> readBatches;

    // free -> unprocessed input -> unprocessed output -> free -> ...
    SimpleConcurrentQueue<ReadBatch*> freeReadBatches;
    SimpleConcurrentQueue<ReadBatch*> unprocessedInputreadBatches;
    SimpleConcurrentQueue<ReadBatch*> unprocessedOutputreadBatches;

    std::atomic<bool> noMoreInputreadBatches{false};

    constexpr int inputreader_maxbatchsize = 200000;
    static_assert(inputreader_maxbatchsize % 2 == 0);

    for(auto& batch : readBatches){
        freeReadBatches.push(&batch);
    }

    auto pairedEndReaderFunc = [&](){
        try{

            PairedInputReader pairedInputReader(originalReadFiles);

            // TIMERSTARTCPU(inputparsing);

            // std::chrono::time_point<std::chrono::system_clock> abegin, aend;
            // std::chrono::duration<double> adelta{0};

            while(pairedInputReader.next() >= 0){

                ReadBatch* batch = freeReadBatches.pop();

                // abegin = std::chrono::system_clock::now();

                batch->items.resize(inputreader_maxbatchsize);

                std::swap(batch->items[0], pairedInputReader.getCurrent1()); //process element from outer loop next() call
                std::swap(batch->items[1], pairedInputReader.getCurrent2()); //process element from outer loop next() call
                int batchsize = 2;

                while(batchsize < inputreader_maxbatchsize && pairedInputReader.next() >= 0){
                    std::swap(batch->items[batchsize], pairedInputReader.getCurrent1());                
                    batchsize++;
                    std::swap(batch->items[batchsize], pairedInputReader.getCurrent2());        
                    batchsize++;
                }

                // aend = std::chrono::system_clock::now();
                // adelta += aend - abegin;

                batch->processedItems = 0;
                batch->validItems = batchsize;
                unprocessedInputreadBatches.push(batch);                
            }

            unprocessedInputreadBatches.push(nullptr);

            // std::cout << "# elapsed time ("<< "inputparsing without queues" <<"): " << adelta.count()  << " s" << std::endl;

            // TIMERSTOPCPU(inputparsing);

        }catch(...){
            std::cout<< "pairedEndReaderFunc exception\n";
            unprocessedInputreadBatches.push(nullptr);
        }
    };

    auto singleEndReaderFunc = [&](){
        try{
            MultiInputReader multiInputReader(originalReadFiles);

            // TIMERSTARTCPU(inputparsing);

            // std::chrono::time_point<std::chrono::system_clock> abegin, aend;
            // std::chrono::duration<double> adelta{0};

            while(multiInputReader.next() >= 0){
                ReadBatch* batch = freeReadBatches.pop();

                // abegin = std::chrono::system_clock::now();

                batch->items.resize(inputreader_maxbatchsize);

                std::swap(batch->items[0], multiInputReader.getCurrent()); //process element from outer loop next() call
                int batchsize = 1;

                while(batchsize < inputreader_maxbatchsize && multiInputReader.next() >= 0){
                    std::swap(batch->items[batchsize], multiInputReader.getCurrent());
                    
                    batchsize++;
                }

                // aend = std::chrono::system_clock::now();
                // adelta += aend - abegin;

                batch->processedItems = 0;
                batch->validItems = batchsize;

                unprocessedInputreadBatches.push(batch);                
            }

            unprocessedInputreadBatches.push(nullptr);
            // std::cout << "# elapsed time ("<< "inputparsing without queues" <<"): " << adelta.count()  << " s" << std::endl;

            // TIMERSTOPCPU(inputparsing);
        }catch(...){
            std::cout << "singleEndReaderFunc exception\n";
            unprocessedInputreadBatches.push(nullptr);
        }
    };

    auto inputReaderFuture = std::async(std::launch::async,
        [&](){
            if(pairType == SequencePairType::SingleEnd){
                singleEndReaderFunc();
            }else{
                assert(pairType == SequencePairType::PairedEnd);
                pairedEndReaderFunc();
            }
        }
    );

    auto outputWriterFuture = std::async(std::launch::async,
        [&](){
            try{
                std::vector<std::unique_ptr<SequenceFileWriter>> writerVector;

                assert(originalReadFiles.size() == outputfiles.size() || outputfiles.size() == 1);

                for(const auto& outputfile : outputfiles){
                    writerVector.emplace_back(makeSequenceWriter(outputfile, outputFormat));
                }

                const int numOutputfiles = outputfiles.size();

                // TIMERSTARTCPU(outputwriting);

                // std::chrono::time_point<std::chrono::system_clock> abegin, aend;
                // std::chrono::duration<double> adelta{0};

                ReadBatch* outputBatch = unprocessedOutputreadBatches.pop();

                while(outputBatch != nullptr){                

                    // abegin = std::chrono::system_clock::now();
                    
                    int processed = outputBatch->processedItems;
                    const int valid = outputBatch->validItems;
                    while(processed < valid){
                        const auto& readWithId = outputBatch->items[processed];
                        const int writerIndex = numOutputfiles == 1 ? 0 : readWithId.fileId;
                        assert(writerIndex < numOutputfiles);

                        writerVector[writerIndex]->writeRead(readWithId.read);

                        processed++;
                    }

                    if(processed == valid){
                        addProgress(valid);
                    }

                    // aend = std::chrono::system_clock::now();
                    // adelta += aend - abegin;

                    freeReadBatches.push(outputBatch);     
                    outputBatch = unprocessedOutputreadBatches.pop();            
                }

                freeReadBatches.push(nullptr);

                // std::cout << "# elapsed time ("<< "outputwriting without queues" <<"): " << adelta.count()  << " s" << std::endl;

                // TIMERSTOPCPU(outputwriting);

            }catch(...){
                std::cout<< "pairedEndReaderFunc exception\n";
                freeReadBatches.push(nullptr);
            }
        }
    );

    ResultTypeBatch* tcsBatch = unprocessedTcsBatches.pop();
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

            // assert(std::is_sorted(
            //     first1,
            //     last1,
            //     [](const auto& l, const auto& r){
            //         if(l.fileId < r.fileId) return true;
            //         if(l.fileId > r.fileId) return false;
            //         if(l.readIdInFile < r.readIdInFile) return true;
            //         if(l.readIdInFile > r.readIdInFile) return false;
                    
            //         return l.globalReadId < r.globalReadId;
            //     }
            // ));

            // assert(std::is_sorted(
            //     first2,
            //     last2,
            //     [](const auto& l, const auto& r){
            //         return l.readId < r.readId;
            //     }
            // ));

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
                    
                    //if all tempcorrectedsequences for current read are contained within a single batch, we can use them directly
                    //otherwise we need to copy all sequences to a temporary contiguous buffer
                    int numBatchesForCurrentRead = 1;
                    auto directSequencesBegin = first2;
                    auto directSequencesEnd = first2;
                    buffer.clear();

                    while(first2 != last2 && (first1->globalReadId == first2->readId)){
                        if(numBatchesForCurrentRead == 1){
                            ++directSequencesEnd;                            
                        }else{
                            buffer.push_back(*first2);
                        }
                        //buffer.push_back(*first2);
                        ++first2;
                        tcsBatch->processedItems++;

                        if(first2 == last2){
                            //tcsbatch fully processed. there might be more sequences for the current anchor in the next batch. switch to buffer processing instead of direct sequence processing
                            buffer.insert(buffer.end(), directSequencesBegin, directSequencesEnd);

                            freeTcsBatches.push(tcsBatch);
                            tcsBatch = unprocessedTcsBatches.pop();

                            if(tcsBatch != nullptr){
                                //new batch could be fetched. update begin and end accordingly
                                last2 = tcsBatch->items.begin() + tcsBatch->validItems;
                                first2 = tcsBatch->items.begin()+ tcsBatch->processedItems;
                                numBatchesForCurrentRead++;
                            }
                        }
                    }

                    const auto combineStatus = [&](){
                        if(buffer.empty()){
                            return combineResultsWithRead(directSequencesBegin, directSequencesEnd, readWithId); 
                        }else{
                            return combineResultsWithRead(buffer.begin(), buffer.end(), readWithId); 
                        }
                    }();

                    if(outputCorrectionQualityLabels){
                        if(combineStatus.corrected){
                            if(combineStatus.lqCorrectionOnlyAnchor){
                                readWithId.read.header += " care:q=1"; 
                            }else if(combineStatus.lqCorrectionWithCandidates){
                                readWithId.read.header += " care:q=2";  
                            }else if(combineStatus.hqCorrection){
                                readWithId.read.header += " care:q=3";  
                            }
                        }else{
                            readWithId.read.header += " care:q=0";
                        }
                    }

                    ++first1;
                }
            }
        }

        assert(inputBatch != nullptr);

        unprocessedOutputreadBatches.push(inputBatch);

        inputBatch = unprocessedInputreadBatches.pop();  
        
        assert(!(inputBatch == nullptr && tcsBatch != nullptr)); //unprocessed correction results must have a corresponding original read

    }

    unprocessedOutputreadBatches.push(nullptr);
    freeTcsBatches.push(nullptr);

    for(auto& future : decoderFutures){
        future.wait();
    }
    //decoderFuture.wait();
    inputReaderFuture.wait();
    outputWriterFuture.wait();

    mergetimer.print();
}





struct CombinedCorrectionResult{
    bool corrected = false;
    bool hqCorrection = false;
    bool lqCorrectionOnlyAnchor = false;
    bool lqCorrectionWithCandidates = false;
    std::string correctedSequence{};
};

template<class TempCorrectedSequenceIterator>
CombinedCorrectionResult combineMultipleCorrectionResults1_rawtcs2_iterator(
    TempCorrectedSequenceIterator tmpresultsBegin, 
    TempCorrectedSequenceIterator tmpresultsEnd, 
    ReadWithId& readWithId
){
    const int numTmpResults = std::distance(tmpresultsBegin, tmpresultsEnd);
    if(numTmpResults == 0){
        CombinedCorrectionResult result;
        result.corrected = false;        
        return result;
    }

    #ifndef NDEBUG
    const bool sameId = std::all_of(
        tmpresultsBegin,
        tmpresultsEnd,
        [&](const auto& tcs){
            return tcs.readId == readWithId.globalReadId;
        }
    );
    assert(sameId);
    #endif

    constexpr bool outputHQ = true;
    constexpr bool outputLQWithCandidates = true;
    constexpr bool outputLQOnlyAnchor = true;
    // constexpr bool outputOnlyCand = false;

    // auto isValidSequence = [](const std::string& s){
    //     return std::all_of(s.begin(), s.end(), [](char c){
    //         return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
    //     });
    // };

    auto isAnchor = [](const auto& tcs){
        return tcs.type == TempCorrectedSequenceType::Anchor;
    };

    auto anchorIter = std::find_if(tmpresultsBegin, tmpresultsEnd, isAnchor);

    if(anchorIter != tmpresultsEnd){
        //if there is a correction using a high quality alignment, use it
        if(anchorIter->hq){
            if(outputHQ){
                if(anchorIter->useEdits){
                    // for(const auto& edit : anchorIter->edits){
                    //     readWithId.read.sequence[edit.pos()] = edit.base();
                    // }
                    for(int i = 0; i < anchorIter->numEdits; i++){
                        readWithId.read.sequence[anchorIter->editPositions[i]] = anchorIter->editChars[i];
                    }
                }else{
                    std::swap(readWithId.read.sequence, anchorIter->sequence);
                }

                CombinedCorrectionResult result;

                result.corrected = true;
                result.hqCorrection = true;
                
                return result;
            }else{
                CombinedCorrectionResult result;

                result.corrected = false;
                return result;
            }
        }else{

            auto isEqualSequenceOneHasEdits = [&](const auto& tcswithedit, const auto& tcs2){
                assert(tcswithedit.useEdits);
                assert(!tcs2.useEdits);

                const int len1 = readWithId.read.sequence.length();

                int editIndex = 0;
                int pos = 0;
                while(pos != len1 && editIndex != tcswithedit.numEdits){
                    if(pos != tcswithedit.editPositions[editIndex]){
                        if(readWithId.read.sequence[pos] != tcs2.sequence[pos]){
                            return false;
                        }
                    }else{
                        if(tcswithedit.editChars[editIndex] != tcs2.sequence[pos]){
                            return false;
                        }
                        ++editIndex;
                    }

                    pos++;
                }
                while(pos != len1){
                    if(readWithId.read.sequence[pos] != tcs2.sequence[pos]){
                        return false;
                    }
                    pos++;
                }
                return true;
            };

            auto hasEqualEdits = [&](const auto& tcs1, const auto& tcs2){
                if(tcs1.numEdits != tcs2.numEdits) return false;
                for(int i = 0; i < tcs1.numEdits; i++){
                    if(tcs1.editPositions[i] != tcs2.editPositions[i]) return false;
                }
                for(int i = 0; i < tcs1.numEdits; i++){
                    if(tcs1.editChars[i] != tcs2.editChars[i]) return false;
                }
                return true;
            };

            auto isEqualSequence = [&](const auto& tcs1, const auto& tcs2){
                
                const int len1 = tcs1.useEdits ? readWithId.read.sequence.length() : tcs1.sequence.length();
                const int len2 = tcs2.useEdits ? readWithId.read.sequence.length() : tcs2.sequence.length();
                if(len1 == len2){
                    if(tcs1.useEdits && tcs2.useEdits){
                        return hasEqualEdits(tcs1, tcs2);
                    }else if(!tcs1.useEdits && !tcs2.useEdits){
                        return tcs1.sequence == tcs2.sequence;
                    }else if(tcs1.useEdits && !tcs2.useEdits){
                        if(tcs1.numEdits == 0){
                            return readWithId.read.sequence == tcs2.sequence;
                        }else{
                            return isEqualSequenceOneHasEdits(tcs1, tcs2);
                        }
                    }else{
                        //if(!tcs1.useEdits && tcs2.useEdits)
                        if(tcs2.numEdits == 0){
                            return readWithId.read.sequence == tcs1.sequence;
                        }else{
                            return isEqualSequenceOneHasEdits(tcs2, tcs1);
                        }
                    }
                }else{
                    return false;
                }
            };

            if(numTmpResults >= 3){

                const bool sizelimitok = true;

                const bool sameCorrections = std::all_of(tmpresultsBegin+1,
                                                        tmpresultsEnd,
                                                        [&](const auto& tcs){
                                                            return isEqualSequence(*tmpresultsBegin, tcs);
                                                        });     

                if(sameCorrections && sizelimitok){
                    if(outputLQWithCandidates){
                        CombinedCorrectionResult result;
                        result.corrected = true;
                        result.lqCorrectionWithCandidates = true;

                        if(tmpresultsBegin->useEdits){
                            for(int i = 0; i < tmpresultsBegin->numEdits; i++){
                                readWithId.read.sequence[tmpresultsBegin->editPositions[i]] = tmpresultsBegin->editChars[i];
                            }
                            // for(const auto& edit : tmpresultsBegin->edits){
                            //     readWithId.read.sequence[edit.pos()] = edit.base();
                            // }
                        }else{
                            std::swap(readWithId.read.sequence, tmpresultsBegin->sequence);
                        }
                        
                        return result;
                    }else{
                        CombinedCorrectionResult result;
                        result.corrected = false;
                        return result;
                    }
                }else{
                    CombinedCorrectionResult result;
                    result.corrected = false;
                    return result;
                }
            }else{
                if(outputLQOnlyAnchor){
                    if(anchorIter->useEdits){
                        // for(const auto& edit : anchorIter->edits){
                        //     readWithId.read.sequence[edit.pos()] = edit.base();
                        // }
                        for(int i = 0; i < anchorIter->numEdits; i++){
                            readWithId.read.sequence[anchorIter->editPositions[i]] = anchorIter->editChars[i];
                        }
                    }else{
                        std::swap(readWithId.read.sequence, anchorIter->sequence);
                    }
                    
                    CombinedCorrectionResult result;
                    result.corrected = true;
                    result.lqCorrectionOnlyAnchor = true;
                    return result;
                }else{
                    CombinedCorrectionResult result;
                    result.corrected = false;
                    return result;
                }
            }
        }
    }else{

        CombinedCorrectionResult result;
        result.corrected = false;
        return result;
    }

}



void constructOutputFileFromCorrectionResults(
    const std::vector<std::string>& originalReadFiles,
    SerializedObjectStorage& partialResults, 
    FileFormat outputFormat,
    const std::vector<std::string>& outputfiles,
    bool showProgress,
    const ProgramOptions& programOptions
){

    auto addProgress = [total = 0ull, showProgress](auto i) mutable {
        if(showProgress){
            total += i;

            printf("Written %10llu reads\r", total);

            std::cout.flush();
        }
    };

    using Iterator = std::vector<TempCorrectedSequence>::iterator;

    mergeSerializedResultsWithOriginalReads_multithreaded<TempCorrectedSequence>(
        originalReadFiles,
        partialResults, 
        outputFormat,
        outputfiles,
        //combineMultipleCorrectionResults1_rawtcs2,
        //combineMultipleCorrectionResults1_rawtcs2_iterator<TempCorrectedSequence*>,
        combineMultipleCorrectionResults1_rawtcs2_iterator<Iterator>,
        addProgress,
        programOptions.outputCorrectionQualityLabels,
        programOptions.pairType
    );

    if(showProgress){
        std::cout << "\n";
    }
}



}