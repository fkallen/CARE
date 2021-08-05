/*
    include in cpu source file for cpu sorting of memoryfile
    include in gpu source file for cpu / gpu sorting of memoryfile
*/

#include <correctionresultprocessing.hpp>

#include <config.hpp>
#include <hpc_helpers.cuh>
#include <memoryfile.hpp>
#include <readlibraryio.hpp>
#include <threadpool.hpp>
#include <concurrencyhelpers.hpp>

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

template<class ResultType, class MemoryFile_t, class Combiner, class ReadIdComparator, class ProgressFunction>
void mergeResultsWithOriginalReads_multithreaded(
    const std::string& tempdir,
    const std::vector<std::string>& originalReadFiles,
    MemoryFile_t& partialResults, 
    std::size_t memoryForSorting,
    FileFormat outputFormat,
    const std::vector<std::string>& outputfiles,
    bool isSorted,
    Combiner combineResultsWithRead, /* combineResultsWithRead(std::vector<ResultType>& in, ReadWithId& in_out) */
    ReadIdComparator origIdResultIdLessThan,
    ProgressFunction addProgress
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

            read_number previousId = 0;
            std::size_t itemnumber = 0;

            while(partialResultsReader.hasNext()){
                ResultTypeBatch* batch = freeTcsBatches.pop();

                //abegin = std::chrono::system_clock::now();

                batch->items.resize(decoder_maxbatchsize);

                int batchsize = 0;
                while(batchsize < decoder_maxbatchsize && partialResultsReader.hasNext()){
                    batch->items[batchsize] = *(partialResultsReader.next());
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

    auto inputReaderFuture = std::async(std::launch::async,
        [&](){

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

            // std::cout << "# elapsed time ("<< "inputparsing without queues" <<"): " << adelta.count()  << " s" << std::endl;

            // TIMERSTOPCPU(inputparsing);

            noMoreInputreadBatches = true;
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

            std::vector<std::unique_ptr<SequenceFileWriter>> writerVector;

            assert(originalReadFiles.size() == outputfiles.size() || outputfiles.size() == 1);

            for(const auto& outputfile : outputfiles){
                writerVector.emplace_back(makeSequenceWriter(outputfile, format));
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

                outputBatch = unprocessedOutputreadBatches.popOrDefault(
                    [&](){
                        return !noMoreOutputreadBatches;  //its possible that there are no corrections. In that case, return nullptr
                    },
                    nullptr
                );            
            }

            // std::cout << "# elapsed time ("<< "outputwriting without queues" <<"): " << adelta.count()  << " s" << std::endl;

            // TIMERSTOPCPU(outputwriting);
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
                    
                    buffer.clear();
                    while(first2 != last2 && !(origIdResultIdLessThan(first1->globalReadId, first2->readId))){
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

                    combineResultsWithRead(buffer, readWithId);  

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
    // progressThread.finished();

    // std::cout << "\n";

    mergetimer.print();
}



struct CombinedCorrectionResult{
    bool corrected = false;
    bool hqCorrection = false;
    bool lqCorrectionOnlyAnchor = false;
    bool lqCorrectionWithCandidates = false;
    std::string correctedSequence{};
};

CombinedCorrectionResult combineMultipleCorrectionResults1(
    std::vector<TempCorrectedSequence>& tmpresults, 
    std::string& originalSequence
){
    if(tmpresults.empty()){
        CombinedCorrectionResult result;
        result.corrected = false;
        std::swap(result.correctedSequence, originalSequence);
        
        return result;
    }

    constexpr bool outputHQ = true;
    constexpr bool outputLQWithCandidates = true;
    constexpr bool outputLQOnlyAnchor = true;
    // constexpr bool outputOnlyCand = false;

    

    auto isAnchor = [](const auto& tcs){
        return tcs.type == TempCorrectedSequenceType::Anchor;
    };

    auto anchorIter = std::find_if(tmpresults.begin(), tmpresults.end(), isAnchor);

    if(anchorIter != tmpresults.end()){
        //if there is a correction using a high quality alignment, use it
        if(anchorIter->hq){
            if(outputHQ){

                assert(anchorIter->sequence.size() == originalSequence.size());
                CombinedCorrectionResult result;

                result.corrected = true;
                result.hqCorrection = true;
                std::swap(result.correctedSequence, anchorIter->sequence);
                
                return result;
            }else{
                CombinedCorrectionResult result;

                result.corrected = false;
                result.correctedSequence = originalSequence;
                return result;
            }
        }else{

            TempCorrectedSequence& anchor = *anchorIter;

            if(tmpresults.size() >= 3){

                const bool sizelimitok = true; //tmpresults.size() > 3;

                const bool sameCorrections = std::all_of(tmpresults.begin()+1,
                                                        tmpresults.end(),
                                                        [&](const auto& tcs){
                                                            return tmpresults[0].sequence == tcs.sequence;
                                                        });

                if(sameCorrections && sizelimitok){
                    if(outputLQWithCandidates){
                        CombinedCorrectionResult result;
                        result.corrected = true;
                        result.lqCorrectionWithCandidates = true;
                        std::swap(result.correctedSequence, tmpresults[0].sequence);
                        return result;
                    }else{
                        CombinedCorrectionResult result;
                        result.corrected = false;
                        result.correctedSequence = originalSequence;
                        return result;
                    }
                }else{
                    CombinedCorrectionResult result;
                    result.corrected = false;
                    result.correctedSequence = originalSequence;
                    return result;
                }
            }else{
                if(outputLQOnlyAnchor){
                    CombinedCorrectionResult result;
                    result.corrected = true;
                    result.lqCorrectionOnlyAnchor = true;
                    std::swap(result.correctedSequence, anchor.sequence);
                    return result;
                }else{
                    CombinedCorrectionResult result;
                    result.corrected = false;
                    result.correctedSequence = originalSequence;
                    return result;
                }
            }
        }
    }else{

        CombinedCorrectionResult result;
        result.corrected = false;
        result.correctedSequence = originalSequence;
        return result;

        // tmpresults.erase(std::remove_if(tmpresults.begin(),
        //                                 tmpresults.end(),
        //                                 [](const auto& tcs){
        //                                     return std::abs(tcs.shift) > 0;
        //                                 }),
        //                   tmpresults.end());
        //
        // if(tmpresults.size() >= 1){
        //
        //     const bool sameCorrections = std::all_of(tmpresults.begin()+1,
        //                                             tmpresults.end(),
        //                                             [&](const auto& tcs){
        //                                                 return tmpresults[0].sequence == tcs.sequence;
        //                                             });
        //
        //     if(sameCorrections){
        //         return std::make_pair(tmpresults[0].sequence, outputOnlyCand);
        //     }else{
        //         return std::make_pair(std::string{""}, false);
        //     }
        // }else{
        //     return std::make_pair(std::string{""}, false);
        // }

    }

}

//same as combineMultipleCorrectionResults1, but edits have not been applied in the tcs
CombinedCorrectionResult combineMultipleCorrectionResults1_rawtcs(
    std::vector<TempCorrectedSequence>& tmpresults, 
    std::string& originalSequence
){
    if(tmpresults.empty()){
        CombinedCorrectionResult result;
        result.corrected = false;
        std::swap(result.correctedSequence, originalSequence);
        
        return result;
    }

    constexpr bool outputHQ = true;
    constexpr bool outputLQWithCandidates = true;
    constexpr bool outputLQOnlyAnchor = true;
    // constexpr bool outputOnlyCand = false;

    auto isAnchor = [](const auto& tcs){
        return tcs.type == TempCorrectedSequenceType::Anchor;
    };

    auto anchorIter = std::find_if(tmpresults.begin(), tmpresults.end(), isAnchor);

    if(anchorIter != tmpresults.end()){
        //if there is a correction using a high quality alignment, use it
        if(anchorIter->hq){
            if(outputHQ){
                if(anchorIter->useEdits){
                    for(const auto& edit : anchorIter->edits){
                        originalSequence[edit.pos()] = edit.base();
                    }
                }

                //assert(anchorIter->sequence.size() == originalSequence.size());
                CombinedCorrectionResult result;

                result.corrected = true;
                result.hqCorrection = true;
                std::swap(result.correctedSequence, originalSequence);
                
                return result;
            }else{
                CombinedCorrectionResult result;

                result.corrected = false;
                std::swap(result.correctedSequence, originalSequence);
                return result;
            }
        }else{

            if(tmpresults.size() >= 3){

                for(auto& tmpres : tmpresults){
                    if(tmpres.useEdits){
                        tmpres.sequence = originalSequence;
                        for(const auto& edit : tmpres.edits){
                            tmpres.sequence[edit.pos()] = edit.base();
                        }
                    }
                }

                const bool sizelimitok = true; //tmpresults.size() > 3;

                const bool sameCorrections = std::all_of(tmpresults.begin()+1,
                                                        tmpresults.end(),
                                                        [&](const auto& tcs){
                                                            return tmpresults[0].sequence == tcs.sequence;
                                                        });

                if(sameCorrections && sizelimitok){
                    if(outputLQWithCandidates){
                        CombinedCorrectionResult result;
                        result.corrected = true;
                        result.lqCorrectionWithCandidates = true;
                        std::swap(result.correctedSequence, tmpresults[0].sequence);
                        return result;
                    }else{
                        CombinedCorrectionResult result;
                        result.corrected = false;
                        result.correctedSequence = originalSequence;
                        return result;
                    }
                }else{
                    CombinedCorrectionResult result;
                    result.corrected = false;
                    std::swap(result.correctedSequence, originalSequence);
                    return result;
                }
            }else{
                if(outputLQOnlyAnchor){
                    if(anchorIter->useEdits){
                        for(const auto& edit : anchorIter->edits){
                            originalSequence[edit.pos()] = edit.base();
                        }
                    }
                    
                    CombinedCorrectionResult result;
                    result.corrected = true;
                    result.lqCorrectionOnlyAnchor = true;
                    std::swap(result.correctedSequence, originalSequence);
                    return result;
                }else{
                    CombinedCorrectionResult result;
                    result.corrected = false;
                    std::swap(result.correctedSequence, originalSequence);
                    return result;
                }
            }
        }
    }else{

        CombinedCorrectionResult result;
        result.corrected = false;
        std::swap(result.correctedSequence, originalSequence);
        return result;

        // tmpresults.erase(std::remove_if(tmpresults.begin(),
        //                                 tmpresults.end(),
        //                                 [](const auto& tcs){
        //                                     return std::abs(tcs.shift) > 0;
        //                                 }),
        //                   tmpresults.end());
        //
        // if(tmpresults.size() >= 1){
        //
        //     const bool sameCorrections = std::all_of(tmpresults.begin()+1,
        //                                             tmpresults.end(),
        //                                             [&](const auto& tcs){
        //                                                 return tmpresults[0].sequence == tcs.sequence;
        //                                             });
        //
        //     if(sameCorrections){
        //         return std::make_pair(tmpresults[0].sequence, outputOnlyCand);
        //     }else{
        //         return std::make_pair(std::string{""}, false);
        //     }
        // }else{
        //     return std::make_pair(std::string{""}, false);
        // }

    }

}


CombinedCorrectionResult combineMultipleCorrectionResults1_rawtcs2(
    std::vector<TempCorrectedSequence>& tmpresults, 
    ReadWithId& readWithId
){
    if(tmpresults.empty()){
        CombinedCorrectionResult result;
        result.corrected = false;        
        return result;
    }

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

    auto anchorIter = std::find_if(tmpresults.begin(), tmpresults.end(), isAnchor);

    if(anchorIter != tmpresults.end()){
        //if there is a correction using a high quality alignment, use it
        if(anchorIter->hq){
            if(outputHQ){
                if(anchorIter->useEdits){
                    for(const auto& edit : anchorIter->edits){
                        readWithId.read.sequence[edit.pos()] = edit.base();
                    }
                }else{
                    std::swap(readWithId.read.sequence, anchorIter->sequence);
                }

                // if(!isValidSequence(readWithId.read.sequence)){
                //     std::cerr << "Warning. Corrected read " << readWithId.globalReadId
                //             << " with header " << readWithId.read.header
                //             << " does contain an invalid DNA base!\n"
                //             << "Corrected sequence is: "  << readWithId.read.sequence << '\n';
                // }

                //assert(anchorIter->sequence.size() == originalSequence.size());
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

                auto editIter = tcswithedit.edits.begin();
                int pos = 0;
                while(pos != len1 && editIter != tcswithedit.edits.end()){
                    if(pos != editIter->pos()){
                        if(readWithId.read.sequence[pos] != tcs2.sequence[pos]){
                            return false;
                        }
                    }else{
                        if(editIter->base() != tcs2.sequence[pos]){
                            return false;
                        }
                        ++editIter;
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

            auto isEqualSequence = [&](const auto& tcs1, const auto& tcs2){
                
                const int len1 = tcs1.useEdits ? readWithId.read.sequence.length() : tcs1.sequence.length();
                const int len2 = tcs2.useEdits ? readWithId.read.sequence.length() : tcs2.sequence.length();
                if(len1 == len2){
                    if(tcs1.useEdits && tcs2.useEdits){
                        return tcs1.edits == tcs2.edits;
                    }else if(!tcs1.useEdits && !tcs2.useEdits){
                        return tcs1.sequence == tcs2.sequence;
                    }else if(tcs1.useEdits && !tcs2.useEdits){
                        if(tcs1.edits.empty()){
                            return readWithId.read.sequence == tcs2.sequence;
                        }else{
                            return isEqualSequenceOneHasEdits(tcs1, tcs2);
                        }
                    }else{
                        //if(!tcs1.useEdits && tcs2.useEdits)
                        if(tcs2.edits.empty()){
                            return readWithId.read.sequence == tcs1.sequence;
                        }else{
                            return isEqualSequenceOneHasEdits(tcs2, tcs1);
                        }
                    }
                }else{
                    return false;
                }
            };

            if(tmpresults.size() >= 3){

                // for(auto& tmpres : tmpresults){
                //     if(tmpres.useEdits){
                //         tmpres.sequence = readWithId.read.sequence;
                //         for(const auto& edit : tmpres.edits){
                //             tmpres.sequence[edit.pos] = edit.base;
                //         }
                //     }
                // }

                const bool sizelimitok = true; //tmpresults.size() > 3;

                // const bool sameCorrections2 = std::all_of(tmpresults.begin()+1,
                //                                         tmpresults.end(),
                //                                         [&](const auto& tcs){
                //                                             return tmpresults[0].sequence == tcs.sequence;
                //                                         });

                const bool sameCorrections = std::all_of(tmpresults.begin()+1,
                                                        tmpresults.end(),
                                                        [&](const auto& tcs){
                                                            return isEqualSequence(tmpresults[0], tcs);
                                                        });

                //assert(sameCorrections == sameCorrections2);                

                if(sameCorrections && sizelimitok){
                    if(outputLQWithCandidates){
                        CombinedCorrectionResult result;
                        result.corrected = true;
                        result.lqCorrectionWithCandidates = true;

                        if(tmpresults[0].useEdits){
                            for(const auto& edit : tmpresults[0].edits){
                                readWithId.read.sequence[edit.pos()] = edit.base();
                            }
                        }else{
                            std::swap(readWithId.read.sequence, tmpresults[0].sequence);
                        }

                        // if(!isValidSequence(readWithId.read.sequence)){
                        //     std::cerr << "Warning. Corrected read " << readWithId.globalReadId
                        //             << " with header " << readWithId.read.header
                        //             << " does contain an invalid DNA base!\n"
                        //             << "Corrected sequence is: "  << readWithId.read.sequence << '\n';
                        // }

                        
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
                        for(const auto& edit : anchorIter->edits){
                            readWithId.read.sequence[edit.pos()] = edit.base();
                        }
                    }else{
                        std::swap(readWithId.read.sequence, anchorIter->sequence);
                    }

                    // if(!isValidSequence(readWithId.read.sequence)){
                    //     std::cerr << "Warning. Corrected read " << readWithId.globalReadId
                    //             << " with header " << readWithId.read.header
                    //             << " does contain an invalid DNA base!\n"
                    //             << "Corrected sequence is: "  << readWithId.read.sequence << '\n';
                    // }
                    
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

        // tmpresults.erase(std::remove_if(tmpresults.begin(),
        //                                 tmpresults.end(),
        //                                 [](const auto& tcs){
        //                                     return std::abs(tcs.shift) > 0;
        //                                 }),
        //                   tmpresults.end());
        //
        // if(tmpresults.size() >= 1){
        //
        //     const bool sameCorrections = std::all_of(tmpresults.begin()+1,
        //                                             tmpresults.end(),
        //                                             [&](const auto& tcs){
        //                                                 return tmpresults[0].sequence == tcs.sequence;
        //                                             });
        //
        //     if(sameCorrections){
        //         return std::make_pair(tmpresults[0].sequence, outputOnlyCand);
        //     }else{
        //         return std::make_pair(std::string{""}, false);
        //     }
        // }else{
        //     return std::make_pair(std::string{""}, false);
        // }

    }

}



void constructOutputFileFromCorrectionResults(
    const std::string& tempdir,
    const std::vector<std::string>& originalReadFiles,
    MemoryFileFixedSize<EncodedTempCorrectedSequence>& partialResults, 
    std::size_t memoryForSorting,
    FileFormat outputFormat,
    const std::vector<std::string>& outputfiles,
    bool isSorted,
    bool showProgress
){

    std::less<read_number> origIdResultIdLessThan{};

    auto addProgress = [total = 0ull, showProgress](auto i) mutable {
        if(showProgress){
            total += i;

            printf("Written %10llu reads\r", total);

            std::fflush(stdout);
        }
    };

    mergeResultsWithOriginalReads_multithreaded<TempCorrectedSequence>(
        tempdir,
        originalReadFiles,
        partialResults, 
        memoryForSorting,
        outputFormat,
        outputfiles,
        isSorted,
        combineMultipleCorrectionResults1_rawtcs2,
        origIdResultIdLessThan,
        addProgress
    );

    if(showProgress){
        std::cout << "\n";
    }
}




    EncodedTempCorrectedSequence& EncodedTempCorrectedSequence::operator=(const TempCorrectedSequence& rhs){
        rhs.encodeInto(*this);

        return *this;
    }

    bool EncodedTempCorrectedSequence::writeToBinaryStream(std::ostream& os) const{
        //assert(bool(os)); 
        os.write(reinterpret_cast<const char*>(&readId), sizeof(read_number));
        //assert(bool(os));
        os.write(reinterpret_cast<const char*>(&encodedflags), sizeof(std::uint32_t));
        //assert(bool(os));
        const int numBytes = getNumBytes();
        os.write(reinterpret_cast<const char*>(data.get()), sizeof(std::uint8_t) * numBytes);
        //assert(bool(os));
        return bool(os);
    }

    bool EncodedTempCorrectedSequence::readFromBinaryStream(std::istream& is){
        is.read(reinterpret_cast<char*>(&readId), sizeof(read_number));
        is.read(reinterpret_cast<char*>(&encodedflags), sizeof(std::uint32_t));
        const int numBytes = getNumBytes();

        data = std::make_unique<std::uint8_t[]>(numBytes);

        is.read(reinterpret_cast<char*>(data.get()), sizeof(std::uint8_t) * numBytes);

        return bool(is);
    }

    std::uint8_t* EncodedTempCorrectedSequence::copyToContiguousMemory(std::uint8_t* ptr, std::uint8_t* endPtr) const{
        const int dataBytes = getNumBytes();

        const std::size_t availableBytes = std::distance(ptr, endPtr);
        const std::size_t requiredBytes = sizeof(read_number) + sizeof(std::uint32_t) + dataBytes;
        if(requiredBytes <= availableBytes){
            std::memcpy(ptr, &readId, sizeof(read_number));
            ptr += sizeof(read_number);
            std::memcpy(ptr, &encodedflags, sizeof(std::uint32_t));
            ptr += sizeof(std::uint32_t);
            std::memcpy(ptr, data.get(), dataBytes);
            ptr += dataBytes;
            return ptr;
        }else{
            return nullptr;
        }        
    }

    void EncodedTempCorrectedSequence::copyFromContiguousMemory(const std::uint8_t* ptr){
        std::memcpy(&readId, ptr, sizeof(read_number));
        ptr += sizeof(read_number);
        std::memcpy(&encodedflags, ptr, sizeof(std::uint32_t));
        ptr += sizeof(read_number);

        const int numBytes = getNumBytes();
        data = std::make_unique<std::uint8_t[]>(numBytes);

        std::memcpy(data.get(), ptr, numBytes);
        //ptr += numBytes;
    }

    void encodeDataIntoEncodedCorrectedSequence(
        EncodedTempCorrectedSequence& target,
        read_number readId,
        bool hq,
        bool useEdits,
        TempCorrectedSequenceType type,
        int shift,
        int numEdits,
        const TempCorrectedSequence::Edit* edits,
        int sequenceLength,
        const char* sequence
    ){
        const std::uint32_t oldNumBytes = target.getNumBytes(); 

        target.readId = readId;

        target.encodedflags = (std::uint32_t(hq) << 31);
        target.encodedflags |= (std::uint32_t(useEdits) << 30);
        target.encodedflags |= (std::uint32_t(int(type)) << 29);

        constexpr std::uint32_t maxNumBytes = (std::uint32_t(1) << 29)-1;

        std::uint32_t numBytes = 0;
        if(useEdits){
            numBytes += sizeof(int);
            numBytes += numEdits * (sizeof(int) + sizeof(char));
        }else{
            numBytes += sizeof(int);
            numBytes += sizeof(char) * sequenceLength;
        }

        if(type == TempCorrectedSequenceType::Anchor){
            ; //nothing
        }else{
            //candidate shift
            numBytes += sizeof(int);
        }

        assert(numBytes <= maxNumBytes);
        target.encodedflags |= numBytes;

        if(numBytes > oldNumBytes){
            target.data = std::make_unique<std::uint8_t[]>(numBytes);
        }else{
            ; //reuse buffer
        }

        //fill buffer

        std::uint8_t* ptr = target.data.get();

        if(useEdits){
            std::memcpy(ptr, &numEdits, sizeof(int));
            ptr += sizeof(int);
            for(int i = 0; i < numEdits; i++){
                const auto& edit = edits[i];
                const int p = edit.pos();
                std::memcpy(ptr, &p, sizeof(int));
                ptr += sizeof(int);
            }
            for(int i = 0; i < numEdits; i++){
                const auto& edit = edits[i];
                const char c = edit.base();
                std::memcpy(ptr, &c, sizeof(char));
                ptr += sizeof(char);
            }
        }else{
            std::memcpy(ptr, &sequenceLength, sizeof(int));
            ptr += sizeof(int);
            std::memcpy(ptr, sequence, sizeof(char) * sequenceLength);
            ptr += sizeof(char) * sequenceLength;
        }

        if(type == TempCorrectedSequenceType::Anchor){
            ; //nothing
        }else{
            std::memcpy(ptr, &shift, sizeof(int));
            ptr += sizeof(int);
        }
    }


    TempCorrectedSequence::TempCorrectedSequence(const EncodedTempCorrectedSequence& encoded){
        decode(encoded);
    }

    TempCorrectedSequence& TempCorrectedSequence::operator=(const EncodedTempCorrectedSequence& encoded){
        decode(encoded);
        return *this;
    }

    void TempCorrectedSequence::encodeInto(EncodedTempCorrectedSequence& target) const{
        encodeDataIntoEncodedCorrectedSequence(
            target,
            readId,
            hq,
            useEdits,
            type,
            shift,
            edits.size(),
            edits.data(),
            sequence.size(),
            sequence.data()
        );
    }

    EncodedTempCorrectedSequence TempCorrectedSequence::encode() const{
        EncodedTempCorrectedSequence encoded;
        encodeInto(encoded);

        return encoded;
    }

    void TempCorrectedSequence::decode(const EncodedTempCorrectedSequence& encoded){

        readId = encoded.getReadId();

        hq = encoded.isHQ();
        useEdits = encoded.useEdits();
        type = encoded.getType();

        const std::uint8_t* ptr = encoded.data.get();
    

        if(useEdits){
            int size;
            std::memcpy(&size, ptr, sizeof(int));
            ptr += sizeof(int);

            edits.resize(size);

            for(auto& edit : edits){
                int p;
                std::memcpy(&p, ptr, sizeof(int));
                edit.pos(p);
                ptr += sizeof(int);
            }
            for(auto& edit : edits){
                char c;
                std::memcpy(&c, ptr, sizeof(char));
                edit.base(c);
                ptr += sizeof(char);
            }
        }else{
            int length;
            std::memcpy(&length, ptr, sizeof(int));
            ptr += sizeof(int);

            sequence.resize(length);
            sequence.replace(0, length, (const char*)ptr, length);

            ptr += sizeof(char) * length;
        }

        if(type == TempCorrectedSequenceType::Anchor){
            ; //nothing
        }else{
            std::memcpy(&shift, ptr, sizeof(int));
            ptr += sizeof(int);
        }
    }

    bool TempCorrectedSequence::writeToBinaryStream(std::ostream& os) const{
        os.write(reinterpret_cast<const char*>(&readId), sizeof(read_number));
        
        std::uint8_t data = bool(hq);
        data = (data << 1) | bool(useEdits);
        data = (data << 6) | std::uint8_t(int(type));

        os.write(reinterpret_cast<const char*>(&data), sizeof(std::uint8_t));

        if(useEdits){
            os << edits.size() << ' ';
            for(const auto& edit : edits){
                os << edit.pos() << ' ';
            }
            for(const auto& edit : edits){
                os << edit.base();
            }
            if(edits.size() > 0){
                os << ' ';
            }
        }else{
            os << sequence << ' ';
        }

        if(type == TempCorrectedSequenceType::Anchor){
            ; // nothing
        }else{
            os << shift;
        }

        return bool(os);
    }

    bool TempCorrectedSequence::readFromBinaryStream(std::istream& is){
        std::uint8_t data = 0;

        is.read(reinterpret_cast<char*>(&readId), sizeof(read_number));
        is.read(reinterpret_cast<char*>(&data), sizeof(std::uint8_t));

        std::string line;
        if(std::getline(is, line)){
            std::stringstream sstream(line);
            auto& stream = sstream;

            hq = (data >> 7) & 1;
            useEdits = (data >> 6) & 1;
            type = TempCorrectedSequenceType(int(data & 0x3F));

            if(useEdits){
                size_t size;
                stream >> size;
                int numEdits = size;
                edits.resize(size);
                for(int i = 0; i < numEdits; i++){
                    int p;
                    stream >> p;
                    edits[i].pos(p);
                }
                for(int i = 0; i < numEdits; i++){
                    char c;
                    stream >> c;
                    edits[i].base(c);
                }
            }else{
                stream >> sequence;
            }

            if(type == TempCorrectedSequenceType::Anchor){
                ; //nothing
            }else{
                stream >> shift;
                shift = std::abs(shift);
            }
        }

        return bool(is);
    }

    



    std::ostream& operator<<(std::ostream& os, const TempCorrectedSequence& tmp){
        //tmp.writeToBinaryStream(os);
        os << "readid = " << tmp.readId << ", type = " << int(tmp.type) << ", hq = " << tmp.hq 
            << ", useEdits = " << tmp.useEdits << ", numEdits = " << tmp.edits.size();
        if(tmp.edits.size() > 0){
            for(const auto& edit : tmp.edits){
                os << " , (" << edit.pos() << "," << edit.base() << ")";
            }
        }

        return os;
    }

    std::istream& operator>>(std::istream& is, TempCorrectedSequence& tmp){
        tmp.readFromBinaryStream(is);
        return is;
    }




}