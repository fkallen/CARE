#ifndef CARE_RESULT_OUTPUT_PROCESSING_HPP
#define CARE_RESULT_OUTPUT_PROCESSING_HPP

#include <config.hpp>
#include <hpc_helpers.cuh>
#include <memoryfile.hpp>
#include <readlibraryio.hpp>
#include <threadpool.hpp>
#include <concurrencyhelpers.hpp>
#include <util.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <future>
#include <functional>

namespace care{

    template<class ResultType, class MemoryFile_t, class Combiner, class ReadIdComparator>
    void mergeResultsWithOriginalReads_multithreaded(
        const std::string& tempdir,
        const std::vector<std::string>& originalReadFiles,
        MemoryFile_t& partialResults, 
        std::size_t memoryForSorting,
        FileFormat outputFormat,
        const std::vector<std::string>& outputfiles,
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

        // auto showProgress = [&](std::int64_t totalCount, int seconds){
        //     if(true /*runtimeOptions.showProgress*/){

        //         int hours = seconds / 3600;
        //         seconds = seconds % 3600;
        //         int minutes = seconds / 60;
        //         seconds = seconds % 60;
                
        //         printf("Written %10lu reads to output file(Runtime: %03d:%02d:%02d)\r",
        //             totalCount, hours, minutes, seconds);

        //         std::fflush(stdout);
        //     }
        // };

        // auto updateShowProgressInterval = [](auto duration){
        //     return duration;
        // };

        // ProgressThread<std::int64_t> progressThread(std::numeric_limits<std::int64_t>::max(), showProgress, updateShowProgressInterval);

        auto outputWriterFuture = std::async(std::launch::async,
            [&](){
                //no gz output
                auto format = outputFormat;

                if(format == FileFormat::FASTQGZ)
                    format = FileFormat::FASTQ;
                if(format == FileFormat::FASTAGZ)
                    format = FileFormat::FASTA;

                std::vector<std::unique_ptr<SequenceFileWriter>> writerVector;


                for(const auto& outputfile : outputfiles){
                    writerVector.emplace_back(makeSequenceWriter(outputfile, format));
                }

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

                        writerVector[readWithId.fileId]->writeRead(readWithId.read);

                        processed++;
                    }

                    // if(processed == valid){
                    //     progressThread.addProgress(valid);
                    // }

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


    template<class ResultType, class MemoryFile_t, class Combiner>
    void mergeResultsWithOriginalReads_multithreaded(
        const std::string& tempdir,
        const std::vector<std::string>& originalReadFiles,
        MemoryFile_t& partialResults, 
        std::size_t memoryForSorting,
        FileFormat outputFormat,
        const std::vector<std::string>& outputfiles,
        bool isSorted,
        Combiner combineResultsWithRead /* combineResultsWithRead(std::vector<ResultType>& in, ReadWithId& in_out) */
    ){
        std::less<read_number> origIdResultIdLessThan{};

        mergeResultsWithOriginalReads_multithreaded<ResultType>(
            tempdir,
            originalReadFiles,
            partialResults,
            memoryForSorting,
            outputFormat,
            outputfiles,
            isSorted,
            combineResultsWithRead,
            origIdResultIdLessThan
        );
    }



}




#endif