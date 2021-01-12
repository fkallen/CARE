#ifndef CARE_READSTORAGECONSTRUCTION2_HPP 
#define CARE_READSTORAGECONSTRUCTION2_HPP

#include <util.hpp>
#include <config.hpp>
#include <threadpool.hpp>
#include <readlibraryio.hpp>
#include <sequencehelpers.hpp>
#include <dynamic2darray.hpp>
#include <concurrencyhelpers.hpp>
#include <lengthstorage.hpp>
#include <chunkedreadstorage.hpp>
#include <options.hpp>

#include <vector>
#include <array>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <string>
#include <iostream>
#include <future>
#include <limits>
#include <map>
#include <set>

namespace care{





ChunkedReadStorage constructReadStorageFromFiles2(
    RuntimeOptions runtimeOptions,
    MemoryOptions memoryOptions,
    const std::vector<std::string>& inputfiles,
    bool useQualityScores
){
    const bool showProgress = runtimeOptions.showProgress;

    auto showProgressFunc = [showProgress](auto totalCount, auto seconds){
        if(showProgress){
            std::cout << "Processed " << totalCount << " reads in file. Elapsed time: " 
                            << seconds << " seconds." << std::endl;
        }
    };

    auto updateShowProgressInterval = [](auto duration){
        return duration * 2;
    };

    ProgressThread<std::size_t> progressThread(
        std::numeric_limits<std::size_t>::max(), 
        showProgressFunc, 
        updateShowProgressInterval
    );
            

    auto preprocessSequence = [&](std::string& sequence, int& Ncount){

        auto isValidBase = [](char c){
            constexpr std::array<char, 10> validBases{'A','C','G','T','a','c','g','t'};
            return validBases.end() != std::find(validBases.begin(), validBases.end(), c);
        };

        const int numundeterminedBasesInSequence = std::count_if(sequence.begin(), sequence.end(), [&](char c){
            return !isValidBase(c);
        });

        constexpr std::array<char, 4> bases = {'A', 'C', 'G', 'T'};

        for(auto& c : sequence){
            if(c == 'a') c = 'A';
            else if(c == 'c') c = 'C';
            else if(c == 'g') c = 'G';
            else if(c == 't') c = 'T';
            else if(!isValidBase(c)){
                c = bases[Ncount];
                Ncount = (Ncount + 1) % 4;
            }
        }

        return numundeterminedBasesInSequence > 0;
    };

    struct SequenceBatchFromFile{
        int fileId = 0;
        int batchId = 0;
        int validItems = 0;
        std::vector<read_number> readIds{};
        std::vector<std::string> sequences{};
    };

    struct QualityBatchFromFile{
        int fileId = 0;
        int batchId = 0;
        int validItems = 0;
        std::vector<std::string> qualities{};
    };

   
    SimpleConcurrentQueue<SequenceBatchFromFile*> freeSequenceBatchFromFile;
    SimpleConcurrentQueue<SequenceBatchFromFile*> unprocessedSequenceBatchFromFile;
    SimpleConcurrentQueue<QualityBatchFromFile*> freeQualityBatchFromFile;
    SimpleConcurrentQueue<QualityBatchFromFile*> unprocessedQualityBatchFromFile;

    constexpr int fileParserMaxBatchsize = 65535;
    //constexpr int fileParserMaxBatchsize = 250000;

    auto fileParserThreadFunction = [&](const std::string& filename, int fileId){
        int batchId = 0;

        SequenceBatchFromFile* sbatch = nullptr;
        QualityBatchFromFile* qbatch = nullptr;

        auto initSbatch = [&](){
            sbatch = freeSequenceBatchFromFile.pop();
            sbatch->fileId = fileId;
            sbatch->batchId = batchId;
            sbatch->sequences.resize(fileParserMaxBatchsize);
            sbatch->readIds.resize(fileParserMaxBatchsize);
            sbatch->validItems = 0;
        };

        auto initQbatch = [&](){
            qbatch = freeQualityBatchFromFile.pop();
            qbatch->fileId = fileId;
            qbatch->batchId = batchId;
            qbatch->qualities.resize(fileParserMaxBatchsize);
            qbatch->validItems = 0;
        };

        initSbatch();
        if(useQualityScores){
            initQbatch();
        }

        batchId++;

        std::size_t totalNumberOfReads = 0;

        forEachReadInFile(
            filename,
            [&](auto readnum, auto& read){

                std::swap(sbatch->sequences[sbatch->validItems], read.sequence);
                sbatch->readIds[sbatch->validItems] = readnum;
                sbatch->validItems++;

                if(useQualityScores){
                    std::swap(qbatch->qualities[qbatch->validItems], read.quality);
                    qbatch->validItems++;
                }

                if(sbatch->validItems >= fileParserMaxBatchsize){
                    unprocessedSequenceBatchFromFile.push(sbatch);

                    initSbatch();

                    if(useQualityScores){
                        unprocessedQualityBatchFromFile.push(qbatch);
                        initQbatch();
                    }

                    batchId++;
                }

                totalNumberOfReads++;
            }
        );        

        sbatch->readIds.resize(sbatch->validItems);
        sbatch->sequences.resize(sbatch->validItems);
        unprocessedSequenceBatchFromFile.push(sbatch);

        if(useQualityScores){
            qbatch->qualities.resize(sbatch->validItems);
            unprocessedQualityBatchFromFile.push(qbatch);
        }

        return totalNumberOfReads;
    };


    struct EncodedSequenceBatch{
        int fileId = 0;
        int batchId = 0;
        int validItems = 0;
        std::size_t encodedSequencePitchInInts = 0;
        std::vector<int> sequenceLengths{};
        std::vector<unsigned int> encodedSequences{};
        std::vector<read_number> ambiguousReadIds{};
    };

    SimpleConcurrentQueue<EncodedSequenceBatch*> freeEncodedSequenceBatches;
    SimpleConcurrentQueue<EncodedSequenceBatch*> unprocessedEncodedSequenceBatches;

    auto sequenceEncoderThreadFunction = [&](){
        SequenceBatchFromFile* sbatch = unprocessedSequenceBatchFromFile.pop();
        EncodedSequenceBatch* encbatch = nullptr;

        auto initEncBatch = [&](auto pitchInInts){
            encbatch = freeEncodedSequenceBatches.pop();
            assert(encbatch != nullptr);

            encbatch->fileId = sbatch->fileId;
            encbatch->batchId = sbatch->batchId;
            encbatch->validItems = sbatch->validItems;
            encbatch->sequenceLengths.resize(encbatch->validItems);
            encbatch->encodedSequences.resize(encbatch->validItems * pitchInInts);
            encbatch->encodedSequencePitchInInts = pitchInInts;
            encbatch->ambiguousReadIds.clear();
        };

        while(sbatch != nullptr){
            int maxLength = 0;
            int Ncount = 0;

            for(int i = 0; i < sbatch->validItems; i++){
                const int length = sbatch->sequences[i].length();
                maxLength = std::max(maxLength, length);
            }

            const std::size_t pitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maxLength);

            initEncBatch(pitchInInts);

            for(int i = 0; i < sbatch->validItems; i++){
                const int length = sbatch->sequences[i].length();
                encbatch->sequenceLengths[i] = length;

                bool isAmbig = preprocessSequence(sbatch->sequences[i], Ncount);
                if(isAmbig){
                    const read_number readId = sbatch->readIds[i];
                    encbatch->ambiguousReadIds.emplace_back(readId);
                }

                SequenceHelpers::encodeSequence2Bit(
                    encbatch->encodedSequences.data() + i * pitchInInts,
                    sbatch->sequences[i].c_str(),
                    length
                );
                
                encbatch->sequenceLengths[i] = length;
            }

            freeSequenceBatchFromFile.push(sbatch);
            unprocessedEncodedSequenceBatches.push(encbatch);           

            sbatch = unprocessedSequenceBatchFromFile.pop();
        }
    };


    struct EncodedQualityBatch{
        int fileId = 0;
        int batchId = 0;
        int validItems = 0;
        std::size_t qualityPitchInBytes = 0;
        std::vector<char> qualities{};
    };


    SimpleConcurrentQueue<EncodedQualityBatch*> freeEncodedQualityBatches;
    SimpleConcurrentQueue<EncodedQualityBatch*> unprocessedEncodedQualityBatches;

    auto qualityEncoderThreadFunction = [&](){
        QualityBatchFromFile* qbatch = unprocessedQualityBatchFromFile.pop();
        EncodedQualityBatch* encbatch = nullptr;

        auto initEncBatch = [&](auto pitchInBytes){
            encbatch = freeEncodedQualityBatches.pop();
            assert(encbatch != nullptr);

            encbatch->fileId = qbatch->fileId;
            encbatch->batchId = qbatch->batchId;
            encbatch->validItems = qbatch->validItems;
            encbatch->qualities.resize(encbatch->validItems * pitchInBytes);
            encbatch->qualityPitchInBytes = pitchInBytes;
        };

        while(qbatch != nullptr){
            int maxLength = 0;
            for(int i = 0; i < qbatch->validItems; i++){
                const int length = qbatch->qualities[i].length();
                maxLength = std::max(maxLength, length);
            }

            const std::size_t pitchInBytes = maxLength;

            initEncBatch(pitchInBytes);

            for(int i = 0; i < qbatch->validItems; i++){
                std::copy(
                    qbatch->qualities[i].begin(),
                    qbatch->qualities[i].end(),
                    encbatch->qualities.data() + i * pitchInBytes
                );
            }

            freeQualityBatchFromFile.push(qbatch);
            unprocessedEncodedQualityBatches.push(encbatch);         

            qbatch = unprocessedQualityBatchFromFile.pop();
        }

    };


    std::vector<ChunkedReadStorage::StoredEncodedSequences> sequenceStorage;
    std::vector<ChunkedReadStorage::StoredSequenceLengths> lengthStorage;
    std::vector<ChunkedReadStorage::StoredQualities> qualityStorage;
    std::vector<ChunkedReadStorage::StoredAmbigIds> ambigStorage;

    auto inserterThreadFunction = [&](){
        EncodedSequenceBatch* sbatch = nullptr;
        EncodedQualityBatch* qbatch = nullptr;

        sbatch = unprocessedEncodedSequenceBatches.pop();
        if(useQualityScores){
            qbatch = unprocessedEncodedQualityBatches.pop();
        }

        while(sbatch != nullptr || qbatch != nullptr){
            if(sbatch != nullptr){
                const int numSequences = sbatch->validItems;

                ChunkedReadStorage::StoredEncodedSequences finalSeq;
                finalSeq.fileId = sbatch->fileId;
                finalSeq.batchId = sbatch->batchId;
                finalSeq.encodedSequencePitchInInts = sbatch->encodedSequencePitchInInts;
                finalSeq.encodedSequences = std::move(sbatch->encodedSequences);

                sequenceStorage.emplace_back(std::move(finalSeq));

                ChunkedReadStorage::StoredSequenceLengths finalLength;
                finalLength.fileId = sbatch->fileId;
                finalLength.batchId = sbatch->batchId;
                finalLength.sequenceLengths = std::move(sbatch->sequenceLengths);

                lengthStorage.emplace_back(std::move(finalLength));

                if(sbatch->ambiguousReadIds.size() > 0){
                    ChunkedReadStorage::StoredAmbigIds finalAmbig;
                    finalAmbig.fileId = sbatch->fileId;
                    finalAmbig.batchId = sbatch->batchId;
                    finalAmbig.ids = std::move(sbatch->ambiguousReadIds);

                    ambigStorage.emplace_back(std::move(finalAmbig));
                }

                progressThread.addProgress(numSequences);

                freeEncodedSequenceBatches.push(sbatch);

                
            }

            if(qbatch != nullptr){
                ChunkedReadStorage::StoredQualities finalQ;
                finalQ.fileId = qbatch->fileId;
                finalQ.batchId = qbatch->batchId;
                finalQ.qualityPitchInBytes = qbatch->qualityPitchInBytes;
                finalQ.qualities = std::move(qbatch->qualities);

                qualityStorage.emplace_back(std::move(finalQ));

                freeEncodedQualityBatches.push(qbatch);
            }

            if(sbatch != nullptr){
                sbatch = unprocessedEncodedSequenceBatches.pop();        
            }

            if(qbatch != nullptr){
                if(useQualityScores){
                    qbatch = unprocessedEncodedQualityBatches.pop();
                }
            }
        }
    };



    constexpr int numParsers = 1;
    int numSequenceEncoders = 1;
    int numQualityEncoders = 1;
    int numInserters = 1;

    const int numFilebatches = numParsers + numSequenceEncoders + numQualityEncoders;

    std::vector<SequenceBatchFromFile> sequenceBatchFromFile(numFilebatches);
    std::vector<QualityBatchFromFile> qualityBatchFromFile(numFilebatches);

    for(int i = 0; i < numFilebatches; i++){
        freeSequenceBatchFromFile.push(&sequenceBatchFromFile[i]);
        freeQualityBatchFromFile.push(&qualityBatchFromFile[i]);
    }

    const int numEncodedSequenceBatches = numSequenceEncoders + numInserters;

    std::vector<EncodedSequenceBatch> encodedSequenceBatches(numEncodedSequenceBatches);

    for(int i = 0; i < numEncodedSequenceBatches; i++){
        freeEncodedSequenceBatches.push(&encodedSequenceBatches[i]);
    }

    int numEncodedQualityBatches = numQualityEncoders + numInserters;
    std::vector<EncodedQualityBatch> encodedQualityBatches(numEncodedQualityBatches);

    for(int i = 0; i < numEncodedQualityBatches; i++){
        freeEncodedQualityBatches.push(&encodedQualityBatches[i]);
    }

    std::vector<std::size_t> numReadsPerFile;
    std::vector<std::future<void>> encoderFutures;
    std::vector<std::future<void>> inserterFutures;
    
    for(int i = 0; i < numSequenceEncoders; i++){
        encoderFutures.emplace_back(
            std::async(
                std::launch::async,
                sequenceEncoderThreadFunction
            )
        );
    }

    for(int i = 0; i < numQualityEncoders; i++){
        encoderFutures.emplace_back(
            std::async(
                std::launch::async,
                qualityEncoderThreadFunction
            )
        );
    }

    for(int i = 0; i < numInserters; i++){
        inserterFutures.emplace_back(
            std::async(
                std::launch::async,
                inserterThreadFunction
            )
        );
    }

    for(int i = 0; i < int(inputfiles.size()); i++){
        std::string inputfile = inputfiles[i];

        std::future<std::size_t> future = std::async(
            std::launch::async,
            fileParserThreadFunction,
            std::move(inputfile), i
        );

        std::size_t numReads = future.get();

        numReadsPerFile.emplace_back(numReads);
    }

    //parsing done. flush queues to exit other threads

    for(int i = 0; i < numSequenceEncoders + numQualityEncoders; i++){
        unprocessedSequenceBatchFromFile.push(nullptr);
        unprocessedQualityBatchFromFile.push(nullptr);        
    }

    for(auto& f : encoderFutures){
        f.wait();
    }

    for(int i = 0; i < numInserters; i++){
        unprocessedEncodedSequenceBatches.push(nullptr);
        unprocessedEncodedQualityBatches.push(nullptr);
    }
    
    for(auto& f : inserterFutures){
        f.wait();
    }
    
    progressThread.finished();
    if(showProgress){
        std::cout << "\n";
    }

    // auto lessThanFileAndBatch = [](const auto& l, const auto& r){
    //     if(l.fileId < r.fileId) return true;
    //     if(l.fileId > r.fileId) return false;
    //     return l.batchId < r.batchId;
    // };

    // std::sort(sequenceStorage.begin(), sequenceStorage.end(), lessThanFileAndBatch);
    // std::sort(lengthStorage.begin(), lengthStorage.end(), lessThanFileAndBatch);
    // std::sort(qualityStorage.begin(), qualityStorage.end(), lessThanFileAndBatch);
    // std::sort(ambigStorage.begin(), ambigStorage.end(), lessThanFileAndBatch);

    std::cerr << "numReadsPerFile: ";
    for(auto n : numReadsPerFile){
        std::cerr << n << " ";
    }
    std::cerr << "\n";

    helpers::CpuTimer footimer("footimer");

    ChunkedReadStorage fooStorage(useQualityScores);
    
    fooStorage.init(
        std::move(numReadsPerFile),
        std::move(sequenceStorage),
        std::move(lengthStorage),
        std::move(qualityStorage),
        std::move(ambigStorage),
        memoryOptions.memoryTotalLimit
    );

    footimer.print();

    return fooStorage;
    
}







} //namespace care


#endif
