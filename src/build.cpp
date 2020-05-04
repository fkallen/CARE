#include <build.hpp>

#include <config.hpp>

#include <util.hpp>
#include <options.hpp>
#include <filehelpers.hpp>
#include <minhasher.hpp>
#include <minhasher_transform.hpp>
#include "readstorage.hpp"
#include <readlibraryio.hpp>
#include "sequence.hpp"
#include "threadsafe_buffer.hpp"

#include "threadpool.hpp"

#include <stdexcept>
#include <iostream>
#include <limits>
#include <thread>
#include <future>
#include <mutex>
#include <iterator>
#include <random>
#include <omp.h>



// #define VALIDATE_READSTORAGE
// #define VALIDATE_MINHASHER

namespace care{

    void validateReadstorage(const cpu::ContiguousReadStorage& readStorage, const FileOptions& fileOptions){
        std::cerr << "validating data in readstorage\n";

        std::vector<read_number> indicesBuffer;
        std::vector<Read> readsBuffer;

        constexpr int batchsize = 16000;

        const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();
        const int sequencePitchInInts = getEncodedNumInts2Bit(maximumSequenceLength);
        const int qualityPitchInBytes = (SDIV(maximumSequenceLength, 32) * 32);

        ThreadPool threadPool;
        cpu::ContiguousReadStorage::GatherHandle readStorageGatherHandle;

        std::vector<char> h_qualities(qualityPitchInBytes * batchsize);
        std::vector<read_number> h_readids(batchsize);
        std::vector<unsigned int> h_sequences(sequencePitchInInts * batchsize);
        std::vector<int> h_lengths(batchsize);

        bool oneIter = true;
        const bool withQuality = readStorage.canUseQualityScores();

        auto isValidBase = [](char c){
            constexpr std::array<char, 10> validBases{'A','C','G','T','a','c','g','t'};
            return validBases.end() != std::find(validBases.begin(), validBases.end(), c);
        };

        auto validateBatch = [&](){

            std::copy(indicesBuffer.begin(), indicesBuffer.end(), h_readids.data());

            readStorage.gatherSequenceLengths(
                readStorageGatherHandle,
                h_readids.data(),
                indicesBuffer.size(),
                h_lengths.data()
            );

            readStorage.gatherSequenceData(
                readStorageGatherHandle,
                h_readids.data(),
                indicesBuffer.size(),
                h_sequences.data(),
                sequencePitchInInts
            );

            if(withQuality) { 

                readStorage.gatherSequenceQualities(
                    readStorageGatherHandle,
                    h_readids.data(),
                    indicesBuffer.size(),
                    h_qualities.data(),
                    qualityPitchInBytes
                );

            }

            auto func = [&](int begin, int end, int /*threadId*/){
                for(int i = begin; i < end; i++){
                    // std::cerr << indicesBuffer[i] << "\n";
                    // std::cerr << "expected " << readsBuffer[i].quality << "\n";
                    // std::cerr << "got      ";
                    //     for(int l = 0; l < readsBuffer[i].quality.size(); l++){
                    //         std::cerr << h_qualities[qualityPitchInBytes * i + l];
                    //     }
                    //     std::cerr << "\n";

                    bool ok = true;
                    const int sequenceLength = readsBuffer[i].sequence.length();

                    if(sequenceLength != h_lengths[i]){
                        ok = false;
                        std::cerr << "length error at sequence read " << indicesBuffer[i] << "\n";
                        std::cerr << "expected " << sequenceLength << "\n";
                        std::cerr << "got      " << h_lengths[i] << "\n";
                    } 
    
                    const std::string seqstring = get2BitString(h_sequences.data() + i * sequencePitchInInts, sequenceLength);
    
                    for(int k = 0; k < sequenceLength && ok; k++){  

                        if(isValidBase(readsBuffer[i].sequence[k]) && readsBuffer[i].sequence[k] != seqstring[k]){
                            ok = false;
                            std::cerr << "error at sequence read " << indicesBuffer[i] << " position " << k << "\n";
                            std::cerr << "expected " << readsBuffer[i].sequence << "\n";
                            std::cerr << "got      " << seqstring << "\n";
                        }
                    }
                    if(withQuality){
                        const int qualityLength = readsBuffer[i].quality.length();
                        ok = true;
                        for(int k = 0; k < qualityLength && ok; k++){    
                            if(readsBuffer[i].quality[k] != h_qualities[qualityPitchInBytes * i + k]){
                                ok = false;
                                std::cerr << "error at quality read " << indicesBuffer[i] << " position " << k << "\n";
                                std::cerr << "expected " << readsBuffer[i].quality << "\n";
                                std::cerr << "got      ";
                                for(int l = 0; l < qualityLength; l++){
                                    std::cerr << h_qualities[qualityPitchInBytes * i + l];
                                }
                                std::cerr << "\n";
                            }
                        }
		            }
                }
            };

            //func(0, indices.size(), 0);

            ThreadPool::ParallelForHandle pforHandle;

            threadPool.parallelFor(pforHandle, 0, int(indicesBuffer.size()), func);
           

            //oneIter = false;

            indicesBuffer.clear();
            readsBuffer.clear();
        };

        read_number globalReadId = 0;

        for(const auto& inputfile : fileOptions.inputfiles){

            forEachReadInFile(inputfile,
                            [&](auto /*readnum*/, const auto& read){

                if(oneIter){
                    indicesBuffer.emplace_back(globalReadId);
                    readsBuffer.emplace_back(read);         

                    if(indicesBuffer.size() >= batchsize){
                        validateBatch();
                    }

                    ++globalReadId;
                }

            });
        }

        if(indicesBuffer.size() >= 1){
            validateBatch();
        }

        std::cerr << "validated data in readstorage\n";
    }


    void validateMinhasher(const Minhasher& minhasher, const cpu::ContiguousReadStorage& readStorage, const FileOptions& fileOptions){
        std::cerr << "validating data in minhasher\n";

        std::vector<read_number> indicesBuffer;
        std::vector<Read> readsBuffer;
        
        constexpr std::int64_t batchsize = 16000;

        const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();
        const int sequencePitchInInts = getEncodedNumInts2Bit(maximumSequenceLength);
        
        

        ThreadPool threadPool;
        
        cpu::ContiguousReadStorage::GatherHandle readStorageGatherHandle;
        std::vector<read_number> h_readids(batchsize);
        std::vector<unsigned int> h_sequences(sequencePitchInInts * batchsize);
        std::vector<int> h_lengths(batchsize);

        auto validateBatch = [&](){

            ThreadPool::ParallelForHandle pforHandle;

            std::copy(indicesBuffer.begin(), indicesBuffer.end(), h_readids.data());

            readStorage.gatherSequenceLengths(
                readStorageGatherHandle,
                h_readids.data(),
                indicesBuffer.size(),
                h_lengths.data()
            );

            readStorage.gatherSequenceData(
                readStorageGatherHandle,
                h_readids.data(),
                indicesBuffer.size(),
                h_sequences.data(),
                sequencePitchInInts
            );                       
            
            auto func1 = [&](int begin, int end, int /*threadId*/){
                Minhasher::Handle minhashHandle;
                std::vector<std::string> sequences;

                //should not hit assertion if everything is ok with minhasher
                for(int i = begin; i < end; i++){               
                    sequences.emplace_back(get2BitString(h_sequences.data() + i * sequencePitchInInts, h_lengths[i]));

                    minhasher.getCandidates_any_map(
                        minhashHandle,
                        sequences.back(),
                        0
                    );
                }

                // check batch hashing too
                minhasher.calculateMinhashSignatures(
                    minhashHandle,
                    sequences
                );

                minhasher.queryPrecalculatedSignatures(
                    minhashHandle, 
                    sequences.size()
                );

                minhasher.makeUniqueQueryResults(
                    minhashHandle, 
                    sequences.size()
                );
            };

            //func1(0, int(indicesBuffer.size()), func1);
            threadPool.parallelFor(pforHandle, 0, int(indicesBuffer.size()), func1);

            //oneIter = false;

            indicesBuffer.clear();
            readsBuffer.clear();
        };

        const std::int64_t numReads = readStorage.getNumberOfReads();
        const int iters = SDIV(numReads, batchsize);

        for(int iter = 0; iter < iters; iter++){
            const std::int64_t begin = iter * batchsize;
            const std::int64_t end = std::min(numReads, (iter+1) * batchsize);

            std::cerr << begin << " - " << end << "\n";
            indicesBuffer.resize(end - begin);
            std::iota(indicesBuffer.begin(), indicesBuffer.end(), begin);

            validateBatch();
        }

        std::cerr << "validated data in minhasher\n";
    }


    BuiltDataStructure<Minhasher> build_minhasher(const FileOptions& fileOptions,
                                			   const RuntimeOptions& runtimeOptions,
                                               const MemoryOptions& memoryOptions,
                                			   std::uint64_t nReads,
                                               const CorrectionOptions& correctionOptions,
                                			   cpu::ContiguousReadStorage& readStorage){

        BuiltDataStructure<Minhasher> result;
        auto& minhasher = result.data;

        Minhasher::MinhashOptions minhashOptions;
        minhashOptions.k = correctionOptions.kmerlength;
        minhashOptions.maps = correctionOptions.numHashFunctions;
        minhashOptions.numResultsPerMapQueryThreshold 
            = calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage);

        minhasher = std::move(Minhasher{minhashOptions});

        minhasher.init(nReads);

        if(fileOptions.load_hashtables_from != ""){
            minhasher.loadFromFile(fileOptions.load_hashtables_from);
            result.builtType = BuiltType::Loaded;

            std::cout << "Loaded hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }else{
            result.builtType = BuiltType::Constructed;

            //const int oldnumthreads = omp_get_thread_num();

            //omp_set_num_threads(runtimeOptions.threads);

            ThreadPool threadPool(runtimeOptions.threads);
            ThreadPool::ParallelForHandle pforHandle;

            const std::string tmpmapsFilename = fileOptions.tempdirectory + "/tmpmaps";
            std::ofstream outstream(tmpmapsFilename, std::ios::binary);
            if(!outstream){
                throw std::runtime_error("Could not open temp file " + tmpmapsFilename + "!");
            }
            std::size_t writtenTableBytes = 0;

            std::size_t maxMemoryForTables = getAvailableMemoryInKB() * 1024;

            maxMemoryForTables = std::min(maxMemoryForTables, 
                                    std::min(memoryOptions.memoryForHashtables, memoryOptions.memoryTotalLimit));

            std::cerr << "maxMemoryForTables = " << maxMemoryForTables << " bytes\n";
            std::size_t availableMemForTables = maxMemoryForTables;

            int numSavedTables = 0;
            int numConstructedTables = 0;

            while(numConstructedTables < minhashOptions.maps && maxMemoryForTables > writtenTableBytes){
                std::vector<Minhasher::Map_t> minhashTables;

                int maxNumTables = 0;

                {
                    std::size_t requiredMemPerTable = Minhasher::Map_t::getRequiredSizeInBytesBeforeCompaction(nReads);
                    maxNumTables = availableMemForTables / requiredMemPerTable;
                    maxNumTables -= 2; // need free memory of 2 tables to perform transformation 
                    std::cerr << "requiredMemPerTable = " << requiredMemPerTable << "\n";
                    std::cerr << "maxNumTables = " << maxNumTables << "\n";
                }

                if(maxNumTables <= 0){
                    throw std::runtime_error("Not enough memory to construct 1 table");
                }

                int currentIterNumTables = std::min(minhashOptions.maps - numConstructedTables, maxNumTables);
                minhashTables.resize(currentIterNumTables);
                for(auto& table : minhashTables){
                    Minhasher::Map_t tmp(nReads);
                    table = std::move(tmp);
                }

                

                std::vector<int> tableIds(minhashTables.size());                
                std::vector<int> hashIds(minhashTables.size());
                std::vector<int> globalTableIds(minhashTables.size());
                
                std::iota(tableIds.begin(), tableIds.end(), 0);
                std::iota(hashIds.begin(), hashIds.end(), numConstructedTables);
                std::iota(globalTableIds.begin(), globalTableIds.end(), numConstructedTables);

                std::cout << "Constructing maps: ";
                std::copy(globalTableIds.begin(), globalTableIds.end(), std::ostream_iterator<int>(std::cout, " "));
                std::cout << "\n";

                const read_number readIdBegin = 0;
                const read_number readIdEnd = readStorage.getNumberOfReads();

                auto showProgress = [&](auto totalCount, auto seconds){
                    std::cerr << "Hashed " << totalCount << " / " << nReads << " reads. Elapsed time: " 
                                << seconds << " seconds.\n";
                };

                auto updateShowProgressInterval = [](auto duration){
                    return duration * 2;
                };

                ProgressThread<std::uint64_t> progressThread(nReads, 
                        showProgress,
                        updateShowProgressInterval);

                auto lambda = [&, readIdBegin](auto begin, auto end, int threadId) {

                    for (read_number readId = begin; readId < end; readId++){
                        const read_number localId = readId - readIdBegin;

                        const std::uint8_t* sequenceptr = (const std::uint8_t*)readStorage.fetchSequenceData_ptr(localId);
    				    const int sequencelength = readStorage.fetchSequenceLength(readId);
    				    std::string sequencestring = get2BitString((const unsigned int*)sequenceptr, sequencelength);

                        minhasher.insertSequenceIntoExternalTables(sequencestring, 
                                                                    readId, 
                                                                    tableIds,
                                                                    minhashTables,
                                                                    hashIds);

                        progressThread.addProgress(1);
                    }
                };

                threadPool.parallelFor(
                    pforHandle,
                    readIdBegin,
                    readIdEnd,
                    std::move(lambda));

                progressThread.finished();

                //if all tables could be constructed at once, no need to save them to temporary file
                if(minhashOptions.maps == int(minhashTables.size())){
                    for(int i = 0; i < int(minhashTables.size()); i++){
                        int globalTableId = globalTableIds[i];
                        int maxValuesPerKey = minhasher.getResultsPerMapThreshold();                    
                        std::cerr << "Transforming table " << globalTableId << ". ";
                        transform_keyvaluemap(minhashTables[i], maxValuesPerKey);
                        numConstructedTables++;
                        minhasher.moveassignMap(globalTableId, std::move(minhashTables[i]));
                    }
                }else{
                    for(int i = 0; i < int(minhashTables.size()); i++){
                        int globalTableId = globalTableIds[i];
                        int maxValuesPerKey = minhasher.getResultsPerMapThreshold();                    
                        std::cerr << "Transforming table " << globalTableId << ". ";
                        transform_keyvaluemap(minhashTables[i], maxValuesPerKey);
                        numConstructedTables++;
                        
                        minhashTables[i].writeToStream(outstream);
                        numSavedTables++;
                        writtenTableBytes = outstream.tellp();
    
                        std::cerr << "tablesize = " << minhashTables[i].numBytes() << "\n";
                        std::cerr << "written total of " << writtenTableBytes << " / " << maxMemoryForTables << "\n";
                        std::cerr << "numSavedTables = " << numSavedTables << "\n";
    
                        if(maxMemoryForTables <= writtenTableBytes){
                            break;
                        }
                    }
                    minhashTables.clear();

                    if(numConstructedTables >= minhashOptions.maps || maxMemoryForTables < writtenTableBytes){
                        outstream.flush();
    
                        std::cerr << "available before loading maps: " << (getAvailableMemoryInKB() * 1024) << "\n";
                        
                        int usableNumMaps = 0;
    
                        //load as many transformed tables from file as possible and move them to minhasher
                        std::ifstream instream(tmpmapsFilename, std::ios::binary);
                        for(int i = 0; i < numSavedTables; i++){
                            try{
                                std::cerr << "try loading table " << i << "\n";
                                Minhasher::Map_t table{};
                                table.readFromStream(instream);
                                minhasher.moveassignMap(i, std::move(table));
                                std::cerr << "available after loading table " << i << ": " << (getAvailableMemoryInKB() * 1024) << "\n";
                                usableNumMaps++;
                                std::cerr << "usable num maps = " << usableNumMaps << "\n";
                            }catch(...){
                                std::cerr << "Loading table " << i << " failed\n";
                                break;
                            }                        
                        }
    
                        filehelpers::removeFile(tmpmapsFilename);
    
                        minhasher.minhashTables.resize(usableNumMaps);
                        std::cout << "Can use " << usableNumMaps << " out of specified " << minhasher.minparams.maps << " tables\n";
                        minhasher.minparams.maps = usableNumMaps;
                    }   
                }
            }



            //omp_set_num_threads(oldnumthreads);
        }

        

        //TIMERSTARTCPU(finalize_hashtables);
        //minhasher.transform();
        //TIMERSTOPCPU(finalize_hashtables);

        return result;
    }



    BuiltDataStructure<cpu::ContiguousReadStorage> build_readstorage2(const FileOptions& fileOptions,
                                                const RuntimeOptions& runtimeOptions,
                                                bool useQualityScores,
                                                read_number expectedNumberOfReads,
                                                int expectedMinimumReadLength,
                                                int expectedMaximumReadLength){



        if(fileOptions.load_binary_reads_from != ""){
            BuiltDataStructure<cpu::ContiguousReadStorage> result;
            auto& readStorage = result.data;

            readStorage.loadFromFile(fileOptions.load_binary_reads_from);
            result.builtType = BuiltType::Loaded;

            if(useQualityScores && !readStorage.canUseQualityScores())
                throw std::runtime_error("Quality scores are required but not present in preprocessed reads file!");
            if(!useQualityScores && readStorage.canUseQualityScores())
                std::cerr << "Warning. The loaded preprocessed reads file contains quality scores, but program does not use them!\n";

            std::cout << "Loaded preprocessed reads from " << fileOptions.load_binary_reads_from << std::endl;

            return result;
        }else{
            //int nThreads = std::max(1, std::min(runtimeOptions.threads, 4));

            constexpr std::array<char, 4> bases = {'A', 'C', 'G', 'T'};
            //int Ncount = 0;

            BuiltDataStructure<cpu::ContiguousReadStorage> result;
            auto& readStorage = result.data;

            readStorage= std::move(cpu::ContiguousReadStorage{expectedNumberOfReads, 
                                                                useQualityScores, 
                                                                expectedMinimumReadLength, 
                                                                expectedMaximumReadLength});
            result.builtType = BuiltType::Constructed;

            auto checkRead = [&](read_number readIndex, Read& read, int& Ncount){
                const int readLength = int(read.sequence.size());

                if(readIndex >= expectedNumberOfReads){
                    throw std::runtime_error("Error! Expected " + std::to_string(expectedNumberOfReads)
                                            + " reads, but file contains at least "
                                            + std::to_string(readIndex+1) + " reads.");
                }

                if(readLength > expectedMaximumReadLength){
                    throw std::runtime_error("Error! Expected maximum read length = "
                                            + std::to_string(expectedMaximumReadLength)
                                            + ", but read " + std::to_string(readIndex)
                                            + "has length " + std::to_string(readLength));
                }

                auto isValidBase = [](char c){
                    constexpr std::array<char, 10> validBases{'A','C','G','T','a','c','g','t'};
                    return validBases.end() != std::find(validBases.begin(), validBases.end(), c);
                };

                const int undeterminedBasesInRead = std::count_if(read.sequence.begin(), read.sequence.end(), [&](char c){
                    return !isValidBase(c);
                });

                //nmap[undeterminedBasesInRead]++;

                if(undeterminedBasesInRead > 0){
                    readStorage.setReadContainsN(readIndex, true);
                }

                for(auto& c : read.sequence){
                    if(c == 'a') c = 'A';
                    else if(c == 'c') c = 'C';
                    else if(c == 'g') c = 'G';
                    else if(c == 't') c = 'T';
                    else if(!isValidBase(c)){
                        c = bases[Ncount];
                        Ncount = (Ncount + 1) % 4;
                    }
                }
            };


            constexpr size_t maxbuffersize = 1000000;
            constexpr int numBuffers = 2;

            std::array<std::vector<read_number>, numBuffers> indicesBuffers;
            std::array<std::vector<Read>, numBuffers> readsBuffers;
            std::array<bool, numBuffers> canBeUsed;
            std::array<std::mutex, numBuffers> mutex;
            std::array<std::condition_variable, numBuffers> cv;

            ThreadPool threadPool(runtimeOptions.threads);

            for(int i = 0; i < numBuffers; i++){
                indicesBuffers[i].reserve(maxbuffersize);
                readsBuffers[i].reserve(maxbuffersize);
                canBeUsed[i] = true;
            }

            int bufferindex = 0;
            read_number globalReadId = 0;

            auto showProgress = [show = runtimeOptions.showProgress](auto totalCount, auto seconds){
                if(show){
                    std::cout << "Processed " << totalCount << " reads in file. Elapsed time: " 
                                    << seconds << " seconds." << std::endl;
                }
            };
    
            auto updateShowProgressInterval = [](auto duration){
                return duration * 2;
            };
    
            ProgressThread<read_number> progressThread(
                expectedNumberOfReads, 
                showProgress, 
                updateShowProgressInterval
            );

            for(const auto& inputfile : fileOptions.inputfiles){
                std::cout << "Converting reads of file " << inputfile << ", storing them in memory\n";

                forEachReadInFile(inputfile,
                                [&](auto /*readnum*/, const auto& read){

                        if(!canBeUsed[bufferindex]){
                            std::unique_lock<std::mutex> ul(mutex[bufferindex]);
                            if(!canBeUsed[bufferindex]){
                                //std::cerr << "waiting for other buffer\n";
                                cv[bufferindex].wait(ul, [&](){ return canBeUsed[bufferindex]; });
                            }
                        }

                        auto indicesBufferPtr = &indicesBuffers[bufferindex];
                        auto readsBufferPtr = &readsBuffers[bufferindex];
                        indicesBufferPtr->emplace_back(globalReadId);
                        readsBufferPtr->emplace_back(read);

                        ++globalReadId;

                        progressThread.addProgress(1);                

                        if(indicesBufferPtr->size() >= maxbuffersize){
                            canBeUsed[bufferindex] = false;

                            //std::cerr << "launch other thread\n";
                            threadPool.enqueue([&, indicesBufferPtr, readsBufferPtr, bufferindex](){
                                //std::cerr << "buffer " << bufferindex << " running\n";
                                int nmodcounter = 0;

                                for(int i = 0; i < int(readsBufferPtr->size()); i++){
                                    read_number readId = (*indicesBufferPtr)[i];
                                    auto& read = (*readsBufferPtr)[i];
                                    checkRead(readId, read, nmodcounter);
                                    readStorage.insertRead(readId, read.sequence, read.quality);
                                }

                                //TIMERSTARTCPU(clear);
                                indicesBufferPtr->clear();
                                readsBufferPtr->clear();
                                //TIMERSTOPCPU(clear);
                                
                                std::lock_guard<std::mutex> l(mutex[bufferindex]);
                                canBeUsed[bufferindex] = true;
                                cv[bufferindex].notify_one();

                                //std::cerr << "buffer " << bufferindex << " finished\n";
                            });

                            bufferindex = (bufferindex + 1) % numBuffers; //swap buffers
                        }

                });
            }

            auto indicesBufferPtr = &indicesBuffers[bufferindex];
            auto readsBufferPtr = &readsBuffers[bufferindex];

            if(int(readsBufferPtr->size()) > 0){
                if(!canBeUsed[bufferindex]){
                    std::unique_lock<std::mutex> ul(mutex[bufferindex]);
                    if(!canBeUsed[bufferindex]){
                        //std::cerr << "waiting for other buffer\n";
                        cv[bufferindex].wait(ul, [&](){ return canBeUsed[bufferindex]; });
                    }
                }

                int nmodcounter = 0;

                for(int i = 0; i < int(readsBufferPtr->size()); i++){
                    read_number readId = (*indicesBufferPtr)[i];
                    auto& read = (*readsBufferPtr)[i];
                    checkRead(readId, read, nmodcounter);
                    readStorage.insertRead(readId, read.sequence, read.quality);
                }

                indicesBufferPtr->clear();
                readsBufferPtr->clear();
            }

            for(int i = 0; i < numBuffers; i++){
                std::unique_lock<std::mutex> ul(mutex[i]);
                if(!canBeUsed[i]){
                    //std::cerr << "Reading file completed. Waiting for buffer " << i << "\n";
                    cv[i].wait(ul, [&](){ return canBeUsed[i]; });
                }
            }

            progressThread.finished();

            return result;
        }

    }

BuiltDataStructures buildDataStructuresImpl2(
                                            const CorrectionOptions& correctionOptions,
                                            const RuntimeOptions& runtimeOptions,
                                            const MemoryOptions& memoryOptions,
                                            const FileOptions& fileOptions,
                                            bool saveDataStructuresToFile){

        BuiltDataStructures result;

        std::uint64_t maximumNumberOfReads = fileOptions.nReads;
        int maximumSequenceLength = fileOptions.maximum_sequence_length;
        int minimumSequenceLength = fileOptions.minimum_sequence_length;
        bool scanned = false;

        if(fileOptions.load_binary_reads_from == ""){

            if(maximumNumberOfReads == 0 || maximumSequenceLength == 0 || minimumSequenceLength == 0) {
                std::cout << "Scanning file(s) to get number of reads and min/max sequence length." << std::endl;

                maximumNumberOfReads = 0;
                maximumSequenceLength = 0;
                minimumSequenceLength = std::numeric_limits<int>::max();

                for(const auto& inputfile : fileOptions.inputfiles){
                    auto prop = getSequenceFileProperties(inputfile, runtimeOptions.showProgress);
                    maximumNumberOfReads += prop.nReads;
                    maximumSequenceLength = std::max(maximumSequenceLength, prop.maxSequenceLength);
                    minimumSequenceLength = std::min(minimumSequenceLength, prop.minSequenceLength);

                    std::cout << "----------------------------------------\n";
                    std::cout << "File: " << inputfile << "\n";
                    std::cout << "Reads: " << prop.nReads << "\n";
                    std::cout << "Minimum sequence length: " << prop.minSequenceLength << "\n";
                    std::cout << "Maximum sequence length: " << prop.maxSequenceLength << "\n";
                    std::cout << "----------------------------------------\n";

                    //result.inputFileProperties.emplace_back(prop);
                }

                scanned = true;
            }else{
                //std::cout << "Using the supplied max number of reads and min/max sequence length." << std::endl;
            }
        }

        TIMERSTARTCPU(build_readstorage);
        result.builtReadStorage = build_readstorage2(
            fileOptions,
            runtimeOptions,
            correctionOptions.useQualityScores,
            maximumNumberOfReads,
            minimumSequenceLength,
            maximumSequenceLength
        );
        TIMERSTOPCPU(build_readstorage);

        auto& readStorage = result.builtReadStorage.data;

        if(saveDataStructuresToFile && fileOptions.save_binary_reads_to != "") {
            std::cout << "Saving reads to file " << fileOptions.save_binary_reads_to << std::endl;
            readStorage.saveToFile(fileOptions.save_binary_reads_to);
            std::cout << "Saved reads" << std::endl;
        }

#ifdef VALIDATE_READSTORAGE
        validateReadstorage(readStorage, fileOptions);
#endif         

        result.totalInputFileProperties.nReads = readStorage.getNumberOfReads();
        result.totalInputFileProperties.maxSequenceLength = readStorage.getStatistics().maximumSequenceLength;
        result.totalInputFileProperties.minSequenceLength = readStorage.getStatistics().minimumSequenceLength;

        if(!scanned){
            std::cout << "Determined the following read properties:\n";
            std::cout << "----------------------------------------\n";
            std::cout << "Total number of reads: " << result.totalInputFileProperties.nReads << "\n";
            std::cout << "Minimum sequence length: " << result.totalInputFileProperties.minSequenceLength << "\n";
            std::cout << "Maximum sequence length: " << result.totalInputFileProperties.maxSequenceLength << "\n";
            std::cout << "----------------------------------------\n";
        }

        auto corOpts = correctionOptions;
        if(corOpts.autodetectKmerlength){
            const int maxlength = result.totalInputFileProperties.maxSequenceLength;

            corOpts.kmerlength = builddetail::getKmerSizeForHashing(maxlength);

            std::cout << "Will use k-mer length = " << corOpts.kmerlength << " for hashing.\n";

            result.kmerlength = corOpts.kmerlength;
        }

        std::cout << "Reads with ambiguous bases: " << readStorage.getNumberOfReadsWithN() << std::endl;

        TIMERSTARTCPU(build_minhasher);
        result.builtMinhasher = build_minhasher(
            fileOptions, 
            runtimeOptions, 
            memoryOptions,
            result.totalInputFileProperties.nReads, 
            corOpts, 
            readStorage
        );
        TIMERSTOPCPU(build_minhasher);

        if(saveDataStructuresToFile && fileOptions.save_hashtables_to != "") {
            std::cout << "Saving minhasher to file " << fileOptions.save_hashtables_to << std::endl;
            result.builtMinhasher.data.saveToFile(fileOptions.save_hashtables_to);
            std::cout << "Saved minhasher" << std::endl;
        }


#ifdef VALIDATE_MINHASHER        
        const auto& minhasher = result.builtMinhasher.data;
        validateMinhasher(minhasher, readStorage, fileOptions);
#endif  

        return result;
    }



    BuiltDataStructures buildDataStructures2(
                                			const CorrectionOptions& correctionOptions,
                                			const RuntimeOptions& runtimeOptions,
                                            const MemoryOptions& memoryOptions,
                                			const FileOptions& fileOptions){

        return buildDataStructuresImpl2(correctionOptions, runtimeOptions, memoryOptions, fileOptions, false);
    }

    BuiltDataStructures buildAndSaveDataStructures2(
                                            const CorrectionOptions& correctionOptions,
                                            const RuntimeOptions& runtimeOptions,
                                            const MemoryOptions& memoryOptions,
                                            const FileOptions& fileOptions){

        return buildDataStructuresImpl2(correctionOptions, runtimeOptions, memoryOptions, fileOptions, true);
    }
}
