#include <build.hpp>

#include <config.hpp>

#include <util.hpp>
#include <options.hpp>
#include <filehelpers.hpp>
#include <minhasher.hpp>
#include <minhasher_transform.hpp>
#include "readstorage.hpp"
#include "sequencefileio.hpp"
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




namespace care{

BuiltDataStructure<cpu::ContiguousReadStorage> build_readstorage(const FileOptions& fileOptions,
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

            if(useQualityScores && !readStorage.hasQualityScores())
                throw std::runtime_error("Quality scores are required but not present in compressed sequence file!");
            if(!useQualityScores && readStorage.hasQualityScores())
                std::cerr << "Warning. The loaded compressed read file contains quality scores, but program does not use them!\n";

            std::cout << "Loaded binary reads from " << fileOptions.load_binary_reads_from << std::endl;

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

            std::chrono::time_point<std::chrono::system_clock> tpa, tpb;
            std::chrono::duration<double> duration;
            std::uint64_t countlimit = 1000000;
		    std::uint64_t count = 0;
		    std::uint64_t totalCount = 0;

            std::array<std::vector<read_number>, numBuffers> indicesBuffers;
            std::array<std::vector<Read>, numBuffers> readsBuffers;
            std::array<bool, numBuffers> canBeUsed;
            std::array<std::mutex, numBuffers> mutex;
            std::array<std::condition_variable, numBuffers> cv;

            for(int i = 0; i < numBuffers; i++){
                indicesBuffers[i].reserve(maxbuffersize);
                readsBuffers[i].reserve(maxbuffersize);
                canBeUsed[i] = true;
            }

            int bufferindex = 0;

            tpa = std::chrono::system_clock::now();

            auto updateProgress = [](auto totalCount, auto seconds){
                std::cout << "Processed " << totalCount << " reads in file. Elapsed time: " 
                            << seconds << " seconds." << std::endl;
            };

            forEachReadInFile(fileOptions.inputfile,
                            fileOptions.format,
                            [&](auto readnum, const auto& read){

                    if(!canBeUsed[bufferindex]){
                        std::unique_lock<std::mutex> ul(mutex[bufferindex]);
                        if(!canBeUsed[bufferindex]){
                            //std::cerr << "waiting for other buffer\n";
                            cv[bufferindex].wait(ul, [&](){ return canBeUsed[bufferindex]; });
                        }
                    }

                    auto indicesBufferPtr = &indicesBuffers[bufferindex];
                    auto readsBufferPtr = &readsBuffers[bufferindex];
                    indicesBufferPtr->emplace_back(readnum);
                    readsBufferPtr->emplace_back(read);

                    ++count;
                    ++totalCount;

                    if(count == countlimit){
                        tpb = std::chrono::system_clock::now();
                        duration = tpb - tpa;
                        updateProgress(totalCount, duration.count());
                        countlimit *= 2;
                    }
            

                    if(indicesBufferPtr->size() >= maxbuffersize){
                        canBeUsed[bufferindex] = false;

                        //std::cerr << "launch other thread\n";
                        threadpool.enqueue([&, indicesBufferPtr, readsBufferPtr, bufferindex](){
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

            if(count > 0){
                tpb = std::chrono::system_clock::now();
                duration = tpb - tpa;
                updateProgress(totalCount, duration.count());
            }

            return result;
        }

    }


#if 0

    BuiltDataStructure<Minhasher> build_minhasher(const FileOptions& fileOptions,
                                			   const RuntimeOptions& runtimeOptions,
                                			   std::uint64_t nReads,
                                               const MinhashOptions& minhashOptions,
                                			   cpu::ContiguousReadStorage& readStorage){

        BuiltDataStructure<Minhasher> result;
        auto& minhasher = result.data;

        minhasher = std::move(Minhasher{minhashOptions});

        minhasher.init(nReads);

        if(fileOptions.load_hashtables_from != ""){
            minhasher.loadFromFile(fileOptions.load_hashtables_from);
            result.builtType = BuiltType::Loaded;

            std::cout << "Loaded hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }else{
            result.builtType = BuiltType::Constructed;



            std::chrono::time_point<std::chrono::system_clock> tpa, tpb;
            std::chrono::duration<double> duration;
            std::uint64_t countlimit = 1000000;
		    std::uint64_t count = 0;
		    std::uint64_t totalCount = 0;

            tpa = std::chrono::system_clock::now();

            auto updateProgress = [](auto totalCount, auto seconds){
                std::cout << "Hashed " << totalCount << " / " << nReads << " reads. Elapsed time: " 
                            << seconds << " seconds." << std::endl;
            };

            ++count;
            ++totalCount;

            if(count == countlimit){
                tpb = std::chrono::system_clock::now();
                duration = tpb - tpa;
                updateProgress(totalCount, duration.count());
                countlimit *= 2;
            }

            const int oldnumthreads = omp_get_thread_num();

            omp_set_num_threads(runtimeOptions.threads);

            const int numBatches = SDIV(minhashOptions.maps, minhasherConstructionNumMaps);

            for(int batch = 0; batch < numBatches; batch++){
                const int firstMap = batch * minhasherConstructionNumMaps;
                const int lastMap = std::min(minhashOptions.maps, (batch+1) * minhasherConstructionNumMaps);
                const int numMaps = lastMap - firstMap;
                std::vector<int> mapIds(numMaps);
                std::iota(mapIds.begin(), mapIds.end(), firstMap);

                for(auto mapId : mapIds){
                    minhasher.initMap(mapId);
                }

                #pragma omp parallel for
                for(read_number readId = 0; readId < readStorage.getNumberOfReads(); readId++){

    				const std::uint8_t* sequenceptr = (const std::uint8_t*)readStorage.fetchSequenceData_ptr(readId);
    				const int sequencelength = readStorage.fetchSequenceLength(readId);
    				std::string sequencestring = get2BitHiLoString((const unsigned int*)sequenceptr, sequencelength);

                    minhasher.insertSequence(sequencestring, readId, mapIds);
                }

                for(auto mapId : mapIds){
                    transform_minhasher(minhasher, mapId);
                }

            }

            omp_set_num_threads(oldnumthreads);
        }

        //TIMERSTARTCPU(finalize_hashtables);
        //minhasher.transform();
        //TIMERSTOPCPU(finalize_hashtables);

        return result;
    }

#else 



    BuiltDataStructure<Minhasher> build_minhasher(const FileOptions& fileOptions,
                                			   const RuntimeOptions& runtimeOptions,
                                               const MemoryOptions& memoryOptions,
                                			   std::uint64_t nReads,
                                               const MinhashOptions& minhashOptions,
                                			   cpu::ContiguousReadStorage& readStorage){

        BuiltDataStructure<Minhasher> result;
        auto& minhasher = result.data;

        minhasher = std::move(Minhasher{minhashOptions});

        minhasher.init(nReads);

        if(fileOptions.load_hashtables_from != ""){
            minhasher.loadFromFile(fileOptions.load_hashtables_from);
            result.builtType = BuiltType::Loaded;

            std::cout << "Loaded hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }else{
            result.builtType = BuiltType::Constructed;

            const int oldnumthreads = omp_get_thread_num();

            omp_set_num_threads(runtimeOptions.threads);

            const std::string tmpmapsFilename = fileOptions.tempdirectory + "/tmpmaps";
            std::ofstream outstream(tmpmapsFilename, std::ios::binary);
            if(!outstream){
                throw std::runtime_error("Could not open temp file " + tmpmapsFilename + "!");
            }
            std::size_t writtenTableBytes = 0;

            constexpr std::size_t GB1 = std::size_t(1) << 30;
            const std::size_t maxMemoryForTransformedTables = getAvailableMemoryInKB() * 1024 - GB1;

	        std::cerr << "maxMemoryForTransformedTables = " << maxMemoryForTransformedTables << " bytes\n";

            

            std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();        
            std::mutex progressMutex;
		    std::uint64_t totalCount = 0;

            // auto updateProgress = [&](auto totalCount, auto seconds){
            //     std::cout << "Hashed " << totalCount << " / " << nReads << " reads. Elapsed time: " 
            //                 << seconds << " seconds." << std::endl;
            // };

            int numSavedTables = 0;
            int numConstructedTables = 0;

            while(numConstructedTables < minhashOptions.maps && maxMemoryForTransformedTables > writtenTableBytes){
                std::vector<Minhasher::Map_t> minhashTables;

                int maxNumTables = 0;

                {
                    constexpr std::size_t MB128 = std::size_t(1) << 27;
                    std::size_t availableMemBefore = getAvailableMemoryInKB() * 1024;
                    Minhasher::Map_t table(nReads, runtimeOptions.deviceIds);
                    std::size_t availableMemAfter = getAvailableMemoryInKB() * 1024;
                    std::size_t tableMemApprox = availableMemBefore - availableMemAfter + MB128;
                    maxNumTables = 1 + (getAvailableMemoryInKB() * 1024) / tableMemApprox;
                    maxNumTables -= 2; // need free memory of 2 table to perform transformation
                }

                assert(maxNumTables > 0);
                int currentIterNumTables = std::min(minhashOptions.maps - numConstructedTables, maxNumTables);
                minhashTables.resize(currentIterNumTables);
                for(auto& table : minhashTables){
                    Minhasher::Map_t tmp(nReads, runtimeOptions.deviceIds);
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

                ProgressThread<std::uint64_t> progressThread(nReads, 
                        [&](auto totalCount, auto seconds){
                            std::cout << "progress: Hashed " << totalCount << " / " << nReads << " reads. Elapsed time: " 
                                        << seconds << " seconds." << std::endl;
                        },
                        [](auto seconds){return seconds * 2;});

                auto lambda = [&, readIdBegin](auto begin, auto end, int threadId) {
                    std::uint64_t countlimit = 1000000;
                    std::uint64_t count = 0;
                    std::uint64_t oldcount = 0;

                    for (read_number readId = begin; readId < end; readId++){
                        const read_number localId = readId - readIdBegin;

                        const std::uint8_t* sequenceptr = (const std::uint8_t*)readStorage.fetchSequenceData_ptr(localId);
    				    const int sequencelength = readStorage.fetchSequenceLength(readId);
    				    std::string sequencestring = get2BitHiLoString((const unsigned int*)sequenceptr, sequencelength);

                        minhasher.insertSequenceIntoExternalTables(sequencestring, 
                                                                    readId, 
                                                                    tableIds,
                                                                    minhashTables,
                                                                    hashIds);
                        // count++;
                        // if(count == countlimit){
                        //     const auto tpb = std::chrono::system_clock::now();
                        //     const std::chrono::duration<double> duration = tpb - tpa;
                        //     countlimit *= 2;

                        //     std::lock_guard<std::mutex> lg(progressMutex);
                        //     totalCount += count - oldcount;                            
                        //     updateProgress(totalCount, duration.count());
                        //     oldcount = count;                            
                        // }

                        progressThread.addProgress(1);
                    }
                    // if(count > 0){
                    //     const auto tpb = std::chrono::system_clock::now();
                    //     const std::chrono::duration<double> duration = tpb - tpa;
                    //     std::lock_guard<std::mutex> lg(progressMutex);
                    //     totalCount += count - oldcount;                            
                    //     updateProgress(totalCount, duration.count());
                    // }
                };

                threadpool.parallelFor(readIdBegin,
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
                        std::cerr << "written total of " << writtenTableBytes << " / " << maxMemoryForTransformedTables << "\n";
                        std::cerr << "numSavedTables = " << numSavedTables << "\n";
    
                        if(maxMemoryForTransformedTables <= writtenTableBytes){
                            break;
                        }
                    }
                    minhashTables.clear();

                    if(numConstructedTables >= minhashOptions.maps || maxMemoryForTransformedTables < writtenTableBytes){
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
    
                        removeFile(tmpmapsFilename);
    
                        minhasher.minhashTables.resize(usableNumMaps);
                        std::cout << "Can use " << usableNumMaps << " out of specified " << minhasher.minparams.maps << " tables\n";
                        minhasher.minparams.maps = usableNumMaps;
                    }   
                }
            }



            omp_set_num_threads(oldnumthreads);
        }

        

        //TIMERSTARTCPU(finalize_hashtables);
        //minhasher.transform();
        //TIMERSTOPCPU(finalize_hashtables);

        return result;
    }




#endif

    BuiltDataStructures buildDataStructuresImpl(const MinhashOptions& minhashOptions,
                                            const CorrectionOptions& correctionOptions,
                                            const RuntimeOptions& runtimeOptions,
                                            const MemoryOptions& memoryOptions,
                                            const FileOptions& fileOptions,
                                            bool saveDataStructuresToFile){

        BuiltDataStructures result;

        auto& sequenceFileProperties = result.sequenceFileProperties;

        if(fileOptions.load_binary_reads_from == "") {
            sequenceFileProperties = detail::getSequenceFilePropertiesFromFileOptions(fileOptions);
        }

        TIMERSTARTCPU(build_readstorage);
        result.builtReadStorage = build_readstorage(fileOptions,
                                                  runtimeOptions,
                                                  correctionOptions.useQualityScores,
                                                  sequenceFileProperties.nReads,
                                                  sequenceFileProperties.minSequenceLength,
                                                  sequenceFileProperties.maxSequenceLength);
        TIMERSTOPCPU(build_readstorage);

        auto& readStorage = result.builtReadStorage.data;

        if(saveDataStructuresToFile && fileOptions.save_binary_reads_to != "") {
            std::cout << "Saving reads to file " << fileOptions.save_binary_reads_to << std::endl;
            readStorage.saveToFile(fileOptions.save_binary_reads_to);
            std::cout << "Saved reads" << std::endl;
        }

        sequenceFileProperties.nReads = readStorage.getNumberOfReads();
        sequenceFileProperties.maxSequenceLength = readStorage.getStatistics().maximumSequenceLength;
        sequenceFileProperties.minSequenceLength = readStorage.getStatistics().minimumSequenceLength;

        std::cout << "After construction of read storage, the following file properties are known "
                    << "which may be different from supplied parameters" << std::endl;      

        detail::printInputFileProperties(std::cout, fileOptions.inputfile, sequenceFileProperties);

        TIMERSTARTCPU(build_minhasher);
        result.builtMinhasher = build_minhasher(fileOptions, 
                                                runtimeOptions, 
                                                memoryOptions,
                                                sequenceFileProperties.nReads, 
                                                minhashOptions, 
                                                readStorage);
        TIMERSTOPCPU(build_minhasher);

        if(saveDataStructuresToFile && fileOptions.save_hashtables_to != "") {
            std::cout << "Saving minhasher to file " << fileOptions.save_hashtables_to << std::endl;
            result.builtMinhasher.data.saveToFile(fileOptions.save_hashtables_to);
            std::cout << "Saved minhasher" << std::endl;
        }

        return result;
    }

    BuiltDataStructures buildDataStructures(const MinhashOptions& minhashOptions,
                                			const CorrectionOptions& correctionOptions,
                                			const RuntimeOptions& runtimeOptions,
                                            const MemoryOptions& memoryOptions,
                                			const FileOptions& fileOptions){

        return buildDataStructuresImpl(minhashOptions, correctionOptions, runtimeOptions, memoryOptions, fileOptions, false);
    }

    BuiltDataStructures buildAndSaveDataStructures(const MinhashOptions& minhashOptions,
                                            const CorrectionOptions& correctionOptions,
                                            const RuntimeOptions& runtimeOptions,
                                            const MemoryOptions& memoryOptions,
                                            const FileOptions& fileOptions){

        return buildDataStructuresImpl(minhashOptions, correctionOptions, runtimeOptions, memoryOptions, fileOptions, true);
    }
}
