#include <build.hpp>

#include <config.hpp>

#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>
#include <minhasher.hpp>
#include <minhasher_transform.hpp>
#include <readstorage.hpp>
#include <sequencefileio.hpp>
#include <sequence.hpp>
#include <threadsafe_buffer.hpp>

#include <threadpool.hpp>


#include <map>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <thread>
#include <future>
#include <mutex>
#include <iterator>
#include <random>
#include <omp.h>
#include <mutex>
#include <condition_variable>

#ifdef __NVCC__

namespace care{
namespace gpu{



    BuiltDataStructure<GpuReadStorageWithFlags> buildGpuReadStorage(const FileOptions& fileOptions,
                                                const RuntimeOptions& runtimeOptions,
                                                bool useQualityScores,
                                                read_number expectedNumberOfReads,
                                                int expectedMinimumReadLength,
                                                int expectedMaximumReadLength){



        if(fileOptions.load_binary_reads_from != ""){
            BuiltDataStructure<GpuReadStorageWithFlags> result;
            auto& readStorage = result.data.readStorage;

            readStorage.loadFromFile(fileOptions.load_binary_reads_from, runtimeOptions.deviceIds);
            result.builtType = BuiltType::Loaded;

            if(useQualityScores && !readStorage.canUseQualityScores())
                throw std::runtime_error("Quality scores are required but not present in compressed sequence file!");
            if(!useQualityScores && readStorage.canUseQualityScores())
                std::cerr << "Warning. The loaded compressed read file contains quality scores, but program does not use them!\n";

            std::cout << "Loaded binary reads from " << fileOptions.load_binary_reads_from << std::endl;

            readStorage.constructionIsComplete();

            return result;
        }else{
            //int nThreads = std::max(1, std::min(runtimeOptions.threads, 2));
            const int nThreads = std::max(1, runtimeOptions.threads);

            constexpr std::array<char, 4> bases = {'A', 'C', 'G', 'T'};
            int Ncount = 0;
            //std::map<int,int> nmap{};

            BuiltDataStructure<GpuReadStorageWithFlags> result;
            DistributedReadStorage& readstorage = result.data.readStorage;
            //auto& validFlags = result.data.readIsValidFlags;

            readstorage = std::move(DistributedReadStorage{runtimeOptions.deviceIds, expectedNumberOfReads, useQualityScores, 
                                                            expectedMinimumReadLength, expectedMaximumReadLength});
            //validFlags.resize(expectedNumberOfReads, false);
            result.builtType = BuiltType::Constructed;

#if 0
            auto flushBuffers = [&](std::vector<read_number>& indicesBuffer, std::vector<Read>& readsBuffer){
                if(indicesBuffer.size() > 0){
                    readstorage.setReads(indicesBuffer, readsBuffer);
                    indicesBuffer.clear();
                    readsBuffer.clear();
                }
            };

            auto handle_read = [&](std::uint64_t readIndex, Read& read, std::vector<read_number>& indicesBuffer, std::vector<Read>& readsBuffer){
                const int readLength = int(read.sequence.size());

                if(readIndex >= expectedNumberOfReads){
                    throw std::runtime_error("Error! Expected " + std::to_string(expectedNumberOfReads)
                                            + " reads, but file contains at least "
                                            + std::to_string(readIndex) + " reads.");
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

                if(undeterminedBasesInRead > 0){
                    readstorage.setReadContainsN(readIndex, true);
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

                indicesBuffer.emplace_back(readIndex);
                readsBuffer.emplace_back(read);
            };

            constexpr size_t maxbuffersize = 1000000;

            std::chrono::time_point<std::chrono::system_clock> tpa, tpb;
            std::chrono::duration<double> duration;
            std::uint64_t countlimit = 1000000;
		    std::uint64_t count = 0;
		    std::uint64_t totalCount = 0;

            std::vector<read_number> indicesBuffer;
            std::vector<Read> readsBuffer;

            indicesBuffer.reserve(maxbuffersize);
            readsBuffer.reserve(maxbuffersize);

            tpa = std::chrono::system_clock::now();

            forEachReadInFile(fileOptions.inputfile,
                            fileOptions.format,
                            [&](auto readnum, auto& read){

                    handle_read(readnum, read, indicesBuffer, readsBuffer);

                    ++count;
                    ++totalCount;

                    if(count == countlimit){
                        tpb = std::chrono::system_clock::now();
                        duration = tpb - tpa;
                        std::cout << "Processed " << totalCount << " reads in file. Elapsed time: " 
                                << duration.count() << " seconds." << std::endl;
                        countlimit *= 2;
                    }

                    if(indicesBuffer.size() >= maxbuffersize){
                        flushBuffers(indicesBuffer, readsBuffer);
                    }
                }
            );

            if(indicesBuffer.size() > 0){
                flushBuffers(indicesBuffer, readsBuffer);
            }

            if(count > 0){
                tpb = std::chrono::system_clock::now();
                duration = tpb - tpa;
                std::cout << "Processed " << totalCount << " reads in file. Elapsed time: " 
                        << duration.count() << " seconds." << std::endl;
            }

#else 

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
                    readstorage.setReadContainsN(readIndex, true);
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
                        std::cout << "Processed " << totalCount << " reads in file. Elapsed time: " 
                                << duration.count() << " seconds." << std::endl;
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
                            }

                            readstorage.setReads(*indicesBufferPtr, *readsBufferPtr);

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
                }

                readstorage.setReads(*indicesBufferPtr, *readsBufferPtr);

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
                std::cout << "Processed " << totalCount << " reads in file. Elapsed time: " 
                                << duration.count() << " seconds." << std::endl;
            }

#endif

            // std::cerr << "occurences of n/N:\n";
            // for(const auto& p : nmap){
            //     std::cerr << p.first << " " << p.second << '\n';
            // }

            readstorage.constructionIsComplete();

            return result;
        }

    }

#if 0
    BuiltDataStructure<Minhasher> build_minhasher(const FileOptions& fileOptions,
                                               const RuntimeOptions& runtimeOptions,
                                               const MemoryOptions& memoryOptions,
                                			   std::uint64_t nReads,
                                               const MinhashOptions& minhashOptions,
                                			   const GpuReadStorageWithFlags& readStoragewFlags){

        BuiltDataStructure<Minhasher> result;
        auto& minhasher = result.data;

        auto identity = [](auto i){return i;};

        minhasher = std::move(Minhasher{minhashOptions});

        minhasher.init(nReads);

        if(fileOptions.load_hashtables_from != ""){
            minhasher.loadFromFile(fileOptions.load_hashtables_from);
            result.builtType = BuiltType::Loaded;

            std::cout << "Loaded hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }else{
            result.builtType = BuiltType::Constructed;

            const auto& readStorage = readStoragewFlags.readStorage;
            //const auto& validFlags = readStoragewFlags.readIsValidFlags;

            constexpr read_number parallelReads = 10000000;

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

                read_number numReads = readStorage.getNumberOfReads();
                int numIters = SDIV(numReads, parallelReads);

                auto sequencehandle = readStorage.makeGatherHandleSequences();
                //auto lengthhandle = readStorage.makeGatherHandleLengths();
                size_t sequencepitch = getEncodedNumInts2BitHiLo(readStorage.getSequenceLengthUpperBound()) * sizeof(int);

                //TIMERSTARTCPU(iter);
                for(int iter = 0; iter < numIters; iter++){
                    read_number readIdBegin = iter * parallelReads;
                    read_number readIdEnd = std::min((iter+1) * parallelReads, numReads);

                    std::vector<read_number> indices(readIdEnd - readIdBegin);
                    std::iota(indices.begin(), indices.end(), readIdBegin);

                    std::vector<char> sequenceData(indices.size() * sequencepitch);
                    std::vector<DistributedReadStorage::Length_t> lengths(indices.size());

                    //TIMERSTARTCPU(gather);

                    auto future1 = readStorage.gatherSequenceDataToHostBufferAsync(
                                                sequencehandle,
                                                sequenceData.data(),
                                                sequencepitch,
                                                indices.data(),
                                                indices.size(),
                                                1);
                    // auto future2 = readStorage.gatherSequenceLengthsToHostBufferAsync(
                    //                             lengthhandle,
                    //                             lengths.data(),
                    //                             indices.data(),
                    //                             indices.size(),
                    //                             1);

                    readStorage.gatherSequenceLengthsToHostBufferNew(
                        lengths.data(),
                        indices.data(),
                        int(indices.size()));

                    future1.wait();
                    //future2.wait();

                    //TIMERSTOPCPU(gather);

                    //TIMERSTARTCPU(insert);

                    auto lambda = [&, readIdBegin](auto begin, auto end, int threadId){
                        for(read_number readId = begin; readId < end; readId++){
                            read_number localId = readId - readIdBegin;
                            const char* encodedsequence = (const char*)&sequenceData[localId * sequencepitch];
                            const int sequencelength = lengths[localId];
                            std::string sequencestring = get2BitHiLoString((const unsigned int*)encodedsequence, sequencelength);
                            minhasher.insertSequence(sequencestring, readId, mapIds);
                        }
                    };

                    threadpool.parallelFor(readIdBegin, 
                                             readIdEnd,
                                             std::move(lambda));

                    //TIMERSTOPCPU(insert);
                }
                //TIMERSTOPCPU(iter);

                for(auto mapId : mapIds){
                    transform_minhasher_gpu(minhasher, mapId, runtimeOptions.deviceIds);
                }
            }
        }

        return result;
    }

#else 
    BuiltDataStructure<Minhasher> build_minhasher(const FileOptions &fileOptions,
                                                const RuntimeOptions &runtimeOptions,
                                                const MemoryOptions& memoryOptions,
                                                std::uint64_t nReads,
                                                const MinhashOptions &minhashOptions,
                                                const GpuReadStorageWithFlags &readStoragewFlags)
    {

        BuiltDataStructure<Minhasher> result;
        auto &minhasher = result.data;

        auto identity = [](auto i) { return i; };

        minhasher = std::move(Minhasher{minhashOptions});

        minhasher.init(nReads);

        if (fileOptions.load_hashtables_from != "")
        {
            minhasher.loadFromFile(fileOptions.load_hashtables_from);
            result.builtType = BuiltType::Loaded;

            std::cout << "Loaded hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }
        else
        {
            result.builtType = BuiltType::Constructed;

            const auto &readStorage = readStoragewFlags.readStorage;
            //const auto& validFlags = readStoragewFlags.readIsValidFlags;

            constexpr read_number parallelReads = 10000000;
            read_number numReads = readStorage.getNumberOfReads();
            int numIters = SDIV(numReads, parallelReads);

            auto sequencehandle = readStorage.makeGatherHandleSequences();
            //auto lengthhandle = readStorage.makeGatherHandleLengths();
            size_t sequencepitch = getEncodedNumInts2BitHiLo(readStorage.getSequenceLengthUpperBound()) * sizeof(int);

            const std::string tmpmapsFilename = fileOptions.tempdirectory + "/tmpmaps";
            std::ofstream outstream(tmpmapsFilename, std::ios::binary);
            if(!outstream){
                throw std::runtime_error("Could not open temp file " + tmpmapsFilename + "!");
            }


            std::size_t writtenTableBytes = 0;

            constexpr std::size_t GB1 = std::size_t(1) << 30;
            std::size_t maxMemoryForTables = getAvailableMemoryInKB() * 1024 - GB1;
            if(memoryOptions.memoryForHashtables > 0){
                maxMemoryForTables = std::min(memoryOptions.memoryForHashtables, maxMemoryForTables);
            }

            std::cerr << "maxMemoryForTables = " << maxMemoryForTables << " bytes\n";
            std::size_t availableMemForTables = maxMemoryForTables;



            
            std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();        
            std::mutex progressMutex;
		    std::uint64_t totalCount = 0;

            auto showProgress = [&](auto totalCount, auto seconds){
                std::cerr << "Hashed " << totalCount << " / " << nReads << " reads. Elapsed time: " 
                            << seconds << " seconds." << '\r';
                if(totalCount == nReads){
                    std::cerr << '\n';
                }
            };

            auto updateShowProgressInterval = [](auto duration){
                return duration;
            };

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

                ProgressThread<read_number> progressThread(numReads, showProgress, updateShowProgressInterval);

                for (int iter = 0; iter < numIters; iter++){
                    read_number readIdBegin = iter * parallelReads;
                    read_number readIdEnd = std::min((iter + 1) * parallelReads, numReads);

                    std::vector<read_number> indices(readIdEnd - readIdBegin);
                    std::iota(indices.begin(), indices.end(), readIdBegin);

                    std::vector<char> sequenceData(indices.size() * sequencepitch);
                    std::vector<DistributedReadStorage::Length_t> lengths(indices.size());

                    //TIMERSTARTCPU(gather);

                    auto future1 = readStorage.gatherSequenceDataToHostBufferAsync(
                        sequencehandle,
                        sequenceData.data(),
                        sequencepitch,
                        indices.data(),
                        indices.size(),
                        1);
                    // auto future2 = readStorage.gatherSequenceLengthsToHostBufferAsync(
                    //                             lengthhandle,
                    //                             lengths.data(),
                    //                             indices.data(),
                    //                             indices.size(),
                    //                             1); 

                    readStorage.gatherSequenceLengthsToHostBufferNew(
                        lengths.data(),
                        indices.data(),
                        int(indices.size()));

                    try{
                        future1.get();
                    }catch(...){
                        throw std::runtime_error("error gathering sequence data for minhasher construction!");
                    }
                    //future2.wait();

                    //TIMERSTOPCPU(gather);

                    //TIMERSTARTCPU(insert);

                    auto lambda = [&, readIdBegin](auto begin, auto end, int threadId) {
                        std::uint64_t countlimit = 10000;
                        std::uint64_t count = 0;
                        std::uint64_t oldcount = 0;

                        for (read_number readId = begin; readId < end; readId++){
                            read_number localId = readId - readIdBegin;
                            const char *encodedsequence = (const char *)&sequenceData[localId * sequencepitch];
                            const int sequencelength = lengths[localId];
                            std::string sequencestring = get2BitHiLoString((const unsigned int *)encodedsequence, sequencelength);
                            minhasher.insertSequenceIntoExternalTables(sequencestring, 
                                                                        readId, 
                                                                        tableIds,
                                                                        minhashTables,
                                                                        hashIds);

                            count++;
                            if(count == countlimit){
                                progressThread.addProgress(count);
                                count = 0;                                                         
                            }
                        }
                        if(count > 0){
                            progressThread.addProgress(count);
                        }
                    };

                    threadpool.parallelFor(readIdBegin,
                                        readIdEnd,
                                        std::move(lambda));

                    //TIMERSTOPCPU(insert);
                }

                progressThread.finished();

                //check free gpu mem for transformation
                std::size_t estRequiredFreeGpuMem = estimateGpuMemoryForTransformKeyValueMap(minhashTables[0]);
                std::size_t freeGpuMem, totalGpuMem;
                cudaMemGetInfo(&freeGpuMem, &totalGpuMem); CUERR;

                DistributedReadStorage::SavedGpuData savedReadstorageGpuData;
                const std::string rstempfile = fileOptions.tempdirectory+"/rstemp";
                bool didSaveGpudata = false;

                //if there is more than 10% gpu memory missing, make room for it
                //if(std::size_t(freeGpuMem * 1.1) < estRequiredFreeGpuMem){
                {
                    std::ofstream rstempostream(rstempfile, std::ios::binary);
                    std::size_t requiredMemPerTable = Minhasher::Map_t::getRequiredSizeInBytesBeforeCompaction(nReads);
                    savedReadstorageGpuData = std::move(readStorage.saveGpuDataAndFreeGpuMem(rstempostream, 2*requiredMemPerTable + GB1));

                    didSaveGpudata = true;
                }

                
                
                //if all tables could be constructed at once, no need to save them to temporary file
                if(minhashOptions.maps == int(minhashTables.size())){

                    for(int i = 0; i < int(minhashTables.size()); i++){
                        int globalTableId = globalTableIds[i];
                        int maxValuesPerKey = minhasher.getResultsPerMapThreshold();                    
                        std::cerr << "Transforming table " << globalTableId << ". ";
                        transform_keyvaluemap_gpu(minhashTables[i], runtimeOptions.deviceIds, maxValuesPerKey);
                        numConstructedTables++;
                        minhasher.moveassignMap(globalTableId, std::move(minhashTables[i]));
                    }

                    if(didSaveGpudata){
                        std::ifstream rstempistream(rstempfile, std::ios::binary);
                        readStorage.allocGpuMemAndLoadGpuData(rstempistream, savedReadstorageGpuData);
                        savedReadstorageGpuData.clear();
                        removeFile(rstempfile);
                    }
                    
                }else{
                    
                    for(int i = 0; i < int(minhashTables.size()); i++){
                        int globalTableId = globalTableIds[i];
                        int maxValuesPerKey = minhasher.getResultsPerMapThreshold();                    
                        std::cerr << "Transforming table " << globalTableId << ". ";
                        transform_keyvaluemap_gpu(minhashTables[i], runtimeOptions.deviceIds, maxValuesPerKey);
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

                    if(didSaveGpudata){
                        std::ifstream rstempistream(rstempfile, std::ios::binary);
                        readStorage.allocGpuMemAndLoadGpuData(rstempistream, savedReadstorageGpuData);
                        savedReadstorageGpuData.clear();
                    }

                    if(numConstructedTables >= minhashOptions.maps || maxMemoryForTables < writtenTableBytes){
                        outstream.flush();

                        std::size_t available = maxMemoryForTables;
    
                        std::cerr << "available before loading maps: " << available << "\n";
                        
                        int usableNumMaps = 0;
    
                        //load as many transformed tables from file as possible and move them to minhasher
                        std::ifstream instream(tmpmapsFilename, std::ios::binary);
                        for(int i = 0; i < numSavedTables; i++){
                            try{
                                std::cerr << "try loading table " << i << "\n";
                                Minhasher::Map_t table{};
                                table.readFromStream(instream);
                                std::size_t tablesize = table.allocationSizeInBytes();
                                if(available > tablesize){
                                    available -= tablesize;

                                    minhasher.moveassignMap(i, std::move(table));

                                    std::cerr << "available after loading table " << i << ": " << (getAvailableMemoryInKB() * 1024) << "\n";
                                    usableNumMaps++;
                                    std::cerr << "usable num maps = " << usableNumMaps << "\n";
                                }else if(available == tablesize){
                                    available -= tablesize;

                                    minhasher.moveassignMap(i, std::move(table));
                                    
                                    std::cerr << "available after loading table " << i << ": " << (getAvailableMemoryInKB() * 1024) << "\n";
                                    usableNumMaps++;
                                    std::cerr << "usable num maps = " << usableNumMaps << "\n";
                                    break;
                                }else{
                                    std::cerr << "Loading table " << i << " failed\n";
                                    break;
                                }
                            }catch(...){
                                std::cerr << "Loading table " << i << " failed\n";
                                break;
                            }                        
                        }
    
                        removeFile(tmpmapsFilename);
                        if(didSaveGpudata){
                            removeFile(rstempfile);
                        }
    
                        minhasher.minhashTables.resize(usableNumMaps);
                        std::cout << "Can use " << usableNumMaps << " out of specified " << minhasher.minparams.maps << " tables\n";
                        minhasher.minparams.maps = usableNumMaps;
                    }   
                } 
            }
        }

        return result;
    }




#endif

    BuiltGpuDataStructures buildGpuDataStructuresImpl(const MinhashOptions& minhashOptions,
                                                        const CorrectionOptions& correctionOptions,
                                                        const RuntimeOptions& runtimeOptions,
                                                        const MemoryOptions& memoryOptions,
                                                        const FileOptions& fileOptions,
                                                        bool saveDataStructuresToFile){                                                     

        BuiltGpuDataStructures result;

        auto& sequenceFileProperties = result.sequenceFileProperties;

        if(fileOptions.load_binary_reads_from == "") {
            sequenceFileProperties = detail::getSequenceFilePropertiesFromFileOptions(fileOptions);
        }

        TIMERSTARTCPU(build_readstorage);
        result.builtReadStorage = buildGpuReadStorage(fileOptions,
                                                  runtimeOptions,
                                                  correctionOptions.useQualityScores,
                                                  sequenceFileProperties.nReads,
                                                  sequenceFileProperties.minSequenceLength,
                                                  sequenceFileProperties.maxSequenceLength);
        TIMERSTOPCPU(build_readstorage);

        const auto& readStorage = result.builtReadStorage.data.readStorage;
        std::cout << "Using " << readStorage.lengthStorage.getRawBitsPerLength() << " bits per read to store its length\n";

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
            result.builtReadStorage.data);
        TIMERSTOPCPU(build_minhasher);

        if(saveDataStructuresToFile && fileOptions.save_hashtables_to != "") {
            std::cout << "Saving minhasher to file " << fileOptions.save_hashtables_to << std::endl;
    		result.builtMinhasher.data.saveToFile(fileOptions.save_hashtables_to);
    		std::cout << "Saved minhasher" << std::endl;
        }
        
        const auto& minhasher = result.builtMinhasher.data;
        const auto histograms = minhasher.getBinSizeHistogramsOfMaps();

        std::ofstream outhist("histograms.txt");
        for(int i = 0; i < int(histograms.size()); i++){
            outhist << "table " << i << '\n';
            for(auto pair : histograms[i]){
                outhist << pair.first << ' '  << pair.second << '\n';
            }
            outhist << '\n';
        }

        return result;
    }


    BuiltGpuDataStructures buildGpuDataStructures(const MinhashOptions& minhashOptions,
                                			const CorrectionOptions& correctionOptions,
                                            const RuntimeOptions& runtimeOptions,
                                            const MemoryOptions& memoryOptions,
                                			const FileOptions& fileOptions){

        return buildGpuDataStructuresImpl(minhashOptions,
                                        correctionOptions,
                                        runtimeOptions,
                                        memoryOptions,
                                        fileOptions,
                                        false);
    }

    BuiltGpuDataStructures buildAndSaveGpuDataStructures(const MinhashOptions& minhashOptions,
                                                        const CorrectionOptions& correctionOptions,
                                                        const RuntimeOptions& runtimeOptions,
                                                        const MemoryOptions& memoryOptions,
                                                        const FileOptions& fileOptions){                                                     

        return buildGpuDataStructuresImpl(minhashOptions,
                                        correctionOptions,
                                        runtimeOptions,
                                        memoryOptions,
                                        fileOptions,
                                        true);
    }

}
}


#endif
