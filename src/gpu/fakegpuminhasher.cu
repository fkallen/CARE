#include <gpu/fakegpuminhasher.cuh>
#include <hpc_helpers.cuh>
#include <gpu/kernels.hpp>

namespace care{
    namespace gpu{


        


void FakeGpuMinhasher::queryPrecalculatedSignatures(
    const std::uint64_t* signatures, //getNumberOfMaps() elements per sequence
    FakeGpuMinhasher::Range_t* ranges, //getNumberOfMaps() elements per sequence
    int* totalNumResultsInRanges, 
    int numSequences) const{ 
    
    int numResults = 0;
    const std::uint64_t kmer_mask = getKmerMask();

    for(int i = 0; i < numSequences; i++){
        const std::uint64_t* const signature = &signatures[i * getNumberOfMaps()];
        FakeGpuMinhasher::Range_t* const range = &ranges[i * getNumberOfMaps()];            

        for(int map = 0; map < getNumberOfMaps(); ++map){
            kmer_type key = signature[map] & kmer_mask;
            auto entries_range = queryMap(map, key);
            numResults += std::distance(entries_range.first, entries_range.second);
            range[map] = entries_range;
        }
    }   

    *totalNumResultsInRanges = numResults;   
}



void FakeGpuMinhasher::writeToStream(std::ostream& os) const{

    os.write(reinterpret_cast<const char*>(&kmerSize), sizeof(int));
    os.write(reinterpret_cast<const char*>(&resultsPerMapThreshold), sizeof(int));

    const int numTables = getNumberOfMaps();
    os.write(reinterpret_cast<const char*>(&numTables), sizeof(int));

    for(const auto& tableptr : minhashTables){
        tableptr->writeToStream(os);
    }
}

int FakeGpuMinhasher::loadFromStream(std::ifstream& is, int numMapsUpperLimit){
    destroy();

    is.read(reinterpret_cast<char*>(&kmerSize), sizeof(int));
    is.read(reinterpret_cast<char*>(&resultsPerMapThreshold), sizeof(int));

    int numMaps = 0;

    is.read(reinterpret_cast<char*>(&numMaps), sizeof(int));

    const int mapsToLoad = std::min(numMapsUpperLimit, numMaps);

    for(int i = 0; i < mapsToLoad; i++){
        HashTable table;
        table.loadFromStream(is);
        addHashTable(std::move(table));
    }

    return mapsToLoad;
}

int FakeGpuMinhasher::calculateResultsPerMapThreshold(int coverage){
    int result = int(coverage * 2.5f);
    result = std::min(result, int(std::numeric_limits<BucketSize>::max()));
    result = std::max(10, result);
    return result;
}


void FakeGpuMinhasher::constructFromReadStorage(
    const FileOptions &fileOptions,
    const RuntimeOptions &runtimeOptions,
    const MemoryOptions& memoryOptions,
    std::uint64_t nReads,
    const CorrectionOptions& correctionOptions,
    const DistributedReadStorage& gpuReadStorage
){
    
    auto& readStorage = gpuReadStorage;
    const auto& deviceIds = runtimeOptions.deviceIds;

    const int requestedNumberOfMaps = correctionOptions.numHashFunctions;

    const read_number numReads = readStorage.getNumberOfReads();
    const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();

    auto sequencehandle = readStorage.makeGatherHandleSequences();
    std::size_t sequencepitch = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength) * sizeof(int);

    const std::string tmpmapsFilename = fileOptions.tempdirectory + "/tmpmaps";
    std::ofstream outstream(tmpmapsFilename, std::ios::binary);
    if(!outstream){
        throw std::runtime_error("Could not open temp file " + tmpmapsFilename + "!");
    }


    const MemoryUsage memoryUsageOfReadStorage = readStorage.getMemoryInfo();
    std::size_t totalLimit = memoryOptions.memoryTotalLimit;
    if(totalLimit > memoryUsageOfReadStorage.host){
        totalLimit -= memoryUsageOfReadStorage.host;
    }else{
        totalLimit = 0;
    }
    if(totalLimit == 0){
        throw std::runtime_error("Not enough memory available for hash tables. Abort!");
    }
    std::size_t maxMemoryForTables = getAvailableMemoryInKB() * 1024;
    // std::cerr << "available: " << maxMemoryForTables 
    //         << ",memoryForHashtables: " << memoryOptions.memoryForHashtables
    //         << ", memoryTotalLimit: " << memoryOptions.memoryTotalLimit
    //         << ", rsHostUsage: " << memoryUsageOfReadStorage.host << "\n";

    maxMemoryForTables = std::min(maxMemoryForTables, 
                            std::min(memoryOptions.memoryForHashtables, totalLimit));

    std::cerr << "maxMemoryForTables = " << maxMemoryForTables << " bytes\n";


    std::size_t writtenTableBytes = 0;
    int numSavedTables = 0;

    int numConstructedTables = 0;
    std::vector<HashTable> cachedConstructedTables;
    std::size_t bytesOfCachedConstructedTables = 0;
    bool allowCaching = false;

    // std::vector<std::ofstream> keysoutput;
    // for(int i = 0; i < requestedNumberOfMaps; i++){
    //     keysoutput.emplace_back("hashkeys" + std::to_string(i));
    //     std::size_t num = numReads;
    //     keysoutput.back().write((const char*)&num, sizeof(std::size_t));
    // }

    while(numConstructedTables < requestedNumberOfMaps && maxMemoryForTables > (writtenTableBytes + bytesOfCachedConstructedTables)){

        int maxNumTables = 0;

        auto updateMaxNumTables = [&](){
            // (1 kmer + readid) per read
            std::size_t requiredMemPerTable = (sizeof(kmer_type) + sizeof(read_number)) * numReads;
            maxNumTables = (maxMemoryForTables - bytesOfCachedConstructedTables) / requiredMemPerTable;
            maxNumTables -= 2; // keep free memory of 2 tables to perform transformation 
            std::cerr << "requiredMemPerTable = " << requiredMemPerTable << "\n";
            std::cerr << "maxNumTables = " << maxNumTables << "\n";
        };

        updateMaxNumTables();

        //if at least 75 percent of all tables can be constructed in first iteration, keep all constructed tables in memory
        //else save constructed tables to file if there are less than requestedNumberOfMaps
        if(numConstructedTables == 0 && float(maxNumTables) / requestedNumberOfMaps >= 0.75f){
            allowCaching = true;
        }

        bool savedTooManyTablesToFile = false;

        if(maxNumTables <= 0){
            if(cachedConstructedTables.empty() && !allowCaching){
                throw std::runtime_error("Not enough memory to construct 1 table");
            }else{
                //save cached constructed tables to file to make room for more tables

                std::cerr << "saving cached constructed tables to file to make room for more tables\n";
                for(int i = 0; i < int(cachedConstructedTables.size()); i++){                            
                    const auto& hashTable = cachedConstructedTables[i];

                    auto memoryUsage = hashTable.getMemoryInfo();
                    hashTable.writeToStream(outstream);
                    numSavedTables++;
                    writtenTableBytes = outstream.tellp();

                    std::cerr << "tablesize = " << memoryUsage.host << "\n";
                    std::cerr << "written total of " << writtenTableBytes << " / " << maxMemoryForTables << "\n";
                    std::cerr << "numSavedTables = " << numSavedTables << "\n";

                    if(maxMemoryForTables <= writtenTableBytes){
                        savedTooManyTablesToFile = true;
                        std::cerr << "savedTooManyTablesToFile\n";
                        break;
                    }
                }
                cachedConstructedTables.clear();
                bytesOfCachedConstructedTables = 0;

                updateMaxNumTables();

                if(maxNumTables <= 0){                        
                    throw std::runtime_error("Not enough memory to construct 1 table");
                }
            }
        }

        if(!savedTooManyTablesToFile){

            const int currentIterNumTables = std::min(requestedNumberOfMaps - numConstructedTables, maxNumTables);

            std::pair< std::vector<std::vector<kmer_type>>, std::vector<std::vector<read_number>> >              
            initialMinhashes = computeKeyValuePairsForHashtableUsingGpu(                      
                currentIterNumTables, 
                numConstructedTables,
                readStorage.getNumberOfReads(),
                readStorage.getSequenceLengthUpperBound(),
                runtimeOptions,
                readStorage
            );

            std::size_t availableMemoryToSaveGpuPartitions = totalLimit;
            // account for the currently calculated minhash signatures
            for(const auto& vec : initialMinhashes.first){
                const std::size_t sub = vec.capacity() * sizeof(kmer_type);
                if(availableMemoryToSaveGpuPartitions > sub){
                    availableMemoryToSaveGpuPartitions -= sub;
                }else{
                    availableMemoryToSaveGpuPartitions = 0;
                }
            }
            for(const auto& vec : initialMinhashes.second){
                const std::size_t sub = vec.capacity() * sizeof(read_number);
                if(availableMemoryToSaveGpuPartitions > sub){
                    availableMemoryToSaveGpuPartitions -= sub;
                }else{
                    availableMemoryToSaveGpuPartitions = 0;
                }
            }
            //account for constructed tables in previous iteration
            if(availableMemoryToSaveGpuPartitions > bytesOfCachedConstructedTables){
                availableMemoryToSaveGpuPartitions -= bytesOfCachedConstructedTables;
            }else{
                availableMemoryToSaveGpuPartitions = 0;
            }

            for(int i = 0; i < 2 + int(minhashTables.size()); i++){
                const std::size_t requiredMemPerTable = nReads * sizeof(Key_t) //keys
                                                        + nReads * sizeof(Value_t) // values
                                                        + nReads * sizeof(Value_t) // counts prefix sum
                                                        + 4 * 1024;
                if(availableMemoryToSaveGpuPartitions > requiredMemPerTable){
                    availableMemoryToSaveGpuPartitions -= requiredMemPerTable;
                }else{
                    availableMemoryToSaveGpuPartitions = 0;
                    break;
                }
            }         

            // std::cerr << "availableMemoryToSaveGpuPartitions: " << availableMemoryToSaveGpuPartitions << "\n";

            DistributedReadStorage::SavedGpuData savedReadstorageGpuData;
            const std::string rstempfile = fileOptions.tempdirectory+"/rstemp";
            const bool didSaveGpudata = true;

            std::ofstream rstempostream(rstempfile, std::ios::binary);
            savedReadstorageGpuData = std::move(readStorage.saveGpuDataAndFreeGpuMem(rstempostream, availableMemoryToSaveGpuPartitions));
            
            constexpr bool valuesOfSameKeyMustBeSorted = false;
            
            //if all tables could be constructed at once, no need to save them to temporary file
            if(requestedNumberOfMaps == currentIterNumTables){

                for(int i = 0; i < currentIterNumTables; i++){
                    int globalTableId = numConstructedTables;
                    
                    if(runtimeOptions.showProgress){
                        std::cout << "Constructing hash table " << globalTableId << "." << std::endl;
                    }                           
                    
                    auto& kmers = initialMinhashes.first[i];
                    auto& readIds = initialMinhashes.second[i];

                    // for(int i = 0; i < 10; i++){
                    //     std::cerr << kmers[i] << " " << readIds[i] << "\n";
                    // }
                    const int maxValuesPerKey = getNumResultsPerMapThreshold();

                    //keysoutput[globalTableId].write((const char*)(kmers.data()), kmers.size() * sizeof(kmer_type));

                    HashTable hashTable(
                        std::move(kmers),
                        std::move(readIds), 
                        maxValuesPerKey,
                        deviceIds,
                        valuesOfSameKeyMustBeSorted
                    );

                    addHashTable(std::move(hashTable));

                    numConstructedTables++;
                }

                if(didSaveGpudata){
                    std::ifstream rstempistream(rstempfile, std::ios::binary);
                    readStorage.allocGpuMemAndLoadGpuData(rstempistream, savedReadstorageGpuData);
                    savedReadstorageGpuData.clear();
                    filehelpers::removeFile(rstempfile);
                }
                
            }else{

                for(int i = 0; i < currentIterNumTables; i++){
                    const int globalTableId = numConstructedTables;
                    
                    if(runtimeOptions.showProgress){
                        std::cout << "Constructing hash table " << globalTableId << "." << std::endl;
                    }                           
                    
                    auto& kmers = initialMinhashes.first[i];
                    auto& readIds = initialMinhashes.second[i];
                    const int maxValuesPerKey = getNumResultsPerMapThreshold();

                    HashTable hashTable(
                        std::move(kmers), 
                        std::move(readIds), 
                        maxValuesPerKey,
                        deviceIds,
                        valuesOfSameKeyMustBeSorted
                    );

                    numConstructedTables++;     

                    auto memoryUsage = hashTable.getMemoryInfo();
                    if(allowCaching){
                        bytesOfCachedConstructedTables += memoryUsage.host;
                        cachedConstructedTables.emplace_back(std::move(hashTable));

                        std::cerr << "cached " << cachedConstructedTables.size() << " constructed tables in memory\n";

                        if(maxMemoryForTables <= bytesOfCachedConstructedTables){
                            break;
                        }
                    }else{
                        hashTable.writeToStream(outstream);
                        numSavedTables++;
                        writtenTableBytes = outstream.tellp();
    
                        std::cerr << "tablesize = " << memoryUsage.host << "\n";
                        std::cerr << "written total of " << writtenTableBytes << " / " << maxMemoryForTables << "\n";
                        std::cerr << "numSavedTables = " << numSavedTables << "\n";

                        if(maxMemoryForTables <= writtenTableBytes){
                            break;
                        }
                    }                            
                }

                initialMinhashes.first.clear();
                initialMinhashes.second.clear();

                if(didSaveGpudata){
                    std::ifstream rstempistream(rstempfile, std::ios::binary);
                    readStorage.allocGpuMemAndLoadGpuData(rstempistream, savedReadstorageGpuData);
                    savedReadstorageGpuData.clear();
                }

                if(int(cachedConstructedTables.size()) + numSavedTables >= requestedNumberOfMaps 
                            || maxMemoryForTables < writtenTableBytes){

                    outstream.flush();

                    //discard any cached table such that size of cached tables + size of tables in file < memory limit
                    std::size_t totalTableBytes = writtenTableBytes;
                    int end = 0;
                    for(int i = 0; i < int(cachedConstructedTables.size()); i++){
                        const auto& table = cachedConstructedTables[i];
                        auto memoryUsage = table.getMemoryInfo();

                        if(totalTableBytes + memoryUsage.host <= maxMemoryForTables){
                            totalTableBytes += memoryUsage.host;
                            end++;
                        }else{
                            break;
                        }
                    }
                    cachedConstructedTables.erase(cachedConstructedTables.begin() + end, cachedConstructedTables.end());
                    
                    int usableNumMaps = loadConstructedTablesFromFile(
                                                tmpmapsFilename, 
                                                numSavedTables, 
                                                maxMemoryForTables);

                    for(int i = 0; i < int(cachedConstructedTables.size()) && usableNumMaps < requestedNumberOfMaps; i++){
                        auto& table = cachedConstructedTables[i];
                        addHashTable(std::move(table));
                        
                        usableNumMaps++;
                    }

                    filehelpers::removeFile(tmpmapsFilename);
                    if(didSaveGpudata){
                        filehelpers::removeFile(rstempfile);
                    }

                    std::cout << "Can use " << usableNumMaps 
                        << " out of specified " << requestedNumberOfMaps
                        << " tables\n";
                } 
            } 
        }else{
            //all constructed tables have been saved to file, and no table is cached

            outstream.flush();

            int usableNumMaps = loadConstructedTablesFromFile(
                                            tmpmapsFilename, 
                                            numSavedTables, 
                                            maxMemoryForTables);

            std::cout << "Can use " << usableNumMaps 
                << " out of specified " << requestedNumberOfMaps
                << " tables\n";
        }
    }
}




FakeGpuMinhasher::Range_t FakeGpuMinhasher::queryMap(int id, const Key_t& key) const{
    HashTable::QueryResult qr = minhashTables[id]->query(key);

    return std::make_pair(qr.valuesBegin, qr.valuesBegin + qr.numValues);
}

void FakeGpuMinhasher::addHashTable(HashTable&& hm){
    minhashTables.emplace_back(std::make_unique<HashTable>(std::move(hm)));
}




std::pair< std::vector<std::vector<kmer_type>>, std::vector<std::vector<read_number>> > 
FakeGpuMinhasher::computeKeyValuePairsForHashtableUsingGpu(
    int numTables, 
    int firstTableId,
    std::int64_t numberOfReads,
    int upperBoundSequenceLength,
    const RuntimeOptions& runtimeOptions,
    const DistributedReadStorage& readStorage
){

    constexpr read_number parallelReads = 1000000;
    read_number numReads = numberOfReads;
    const int numIters = SDIV(numReads, parallelReads);
    const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(upperBoundSequenceLength);

    const auto& deviceIds = runtimeOptions.deviceIds;
    const int numThreads = runtimeOptions.threads;

    const std::uint64_t kmer_mask = getKmerMask();

    assert(deviceIds.size() > 0);

    const int deviceId = deviceIds[0];

    cudaSetDevice(deviceId); CUERR;

    const int numHashFuncs = numTables;
    const int firstHashFunc = firstTableId;
    const std::size_t signaturesRowPitchElements = numHashFuncs;

    ThreadPool::ParallelForHandle pforHandle;

    std::vector<std::vector<kmer_type>> kmersPerFunc(numTables);
    std::vector<std::vector<read_number>> readIdsPerFunc(numTables);

    for(auto& v : kmersPerFunc){
        v.resize(numberOfReads);
    }

    for(auto& v : readIdsPerFunc){
        v.resize(numberOfReads);
    }

    std::vector<int> tableIds(numTables);                
    std::vector<int> hashIds(numTables);
    
    std::iota(tableIds.begin(), tableIds.end(), 0);

    std::cout << "Constructing maps: ";
    for(int i = 0; i < numTables; i++){
        std::cout << (firstTableId + i) << ' ';
    }
    std::cout << '\n';

    auto showProgress = [&](auto totalCount, auto seconds){
        if(runtimeOptions.showProgress){
            std::cout << "Hashed " << totalCount << " / " << numReads << " reads. Elapsed time: " 
                    << seconds << " seconds.\n";
        }
    };

    auto updateShowProgressInterval = [](auto duration){
        return duration * 2;
    };

    ProgressThread<read_number> progressThread(numReads, showProgress, updateShowProgressInterval);

    ThreadPool threadPool(numThreads);

    helpers::SimpleAllocationDevice<unsigned int, 1> d_sequenceData(encodedSequencePitchInInts * parallelReads);
    helpers::SimpleAllocationDevice<int, 0> d_lengths(parallelReads);

    helpers::SimpleAllocationPinnedHost<read_number, 0> h_indices(parallelReads);
    helpers::SimpleAllocationDevice<read_number, 0> d_indices(parallelReads);

    helpers::SimpleAllocationPinnedHost<std::uint64_t, 0> h_signatures(signaturesRowPitchElements * parallelReads);
    helpers::SimpleAllocationDevice<std::uint64_t, 0> d_signatures(signaturesRowPitchElements * parallelReads);

    cudaStream_t stream;
    cudaStreamCreate(&stream); CUERR;

    auto sequencehandle = readStorage.makeGatherHandleSequences();


    for (int iter = 0; iter < numIters; iter++){
        read_number readIdBegin = iter * parallelReads;
        read_number readIdEnd = std::min((iter + 1) * parallelReads, numReads);

        const std::size_t curBatchsize = readIdEnd - readIdBegin;

        std::iota(h_indices.get(), h_indices.get() + curBatchsize, readIdBegin);

        cudaMemcpyAsync(d_indices, h_indices, sizeof(read_number) * curBatchsize, H2D, stream); CUERR;

        readStorage.gatherSequenceDataToGpuBufferAsync(
            &threadPool,
            sequencehandle,
            d_sequenceData,
            encodedSequencePitchInInts,
            h_indices,
            d_indices,
            curBatchsize,
            deviceId,
            stream
        );
    
        readStorage.gatherSequenceLengthsToGpuBufferAsync(
            d_lengths,
            deviceId,
            d_indices,
            curBatchsize,
            stream
        );

        callMinhashSignaturesKernel(
            d_signatures,
            signaturesRowPitchElements,
            d_sequenceData,
            encodedSequencePitchInInts,
            curBatchsize,
            d_lengths,
            getKmerSize(),
            numHashFuncs,
            firstHashFunc,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            h_signatures, 
            d_signatures, 
            signaturesRowPitchElements * sizeof(std::uint64_t) * curBatchsize, 
            D2H, 
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;


        auto lambda = [&, readIdBegin](auto begin, auto end, int threadId) {
            std::uint64_t countlimit = 10000;
            std::uint64_t count = 0;

            for (read_number readId = begin; readId < end; readId++){
                read_number localId = readId - readIdBegin;

                for(int i = 0; i < numHashFuncs; i++){
                    const kmer_type kmer = kmer_mask & h_signatures[signaturesRowPitchElements * localId + i];
                    kmersPerFunc[i][readId] = kmer;
                    readIdsPerFunc[i][readId] = readId;
                }
                
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

        threadPool.parallelFor(
            pforHandle,
            readIdBegin,
            readIdEnd,
            std::move(lambda));

        //TIMERSTOPCPU(insert);
    }

    progressThread.finished();

    cudaStreamDestroy(stream); CUERR;

    return {std::move(kmersPerFunc), std::move(readIdsPerFunc)};
}





int FakeGpuMinhasher::loadConstructedTablesFromFile(
    const std::string& filename,
    int numTablesToLoad, 
    std::size_t availableMemory
){

    std::cerr << "available before loading maps: " << availableMemory << "\n";
    
    int assignedNumMaps = 0;

    //load as many transformed tables from file as possible and move them to minhasher
    std::ifstream instream(filename, std::ios::binary);
    for(int i = 0; i < numTablesToLoad; i++){
        try{
            std::cerr << "try loading table " << i << "\n";
            HashTable table;
            table.loadFromStream(instream);
            const auto memoryUsage = table.getMemoryInfo();
            const std::size_t tablesize = memoryUsage.host;

            if(availableMemory > tablesize){
                availableMemory -= tablesize;

                addHashTable(std::move(table));

                std::cerr << "available after loading table " << i << ": " << (getAvailableMemoryInKB() * 1024) << "\n";
                assignedNumMaps++;
                std::cerr << "usable num maps = " << assignedNumMaps << "\n";
            }else if(availableMemory == tablesize){
                availableMemory -= tablesize;

                addHashTable(std::move(table));
                
                std::cerr << "available after loading table " << i << ": " << (getAvailableMemoryInKB() * 1024) << "\n";
                assignedNumMaps++;
                std::cerr << "usable num maps = " << assignedNumMaps << "\n";
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

    return assignedNumMaps;
}




} //namespace gpu
} //namespace care
