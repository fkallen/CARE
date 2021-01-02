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
        auto ptr = std::make_unique<HashTable>();
        ptr->loadFromStream(is);
        minhashTables.emplace_back(std::move(ptr));
    }

    return mapsToLoad;
}

void FakeGpuMinhasher::constructFromReadStorage(
    const FileOptions &fileOptions,
    const RuntimeOptions &runtimeOptions,
    const MemoryOptions& memoryOptions,
    std::uint64_t nReads,
    const CorrectionOptions& correctionOptions,
    const GpuReadStorage& gpuReadStorage
){
    
    auto& readStorage = gpuReadStorage;
    const auto& deviceIds = runtimeOptions.deviceIds;

    int deviceId = deviceIds[0];

    cub::SwitchDevice sd{deviceId};

    const int requestedNumberOfMaps = correctionOptions.numHashFunctions;

    const read_number numReads = readStorage.getNumberOfReads();
    const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();

    auto sequencehandle = gpuReadStorage.makeHandle();
    const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength) * sizeof(unsigned int);

    constexpr read_number parallelReads = 1000000;
    const int numIters = SDIV(numReads, parallelReads);

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

    const int hashFunctionOffset = 0;

    
    std::vector<int> usedHashFunctionNumbers;
    
    helpers::SimpleAllocationDevice<unsigned int, 1> d_sequenceData(encodedSequencePitchInInts * parallelReads);
    helpers::SimpleAllocationDevice<int, 0> d_lengths(parallelReads);
    
    helpers::SimpleAllocationPinnedHost<read_number, 0> h_indices(parallelReads);
    helpers::SimpleAllocationDevice<read_number, 0> d_indices(parallelReads);
    
    std::size_t insert_temp_size = 0;
    insert(
        nullptr,
        insert_temp_size,
        (const unsigned int*)nullptr,
        int(parallelReads),
        (const int*)nullptr,
        encodedSequencePitchInInts,
        (const read_number*)nullptr,
        0,
        requestedNumberOfMaps,
        (const int*)nullptr,
        (cudaStream_t)0
    );
    
    helpers::SimpleAllocationDevice<char, 0> d_temp(insert_temp_size);
    
    CudaStream stream{};
    ThreadPool tp(runtimeOptions.threads);

    setThreadPool(&tp);
    setMemoryLimitForConstruction(maxMemoryForTables);
    
    //std::size_t bytesOfCachedConstructedTables = 0;
    int remainingHashFunctions = requestedNumberOfMaps;
    bool keepGoing = true;

    while(remainingHashFunctions > 0 && keepGoing){

        const int alreadyExistingHashFunctions = requestedNumberOfMaps - remainingHashFunctions;
        int addedHashFunctions = addHashfunctions(remainingHashFunctions);

        if(addedHashFunctions == 0){
            keepGoing = false;
            break;
        }

        std::cout << "Constructing maps: ";
        for(int i = 0; i < addedHashFunctions; i++){
            std::cout << (alreadyExistingHashFunctions + i) << "(" << (hashFunctionOffset + alreadyExistingHashFunctions + i) << ") ";
        }
        std::cout << '\n';

        std::vector<int> h_hashfunctionNumbers(addedHashFunctions);
        std::iota(
            h_hashfunctionNumbers.begin(),
            h_hashfunctionNumbers.end(),
            alreadyExistingHashFunctions + hashFunctionOffset
        );

        usedHashFunctionNumbers.insert(usedHashFunctionNumbers.end(), h_hashfunctionNumbers.begin(), h_hashfunctionNumbers.end());

        for (int iter = 0; iter < numIters; iter++){
            read_number readIdBegin = iter * parallelReads;
            read_number readIdEnd = std::min((iter + 1) * parallelReads, numReads);

            const std::size_t curBatchsize = readIdEnd - readIdBegin;

            std::iota(h_indices.get(), h_indices.get() + curBatchsize, readIdBegin);

            cudaMemcpyAsync(d_indices, h_indices, sizeof(read_number) * curBatchsize, H2D, stream); CUERR;

            gpuReadStorage.gatherSequences(
                sequencehandle,
                d_sequenceData,
                encodedSequencePitchInInts,
                h_indices,
                d_indices,
                curBatchsize,
                stream
            );
        
            gpuReadStorage.gatherSequenceLengths(
                sequencehandle,
                d_lengths,
                d_indices,
                curBatchsize,
                stream
            );

            insert(
                d_temp.data(),
                insert_temp_size,
                d_sequenceData,
                curBatchsize,
                d_lengths,
                encodedSequencePitchInInts,
                d_indices,
                alreadyExistingHashFunctions,
                addedHashFunctions,
                h_hashfunctionNumbers.data(),
                stream
            );

            cudaStreamSynchronize(stream); CUERR;
        }

        std::cerr << "Compacting\n";
        finalize();

        remainingHashFunctions -= addedHashFunctions;
    }

    setThreadPool(nullptr);      
}


FakeGpuMinhasher::Range_t FakeGpuMinhasher::queryMap(int id, const Key_t& key) const{
    HashTable::QueryResult qr = minhashTables[id]->query(key);

    return std::make_pair(qr.valuesBegin, qr.valuesBegin + qr.numValues);
}





} //namespace gpu
} //namespace care
