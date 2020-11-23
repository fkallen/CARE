#include <gpu/warpcoreminhasher.cuh>
#include <hpc_helpers.cuh>

namespace care{
    namespace gpu{
namespace warpcoreminhasher{

template<class T>
__global__
void iotaKernel(T* __restrict__ begin, T* __restrict__ end, T firstvalue){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    const int size = end - begin;

    for(int i = tid; i < size; i += stride){
        begin[i] = firstvalue + T{i};
    }
}

template<class T>
__global__
void fixTableKeysKernel(T* __restrict__ keys, int numKeys, T emptyKey, T tombstoneKey){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    for(int i = tid; i < numKeys; i += stride){
        T key = keys[i];
        bool changed = false;
        while(key == emptyKey || key == tombstoneKey){
            key++;
            changed = true;
        }
        if(changed){
            keys[i] = key;
        }
    }
}

__global__
void minhashSignaturesKernel(
        std::uint64_t* __restrict__ signatures,
        std::size_t signaturesRowPitchElements,
        const unsigned int* __restrict__ sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* __restrict__ sequenceLengths,
        int k,
        int numHashFuncs,
        int firstHashFunc){
            
    //constexpr int blocksize = 128;
    constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < numSequences * numHashFuncs){
        const int mySequenceIndex = tid / numHashFuncs;
        const int myNumHashFunc = tid % numHashFuncs;
        const int hashFuncId = myNumHashFunc + firstHashFunc;

        const unsigned int* const mySequence = sequences2Bit + mySequenceIndex * sequenceRowPitchElements;
        const int myLength = sequenceLengths[mySequenceIndex];

        std::uint64_t* const mySignature = signatures + mySequenceIndex * signaturesRowPitchElements;

        

        if(myLength >= k){
            const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - k) * 2);
            std::uint64_t minHashValue = std::numeric_limits<std::uint64_t>::max();

            auto handlekmer = [&](auto fwd, auto rc){
                using hasher = hashers::MurmurHash<std::uint64_t>;

                const auto smallest = min(fwd, rc);
                const auto hashvalue = hasher::hash(smallest + hashFuncId);
                minHashValue = min(minHashValue, hashvalue) & kmer_mask;
            };

            //const int numKmers = myLength - k + 1;
            
            const int rcshiftamount = (maximum_kmer_length - k) * 2;

            //Compute the first kmer
            std::uint64_t kmer_encoded = mySequence[0];
            if(k <= 16){
                kmer_encoded >>= (16 - k) * 2;
            }else{
                kmer_encoded = (kmer_encoded << 32) | mySequence[1];
                kmer_encoded >>= (32 - k) * 2;
            }

            kmer_encoded >>= 2; //k-1 bases, allows easier loop

            std::uint64_t rc_kmer_encoded = SequenceHelpers::reverseComplementInt2Bit(kmer_encoded);

            auto addBase = [&](std::uint64_t encBase){
                kmer_encoded <<= 2;
                rc_kmer_encoded >>= 2;

                const std::uint64_t revcBase = (~encBase) & 3;
                kmer_encoded |= encBase;
                rc_kmer_encoded |= revcBase << 62;
            };

            constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();

            const int itersend1 = min(SDIV(k-1, basesPerInt) * basesPerInt, myLength);

            //process sequence positions one by one
            // until the next encoded sequence data element is reached
            for(int nextSequencePos = k - 1; nextSequencePos < itersend1; nextSequencePos++){
                const int nextIntIndex = nextSequencePos / basesPerInt;
                const int nextPositionInInt = nextSequencePos % basesPerInt;

                const std::uint64_t nextBase = mySequence[nextIntIndex] >> (30 - 2 * nextPositionInInt);

                addBase(nextBase);

                handlekmer(
                    kmer_encoded & kmer_mask, 
                    (rc_kmer_encoded >> rcshiftamount) & kmer_mask
                );
            }

            const int fullIntIters = (myLength - itersend1) / basesPerInt;

            //process all fully occupied encoded sequence data elements
            // improves memory access
            for(int iter = 0; iter < fullIntIters; iter++){
                const int intIndex = (itersend1 + iter * basesPerInt) / basesPerInt;
                const unsigned int data = mySequence[intIndex];

                #pragma unroll
                for(int posInInt = 0; posInInt < basesPerInt; posInInt++){
                    const std::uint64_t nextBase = data >> (30 - 2 * posInInt);

                    addBase(nextBase);

                    handlekmer(
                        kmer_encoded & kmer_mask, 
                        (rc_kmer_encoded >> rcshiftamount) & kmer_mask
                    );
                }
            }

            //process remaining positions one by one
            for(int nextSequencePos = fullIntIters * basesPerInt + itersend1; nextSequencePos < myLength; nextSequencePos++){
                const int nextIntIndex = nextSequencePos / basesPerInt;
                const int nextPositionInInt = nextSequencePos % basesPerInt;

                const std::uint64_t nextBase = mySequence[nextIntIndex] >> (30 - 2 * nextPositionInInt);

                addBase(nextBase);

                handlekmer(
                    kmer_encoded & kmer_mask, 
                    (rc_kmer_encoded >> rcshiftamount) & kmer_mask
                );
            }

            mySignature[myNumHashFunc] = minHashValue;

        }else{
            mySignature[myNumHashFunc] = std::numeric_limits<std::uint64_t>::max();
        }
    }
}


void callMinhashSignaturesKernel_async(
        std::uint64_t* d_signatures,
        std::size_t signaturesRowPitchElements,
        const unsigned int* d_sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* d_sequenceLengths,
        int k,
        int numHashFuncs,
        int firstHashFunc,
        cudaStream_t stream){

    constexpr int blocksize = 128;

    if(numSequences <= 0){
        return;
    }

    dim3 block(blocksize, 1, 1);
    dim3 grid(SDIV(numSequences * numHashFuncs, blocksize), 1, 1);
    std::size_t smem = 0;

    minhashSignaturesKernel<<<grid, block, smem, stream>>>(
        d_signatures,
        signaturesRowPitchElements,
        d_sequences2Bit,
        sequenceRowPitchElements,
        numSequences,
        d_sequenceLengths,
        k,
        numHashFuncs,
        firstHashFunc
    );

    CUERR;
}

void callMinhashSignaturesKernel_async(
        std::uint64_t* d_signatures,
        std::size_t signaturesRowPitchElements,
        const unsigned int* d_sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* d_sequenceLengths,
        int k,
        int numHashFuncs,
        cudaStream_t stream){
            
    constexpr int blocksize = 128;

    if(numSequences <= 0){
        return;
    }
            
    dim3 block(blocksize, 1, 1);
    dim3 grid(SDIV(numSequences * numHashFuncs, blocksize), 1, 1);
    std::size_t smem = 0;
    
    const int firstHashFunc = 0;

    minhashSignaturesKernel<<<grid, block, smem, stream>>>(
        d_signatures,
        signaturesRowPitchElements,
        d_sequences2Bit,
        sequenceRowPitchElements,
        numSequences,
        d_sequenceLengths,
        k,
        numHashFuncs,
        firstHashFunc
    );

    CUERR;
}






void WarpcoreMinhasher::queryPrecalculatedSignatures(
    const std::uint64_t* signatures, //getNumberOfMaps() elements per sequence
    WarpcoreMinhasher::Range_t* ranges, //getNumberOfMaps() elements per sequence
    int* totalNumResultsInRanges, 
    int numSequences) const{ 
    
    int numResults = 0;
    const std::uint64_t kmer_mask = getKmerMask();

    for(int i = 0; i < numSequences; i++){
        const std::uint64_t* const signature = &signatures[i * getNumberOfMaps()];
        WarpcoreMinhasher::Range_t* const range = &ranges[i * getNumberOfMaps()];            

        for(int map = 0; map < getNumberOfMaps(); ++map){
            kmer_type key = signature[map] & kmer_mask;
            auto entries_range = queryMap(map, key);
            numResults += std::distance(entries_range.first, entries_range.second);
            range[map] = entries_range;
        }
    }   

    *totalNumResultsInRanges = numResults;   
}


MemoryUsage WarpcoreMinhasher::getMemoryInfo() const{
    MemoryUsage result;

    result.host = sizeof(HashTable) * minhashTables.size();
    
    for(const auto& tableptr : minhashTables){
        auto m = tableptr->getMemoryInfo();
        result.host += m.host;

        for(auto pair : m.device){
            result.device[pair.first] += pair.second;
        }
    }

    return result;
}

void WarpcoreMinhasher::destroy(){
    minhashTables.clear();
}

void WarpcoreMinhasher::writeToStream(std::ostream& os) const{

    os.write(reinterpret_cast<const char*>(&kmerSize), sizeof(int));
    os.write(reinterpret_cast<const char*>(&resultsPerMapThreshold), sizeof(int));

    const int numTables = getNumberOfMaps();
    os.write(reinterpret_cast<const char*>(&numTables), sizeof(int));

    for(const auto& tableptr : minhashTables){
        tableptr->writeToStream(os);
    }
}

int WarpcoreMinhasher::loadFromStream(std::ifstream& is, int numMapsUpperLimit){
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

int WarpcoreMinhasher::calculateResultsPerMapThreshold(int coverage){
    int result = int(coverage * 2.5f);
    result = std::min(result, int(std::numeric_limits<BucketSize>::max()));
    result = std::max(10, result);
    return result;
}

void WarpcoreMinhasher::computeReadHashesOnGpu(
    std::uint64_t* d_hashValues,
    std::size_t hashValuesPitchInElements,
    const unsigned int* d_encodedSequenceData,
    std::size_t encodedSequencePitchInInts,
    int numSequences,
    const int* d_sequenceLengths,
    cudaStream_t stream
) const{
    callMinhashSignaturesKernel_async(
        d_hashValues,
        hashValuesPitchInElements,
        d_encodedSequenceData,
        encodedSequencePitchInInts,
        numSequences,
        d_sequenceLengths,
        getKmerSize(),
        getNumberOfMaps(),
        stream
    );
}






void WarpcoreMinhasher::construct(
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

    using MultiValueHashTable =  warpcore::MultiValueHashTable<
        kmer_type,
        read_number,
        warpcore::defaults::empty_key<kmer_type>(),
        warpcore::defaults::tombstone_key<kmer_type>(),
        warpcore::defaults::probing_scheme_t<kmer_type, 8>,
        warpcore::defaults::table_storage_t<kmer_type, read_number>,
        warpcore::defaults::temp_memory_bytes()>;

    using BucketListHashTable = warpcore::BucketListHashTable<
        kmer_type,
        read_number,
        warpcore::defaults::empty_key<kmer_type>(),
        warpcore::defaults::tombstone_key<kmer_type>(),
        warpcore::defaults::value_storage_t<read_number>,
        warpcore::defaults::probing_scheme_t<kmer_type, 8>>;

    using HashTable_t = BucketListHashTable;

    constexpr float loadKeys = 0.8f;
    constexpr float loadValues = 0.8f;
    const std::size_t key_capacity = numReads / loadKeys; 
    const std::size_t value_capacity = numReads / loadValues; 
    const std::size_t max_values_per_key = getNumResultsPerMapThreshold() + 1;

    std::vector<HashTable_t> mvHashTables;

    for(int i = 0; i < requestedNumberOfMaps; i++){
        // mvHashTables.emplace_back(
        //     min_capacity,
        //     warpcore::defaults::seed<kmer_type>(),
        //     max_values_per_key
        // );

        mvHashTables.emplace_back(
            key_capacity,
            value_capacity,
            warpcore::defaults::seed<kmer_type>(),
            1.1, // grow_factor
            1, //min_bucket_size
            max_values_per_key, //max_bucket_size
            max_values_per_key //max_values_per_key
        );

        auto status = mvHashTables.back().pop_status();
        cudaDeviceSynchronize();
        assert(!status.has_any());
    }


    auto populateHashTables = [this](
        HashTable_t* tables,
        int numTables, 
        int firstTableId,
        std::int64_t numberOfReads,
        int upperBoundSequenceLength,
        int deviceId,
        const RuntimeOptions& runtimeOptions,
        const DistributedReadStorage& readStorage
    ){

        int oldDevice = 0;
        cudaGetDevice(&oldDevice); CUERR;


        cudaSetDevice(deviceId); CUERR;

    
        constexpr read_number batchsize = SDIV(1000000,1024) * 1024;
        read_number numReads = numberOfReads;
        const int numIters = SDIV(numReads, batchsize);
        const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(upperBoundSequenceLength);

        const int numHashFuncs = numTables;
        const int firstHashFunc = firstTableId;
        const std::size_t signaturesRowPitchElements = numHashFuncs;

        helpers::SimpleAllocationDevice<unsigned int, 1> d_sequenceData(encodedSequencePitchInInts * batchsize);
        helpers::SimpleAllocationDevice<int, 0> d_lengths(batchsize);
    
        helpers::SimpleAllocationPinnedHost<read_number, 0> h_indices(batchsize);
        helpers::SimpleAllocationDevice<read_number, 0> d_indices(batchsize);
    
        helpers::SimpleAllocationDevice<std::uint64_t, 0> d_signatures(signaturesRowPitchElements * batchsize);
        helpers::SimpleAllocationPinnedHost<std::uint64_t, 0> d_signatures_transposed(signaturesRowPitchElements * batchsize);

        using StatusHandler = warpcore::status_handlers::ReturnStatus;
        helpers::SimpleAllocationPinnedHost<StatusHandler::base_type, 0> h_insertionStatus(batchsize);
        helpers::SimpleAllocationDevice<StatusHandler::base_type, 0> d_insertionStatus(batchsize);

        cudaStream_t stream;
        cudaStreamCreate(&stream); CUERR;
    
        auto sequencehandle = readStorage.makeGatherHandleSequences();    
        
        const int numThreads = runtimeOptions.threads;
    
        const std::uint64_t kmer_mask = getKmerMask(); 
    
        ThreadPool::ParallelForHandle pforHandle;
    
        // std::vector<std::ofstream> keysoutput;
        // for(int i = 0; i < numTables; i++){
        //     keysoutput.emplace_back("hashkeys" + std::to_string(i));
        //     std::size_t num = numReads;
        //     keysoutput.back().write((const char*)&num, sizeof(std::size_t));
        // }

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
    
        for (int iter = 0; iter < numIters; iter++){
            //std::cerr << "iter " << iter << " / " << numIters << "\n";

            read_number readIdBegin = iter * batchsize;
            read_number readIdEnd = std::min((iter + 1) * batchsize, numReads);
    
            const std::size_t curBatchsize = readIdEnd - readIdBegin;

            //iotaKernel<<<SDIV(curBatchsize, 128), 128, 0, stream>>>(d_indices.begin(), d_indices.end(), readIdBegin); CUERR;
    
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
    
            computeReadHashesOnGpu(
                d_signatures,
                signaturesRowPitchElements,
                d_sequenceData,
                encodedSequencePitchInInts,
                curBatchsize,
                d_lengths,
                numHashFuncs,
                firstHashFunc,
                stream
            );
    
            helpers::call_transpose_kernel(
                d_signatures_transposed.data(), 
                d_signatures.data(), 
                curBatchsize, 
                signaturesRowPitchElements, 
                signaturesRowPitchElements,
                stream
            );

            fixTableKeysKernel<<<SDIV(curBatchsize * numHashFuncs, 128), 128, 0, stream>>>(
                d_signatures_transposed.data(), 
                curBatchsize * numHashFuncs, 
                HashTable_t::empty_key(), 
                HashTable_t::tombstone_key()
            ); CUERR;

            for(int i = 0; i < numTables; i++){
                //std::cerr << "table " << i << " / " << numTables << "\n";
                // if(iter == 21 && i == 3){
                //     std::cerr << "saving\n";
                //     std::ofstream debugstream("debughashes.txt");
                //     for(std::size_t k = 0; k < curBatchsize; k++){
                //         debugstream << (d_signatures_transposed + i * curBatchsize)[k] << "\n";
                //     }
                //     std::cerr << "saved\n";
                // }
                tables[i].insert(
                    d_signatures_transposed + i * curBatchsize,
                    d_indices,
                    curBatchsize,
                    stream,
                    10000, //warpcore::defaults::probing_length(),
                    d_insertionStatus
                );

                cudaMemcpyAsync(
                    h_insertionStatus,
                    d_insertionStatus,
                    d_insertionStatus.sizeInBytes(),
                    D2H,
                    stream
                ); CUERR;

                 cudaStreamSynchronize(stream); CUERR;

                for(std::size_t k = 0; k < curBatchsize; k++){
                    if(h_insertionStatus[k].has_any()){
                        std::cerr << "Error table " << i << ", batch " << iter << ", position " << k << ": " << h_insertionStatus[k] << "\n";
                    }
                }

                //keysoutput[i].write((const char*)(d_signatures_transposed + i * curBatchsize), curBatchsize * sizeof(kmer_type));
            }

            progressThread.addProgress(curBatchsize);

            cudaStreamSynchronize(stream); CUERR;
    
            //TIMERSTOPCPU(insert);
        }
    
        progressThread.finished();
    
        cudaStreamDestroy(stream); CUERR;

        cudaSetDevice(oldDevice); CUERR;
    };

    populateHashTables(
        mvHashTables.data(), 
        mvHashTables.size(),
        0,
        readStorage.getNumberOfReads(),
        readStorage.getSequenceLengthUpperBound(),
        runtimeOptions.deviceIds[0],
        runtimeOptions,
        readStorage
    );

    std::cout << "Can use " << requestedNumberOfMaps 
        << " out of specified " << requestedNumberOfMaps
        << " tables\n";

    std::cerr << "press key\n";
    char c;
    std::cin >> c;
}




WarpcoreMinhasher::Range_t WarpcoreMinhasher::queryMap(int id, const Key_t& key) const{
    HashTable::QueryResult qr = minhashTables[id]->query(key);

    return std::make_pair(qr.valuesBegin, qr.valuesBegin + qr.numValues);
}

void WarpcoreMinhasher::addHashTable(HashTable&& hm){
    minhashTables.emplace_back(std::make_unique<HashTable>(std::move(hm)));
}

void WarpcoreMinhasher::computeReadHashesOnGpu(
    std::uint64_t* d_hashValues,
    std::size_t hashValuesPitchInElements,
    const unsigned int* d_encodedSequenceData,
    std::size_t encodedSequencePitchInInts,
    int numSequences,
    const int* d_sequenceLengths,
    int numHashFuncs,
    cudaStream_t stream
) const{
    callMinhashSignaturesKernel_async(
        d_hashValues,
        hashValuesPitchInElements,
        d_encodedSequenceData,
        encodedSequencePitchInInts,
        numSequences,
        d_sequenceLengths,
        getKmerSize(),
        numHashFuncs,
        stream
    );
}

void WarpcoreMinhasher::computeReadHashesOnGpu(
    std::uint64_t* d_hashValues,
    std::size_t hashValuesPitchInElements,
    const unsigned int* d_encodedSequenceData,
    std::size_t encodedSequencePitchInInts,
    int numSequences,
    const int* d_sequenceLengths,
    int numHashFuncs,
    int firstHashFunc,
    cudaStream_t stream
) const{
    callMinhashSignaturesKernel_async(
        d_hashValues,
        hashValuesPitchInElements,
        d_encodedSequenceData,
        encodedSequencePitchInInts,
        numSequences,
        d_sequenceLengths,
        getKmerSize(),
        numHashFuncs,
        firstHashFunc,
        stream
    );
}


std::pair< std::vector<std::vector<kmer_type>>, std::vector<std::vector<read_number>> > 
WarpcoreMinhasher::computeKeyValuePairsForHashtableUsingGpu(
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

        computeReadHashesOnGpu(
            d_signatures,
            signaturesRowPitchElements,
            d_sequenceData,
            encodedSequencePitchInInts,
            curBatchsize,
            d_lengths,
            numHashFuncs,
            firstHashFunc,
            stream
        );

        CUERR;

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





int WarpcoreMinhasher::loadConstructedTablesFromFile(
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



}
} //namespace gpu
} //namespace care