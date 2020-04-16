#include <build.hpp>

#include <config.hpp>

#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>
#include <minhasher.hpp>
#include <minhasher_transform.hpp>
#include <readstorage.hpp>
#include <readlibraryio.hpp>
#include <sequence.hpp>
#include <threadsafe_buffer.hpp>

#include <gpu/simpleallocation.cuh>
#include <gpu/minhashkernels.hpp>

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



//#define VALIDATE_READSTORAGE
//#define VALIDATE_MINHASHER


#ifdef __NVCC__

namespace care{
namespace gpu{

    template<
             // SequenceProviderFunc::operator()(char* dest, int sequencepitch, const read_number* indices, int numIndices);
             class SequenceProviderFunc, 
             // SequenceLengthProviderFunc::operator()(DistributedReadStorage::Length_t* dest, const read_number* indices, int numIndices);
             class SequenceLengthProviderFunc> 
    std::vector<Minhasher::Map_t> constructTables(const Minhasher& minhasher, 
                                                    int numTables, 
                                                    int firstTableId,
                                                    std::int64_t numberOfReads,
                                                    int upperBoundSequenceLength,
                                                    int numThreads,
                                                    SequenceProviderFunc&& getSequenceData,
                                                    SequenceLengthProviderFunc&& getSequenceLength,
                                                    const DistributedReadStorage& readStorage){

        constexpr read_number parallelReads = 10000000;
        read_number numReads = numberOfReads;
        const int numIters = SDIV(numReads, parallelReads);
        const size_t sequencepitch = getEncodedNumInts2Bit(upperBoundSequenceLength) * sizeof(int);

        ThreadPool::ParallelForHandle pforHandle;

        std::vector<Minhasher::Map_t> minhashTables(numTables);

        std::vector<int> tableIds(numTables);                
        std::vector<int> hashIds(numTables);
        
        std::iota(tableIds.begin(), tableIds.end(), 0);
        std::iota(hashIds.begin(), hashIds.end(), firstTableId);

        std::cout << "Constructing maps: ";
        for(int i = 0; i < numTables; i++){
            std::cout << (firstTableId + i) << ' ';
        }
        std::cout << '\n';

        for(auto& table : minhashTables){
            Minhasher::Map_t tmp(numReads);
            table = std::move(tmp);
        }

        auto showProgress = [&](auto totalCount, auto seconds){
            std::cerr << "Hashed " << totalCount << " / " << numReads << " reads. Elapsed time: " 
                        << seconds << " seconds." << '\r';
            if(totalCount == numReads){
                std::cerr << '\n';
            }
        };

        auto updateShowProgressInterval = [](auto duration){
            return duration;
        };

        ProgressThread<read_number> progressThread(numReads, showProgress, updateShowProgressInterval);

        ThreadPool threadPool(numThreads);

        for (int iter = 0; iter < numIters; iter++){
            read_number readIdBegin = iter * parallelReads;
            read_number readIdEnd = std::min((iter + 1) * parallelReads, numReads);

            std::vector<read_number> indices(readIdEnd - readIdBegin);
            std::iota(indices.begin(), indices.end(), readIdBegin);

            std::vector<char> sequenceData(indices.size() * sequencepitch);
            std::vector<DistributedReadStorage::Length_t> lengths(indices.size());

            //TIMERSTARTCPU(gather);

            getSequenceData(
                sequenceData.data(),
                sequencepitch,
                indices.data(),
                indices.size());

            getSequenceLength(
                lengths.data(),
                indices.data(),
                indices.size());

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
                    std::string sequencestring = get2BitString((const unsigned int *)encodedsequence, sequencelength);
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

            threadPool.parallelFor(
                pforHandle,
                readIdBegin,
                readIdEnd,
                std::move(lambda));

            //TIMERSTOPCPU(insert);
        }

        progressThread.finished();

        return minhashTables;
    }


    std::vector<Minhasher::Map_t> constructTablesWithGpuHashing(const Minhasher& minhasher, 
                                                    int numTables, 
                                                    int firstTableId,
                                                    int kmersize,
                                                    std::int64_t numberOfReads,
                                                    int upperBoundSequenceLength,
                                                    int numThreads,
                                                    const DistributedReadStorage& readStorage){

        constexpr read_number parallelReads = 1000000;
        read_number numReads = numberOfReads;
        const int numIters = SDIV(numReads, parallelReads);
        const std::size_t encodedSequencePitchInInts = getEncodedNumInts2Bit(upperBoundSequenceLength);

        cudaSetDevice(0); CUERR;

        const int numHashFuncs = numTables;
        const int firstHashFunc = firstTableId;
        const std::size_t signaturesRowPitchElements = numHashFuncs;

        ThreadPool::ParallelForHandle pforHandle;

        std::vector<Minhasher::Map_t> minhashTables(numTables);

        std::vector<int> tableIds(numTables);                
        std::vector<int> hashIds(numTables);
        
        std::iota(tableIds.begin(), tableIds.end(), 0);

        std::cout << "Constructing maps: ";
        for(int i = 0; i < numTables; i++){
            std::cout << (firstTableId + i) << ' ';
        }
        std::cout << '\n';

        for(auto& table : minhashTables){
            Minhasher::Map_t tmp(numReads);
            table = std::move(tmp);
        }

        auto showProgress = [&](auto totalCount, auto seconds){
            std::cerr << "Hashed " << totalCount << " / " << numReads << " reads. Elapsed time: " 
                        << seconds << " seconds." << '\r';
            if(totalCount == numReads){
                std::cerr << '\n';
            }
        };

        auto updateShowProgressInterval = [](auto duration){
            return duration;
        };

        ProgressThread<read_number> progressThread(numReads, showProgress, updateShowProgressInterval);

        ThreadPool threadPool(numThreads);

        SimpleAllocationDevice<unsigned int, 1> d_sequenceData(encodedSequencePitchInInts * parallelReads);
        SimpleAllocationDevice<int, 0> d_lengths(parallelReads);

        SimpleAllocationPinnedHost<read_number, 0> h_indices(parallelReads);
        SimpleAllocationDevice<read_number, 0> d_indices(parallelReads);

        SimpleAllocationPinnedHost<std::uint64_t, 0> h_signatures(signaturesRowPitchElements * parallelReads);
        SimpleAllocationDevice<std::uint64_t, 0> d_signatures(signaturesRowPitchElements * parallelReads);

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
                0,
                stream,
                numThreads
            );
        
            readStorage.gatherSequenceLengthsToGpuBufferAsync(
                d_lengths,
                0,
                d_indices,
                curBatchsize,
                stream
            );

            callMinhashSignaturesKernel_async(
                d_signatures,
                signaturesRowPitchElements,
                d_sequenceData,
                encodedSequencePitchInInts,
                curBatchsize,
                d_lengths,
                kmersize,
                numHashFuncs,
                firstHashFunc,
                stream
            );

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

                    minhasher.insertSequenceIntoExternalTables(h_signatures + signaturesRowPitchElements * localId, 
                        numHashFuncs,
                        readId,                                                     
                        tableIds,
                        minhashTables
                    );
                    
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

        return minhashTables;
    }

    int loadTablesFromFileAndAssignToMinhasher(const std::string& filename, 
                                            Minhasher& minhasher, 
                                            int numTablesToLoad, 
                                            int firstTableId,
                                            std::size_t availableMemory){

        std::cerr << "available before loading maps: " << availableMemory << "\n";
        
        int assignedNumMaps = 0;

        //load as many transformed tables from file as possible and move them to minhasher
        std::ifstream instream(filename, std::ios::binary);
        for(int i = 0; i < numTablesToLoad; i++){
            try{
                std::cerr << "try loading table " << i << "\n";
                Minhasher::Map_t table{};
                table.readFromStream(instream);
                std::size_t tablesize = table.allocationSizeInBytes();
                if(availableMemory > tablesize){
                    availableMemory -= tablesize;

                    minhasher.moveassignMap(firstTableId + i, std::move(table));

                    std::cerr << "available after loading table " << i << ": " << (getAvailableMemoryInKB() * 1024) << "\n";
                    assignedNumMaps++;
                    std::cerr << "usable num maps = " << assignedNumMaps << "\n";
                }else if(availableMemory == tablesize){
                    availableMemory -= tablesize;

                    minhasher.moveassignMap(firstTableId + i, std::move(table));
                    
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


    void validateReadstorage(const DistributedReadStorage& readStorage, const FileOptions& fileOptions){
        std::cerr << "validating data in readstorage\n";

        std::vector<read_number> indicesBuffer;
        std::vector<Read> readsBuffer;

        constexpr int batchsize = 16000;

        const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();
        const int sequencePitchInInts = getEncodedNumInts2Bit(maximumSequenceLength);
        const int qualityPitchInBytes = (SDIV(maximumSequenceLength, 32) * 32);

        ThreadPool threadPool;
        auto gatherHandleQ = readStorage.makeGatherHandleQualities();
        auto gatherHandleS = readStorage.makeGatherHandleSequences();
        SimpleAllocationPinnedHost<char> h_qualities(qualityPitchInBytes * batchsize);
        SimpleAllocationDevice<char> d_qualities(qualityPitchInBytes * batchsize);
        SimpleAllocationPinnedHost<read_number> h_readids(batchsize);
        SimpleAllocationDevice<read_number> d_readids(batchsize);
        SimpleAllocationPinnedHost<unsigned int> h_sequences(sequencePitchInInts * batchsize);
        SimpleAllocationDevice<unsigned int> d_sequences(sequencePitchInInts * batchsize);

        cudaStream_t stream;
        cudaStreamCreate(&stream); CUERR;

        bool oneIter = true;
        const bool withQuality = readStorage.canUseQualityScores();

        auto isValidBase = [](char c){
            constexpr std::array<char, 10> validBases{'A','C','G','T','a','c','g','t'};
            return validBases.end() != std::find(validBases.begin(), validBases.end(), c);
        };

        auto validateBatch = [&](){
            std::copy(indicesBuffer.begin(), indicesBuffer.end(), h_readids.get());
            cudaMemcpyAsync(d_readids.get(), h_readids.get(), indicesBuffer.size() * sizeof(read_number), H2D, stream); CUERR;

            readStorage.gatherSequenceDataToGpuBufferAsync(
                &threadPool,
                gatherHandleS,
                d_sequences.get(),
                sequencePitchInInts,
                h_readids.get(),
                d_readids.get(),
                indicesBuffer.size(),
                0,
                stream,
                16
            );
            if(withQuality){
                readStorage.gatherQualitiesToGpuBufferAsync(
                    &threadPool,
                    gatherHandleQ,
                    d_qualities.get(),
                    qualityPitchInBytes,
                    h_readids.get(),
                    d_readids.get(),
                    indicesBuffer.size(),
                    0,
                    stream,
                    16
                );
            }

            if(withQuality) { 
                cudaMemcpyAsync(h_qualities.get(), d_qualities.get(), sizeof(char) * qualityPitchInBytes * indicesBuffer.size(), D2H, stream); CUERR; 
            }
            cudaMemcpyAsync(h_sequences.get(), d_sequences.get(), sizeof(unsigned int) * sequencePitchInInts * indicesBuffer.size(), D2H, stream); CUERR;

            cudaStreamSynchronize(stream); CUERR;

            auto func = [&](int begin, int end, int /*threadId*/){
                for(int i = begin; i < end; i++){
                    // std::cerr << indicesBuffer[i] << "\n";
                    // std::cerr << "expected " << readsBuffer[i].quality << "\n";
                    // std::cerr << "got      ";
                    //     for(int l = 0; l < readsBuffer[i].quality.size(); l++){
                    //         std::cerr << h_qualities[qualityPitchInBytes * i + l];
                    //     }
                    //     std::cerr << "\n";

                    const int sequenceLength = readsBuffer[i].sequence.size();
    
                    std::string seqstring = get2BitString(h_sequences.get() + i * sequencePitchInInts, sequenceLength);
    
                    bool ok = true;
                    for(int k = 0; k < sequenceLength && ok; k++){                                
    
                        if(isValidBase(readsBuffer[i].sequence[k]) && readsBuffer[i].sequence[k] != seqstring[k]){
                            ok = false;
                            std::cerr << "error at sequence read " << indicesBuffer[i] << " position " << k << "\n";
                            std::cerr << "expected " << readsBuffer[i].sequence << "\n";
                            std::cerr << "got      " << seqstring << "\n";
                        }
                    }
                    if(withQuality){
                        const int qualityLength = readsBuffer[i].sequence.size();

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

        cudaStreamDestroy(stream);

        std::cerr << "validated data in readstorage\n";
    }



    void validateMinhasher(const Minhasher& minhasher, const DistributedReadStorage& readStorage, const FileOptions& fileOptions){
        std::cerr << "validating data in minhasher\n";

        std::vector<read_number> indicesBuffer;
        std::vector<Read> readsBuffer;
        
        constexpr std::int64_t batchsize = 16000;

        const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();
        const int sequencePitchInInts = getEncodedNumInts2Bit(maximumSequenceLength);
        
        

        ThreadPool threadPool;
        
        auto gatherHandleS = readStorage.makeGatherHandleSequences();
        
        
        SimpleAllocationPinnedHost<int> h_lengths(batchsize);
        SimpleAllocationDevice<int> d_lengths(batchsize);
        SimpleAllocationPinnedHost<read_number> h_readids(batchsize);
        SimpleAllocationDevice<read_number> d_readids(batchsize);
        SimpleAllocationPinnedHost<unsigned int> h_sequences(sequencePitchInInts * batchsize);
        SimpleAllocationDevice<unsigned int> d_sequences(sequencePitchInInts * batchsize);

        cudaStream_t stream;
        cudaStreamCreate(&stream); CUERR;

        auto validateBatch = [&](){

            ThreadPool::ParallelForHandle pforHandle;

            std::copy(indicesBuffer.begin(), indicesBuffer.end(), h_readids.get());
            cudaMemcpyAsync(d_readids.get(), h_readids.get(), indicesBuffer.size() * sizeof(read_number), H2D, stream); CUERR;

            readStorage.gatherSequenceDataToGpuBufferAsync(
                &threadPool,
                gatherHandleS,
                d_sequences.get(),
                sequencePitchInInts,
                h_readids.get(),
                d_readids.get(),
                indicesBuffer.size(),
                0,
                stream,
                16
            );

            readStorage.gatherSequenceLengthsToGpuBufferAsync(
                d_lengths.get(),
                0,
                d_readids.get(),
                indicesBuffer.size(),            
                stream
            );

            

            cudaMemcpyAsync(h_lengths.get(), d_lengths.get(), sizeof(int) * indicesBuffer.size(), D2H, stream); CUERR;
            cudaMemcpyAsync(h_sequences.get(), d_sequences.get(), sizeof(unsigned int) * sequencePitchInInts * indicesBuffer.size(), D2H, stream); CUERR;
            

            cudaStreamSynchronize(stream); CUERR;
            
            
            auto func1 = [&](int begin, int end, int /*threadId*/){
                Minhasher::Handle minhashHandle;
                std::vector<std::string> sequences;

                //should not hit assertion if everything is ok with minhasher
                for(int i = begin; i < end; i++){               
                    sequences.emplace_back(get2BitString(h_sequences.get() + i * sequencePitchInInts, h_lengths[i]));

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


    BuiltDataStructure<Minhasher> build_minhasher(const FileOptions &fileOptions,
                                                const RuntimeOptions &runtimeOptions,
                                                const MemoryOptions& memoryOptions,
                                                std::uint64_t nReads,
                                                const CorrectionOptions &correctionOptions,
                                                const GpuReadStorageWithFlags &readStoragewFlags)
    {

        BuiltDataStructure<Minhasher> result;
        auto &minhasher = result.data;

        auto identity = [](auto i) { return i; };

        Minhasher::MinhashOptions minhashOptions;
        minhashOptions.k = correctionOptions.kmerlength;
        minhashOptions.maps = correctionOptions.numHashFunctions;
        minhashOptions.numResultsPerMapQueryThreshold 
            = calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage);


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

            read_number numReads = readStorage.getNumberOfReads();

            auto sequencehandle = readStorage.makeGatherHandleSequences();
            //auto lengthhandle = readStorage.makeGatherHandleLengths();
            size_t sequencepitch = getEncodedNumInts2Bit(readStorage.getSequenceLengthUpperBound()) * sizeof(int);

            const std::string tmpmapsFilename = fileOptions.tempdirectory + "/tmpmaps";
            std::ofstream outstream(tmpmapsFilename, std::ios::binary);
            if(!outstream){
                throw std::runtime_error("Could not open temp file " + tmpmapsFilename + "!");
            }


            std::size_t writtenTableBytes = 0;

            
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
            std::cerr << "available: " << maxMemoryForTables 
                    << ",memoryForHashtables: " << memoryOptions.memoryForHashtables
                    << ", memoryTotalLimit: " << memoryOptions.memoryTotalLimit
                    << ", rsHostUsage: " << memoryUsageOfReadStorage.host << "\n";

            maxMemoryForTables = std::min(maxMemoryForTables, 
                                    std::min(memoryOptions.memoryForHashtables, totalLimit));

            std::cerr << "maxMemoryForTables = " << maxMemoryForTables << " bytes\n";


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

            auto getSequenceData = [&](char* dest, int sequencepitchInBytes, const read_number* indices, int numIndices){
                readStorage.gatherSequenceDataToHostBuffer(
                    sequencehandle,
                    (unsigned int*)dest,
                    sequencepitchInBytes / sizeof(unsigned int),
                    indices,
                    numIndices,
                    1);          
            };

            auto getSequenceLength = [&](DistributedReadStorage::Length_t* dest, const read_number* indices, int numIndices){
                readStorage.gatherSequenceLengthsToHostBuffer(
                    dest,
                    indices,
                    numIndices);
            };



            int numSavedTables = 0;

            int numConstructedTables = 0;
            std::vector<Minhasher::Map_t> cachedConstructedTables;
            std::size_t bytesOfCachedConstructedTables = 0;
            bool allowCaching = false;

            while(numConstructedTables < minhashOptions.maps && maxMemoryForTables > (writtenTableBytes + bytesOfCachedConstructedTables)){

                int maxNumTables = 0;

                auto updateMaxNumTables = [&](){
                    std::size_t requiredMemPerTable = Minhasher::Map_t::getRequiredSizeInBytesBeforeCompaction(nReads);
                    maxNumTables = (maxMemoryForTables - bytesOfCachedConstructedTables) / requiredMemPerTable;
                    maxNumTables -= 2; // need free memory of 2 tables to perform transformation 
                    std::cerr << "requiredMemPerTable = " << requiredMemPerTable << "\n";
                    std::cerr << "maxNumTables = " << maxNumTables << "\n";
                };

                updateMaxNumTables();

                //if at least 75 percent of all tables can be constructed in first iteration, keep all constructed tables in memory
                //else save constructed tables to file if there are less than minhashOptions.maps
                if(numConstructedTables == 0 && float(maxNumTables) / minhashOptions.maps >= 0.75){
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
                            cachedConstructedTables[i].writeToStream(outstream);
                            numSavedTables++;
                            writtenTableBytes = outstream.tellp();
        
                            std::cerr << "tablesize = " << cachedConstructedTables[i].numBytes() << "\n";
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

                    int currentIterNumTables = std::min(minhashOptions.maps - numConstructedTables, maxNumTables);

                    std::vector<Minhasher::Map_t> minhashTables = constructTablesWithGpuHashing(
                        minhasher, 
                        currentIterNumTables, 
                        numConstructedTables,
                        minhasher.minparams.k,
                        readStorage.getNumberOfReads(),
                        readStorage.getSequenceLengthUpperBound(),
                        runtimeOptions.threads,
                        readStorage
                    );

                    // std::vector<Minhasher::Map_t> minhashTables = constructTables(
                    //     minhasher, 
                    //     currentIterNumTables, 
                    //     numConstructedTables,
                    //     readStorage.getNumberOfReads(),
                    //     readStorage.getSequenceLengthUpperBound(),
                    //     runtimeOptions.threads,
                    //     getSequenceData,
                    //     getSequenceLength,
                    //     readStorage
                    // );


                    //check free gpu mem for transformation
                    std::size_t estRequiredFreeGpuMem = estimateGpuMemoryForTransformKeyValueMap(minhashTables[0]);
                    std::size_t freeGpuMem, totalGpuMem;
                    cudaMemGetInfo(&freeGpuMem, &totalGpuMem); CUERR;

                    std::size_t availableMemoryToSaveGpuPartitions = totalLimit;
                    //account for constructed tables in previous iteration
                    if(availableMemoryToSaveGpuPartitions > bytesOfCachedConstructedTables){
                        availableMemoryToSaveGpuPartitions -= bytesOfCachedConstructedTables;
                    }else{
                        availableMemoryToSaveGpuPartitions = 0;
                    }
                    //account for constructed tables in current iteration and space needed by transformation
                    for(int i = 0; i < 2 + int(minhashTables.size()); i++){
                        const std::size_t requiredMemPerTable = Minhasher::Map_t::getRequiredSizeInBytesBeforeCompaction(nReads);
                        if(availableMemoryToSaveGpuPartitions > requiredMemPerTable){
                            availableMemoryToSaveGpuPartitions -= requiredMemPerTable;
                        }else{
                            availableMemoryToSaveGpuPartitions = 0;
                            break;
                        }
                    }               
                    
                    std::cerr << "availableMemoryToSaveGpuPartitions: " << availableMemoryToSaveGpuPartitions << "\n";

                    DistributedReadStorage::SavedGpuData savedReadstorageGpuData;
                    const std::string rstempfile = fileOptions.tempdirectory+"/rstemp";
                    bool didSaveGpudata = false;

                    //if there is more than 10% gpu memory missing, make room for it
                    //if(std::size_t(freeGpuMem * 1.1) < estRequiredFreeGpuMem){
                    {
                        std::ofstream rstempostream(rstempfile, std::ios::binary);
                        std::size_t requiredMemPerTable = Minhasher::Map_t::getRequiredSizeInBytesBeforeCompaction(nReads);
                        savedReadstorageGpuData = std::move(readStorage.saveGpuDataAndFreeGpuMem(rstempostream, availableMemoryToSaveGpuPartitions));

                        didSaveGpudata = true;
                    }

                    
                    
                    //if all tables could be constructed at once, no need to save them to temporary file
                    if(minhashOptions.maps == int(minhashTables.size())){

                        for(int i = 0; i < int(minhashTables.size()); i++){
                            int globalTableId = numConstructedTables;
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
                            filehelpers::removeFile(rstempfile);
                        }
                        
                    }else{

                        for(int i = 0; i < int(minhashTables.size()); i++){
                            int globalTableId = numConstructedTables;
                            int maxValuesPerKey = minhasher.getResultsPerMapThreshold();                    
                            std::cerr << "Transforming table " << globalTableId << ". ";
                            transform_keyvaluemap_gpu(minhashTables[i], runtimeOptions.deviceIds, maxValuesPerKey);

                            numConstructedTables++;

                            if(allowCaching){
                                bytesOfCachedConstructedTables += minhashTables[i].numBytes();
                                cachedConstructedTables.emplace_back(std::move(minhashTables[i]));
    
                                std::cerr << "cached " << cachedConstructedTables.size() << " constructed tables in memory\n";
    
                                if(maxMemoryForTables <= bytesOfCachedConstructedTables){
                                    break;
                                }
                            }else{
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
                        }
                        minhashTables.clear();

                        if(didSaveGpudata){
                            std::ifstream rstempistream(rstempfile, std::ios::binary);
                            readStorage.allocGpuMemAndLoadGpuData(rstempistream, savedReadstorageGpuData);
                            savedReadstorageGpuData.clear();
                        }

                        if(int(cachedConstructedTables.size()) + numSavedTables >= minhashOptions.maps 
                                    || maxMemoryForTables < writtenTableBytes){

                            outstream.flush();

                            //discard any cached table such that size of cached tables + size of tables in file < memory limit
                            std::size_t totalTableBytes = writtenTableBytes;
                            int end = 0;
                            for(int i = 0; i < int(cachedConstructedTables.size()); i++){
                                const auto& table = cachedConstructedTables[i];
                                if(totalTableBytes + table.numBytes() <= maxMemoryForTables){
                                    totalTableBytes += table.numBytes();
                                    end++;
                                }else{
                                    break;
                                }
                            }
                            cachedConstructedTables.erase(cachedConstructedTables.begin() + end, cachedConstructedTables.end());
                            
                            int usableNumMaps = loadTablesFromFileAndAssignToMinhasher(
                                                        tmpmapsFilename, 
                                                        minhasher, 
                                                        numSavedTables, 
                                                        0,
                                                        maxMemoryForTables);

                            for(int i = 0; i < int(cachedConstructedTables.size()) && usableNumMaps < minhashOptions.maps; i++){
                                auto& table = cachedConstructedTables[i];
                                minhasher.moveassignMap(usableNumMaps, std::move(table));
                                
                                usableNumMaps++;
                            }
        
                            filehelpers::removeFile(tmpmapsFilename);
                            if(didSaveGpudata){
                                filehelpers::removeFile(rstempfile);
                            }
        
                            minhasher.minhashTables.resize(usableNumMaps);
                            std::cout << "Can use " << usableNumMaps << " out of specified " << minhasher.minparams.maps << " tables\n";
                            minhasher.minparams.maps = usableNumMaps;
                        } 
                    } 
                }else{
                    //all constructed tables have been saved to file, and no table is cached

                    outstream.flush();

                    int usableNumMaps = loadTablesFromFileAndAssignToMinhasher(
                                                    tmpmapsFilename, 
                                                    minhasher, 
                                                    numSavedTables, 
                                                    0,
                                                    maxMemoryForTables);

                    minhasher.minhashTables.resize(usableNumMaps);
                    std::cout << "Can use " << usableNumMaps << " out of specified " << minhasher.minparams.maps << " tables\n";
                    minhasher.minparams.maps = usableNumMaps;
                }
            }
        }

        return result;
    }







//###########################################################################
//###########################################################################
//###########################################################################
//###########################################################################
//###########################################################################
//###########################################################################
//###########################################################################
//###########################################################################
//###########################################################################
//###########################################################################








BuiltDataStructure<GpuReadStorageWithFlags> buildGpuReadStorage2(const FileOptions& fileOptions,
                                                const RuntimeOptions& runtimeOptions,
                                                bool useQualityScores,
                                                read_number expectedNumberOfReads,
                                                int expectedMinimumReadLength,
                                                int expectedMaximumReadLength){

        



        if(fileOptions.load_binary_reads_from != ""){
            BuiltDataStructure<GpuReadStorageWithFlags> result;
            auto& readStorage = result.data.readStorage;

            TIMERSTARTCPU(load_from_file);
            readStorage.loadFromFile(fileOptions.load_binary_reads_from, runtimeOptions.deviceIds);
            TIMERSTOPCPU(load_from_file);
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
            //int Ncount = 0;
            //std::map<int,int> nmap{};

            BuiltDataStructure<GpuReadStorageWithFlags> result;
            DistributedReadStorage& readstorage = result.data.readStorage;
            //auto& validFlags = result.data.readIsValidFlags;

            readstorage = std::move(DistributedReadStorage{runtimeOptions.deviceIds, expectedNumberOfReads, useQualityScores, 
                                                            expectedMinimumReadLength, expectedMaximumReadLength});
            //validFlags.resize(expectedNumberOfReads, false);
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

            ThreadPool threadPool(runtimeOptions.threads);

            for(int i = 0; i < numBuffers; i++){
                indicesBuffers[i].reserve(maxbuffersize);
                readsBuffers[i].reserve(maxbuffersize);
                canBeUsed[i] = true;
            }

            int bufferindex = 0;
            read_number globalReadId = 0;

            tpa = std::chrono::system_clock::now();

            for(const auto& inputfile : fileOptions.inputfiles){
                std::cout << "Parsing " << inputfile << "\n";

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
                            threadPool.enqueue([&, indicesBufferPtr, readsBufferPtr, bufferindex](){
                                //std::cerr << "buffer " << bufferindex << " running\n";
                                int nmodcounter = 0;

                                for(int i = 0; i < int(readsBufferPtr->size()); i++){
                                    read_number readId = (*indicesBufferPtr)[i];
                                    auto& read = (*readsBufferPtr)[i];
                                    checkRead(readId, read, nmodcounter);
                                }

                                readstorage.setReads(&threadPool, *indicesBufferPtr, *readsBufferPtr);

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
                }

                readstorage.setReads(&threadPool, *indicesBufferPtr, *readsBufferPtr);

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

            // std::cerr << "occurences of n/N:\n";
            // for(const auto& p : nmap){
            //     std::cerr << p.first << " " << p.second << '\n';
            // }

            readstorage.constructionIsComplete();

            return result;
        }

    }



BuiltGpuDataStructures buildGpuDataStructuresImpl2(
                        const CorrectionOptions& correctionOptions,
                        const RuntimeOptions& runtimeOptions,
                        const MemoryOptions& memoryOptions,
                        const FileOptions& fileOptions,
                        bool saveDataStructuresToFile){                                                     

        BuiltGpuDataStructures result;

        auto inputFileProperties = detail::getSequenceFilePropertiesFromFileOptions2(fileOptions);

        SequenceFileProperties totalInputFileProperties;
        totalInputFileProperties.nReads = std::accumulate(
            inputFileProperties.begin(), 
            inputFileProperties.end(), 
            std::uint64_t{0}, 
            [](const auto acc, const auto& e){return acc + e.nReads;}
        );
        totalInputFileProperties.minSequenceLength = std::min_element(
            inputFileProperties.begin(), 
            inputFileProperties.end(), 
            [](const auto& l, const auto& r){return l.minSequenceLength < r.minSequenceLength;}
        )->minSequenceLength;
        totalInputFileProperties.maxSequenceLength = std::max_element(
            inputFileProperties.begin(), 
            inputFileProperties.end(), 
            [](const auto& l, const auto& r){return l.maxSequenceLength < r.maxSequenceLength;}
        )->maxSequenceLength;

        TIMERSTARTCPU(build_readstorage);
        result.builtReadStorage = buildGpuReadStorage2(fileOptions,
                                                  runtimeOptions,
                                                  correctionOptions.useQualityScores,
                                                  totalInputFileProperties.nReads,
                                                  totalInputFileProperties.minSequenceLength,
                                                  totalInputFileProperties.maxSequenceLength);
        TIMERSTOPCPU(build_readstorage);

        const auto& readStorage = result.builtReadStorage.data.readStorage;

#ifdef VALIDATE_READSTORAGE
        validateReadstorage(readStorage, fileOptions);
#endif 

        std::cout << "Using " << readStorage.lengthStorage.getRawBitsPerLength() << " bits per read to store its length\n";

        if(saveDataStructuresToFile && fileOptions.save_binary_reads_to != "") {
            std::cout << "Saving reads to file " << fileOptions.save_binary_reads_to << std::endl;
            TIMERSTARTCPU(save_to_file);
            readStorage.saveToFile(fileOptions.save_binary_reads_to);
            TIMERSTOPCPU(save_to_file);
    		std::cout << "Saved reads" << std::endl;
    	}

        result.sequenceFileProperties.nReads = readStorage.getNumberOfReads();
        result.sequenceFileProperties.maxSequenceLength = readStorage.getStatistics().maximumSequenceLength;
        result.sequenceFileProperties.minSequenceLength = readStorage.getStatistics().minimumSequenceLength;
        result.inputFileProperties = std::move(inputFileProperties);

        std::cout << "Reads with ambiguous bases: " << readStorage.getNumberOfReadsWithN() << std::endl;

        TIMERSTARTCPU(build_minhasher);
        result.builtMinhasher = build_minhasher(fileOptions, 
            runtimeOptions, 
            memoryOptions,
            result.sequenceFileProperties.nReads, 
            correctionOptions,
            result.builtReadStorage.data);
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



    BuiltGpuDataStructures buildGpuDataStructures2(
        const CorrectionOptions& correctionOptions,
        const RuntimeOptions& runtimeOptions,
        const MemoryOptions& memoryOptions,
        const FileOptions& fileOptions){

        return buildGpuDataStructuresImpl2(
            correctionOptions,
            runtimeOptions,
            memoryOptions,
            fileOptions,
            false
        );
    }

    BuiltGpuDataStructures buildAndSaveGpuDataStructures2(
        const CorrectionOptions& correctionOptions,
        const RuntimeOptions& runtimeOptions,
        const MemoryOptions& memoryOptions,
        const FileOptions& fileOptions){                                                     

        return buildGpuDataStructuresImpl2(
            correctionOptions,
            runtimeOptions,
            memoryOptions,
            fileOptions,
            true
        );
    }

}
}


#endif
