
#include <gpu/gpuminhasherconstruction.cuh>

#include <gpu/singlegpuminhasher.cuh>
#include <minhasherlimit.hpp>

#include <options.hpp>

#include <memory>
#include <utility>


namespace care{
namespace gpu{
    
    void constructFromReadStorageSingle(
        const ProgramOptions& programOptions,
        const GpuReadStorage& gpuReadStorage,
        GpuMinhasher* gpuMinhasher
    ){
        
        auto& readStorage = gpuReadStorage;
        const auto& deviceIds = programOptions.deviceIds;

        int deviceId = deviceIds[0];

        cub::SwitchDevice sd{deviceId};

        const int requestedNumberOfMaps = programOptions.numHashFunctions;

        const read_number numReads = readStorage.getNumberOfReads();
        const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();

        auto sequencehandle = gpuReadStorage.makeHandle();
        const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);

        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

        constexpr read_number parallelReads = 1000000;
        const int numIters = SDIV(numReads, parallelReads);

        const MemoryUsage memoryUsageOfReadStorage = readStorage.getMemoryInfo();
        std::size_t totalLimit = programOptions.memoryTotalLimit;
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
        //         << ",memoryForHashtables: " << programOptions.memoryForHashtables
        //         << ", memoryTotalLimit: " << programOptions.memoryTotalLimit
        //         << ", rsHostUsage: " << memoryUsageOfReadStorage.host << "\n";

        maxMemoryForTables = std::min(maxMemoryForTables, 
                                std::min(programOptions.memoryForHashtables, totalLimit));

        std::cerr << "maxMemoryForTables = " << maxMemoryForTables << " bytes\n";

        const int hashFunctionOffset = 0;

        
        std::vector<int> usedHashFunctionNumbers;

        cudaStream_t stream = cudaStreamPerThread;
        
        rmm::device_uvector<unsigned int> d_sequenceData(encodedSequencePitchInInts * parallelReads, stream, mr);
        rmm::device_uvector<int> d_lengths(parallelReads, stream, mr);
        rmm::device_uvector<read_number> d_indices(parallelReads, stream, mr);
        
        helpers::SimpleAllocationPinnedHost<read_number, 0> h_indices(parallelReads);

        
        ThreadPool tpForHashing(programOptions.threads);
        ThreadPool tpForCompacting(std::min(2,programOptions.threads));

        
        gpuMinhasher->setHostMemoryLimitForConstruction(maxMemoryForTables);
        gpuMinhasher->setDeviceMemoryLimitsForConstruction({1*1024*1024*1024});
        
        //std::size_t bytesOfCachedConstructedTables = 0;
        int remainingHashFunctions = requestedNumberOfMaps;
        bool keepGoing = true;

        while(remainingHashFunctions > 0 && keepGoing){

            gpuMinhasher->setThreadPool(&tpForHashing);

            const int alreadyExistingHashFunctions = requestedNumberOfMaps - remainingHashFunctions;
            std::vector<int> h_hashfunctionNumbers(remainingHashFunctions);
            std::iota(
                h_hashfunctionNumbers.begin(),
                h_hashfunctionNumbers.end(),
                alreadyExistingHashFunctions + hashFunctionOffset
            );

            int addedHashFunctions = gpuMinhasher->addHashTables(remainingHashFunctions,h_hashfunctionNumbers.data(), stream);

            if(addedHashFunctions == 0){
                keepGoing = false;
                break;
            }

            std::cout << "Constructing maps: ";
            for(int i = 0; i < addedHashFunctions; i++){
                std::cout << (alreadyExistingHashFunctions + i) << "(" << (hashFunctionOffset + alreadyExistingHashFunctions + i) << ") ";
            }
            std::cout << '\n';

            usedHashFunctionNumbers.insert(usedHashFunctionNumbers.end(), h_hashfunctionNumbers.begin(), h_hashfunctionNumbers.begin() + addedHashFunctions);

            for (int iter = 0; iter < numIters; iter++){
                read_number readIdBegin = iter * parallelReads;
                read_number readIdEnd = std::min((iter + 1) * parallelReads, numReads);

                const std::size_t curBatchsize = readIdEnd - readIdBegin;

                std::iota(h_indices.get(), h_indices.get() + curBatchsize, readIdBegin);

                CUDACHECK(cudaMemcpyAsync(d_indices.data(), h_indices, sizeof(read_number) * curBatchsize, H2D, stream));

                gpuReadStorage.gatherSequences(
                    sequencehandle,
                    d_sequenceData.data(),
                    encodedSequencePitchInInts,
                    makeAsyncConstBufferWrapper(h_indices.data()),
                    d_indices.data(),
                    curBatchsize,
                    stream,
                    mr
                );
            
                gpuReadStorage.gatherSequenceLengths(
                    sequencehandle,
                    d_lengths.data(),
                    d_indices.data(),
                    curBatchsize,
                    stream
                );

                gpuMinhasher->insert(
                    d_sequenceData.data(),
                    curBatchsize,
                    d_lengths.data(),
                    encodedSequencePitchInInts,
                    d_indices.data(),
                    h_indices,
                    alreadyExistingHashFunctions,
                    addedHashFunctions,
                    h_hashfunctionNumbers.data(),
                    stream,
                    mr
                );

                CUDACHECK(cudaStreamSynchronize(stream));
            }

            CUDACHECK(cudaStreamSynchronize(stream));

            std::cerr << "Compacting\n";
            if(tpForCompacting.getConcurrency() > 1){
                gpuMinhasher->setThreadPool(&tpForCompacting);
            }else{
                gpuMinhasher->setThreadPool(nullptr);
            }
            
            gpuMinhasher->compact(stream);
            CUDACHECK(cudaStreamSynchronize(stream));

            remainingHashFunctions -= addedHashFunctions;
        }

        gpuMinhasher->setThreadPool(nullptr); 
        
        gpuReadStorage.destroyHandle(sequencehandle);

        gpuMinhasher->constructionIsFinished(stream);
        CUDACHECK(cudaStreamSynchronize(stream));
    }


    #ifdef CARE_HAS_WARPCORE
        std::unique_ptr<SingleGpuMinhasher>
        constructSingleGpuMinhasherFromGpuReadStorage(
            const ProgramOptions& programOptions,
            const GpuReadStorage& gpuReadStorage
        ){
            
            auto gpuMinhasher = std::make_unique<SingleGpuMinhasher>(
                gpuReadStorage.getNumberOfReads(), 
                calculateResultsPerMapThreshold(programOptions.estimatedCoverage), 
                programOptions.kmerlength
            );

            constructFromReadStorageSingle(
                programOptions,
                gpuReadStorage,
                gpuMinhasher.get()
            );

            // gpuMinhasher->constructFromReadStorage(
            //     programOptions,
            //     gpuReadStorage.getNumberOfReads(),
            //     gpuReadStorage,
            //     gpuReadStorage.getSequenceLengthUpperBound(),
            //     programOptions.numHashFunctions
            // );

            return gpuMinhasher;
        }

    #endif
    
}
}