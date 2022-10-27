
#include <gpu/gpuminhasherconstruction.cuh>

#include <gpu/gpuminhasher.cuh>
#include <gpu/fakegpuminhasher.cuh>
#include <gpu/singlegpuminhasher.cuh>
#include <gpu/multigpuminhasher.cuh>
#include <gpu/global_cuda_stream_pool.cuh>
#include <minhasherlimit.hpp>

#include <options.hpp>

#include <array>
#include <vector>
#include <memory>
#include <utility>

#include <thrust/sequence.h>


namespace care{
namespace gpu{

        std::string to_string(GpuMinhasherType type){
            switch(type){
                case GpuMinhasherType::Fake: return "FakeGpu";
                case GpuMinhasherType::FakeSingleHash: return "FakeGpuSingleHash";
                case GpuMinhasherType::Single: return "SingleGpu";
                case GpuMinhasherType::SingleSingleHash: return "SingleGpuSingleHash";
                case GpuMinhasherType::Multi: return "MultiGpu";
                case GpuMinhasherType::MultiSingleHash: return "MultiGpuSingleHash";
                case GpuMinhasherType::ReplicatedSingle: return "ReplicatedSingle";
                case GpuMinhasherType::None: return "None";
                default: return "Unknown";
            }
        }

        void constructGpuMinhasherFromReadStorage(
            const ProgramOptions& programOptions,
            const GpuReadStorage& gpuReadStorage,
            GpuMinhasher* gpuMinhasher
        ){

            nvtx::ScopedRange sr("constructGpuMinhasherFromReadStorage", 0);

            auto& readStorage = gpuReadStorage;
            const auto& deviceIds = programOptions.deviceIds;
    
            int deviceId = deviceIds[0];
    
            cub::SwitchDevice sd{deviceId};
    
            const int requestedNumberOfMaps = programOptions.numHashFunctions;
    
            const read_number numReads = readStorage.getNumberOfReads();
            const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();

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

            constexpr int numStreams = 2;

            rmm::cuda_stream_pool* streamPool = streampool::get_current_device_pool();

            std::vector<ReadStorageHandle> sequenceHandles(numStreams);
            std::vector<cudaStream_t> streams(numStreams);
            std::vector<std::vector<read_number>> vec_h_indices(numStreams, std::vector<read_number>(parallelReads));

            std::vector<rmm::device_uvector<unsigned int>> vec_d_sequenceData;
            std::vector<rmm::device_uvector<int>> vec_d_lengths;
            std::vector<rmm::device_uvector<read_number>> vec_d_indices;

            for(int i = 0; i < numStreams; i++){
                sequenceHandles[i] = gpuReadStorage.makeHandle();
                streams[i] = streamPool->get_stream();

                vec_d_sequenceData.emplace_back(encodedSequencePitchInInts * parallelReads, streams[i], mr);
                vec_d_lengths.emplace_back(parallelReads, streams[i], mr);
                vec_d_indices.emplace_back(parallelReads, streams[i], mr);
            }    
            
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
    
                //Hacky way to limit gpu memory usage of hash tables
                constexpr std::size_t hackbytes = 1024ul * 1024ul * 1024ul * 1ul;
                std::vector<char*> hackbuffers(programOptions.deviceIds.size());
                for(std::size_t i = 0; i < programOptions.deviceIds.size(); i++){
                    cub::SwitchDevice sd{programOptions.deviceIds[i]};
                    std::size_t free, total;
                    CUDACHECK(cudaMemGetInfo(&free, &total));
                    if(free > hackbytes){
                        CUDACHECK(cudaMalloc(&hackbuffers[i], hackbytes));
                    }
                }

                int addedHashFunctions = gpuMinhasher->addHashTables(remainingHashFunctions,h_hashfunctionNumbers.data(), streams[0]);
                CUDACHECK(cudaStreamSynchronize(streams[0]));

                for(std::size_t i = 0; i < programOptions.deviceIds.size(); i++){
                    cub::SwitchDevice sd{programOptions.deviceIds[i]};
                    CUDACHECK(cudaFree(hackbuffers[i]));
                }
    
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
    
                for (int iter = 0, streamIndex = 0; iter < numIters; iter++, streamIndex = (streamIndex + 1) % numStreams){
                    read_number readIdBegin = iter * parallelReads;
                    read_number readIdEnd = std::min((iter + 1) * parallelReads, numReads);
    
                    const std::size_t curBatchsize = readIdEnd - readIdBegin;

                    thrust::sequence(
                        rmm::exec_policy_nosync(streams[streamIndex], mr),
                        vec_d_indices[streamIndex].begin(),
                        vec_d_indices[streamIndex].end(),
                        readIdBegin
                    );

                    gpuReadStorage.gatherSequenceLengths(
                        sequenceHandles[streamIndex],
                        vec_d_lengths[streamIndex].data(),
                        vec_d_indices[streamIndex].data(),
                        curBatchsize,
                        streams[streamIndex]
                    );
    
                    std::iota(vec_h_indices[streamIndex].begin(), vec_h_indices[streamIndex].begin() + curBatchsize, readIdBegin);

                    gpuReadStorage.gatherContiguousSequences(
                        sequenceHandles[streamIndex],
                        vec_d_sequenceData[streamIndex].data(),
                        encodedSequencePitchInInts,
                        readIdBegin,
                        curBatchsize,
                        streams[streamIndex],
                        mr
                    );
    
                    gpuMinhasher->insert(
                        vec_d_sequenceData[streamIndex].data(),
                        curBatchsize,
                        vec_d_lengths[streamIndex].data(),
                        encodedSequencePitchInInts,
                        vec_d_indices[streamIndex].data(),
                        vec_h_indices[streamIndex].data(),
                        alreadyExistingHashFunctions,
                        addedHashFunctions,
                        h_hashfunctionNumbers.data(),
                        streams[streamIndex],
                        mr
                    );
                }

                for(int i = 0; i < numStreams; i++){
                    CUDACHECK(cudaStreamSynchronize(streams[i]));
                }

                int errorcount = gpuMinhasher->checkInsertionErrors(
                    alreadyExistingHashFunctions,
                    addedHashFunctions,
                    streams[0]
                );
                if(errorcount > 0){
                    throw std::runtime_error("An error occurred during hash table construction.");
                }
    
                CUDACHECK(cudaStreamSynchronize(streams[0]));
    
                std::cerr << "Compacting\n";
                if(tpForCompacting.getConcurrency() > 1){
                    gpuMinhasher->setThreadPool(&tpForCompacting);
                }else{
                    gpuMinhasher->setThreadPool(nullptr);
                }
                
                gpuMinhasher->compact(streams[0]);
                CUDACHECK(cudaStreamSynchronize(streams[0]));
    
                remainingHashFunctions -= addedHashFunctions;
            }
    
            gpuMinhasher->setThreadPool(nullptr); 
            
            for(int i = 0; i < numStreams; i++){
                gpuReadStorage.destroyHandle(sequenceHandles[i]);
            }
    
            gpuMinhasher->constructionIsFinished(streams[0]);
            CUDACHECK(cudaStreamSynchronize(streams[0]));
        }
    
    
        std::pair<std::unique_ptr<GpuMinhasher>, GpuMinhasherType>
        constructGpuMinhasherFromGpuReadStorage(
            const ProgramOptions& programOptions,
            const GpuReadStorage& gpuReadStorage,
            GpuMinhasherType requestedType
        ){
            #ifdef CARE_HAS_WARPCORE
                if(requestedType != GpuMinhasherType::Fake && programOptions.warpcore == 1 && programOptions.replicateGpuHashtables){
                    auto sgpuminhasher = std::make_unique<SingleGpuMinhasher>(
                        gpuReadStorage.getNumberOfReads(), 
                        calculateResultsPerMapThreshold(programOptions.estimatedCoverage), 
                        programOptions.kmerlength,
                        programOptions.hashtableLoadfactor
                    );

                    constructGpuMinhasherFromReadStorage(
                        programOptions,
                        gpuReadStorage,
                        sgpuminhasher.get()
                    );

                    std::cerr << "Creating ReplicatedSingleGpuMinhasher\n";

                    nvtx::push_range("replicate gpu minhasher", 1);
                    auto replicated = std::make_unique<ReplicatedSingleGpuMinhasher>(std::move(sgpuminhasher), programOptions.deviceIds);
                    nvtx::pop_range();

                    return {std::move(replicated), GpuMinhasherType::ReplicatedSingle};
                }else{
            #endif

                    std::unique_ptr<GpuMinhasher> gpuMinhasher;
                    GpuMinhasherType gpuMinhasherType = GpuMinhasherType::None;

                    auto makeFake = [&](){                
                        gpuMinhasher = std::make_unique<FakeGpuMinhasher>(
                            gpuReadStorage.getNumberOfReads(),
                            calculateResultsPerMapThreshold(programOptions.estimatedCoverage),
                            programOptions.kmerlength,
                            programOptions.hashtableLoadfactor
                        );

                        gpuMinhasherType = GpuMinhasherType::Fake;
                    };

                    #ifdef CARE_HAS_WARPCORE

                    auto makeSingle = [&](){
                        gpuMinhasher = std::make_unique<SingleGpuMinhasher>(
                            gpuReadStorage.getNumberOfReads(), 
                            calculateResultsPerMapThreshold(programOptions.estimatedCoverage), 
                            programOptions.kmerlength,
                            programOptions.hashtableLoadfactor
                        );

                        gpuMinhasherType = GpuMinhasherType::Single;
                    };

                    auto makeMulti = [&](){
                        auto layout = programOptions.gpuHashtableLayout == GpuDataLayout::FirstFit ? gpu::MultiGpuMinhasher::Layout::FirstFit 
                            : gpu::MultiGpuMinhasher::Layout::EvenShare;

                        gpuMinhasher = std::make_unique<MultiGpuMinhasher>(
                            layout,
                            gpuReadStorage.getNumberOfReads(), 
                            calculateResultsPerMapThreshold(programOptions.estimatedCoverage), 
                            programOptions.kmerlength,
                            programOptions.hashtableLoadfactor,
                            programOptions.deviceIds
                        );

                        gpuMinhasherType = GpuMinhasherType::Multi;
                    };

                    #endif

                    if(requestedType == GpuMinhasherType::Fake || programOptions.warpcore == 0){
                        makeFake();

                    #ifdef CARE_HAS_WARPCORE
                    }else if(requestedType == GpuMinhasherType::Single || programOptions.deviceIds.size() < 2){
                        makeSingle();
                    }else if(requestedType == GpuMinhasherType::Multi){
                        makeMulti();
                    #endif
                    }else{
                        makeFake();
                    }

                    if(programOptions.load_hashtables_from != "" && gpuMinhasher->canLoadFromStream()){

                        std::ifstream is(programOptions.load_hashtables_from);
                        assert((bool)is);

                        const int loadedMaps = gpuMinhasher->loadFromStream(is, programOptions.numHashFunctions);

                        std::cout << "Loaded " << loadedMaps << " hash tables from " << programOptions.load_hashtables_from << std::endl;
                    }else{
                        constructGpuMinhasherFromReadStorage(
                            programOptions,
                            gpuReadStorage,
                            gpuMinhasher.get()
                        );
                    }

                    return {std::move(gpuMinhasher), gpuMinhasherType};
            #ifdef CARE_HAS_WARPCORE
                }
            #endif
        }
    
    
}
}
