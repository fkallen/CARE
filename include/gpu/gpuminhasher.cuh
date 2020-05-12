#ifndef CARE_GPUMINHASHER_CUH
#define CARE_GPUMINHASHER_CUH

#include <config.hpp>

#include <gpu/gpuhashtable.cuh>
#include <gpu/distributedreadstorage.hpp>
#include <gpu/simpleallocation.cuh>
#include <gpu/minhashkernels.hpp>

#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>
#include <minhasher_transform.hpp>


#include <threadpool.hpp>


#include <vector>
#include <memory>
#include <limits>

namespace care{
namespace gpu{

    class GpuMinhasher{
    private:
        using HashTable = CpuReadOnlyMultiValueHashTable<kmer_type, read_number>;
    public:
        using Key_t = kmer_type;
        using Value_t = read_number;

        using Range_t = std::pair<const Value_t*, const Value_t*>;

        static constexpr int bits_kmer = sizeof(kmer_type) * 8;
        static constexpr std::uint64_t kmer_mask = (std::uint64_t(1) << (bits_kmer - 1)) 
                                                    | ((std::uint64_t(1) << (bits_kmer - 1)) - 1);

        void queryPrecalculatedSignatures(
            const std::uint64_t* signatures, //getNumberOfMaps() elements per sequence
            GpuMinhasher::Range_t* ranges, //getNumberOfMaps() elements per sequence
            int* totalNumResultsInRanges, 
            int numSequences) const{ 
            
            int numResults = 0;
    
            for(int i = 0; i < numSequences; i++){
                const std::uint64_t* const signature = &signatures[i * getNumberOfMaps()];
                GpuMinhasher::Range_t* const range = &ranges[i * getNumberOfMaps()];            
    
                for(int map = 0; map < getNumberOfMaps(); ++map){
                    kmer_type key = signature[map] & kmer_mask;
                    auto entries_range = queryMap(map, key);
                    numResults += std::distance(entries_range.first, entries_range.second);
                    range[map] = entries_range;
                }
            }   
    
            *totalNumResultsInRanges = numResults;   
        }

        int getNumberOfMaps() const{
            return minhashTables.size();
        }

        int getKmerSize() const{
            return kmerSize;
        }

        int getNumResultsPerMapThreshold() const{
            return resultsPerMapThreshold;
        }

        void addHashTable(HashTable&& hm){
            minhashTables.emplace_back(std::make_unique<HashTable>(std::move(hm)));
        }

        int calculateResultsPerMapThreshold(int coverage){
            int result = int(coverage * 2.5f);
            result = std::min(result, int(std::numeric_limits<BucketSize>::max()));
            result = std::max(10, result);
            return result;
        }

        void computeReadHashesOnGpu(
            std::uint64_t* d_hashValues,
            std::size_t hashValuesPitchInElements,
            const unsigned int* d_encodedSequenceData,
            std::size_t encodedSequencePitchInInts,
            int numSequences,
            const int* d_sequenceLengths,
            cudaStream_t stream
        ){
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

        void computeReadHashesOnGpu(
            std::uint64_t* d_hashValues,
            std::size_t hashValuesPitchInElements,
            const unsigned int* d_encodedSequenceData,
            std::size_t encodedSequencePitchInInts,
            int numSequences,
            const int* d_sequenceLengths,
            int firstHashFunc,
            cudaStream_t stream
        ){
            callMinhashSignaturesKernel_async(
                d_hashValues,
                hashValuesPitchInElements,
                d_encodedSequenceData,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                getNumberOfMaps(),
                firstHashFunc,
                stream
            );
        }

        



    private:
        

        Range_t queryMap(int id, const Key_t& key) const{
            HashTable::QueryResult qr = minhashTables[id]->query(key);

            return std::make_pair(qr.valuesBegin, qr.valuesBegin + qr.numValues);
        }

        std::pair< std::vector<std::vector<kmer_type>>, std::vector<std::vector<read_number>> > 
        constructTablesWithGpuHashing(
            int numTables, 
            int firstTableId,
            int kmersize,
            std::int64_t numberOfReads,
            int upperBoundSequenceLength,
            const RuntimeOptions& runtimeOptions,
            const DistributedReadStorage& readStorage
        ){

            constexpr read_number parallelReads = 1000000;
            read_number numReads = numberOfReads;
            const int numIters = SDIV(numReads, parallelReads);
            const std::size_t encodedSequencePitchInInts = getEncodedNumInts2Bit(upperBoundSequenceLength);

            const auto& deviceIds = runtimeOptions.deviceIds;
            const int numThreads = runtimeOptions.threads;

            assert(deviceIds.size() > 0);

            const int deviceId = deviceIds[0];

            cudaSetDevice(deviceId); CUERR;

            const int numHashFuncs = numTables;
            const int firstHashFunc = firstTableId;
            const std::size_t signaturesRowPitchElements = numHashFuncs;

            ThreadPool::ParallelForHandle pforHandle;

            std::vector<std::vector<kmer_type>> kmersPerFunc(numTables);
            std::vector<std::vector<read_number>> readIdsPerFunc(numTables);

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



        int kmerSize;
        int resultsPerMapThreshold;
        std::vector<std::unique_ptr<HashTable>> minhashTables;
    };




    
}
}



#endif
