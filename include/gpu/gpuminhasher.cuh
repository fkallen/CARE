#ifndef CARE_GPUMINHASHER_CUH
#define CARE_GPUMINHASHER_CUH

#include <config.hpp>

#include <gpu/distributedreadstorage.hpp>
#include <gpu/cuda_unique.cuh>
#include <cpuhashtable.hpp>

#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>

#include <sequence.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>


#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <fstream>
#include <algorithm>


namespace care{
namespace gpu{


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
        cudaStream_t stream
    );

    void callMinhashSignaturesKernel_async(
        std::uint64_t* d_signatures,
        std::size_t signaturesRowPitchElements,
        const unsigned int* d_sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* d_sequenceLengths,
        int k,
        int numHashFuncs,
        cudaStream_t stream
    );




    class GpuMinhasher{
    private:
        using HashTable = CpuReadOnlyMultiValueHashTable<kmer_type, read_number>;
    public:
        using Key_t = kmer_type;
        using Value_t = read_number;

        using Range_t = std::pair<const Value_t*, const Value_t*>;

        struct QueryHandle{
            static constexpr int overprovisioningPercent = 0;

            template<class T>
            using DeviceBuffer = helpers::SimpleAllocationDevice<T, overprovisioningPercent>;
            
            template<class T>
            using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T, overprovisioningPercent>;

            int deviceId;

            DeviceBuffer<std::uint64_t> d_minhashSignatures;
            PinnedBuffer<std::uint64_t> h_minhashSignatures;            

            PinnedBuffer<read_number> h_candidate_read_ids_tmp;
            DeviceBuffer<read_number> d_candidate_read_ids_tmp;

            PinnedBuffer<int> h_begin_offsets;
            DeviceBuffer<int> d_begin_offsets;

            DeviceBuffer<char> d_cub_temp;

            std::vector<Range_t> allRanges;
            std::vector<int> idsPerChunk;   
            std::vector<int> numAnchorsPerChunk;
            std::vector<int> idsPerChunkPrefixSum;
            std::vector<int> numAnchorsPerChunkPrefixSum;

            DeviceBuffer<std::uint64_t> d_temp;
            DeviceBuffer<int> d_signatureSizePerSequence;
            PinnedBuffer<int> h_signatureSizePerSequence;
            DeviceBuffer<int> d_hashFuncIds;
            PinnedBuffer<int> h_hashFuncIds;

            GpuSegmentedUnique::Handle segmentedUniqueHandle;


            void resize(const GpuMinhasher& minhasher, std::size_t numSequences, int numThreads = 1){
                const std::size_t maximumResultSize 
                    = minhasher.getNumResultsPerMapThreshold() * minhasher.getNumberOfMaps() * numSequences;

                d_minhashSignatures.resize(minhasher.getNumberOfMaps() * numSequences);
                h_minhashSignatures.resize(minhasher.getNumberOfMaps() * numSequences);
                h_candidate_read_ids_tmp.resize(maximumResultSize);
                d_candidate_read_ids_tmp.resize(maximumResultSize);

                h_begin_offsets.resize(numSequences+1);
                d_begin_offsets.resize(numSequences+1);
            
                allRanges.resize(minhasher.getNumberOfMaps() * numSequences);
                idsPerChunk.resize(numThreads, 0);   
                numAnchorsPerChunk.resize(numThreads, 0);
                idsPerChunkPrefixSum.resize(numThreads, 0);
                numAnchorsPerChunkPrefixSum.resize(numThreads, 0);

                d_temp.resize(minhasher.getNumberOfMaps() * numSequences);
                d_signatureSizePerSequence.resize(numSequences);
                h_signatureSizePerSequence.resize(numSequences);

                d_hashFuncIds.resize(minhasher.getNumberOfMaps() * numSequences);
                h_hashFuncIds.resize(minhasher.getNumberOfMaps() * numSequences);
            }

            MemoryUsage getMemoryInfo() const{
                MemoryUsage info;
                info.host = 0;
                info.device[deviceId] = 0;
    
                auto handlehost = [&](const auto& buff){
                    info.host += buff.capacityInBytes();
                };
    
                auto handledevice = [&](const auto& buff){
                    info.device[deviceId] += buff.capacityInBytes();
                };

                auto handlevector = [&](const auto& buff){
                    info.host += 
                        sizeof(typename std::remove_reference<decltype(buff)>::type::value_type) * buff.capacity();
                };
    
                handlehost(h_minhashSignatures);
                handlehost(h_candidate_read_ids_tmp);
                handlehost(h_begin_offsets);
    
                handledevice(d_minhashSignatures);
                handledevice(d_candidate_read_ids_tmp);
                handledevice(d_begin_offsets);

                handledevice(d_cub_temp);

                handlevector(allRanges);
                handlevector(idsPerChunk);
                handlevector(numAnchorsPerChunk);
                handlevector(idsPerChunkPrefixSum);
                handlevector(numAnchorsPerChunkPrefixSum);

                handledevice(d_temp);
                handledevice(d_signatureSizePerSequence);
                handledevice(d_hashFuncIds);
                handlehost(h_signatureSizePerSequence);
                handlehost(h_hashFuncIds);

                info += segmentedUniqueHandle->getMemoryInfo();
    
                return info;
            }

            void destroy(){
                int cur = 0;
                cudaGetDevice(&cur); CUERR;
                cudaSetDevice(deviceId); CUERR;

                d_minhashSignatures.destroy();
                h_minhashSignatures.destroy();
                h_candidate_read_ids_tmp.destroy();
                d_candidate_read_ids_tmp.destroy();
                h_begin_offsets.destroy();
                d_begin_offsets.destroy();

                d_cub_temp.destroy();

                allRanges.clear();
                allRanges.shrink_to_fit();

                d_temp.destroy();
                d_signatureSizePerSequence.destroy();
                h_signatureSizePerSequence.destroy();

                d_hashFuncIds.destroy();
                h_hashFuncIds.destroy();

                segmentedUniqueHandle = nullptr;

                cudaSetDevice(cur); CUERR;
            }
        };

        static QueryHandle makeQueryHandle(){
            QueryHandle handle;
            handle.segmentedUniqueHandle = GpuSegmentedUnique::makeHandle();

            cudaGetDevice(&handle.deviceId); CUERR;

            return handle;
        }

        static void destroyQueryHandle(QueryHandle& handle){
            handle.destroy();
        }


        GpuMinhasher() : GpuMinhasher(16, 50){

        }

        GpuMinhasher(int kmerSize, int resultsPerMapThreshold)
            : kmerSize(kmerSize), resultsPerMapThreshold(resultsPerMapThreshold){

        }

        GpuMinhasher(const GpuMinhasher&) = delete;
        GpuMinhasher(GpuMinhasher&&) = default;
        GpuMinhasher& operator=(const GpuMinhasher&) = delete;
        GpuMinhasher& operator=(GpuMinhasher&&) = default;



        template<class ParallelForLoop>
        void getIdsOfSimilarReads(
            QueryHandle& handle,
            const unsigned int* d_encodedSequences,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int deviceId, 
            cudaStream_t stream,
            ParallelForLoop parallelFor,
            read_number* d_similarReadIds,
            int* d_similarReadsPerSequence,
            int* d_similarReadsPerSequencePrefixSum
        ) const{

            int currentDeviceId = 0;
            cudaGetDevice(&currentDeviceId); CUERR;
            cudaSetDevice(deviceId); CUERR;

            const std::size_t maximumResultSize = getNumResultsPerMapThreshold() * getNumberOfMaps() * numSequences;

            handle.d_minhashSignatures.resize(getNumberOfMaps() * numSequences);
            handle.h_minhashSignatures.resize(getNumberOfMaps() * numSequences);
            handle.h_candidate_read_ids_tmp.resize(maximumResultSize);
            handle.d_candidate_read_ids_tmp.resize(maximumResultSize);
            handle.h_begin_offsets.resize(numSequences+1);
            handle.d_begin_offsets.resize(numSequences+1);

            std::vector<Range_t>& allRanges = handle.allRanges;
            std::vector<int>& idsPerChunk = handle.idsPerChunk;   
            std::vector<int>& numAnchorsPerChunk = handle.numAnchorsPerChunk;
            std::vector<int>& idsPerChunkPrefixSum = handle.idsPerChunkPrefixSum;
            std::vector<int>& numAnchorsPerChunkPrefixSum = handle.numAnchorsPerChunkPrefixSum;

            const int maxNumThreads = parallelFor.getNumThreads();

            allRanges.resize(getNumberOfMaps() * numSequences);
            idsPerChunk.resize(maxNumThreads, 0);   
            numAnchorsPerChunk.resize(maxNumThreads, 0);
            idsPerChunkPrefixSum.resize(maxNumThreads, 0);
            numAnchorsPerChunkPrefixSum.resize(maxNumThreads, 0);

            const std::size_t hashValuesPitchInElements = getNumberOfMaps();

            computeReadHashesOnGpu(
                handle.d_minhashSignatures.get(),
                hashValuesPitchInElements,
                d_encodedSequences,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                stream
            );

            cudaMemcpyAsync(
                handle.h_minhashSignatures.get(),
                handle.d_minhashSignatures.get(),
                handle.h_minhashSignatures.sizeInBytes(),
                H2D,
                stream
            ); CUERR;


            std::fill(idsPerChunk.begin(), idsPerChunk.end(), 0);
            std::fill(numAnchorsPerChunk.begin(), numAnchorsPerChunk.end(), 0);
    
            cudaStreamSynchronize(stream); CUERR; //wait for D2H transfers of signatures anchor data
    
            auto querySignatures2 = [&, this](int begin, int end, int threadId){
    
                const int chunksize = end - begin;
    
                int totalNumResults = 0;
    
                nvtx::push_range("queryPrecalculatedSignatures", 6);
                queryPrecalculatedSignatures(
                    handle.h_minhashSignatures.get() + begin * getNumberOfMaps(),
                    allRanges.data() + begin * getNumberOfMaps(),
                    &totalNumResults, 
                    chunksize
                );
    
                idsPerChunk[threadId] = totalNumResults;
                numAnchorsPerChunk[threadId] = chunksize;
                nvtx::pop_range();
            };
    
            const int numChunksRequired = parallelFor(
                0, 
                numSequences, 
                [=](auto begin, auto end, auto threadId){
                    querySignatures2(begin, end, threadId);
                }
            );

            //exclusive prefix sum
            idsPerChunkPrefixSum[0] = 0;
            for(int i = 0; i < numChunksRequired; i++){
                idsPerChunkPrefixSum[i+1] = idsPerChunkPrefixSum[i] + idsPerChunk[i];
            }

            numAnchorsPerChunkPrefixSum[0] = 0;
            for(int i = 0; i < numChunksRequired; i++){
                numAnchorsPerChunkPrefixSum[i+1] = numAnchorsPerChunkPrefixSum[i] + numAnchorsPerChunk[i];
            }

            const int totalNumIds = idsPerChunkPrefixSum[numChunksRequired-1] + idsPerChunk[numChunksRequired-1];

            handle.h_begin_offsets[0] = 0;

            //map queries return pointers to value ranges. copy all value ranges into a single contiguous pinned buffer,
            //then copy to the device

            auto copyCandidateIdsToContiguousMem = [&](int begin, int end, int threadId){
                nvtx::push_range("copyCandidateIdsToContiguousMem", 1);
                for(int chunkId = begin; chunkId < end; chunkId++){
                    const auto hostdatabegin = handle.h_candidate_read_ids_tmp.get() + idsPerChunkPrefixSum[chunkId];

                    //const auto devicedatabegin = handle.d_candidate_read_ids_tmp.get() + idsPerChunkPrefixSum[chunkId];
                    const auto devicedatabegin = d_similarReadIds + idsPerChunkPrefixSum[chunkId];

                    
                    const size_t elementsInChunk = idsPerChunk[chunkId];
    
                    const auto* ranges = allRanges.data() + numAnchorsPerChunkPrefixSum[chunkId] * getNumberOfMaps();
    
                    auto* dest = hostdatabegin;
    
                    const int lmax = numAnchorsPerChunk[chunkId] * getNumberOfMaps();

                    for(int sequenceIndex = 0; sequenceIndex < numAnchorsPerChunk[chunkId]; sequenceIndex++){

                        const int globalSequenceIndex = sequenceIndex + numAnchorsPerChunkPrefixSum[chunkId];

                        for(int mapIndex = 0; mapIndex < getNumberOfMaps(); mapIndex++){
                            const int k = sequenceIndex * getNumberOfMaps() + mapIndex;
                            
                            constexpr int nextprefetch = 2;
    
                            //prefetch first element of next range if the next range is not empty
                            if(k+nextprefetch < lmax){
                                if(ranges[k+nextprefetch].first != ranges[k+nextprefetch].second){
                                    __builtin_prefetch(ranges[k+nextprefetch].first, 0, 0);
                                }
                            }
                            const auto& range = ranges[k];
                            dest = std::copy(range.first, range.second, dest);

                        }

                        const auto endOfSequenceRange = dest;

                        handle.h_begin_offsets[globalSequenceIndex+1] = std::distance(
                            handle.h_candidate_read_ids_tmp.get(),
                            endOfSequenceRange
                        ); 

                    }
    
                    // for(int k = 0; k < lmax; k++){
                    //     constexpr int nextprefetch = 2;
    
                    //     //prefetch first element of next range if the next range is not empty
                    //     if(k+nextprefetch < lmax){
                    //         if(ranges[k+nextprefetch].first != ranges[k+nextprefetch].second){
                    //             __builtin_prefetch(ranges[k+nextprefetch].first, 0, 0);
                    //         }
                    //     }
                    //     const auto& range = ranges[k];
                    //     dest = std::copy(range.first, range.second, dest);
                    // }
    
                    cudaMemcpyAsync(
                        devicedatabegin,
                        hostdatabegin,
                        sizeof(read_number) * elementsInChunk,
                        H2D,
                        stream
                    ); CUERR;
                }
                nvtx::pop_range();
            };
    
            parallelFor(
                0, 
                numChunksRequired, 
                [=](auto begin, auto end, auto threadId){
                    copyCandidateIdsToContiguousMem(begin, end, threadId);
                }
            );

            cudaMemcpyAsync(
                handle.d_begin_offsets.get(),
                handle.h_begin_offsets.get(),
                sizeof(int) * (numSequences + 1),
                H2D,
                stream
            ); CUERR;

            nvtx::push_range("gpumakeUniqueQueryResults", 2);

            GpuSegmentedUnique::unique(
                handle.segmentedUniqueHandle,
                d_similarReadIds, //input
                totalNumIds,
                handle.d_candidate_read_ids_tmp.get(), //output
                d_similarReadsPerSequence,
                numSequences,
                handle.d_begin_offsets.get(),
                handle.d_begin_offsets.get() + 1,
                handle.h_begin_offsets.get(),
                handle.h_begin_offsets.get() + 1,
                0,
                sizeof(read_number) * 8,
                stream
            );

            CUERR;

            std::size_t cubTempBytes = 0;

            cudaMemsetAsync(d_similarReadsPerSequencePrefixSum, 0, sizeof(int), stream);

            cub::DeviceScan::InclusiveSum(
                nullptr, 
                cubTempBytes, 
                d_similarReadsPerSequence, 
                d_similarReadsPerSequencePrefixSum + 1, 
                numSequences,
                stream
            );

            handle.d_cub_temp.resize(cubTempBytes);

            cub::DeviceScan::InclusiveSum(
                handle.d_cub_temp.get(), 
                cubTempBytes, 
                d_similarReadsPerSequence, 
                d_similarReadsPerSequencePrefixSum + 1, 
                numSequences,
                stream
            );

            //compact copy elements of each segment into output buffer
            helpers::lambda_kernel<<<numSequences, 128, 0, stream>>>(
                [=,
                    d_begin_offsets = handle.d_begin_offsets.get(),
                    input = handle.d_candidate_read_ids_tmp.get(),
                    output = d_similarReadIds
                ] __device__ (){

                    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){
                       
                        const read_number* const blockinput = input + d_begin_offsets[sequenceIndex];
                        read_number* const blockoutput = output + d_similarReadsPerSequencePrefixSum[sequenceIndex];
                        const int numElements = d_similarReadsPerSequence[sequenceIndex];

                        for(int pos = threadIdx.x; pos < numElements; pos += blockDim.x){
                            blockoutput[pos] = blockinput[pos];
                        }
                    }
                }
            );

            nvtx::pop_range();

            cudaSetDevice(currentDeviceId); CUERR;

        }




        template<class ParallelForLoop>
        void getIdsOfSimilarReadsNormalExcludingSelf(
            QueryHandle& handle,
            const read_number* d_readIds,
            const read_number* h_readIds,
            const unsigned int* d_encodedSequences,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int deviceId, 
            cudaStream_t stream,
            ParallelForLoop parallelFor,
            read_number* d_similarReadIds,
            int* d_similarReadsPerSequence,
            int* d_similarReadsPerSequencePrefixSum
        ) const{

            int currentDeviceId = 0;
            cudaGetDevice(&currentDeviceId); CUERR;
            cudaSetDevice(deviceId); CUERR;

            const std::size_t maximumResultSize = getNumResultsPerMapThreshold() * getNumberOfMaps() * numSequences;

            handle.d_minhashSignatures.resize(getNumberOfMaps() * numSequences);
            handle.h_minhashSignatures.resize(getNumberOfMaps() * numSequences);
            handle.h_candidate_read_ids_tmp.resize(maximumResultSize);
            handle.d_candidate_read_ids_tmp.resize(maximumResultSize);
            handle.h_begin_offsets.resize(numSequences+1);
            handle.d_begin_offsets.resize(numSequences+1);

            std::vector<Range_t>& allRanges = handle.allRanges;
            std::vector<int>& idsPerChunk = handle.idsPerChunk;   
            std::vector<int>& numAnchorsPerChunk = handle.numAnchorsPerChunk;
            std::vector<int>& idsPerChunkPrefixSum = handle.idsPerChunkPrefixSum;
            std::vector<int>& numAnchorsPerChunkPrefixSum = handle.numAnchorsPerChunkPrefixSum;

            const int maxNumThreads = parallelFor.getNumThreads();

            allRanges.resize(getNumberOfMaps() * numSequences);
            idsPerChunk.resize(maxNumThreads, 0);   
            numAnchorsPerChunk.resize(maxNumThreads, 0);
            idsPerChunkPrefixSum.resize(maxNumThreads, 0);
            numAnchorsPerChunkPrefixSum.resize(maxNumThreads, 0);

            const std::size_t hashValuesPitchInElements = getNumberOfMaps();

            computeReadHashesOnGpu(
                handle.d_minhashSignatures.get(),
                hashValuesPitchInElements,
                d_encodedSequences,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                stream
            );

            cudaMemcpyAsync(
                handle.h_minhashSignatures.get(),
                handle.d_minhashSignatures.get(),
                handle.h_minhashSignatures.sizeInBytes(),
                H2D,
                stream
            ); CUERR;


            std::fill(idsPerChunk.begin(), idsPerChunk.end(), 0);
            std::fill(numAnchorsPerChunk.begin(), numAnchorsPerChunk.end(), 0);
    
            cudaStreamSynchronize(stream); CUERR; //wait for D2H transfers of signatures anchor data
    
            auto querySignatures2 = [&, this](int begin, int end, int threadId){
    
                const int chunksize = end - begin;
    
                int totalNumResults = 0;
    
                nvtx::push_range("queryPrecalculatedSignatures", 6);
                queryPrecalculatedSignatures(
                    handle.h_minhashSignatures.get() + begin * getNumberOfMaps(),
                    allRanges.data() + begin * getNumberOfMaps(),
                    &totalNumResults, 
                    chunksize
                );
    
                idsPerChunk[threadId] = totalNumResults;
                numAnchorsPerChunk[threadId] = chunksize;
                nvtx::pop_range();
            };
    
            const int numChunksRequired = parallelFor(
                0, 
                numSequences, 
                [=](auto begin, auto end, auto threadId){
                    querySignatures2(begin, end, threadId);
                }
            );

            //exclusive prefix sum
            idsPerChunkPrefixSum[0] = 0;
            for(int i = 0; i < numChunksRequired; i++){
                idsPerChunkPrefixSum[i+1] = idsPerChunkPrefixSum[i] + idsPerChunk[i];
            }

            numAnchorsPerChunkPrefixSum[0] = 0;
            for(int i = 0; i < numChunksRequired; i++){
                numAnchorsPerChunkPrefixSum[i+1] = numAnchorsPerChunkPrefixSum[i] + numAnchorsPerChunk[i];
            }

            const int totalNumIds = idsPerChunkPrefixSum[numChunksRequired-1] + idsPerChunk[numChunksRequired-1];
            
            handle.h_begin_offsets[0] = 0;

            //map queries return pointers to value ranges. copy all value ranges into a single contiguous pinned buffer,
            //then copy to the device

            auto copyCandidateIdsToContiguousMem = [&](int begin, int end, int threadId){
                nvtx::push_range("copyCandidateIdsToContiguousMem", 1);
                for(int chunkId = begin; chunkId < end; chunkId++){
                    const auto hostdatabegin = handle.h_candidate_read_ids_tmp.get() + idsPerChunkPrefixSum[chunkId];

                    //const auto devicedatabegin = handle.d_candidate_read_ids_tmp.get() + idsPerChunkPrefixSum[chunkId];
                    const auto devicedatabegin = d_similarReadIds + idsPerChunkPrefixSum[chunkId];

                    
                    const size_t elementsInChunk = idsPerChunk[chunkId];
    
                    const auto* ranges = allRanges.data() + numAnchorsPerChunkPrefixSum[chunkId] * getNumberOfMaps();
    
                    auto* dest = hostdatabegin;
    
                    const int lmax = numAnchorsPerChunk[chunkId] * getNumberOfMaps();

                    for(int sequenceIndex = 0; sequenceIndex < numAnchorsPerChunk[chunkId]; sequenceIndex++){

                        const int globalSequenceIndex = sequenceIndex + numAnchorsPerChunkPrefixSum[chunkId];

                        for(int mapIndex = 0; mapIndex < getNumberOfMaps(); mapIndex++){
                            const int k = sequenceIndex * getNumberOfMaps() + mapIndex;
                            
                            constexpr int nextprefetch = 2;
    
                            //prefetch first element of next range if the next range is not empty
                            if(k+nextprefetch < lmax){
                                if(ranges[k+nextprefetch].first != ranges[k+nextprefetch].second){
                                    __builtin_prefetch(ranges[k+nextprefetch].first, 0, 0);
                                }
                            }
                            const auto& range = ranges[k];
                            dest = std::copy(range.first, range.second, dest);

                        }

                        const auto endOfSequenceRange = dest;

                        handle.h_begin_offsets[globalSequenceIndex+1] = std::distance(
                            handle.h_candidate_read_ids_tmp.get(),
                            endOfSequenceRange
                        ); 

                    }
    
                    // for(int k = 0; k < lmax; k++){
                    //     constexpr int nextprefetch = 2;
    
                    //     //prefetch first element of next range if the next range is not empty
                    //     if(k+nextprefetch < lmax){
                    //         if(ranges[k+nextprefetch].first != ranges[k+nextprefetch].second){
                    //             __builtin_prefetch(ranges[k+nextprefetch].first, 0, 0);
                    //         }
                    //     }
                    //     const auto& range = ranges[k];
                    //     dest = std::copy(range.first, range.second, dest);
                    // }
    
                    cudaMemcpyAsync(
                        devicedatabegin,
                        hostdatabegin,
                        sizeof(read_number) * elementsInChunk,
                        H2D,
                        stream
                    ); CUERR;
                }
                nvtx::pop_range();
            };
    
            parallelFor(
                0, 
                numChunksRequired, 
                [=](auto begin, auto end, auto threadId){
                    copyCandidateIdsToContiguousMem(begin, end, threadId);
                }
            );

            cudaMemcpyAsync(
                handle.d_begin_offsets.get(),
                handle.h_begin_offsets.get(),
                sizeof(int) * (numSequences + 1),
                H2D,
                stream
            ); CUERR;

            nvtx::push_range("gpumakeUniqueQueryResults", 2);

            GpuSegmentedUnique::unique(
                handle.segmentedUniqueHandle,
                d_similarReadIds, //input
                totalNumIds,
                handle.d_candidate_read_ids_tmp.get(), //output
                d_similarReadsPerSequence,
                numSequences,
                handle.d_begin_offsets.get(),
                handle.d_begin_offsets.get() + 1,
                handle.h_begin_offsets.get(),
                handle.h_begin_offsets.get() + 1,
                0,
                sizeof(read_number) * 8,
                stream
            );

            //if a candidate list is not empty, it must at least contain the id of the querying read.
            //this id will be removed next. however, the prefixsum already requires the numbers with removed ids.
            auto op = [] __device__ (int elem){
                if(elem > 0) return elem - 1;
                else return elem;
            };
            cub::TransformInputIterator<int, decltype(op), int*> itr(d_similarReadsPerSequence, op);

            std::size_t cubTempBytes = 0;

            cudaMemsetAsync(d_similarReadsPerSequencePrefixSum, 0, sizeof(int), stream);

            cub::DeviceScan::InclusiveSum(
                nullptr, 
                cubTempBytes, 
                itr, 
                d_similarReadsPerSequencePrefixSum + 1, 
                numSequences,
                stream
            );

            handle.d_cub_temp.resize(cubTempBytes);

            cub::DeviceScan::InclusiveSum(
                handle.d_cub_temp.get(), 
                cubTempBytes, 
                itr, 
                d_similarReadsPerSequencePrefixSum + 1, 
                numSequences,
                stream
            );

            helpers::lambda_kernel<<<numSequences, 128, 0, stream>>>(
                [=,
                    d_begin_offsets = handle.d_begin_offsets.get(),
                    input = handle.d_candidate_read_ids_tmp.get(),
                    output = d_similarReadIds
                ] __device__ (){

                    using BlockReduce = cub::BlockReduce<int, 128>;

                    __shared__ BlockReduce::TempStorage temp_reduce;
                    __shared__ int broadcast;

                    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

                       
                        const read_number* const blockinput = input + d_begin_offsets[sequenceIndex];
                        read_number* const blockoutput = output + d_similarReadsPerSequencePrefixSum[sequenceIndex];
                        int numElements = d_similarReadsPerSequence[sequenceIndex];
                        const read_number anchorIdToRemove = d_readIds[sequenceIndex];

                        bool foundInvalid = false;

                        const int iters = SDIV(numElements, 128);
                        for(int iter = 0; iter < iters; iter++){
                        
                            read_number elem = 0;
                            int invalidpos = std::numeric_limits<int>::max();

                            if(iter * 128 + threadIdx.x < numElements){
                                elem = blockinput[iter * 128 + threadIdx.x];
                            }



                            //true for at most one thread
                            if(elem == anchorIdToRemove){
                                invalidpos = threadIdx.x;
                            }

                            if(!foundInvalid){
                                invalidpos = BlockReduce(temp_reduce).Reduce(invalidpos, cub::Min{});

                                if(threadIdx.x == 0){
                                    broadcast = invalidpos;
                                }
                            }else{
                                if(threadIdx.x == 0){
                                    broadcast = -1;
                                }
                            }
                            
                            __syncthreads(); // wait until broadcast is set

                            invalidpos = broadcast;

                            //store valid elements to output array
                            if(elem != anchorIdToRemove && (iter * 128 + threadIdx.x < numElements)){

                                const bool doShift = foundInvalid || threadIdx.x >= invalidpos;

                                if(!doShift){
                                    blockoutput[iter * 128 + threadIdx.x] = elem;
                                }else{
                                    blockoutput[iter * 128 + threadIdx.x - 1] = elem;
                                }
                            }

                            if(invalidpos != std::numeric_limits<int>::max()){
                                foundInvalid |= true;
                            }

                            __syncthreads(); // wait until broadcast is read by every thread
                        }

                        if(threadIdx.x == 0 && numElements > 0){
                            d_similarReadsPerSequence[sequenceIndex] -= 1;
                        }
                    }
                }
            );

            CUERR;

            nvtx::pop_range();

            cudaSetDevice(currentDeviceId); CUERR;

        }

        template<class ParallelForLoop>
        void getIdsOfSimilarReadsExcludingSelf(
            QueryHandle& handle,
            const read_number* d_readIds,
            const read_number* h_readIds,
            const unsigned int* d_encodedSequences,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int deviceId, 
            cudaStream_t stream,
            ParallelForLoop parallelFor,
            read_number* d_similarReadIds,
            int* d_similarReadsPerSequence,
            int* d_similarReadsPerSequencePrefixSum
        ) const{

            if(numSequences == 0){
                return;
            }

            getIdsOfSimilarReadsNormalExcludingSelf(
                handle,
                d_readIds, 
                h_readIds,
                d_encodedSequences,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                deviceId,
                stream,
                parallelFor,
                d_similarReadIds,
                d_similarReadsPerSequence,
                d_similarReadsPerSequencePrefixSum
            );
        }
                                                    

        void queryPrecalculatedSignatures(
            const std::uint64_t* signatures, //getNumberOfMaps() elements per sequence
            GpuMinhasher::Range_t* ranges, //getNumberOfMaps() elements per sequence
            int* totalNumResultsInRanges, 
            int numSequences) const;


        int getNumberOfMaps() const{
            return minhashTables.size();
        }

        int getKmerSize() const{
            return kmerSize;
        }

        int getNumResultsPerMapThreshold() const{
            return resultsPerMapThreshold;
        }

        std::uint64_t getKmerMask() const{
            constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

            return std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - getKmerSize()) * 2);
        }

        MemoryUsage getMemoryInfo() const;

        void destroy();

        void writeToStream(std::ostream& os) const;
    
        int loadFromStream(std::ifstream& is, int numMapsUpperLimit = std::numeric_limits<int>::max());

        int calculateResultsPerMapThreshold(int coverage);

        void computeReadHashesOnGpu(
            std::uint64_t* d_hashValues,
            std::size_t hashValuesPitchInElements,
            const unsigned int* d_encodedSequenceData,
            std::size_t encodedSequencePitchInInts,
            int numSequences,
            const int* d_sequenceLengths,
            cudaStream_t stream
        ) const;

        void construct(
            const FileOptions &fileOptions,
            const RuntimeOptions &runtimeOptions,
            const MemoryOptions& memoryOptions,
            std::uint64_t nReads,
            const CorrectionOptions& correctionOptions,
            const DistributedReadStorage& gpuReadStorage
        );

        



    private:
        

        Range_t queryMap(int id, const Key_t& key) const;

        void addHashTable(HashTable&& hm);

        void computeReadHashesOnGpu(
            std::uint64_t* d_hashValues,
            std::size_t hashValuesPitchInElements,
            const unsigned int* d_encodedSequenceData,
            std::size_t encodedSequencePitchInInts,
            int numSequences,
            const int* d_sequenceLengths,
            int numHashFuncs,
            cudaStream_t stream
        ) const;

        void computeReadHashesOnGpu(
            std::uint64_t* d_hashValues,
            std::size_t hashValuesPitchInElements,
            const unsigned int* d_encodedSequenceData,
            std::size_t encodedSequencePitchInInts,
            int numSequences,
            const int* d_sequenceLengths,
            int numHashFuncs,
            int firstHashFunc,
            cudaStream_t stream
        ) const;


        

        std::pair< std::vector<std::vector<kmer_type>>, std::vector<std::vector<read_number>> > 
        constructTablesAAA(
            int numTables, 
            int firstTableId,
            std::int64_t numberOfReads,
            int upperBoundSequenceLength,
            const RuntimeOptions& runtimeOptions,
            const DistributedReadStorage& readStorage
        );

        std::pair< std::vector<std::vector<kmer_type>>, std::vector<std::vector<read_number>> > 
        computeKeyValuePairsForHashtableUsingGpu(
            int numTables, 
            int firstTableId,
            std::int64_t numberOfReads,
            int upperBoundSequenceLength,
            const RuntimeOptions& runtimeOptions,
            const DistributedReadStorage& readStorage
        );      

        int loadConstructedTablesFromFile(
            const std::string& filename,
            int numTablesToLoad, 
            std::size_t availableMemory
        );


        int kmerSize;
        int resultsPerMapThreshold;
        std::vector<std::unique_ptr<HashTable>> minhashTables;
    };






    
}
}



#endif
