#ifndef CARE_GPUMINHASHER_CUH
#define CARE_GPUMINHASHER_CUH

#include <config.hpp>

#include <gpu/nvtxtimelinemarkers.hpp>
#include <gpu/gpuhashtable.cuh>
#include <gpu/distributedreadstorage.hpp>
#include <gpu/simpleallocation.cuh>
#include <gpu/minhashkernels.hpp>
#include <gpu/cuda_unique.cuh>

#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>
#include <minhasher_transform.hpp>
#include <sequence.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>


#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <fstream>
#include <algorithm>

//#define UNIQUE

//#define MAKEUNIQUEAFTERHASHING


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

        struct QueryHandle{
            static constexpr int overprovisioningPercent = 0;

            template<class T>
            using DeviceBuffer = SimpleAllocationDevice<T, overprovisioningPercent>;
            
            template<class T>
            using PinnedBuffer = SimpleAllocationPinnedHost<T, overprovisioningPercent>;

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

            MergeRangesGpuHandle<read_number> mergeHandle;

            DeviceBuffer<std::uint64_t> d_temp;
            DeviceBuffer<int> d_signatureSizePerSequence;
            PinnedBuffer<int> h_signatureSizePerSequence;
            DeviceBuffer<int> d_hashFuncIds;
            PinnedBuffer<int> h_hashFuncIds;


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
    
                return info;
            }

            void destroy(){
                int cur = 0;
                cudaGetDevice(&cur); CUERR;
                cudaSetDevice(deviceId); CUERR;

                destroyMergeRangesGpuHandle(mergeHandle);

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

                cudaSetDevice(cur); CUERR;
            }
        };

        static QueryHandle makeQueryHandle(){
            QueryHandle handle;

            handle.mergeHandle = makeMergeRangesGpuHandle<read_number>();

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
        void getIdsOfSimilarReadsNormal(
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

            std::vector<int> h_begin_offsets;

            const int maxNumThreads = parallelFor.getNumThreads();

            allRanges.resize(getNumberOfMaps() * numSequences);
            idsPerChunk.resize(maxNumThreads, 0);   
            numAnchorsPerChunk.resize(maxNumThreads, 0);
            idsPerChunkPrefixSum.resize(maxNumThreads, 0);
            numAnchorsPerChunkPrefixSum.resize(maxNumThreads, 0);
            h_begin_offsets.resize(numSequences+1);

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


            // SimpleAllocationDevice<read_number> d_normal_candidate_read_ids_tmp(handle.h_candidate_read_ids_tmp.size());
            // SimpleAllocationDevice<read_number> d_normal_similar_read_ids(handle.h_candidate_read_ids_tmp.size());
            // SimpleAllocationDevice<read_number> d_normal_similarReadsPerSequence(numSequences);
            // SimpleAllocationDevice<read_number> d_normal_similarReadsPerSequencePrefixSum(numSequences + 1);

            // cudaMemcpyAsync(
            //     d_normal_candidate_read_ids_tmp.get(),
            //     handle.h_candidate_read_ids_tmp.get(),
            //     handle.h_candidate_read_ids_tmp.sizeInBytes(),
            //     H2D,
            //     stream
            // ); CUERR;


            nvtx::push_range("gpumakeUniqueQueryResults", 2);
#if 1
            GpuSegmentedUnique::unique(
                //HandleData* handle,
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

            // {
            //     std::cerr << "before removing self id\n";
            //     SimpleAllocationPinnedHost<read_number> h_similarReadIds(handle.d_candidate_read_ids_tmp.size());
            //     SimpleAllocationPinnedHost<int> h_similarReadsPerSequence(numSequences);
            //     SimpleAllocationPinnedHost<int> h_similarReadsPerSequencePrefixSum(numSequences+1);

            //     cudaMemcpyAsync(h_similarReadIds, handle.d_candidate_read_ids_tmp.get(), sizeof(read_number) * h_similarReadIds.size(), D2H, stream); CUERR;
            //     cudaMemcpyAsync(h_similarReadsPerSequence, d_similarReadsPerSequence, sizeof(int) * numSequences, D2H, stream); CUERR;

            //     cudaStreamSynchronize(stream); CUERR;

            //     std::cerr << "h_similarReadsPerSequence: ";
            //     for(int i = 0; i < 13; i++){
            //         std::cerr << h_similarReadsPerSequence[i] << " ";
            //     }
            //     std::cerr << "\n";

            //     int b = 0;
            //     for(int i = 0; i < 11; i++){
            //         b += h_similarReadsPerSequence[i];
            //     }

            //     std::cerr << "results of read 11 " << "\n";
            //     const int l = h_similarReadsPerSequence[11];

            //     for(int i = 0; i < l; i++){
            //         std::cerr << h_similarReadIds[b + i] << "\n";
            //     }
            // }

            generic_kernel<<<numSequences, 128, 0, stream>>>(
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

                            invalidpos = BlockReduce(temp_reduce).Reduce(invalidpos, cub::Min{});

                            if(threadIdx.x == 0){
                                broadcast = invalidpos;
                            }
                            
                            __syncthreads();

                            //store valid elements to output array
                            if(elem != anchorIdToRemove && (iter * 128 + threadIdx.x < numElements)){
                                if(threadIdx.x < broadcast){
                                    blockoutput[iter * 128 + threadIdx.x] = elem;
                                }else{
                                    blockoutput[iter * 128 + threadIdx.x - 1] = elem;
                                }
                            }
                        }

                        if(threadIdx.x == 0 && numElements > 0){
                            d_similarReadsPerSequence[sequenceIndex] -= 1;
                        }
                    }
                }
            );


#else 


            mergeRangesGpuAsync(
                handle.mergeHandle, 
                d_similarReadIds,
                d_similarReadsPerSequence,
                d_similarReadsPerSequencePrefixSum,
                handle.d_candidate_read_ids_tmp.get(),
                allRanges.data(), 
                getNumberOfMaps() * numSequences, 
                d_readIds,
                getNumberOfMaps(), 
                stream,
                MergeRangesKernelType::allcub
            );

#endif     
            // cudaDeviceSynchronize(); CUERR;
            // std::cerr << "after removing self id\n";

            // SimpleAllocationPinnedHost<read_number> h_similarReadIds(handle.d_candidate_read_ids_tmp.size());
            // SimpleAllocationPinnedHost<int> h_similarReadsPerSequence(numSequences);
            // SimpleAllocationPinnedHost<int> h_similarReadsPerSequencePrefixSum(numSequences+1);

            // cudaMemcpyAsync(h_similarReadIds, d_similarReadIds, sizeof(read_number) * h_similarReadIds.size(), D2H, stream); CUERR;
            // cudaMemcpyAsync(h_similarReadsPerSequence, d_similarReadsPerSequence, sizeof(int) * numSequences, D2H, stream); CUERR;
            // cudaMemcpyAsync(h_similarReadsPerSequencePrefixSum, d_similarReadsPerSequencePrefixSum, sizeof(int) * (numSequences+1), D2H, stream); CUERR;

            // cudaStreamSynchronize(stream); CUERR;

            // std::cerr << "h_similarReadsPerSequence: ";
            // for(int i = 0; i < 13; i++){
            //     std::cerr << h_similarReadsPerSequence[i] << " ";
            // }
            // std::cerr << "\n";

            // std::cerr << "h_similarReadsPerSequencePrefixSum: ";
            // for(int i = 0; i < 13; i++){
            //     std::cerr << h_similarReadsPerSequencePrefixSum[i] << " ";
            // }
            // std::cerr << "\n";

            // for(int x = 0; x < 12; x++){
            //     std::cerr << "results of read " << x << "\n";
            //     const int l = h_similarReadsPerSequence[x];
            //     const int b = h_similarReadsPerSequencePrefixSum[x];

            //     for(int i = 0; i < l; i++){
            //         std::cerr << h_similarReadIds[b + i] << "\n";
            //     }
            // }

            nvtx::pop_range();

            //std::exit(0);


            cudaSetDevice(currentDeviceId); CUERR;

        }


        template<class ParallelForLoop>
        void getIdsOfSimilarReadsUnique1(
            QueryHandle& handle,
            const read_number* d_readIds,
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

            handle.d_temp.resize(getNumberOfMaps() * numSequences);
            handle.d_signatureSizePerSequence.resize(numSequences);
            handle.h_signatureSizePerSequence.resize(numSequences);
            handle.d_hashFuncIds.resize(getNumberOfMaps() * numSequences);
            handle.h_hashFuncIds.resize(getNumberOfMaps() * numSequences);

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
            const std::size_t hashFuncIdsRowPitchElements = getNumberOfMaps();

            callUniqueMinhashSignaturesKernel_async(
                handle.d_temp,
                handle.d_minhashSignatures,
                hashValuesPitchInElements,
                handle.d_hashFuncIds,
                hashFuncIdsRowPitchElements,
                handle.d_signatureSizePerSequence,
                d_encodedSequences,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                getNumberOfMaps(),
                stream
            );

            cudaMemcpyAsync(
                handle.h_minhashSignatures.get(),
                handle.d_minhashSignatures.get(),
                handle.h_minhashSignatures.sizeInBytes(),
                H2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                handle.h_hashFuncIds.get(),
                handle.d_hashFuncIds.get(),
                handle.h_hashFuncIds.sizeInBytes(),
                H2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                handle.h_signatureSizePerSequence.get(),
                handle.d_signatureSizePerSequence.get(),
                handle.h_signatureSizePerSequence.sizeInBytes(),
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
                    handle.h_hashFuncIds.get() + begin * getNumberOfMaps(),
                    handle.h_signatureSizePerSequence.get() + begin,
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

            //map queries return pointers to value ranges. copy all value ranges into a single contiguous pinned buffer,
            //then copy to the device

            auto copyCandidateIdsToContiguousMem = [&](int begin, int end, int threadId){
                nvtx::push_range("copyCandidateIdsToContiguousMem", 1);
                for(int chunkId = begin; chunkId < end; chunkId++){
                    const auto hostdatabegin = handle.h_candidate_read_ids_tmp.get() + idsPerChunkPrefixSum[chunkId];
                    const auto devicedatabegin = handle.d_candidate_read_ids_tmp.get() + idsPerChunkPrefixSum[chunkId];
                    const size_t elementsInChunk = idsPerChunk[chunkId];
    
                    const auto* ranges = allRanges.data() + numAnchorsPerChunkPrefixSum[chunkId] * getNumberOfMaps();
    
                    auto* dest = hostdatabegin;
    
                    const int lmax = numAnchorsPerChunk[chunkId] * getNumberOfMaps();
    
                    for(int k = 0; k < lmax; k++){
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

            nvtx::push_range("gpumakeUniqueQueryResults", 2);
            mergeRangesGpuAsync(
                handle.mergeHandle, 
                d_similarReadIds,
                d_similarReadsPerSequence,
                d_similarReadsPerSequencePrefixSum,
                handle.d_candidate_read_ids_tmp.get(),
                allRanges.data(), 
                getNumberOfMaps() * numSequences, 
                d_readIds,
                getNumberOfMaps(), 
                stream,
                MergeRangesKernelType::allcub
            );
    
            nvtx::pop_range();


            cudaSetDevice(currentDeviceId); CUERR;

        }



        template<class ParallelForLoop>
        void getIdsOfSimilarReadsUnique2(
            QueryHandle& handle,
            const read_number* d_readIds,
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

            handle.d_temp.resize(getNumberOfMaps() * numSequences);
            handle.d_signatureSizePerSequence.resize(numSequences);
            handle.h_signatureSizePerSequence.resize(numSequences);
            handle.d_hashFuncIds.resize(getNumberOfMaps() * numSequences);
            handle.h_hashFuncIds.resize(getNumberOfMaps() * numSequences);

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
            const std::size_t hashFuncIdsRowPitchElements = getNumberOfMaps();

            callMinhashSignaturesOfUniqueKmersKernel128_async(
                handle.d_minhashSignatures,
                hashValuesPitchInElements,
                d_encodedSequences,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                getNumberOfMaps(),
                stream
            );

            cudaMemcpyAsync(
                handle.h_minhashSignatures.get(),
                handle.d_minhashSignatures.get(),
                handle.h_minhashSignatures.sizeInBytes(),
                H2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                handle.h_hashFuncIds.get(),
                handle.d_hashFuncIds.get(),
                handle.h_hashFuncIds.sizeInBytes(),
                H2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                handle.h_signatureSizePerSequence.get(),
                handle.d_signatureSizePerSequence.get(),
                handle.h_signatureSizePerSequence.sizeInBytes(),
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

            //map queries return pointers to value ranges. copy all value ranges into a single contiguous pinned buffer,
            //then copy to the device

            auto copyCandidateIdsToContiguousMem = [&](int begin, int end, int threadId){
                nvtx::push_range("copyCandidateIdsToContiguousMem", 1);
                for(int chunkId = begin; chunkId < end; chunkId++){
                    const auto hostdatabegin = handle.h_candidate_read_ids_tmp.get() + idsPerChunkPrefixSum[chunkId];
                    const auto devicedatabegin = handle.d_candidate_read_ids_tmp.get() + idsPerChunkPrefixSum[chunkId];
                    const size_t elementsInChunk = idsPerChunk[chunkId];
    
                    const auto* ranges = allRanges.data() + numAnchorsPerChunkPrefixSum[chunkId] * getNumberOfMaps();
    
                    auto* dest = hostdatabegin;
    
                    const int lmax = numAnchorsPerChunk[chunkId] * getNumberOfMaps();
    
                    for(int k = 0; k < lmax; k++){
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

            nvtx::push_range("gpumakeUniqueQueryResults", 2);
            mergeRangesGpuAsync(
                handle.mergeHandle, 
                d_similarReadIds,
                d_similarReadsPerSequence,
                d_similarReadsPerSequencePrefixSum,
                handle.d_candidate_read_ids_tmp.get(),
                allRanges.data(), 
                getNumberOfMaps() * numSequences, 
                d_readIds,
                getNumberOfMaps(), 
                stream,
                MergeRangesKernelType::allcub
            );
    
            nvtx::pop_range();


            cudaSetDevice(currentDeviceId); CUERR;

        }

        template<class ParallelForLoop>
        void getIdsOfSimilarReads(
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
#ifndef UNIQUE
            getIdsOfSimilarReadsNormal(
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
#else 

#ifdef MAKEUNIQUEAFTERHASHING
            getIdsOfSimilarReadsUnique(
                handle,
                d_readIds, 
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
#else            
            getIdsOfSimilarReadsUnique2(
                handle,
                d_readIds, 
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
#endif
#endif
        }
                                                    

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

        void queryPrecalculatedSignatures(
            const std::uint64_t* signatures, //getNumberOfMaps() elements per sequence
            const int* hashFuncIds,  //getNumberOfMaps() elements per sequence
            const int* signatureSizesPerSequence,
            GpuMinhasher::Range_t* ranges, //getNumberOfMaps() elements per sequence
            int* totalNumResultsInRanges, 
            int numSequences
        ) const { 
            int numResults = 0;
    
            for(int i = 0; i < numSequences; i++){
                const std::uint64_t* const signature = &signatures[i * getNumberOfMaps()];
                const int* hashFuncIdsForSequence = &hashFuncIds[i * getNumberOfMaps()];
                GpuMinhasher::Range_t* const range = &ranges[i * getNumberOfMaps()];
                
                const int signatureSize = signatureSizesPerSequence[i];

                int cur = 0;
    
                for(int map = 0; map < getNumberOfMaps(); ++map){
                    if(cur < signatureSize && hashFuncIdsForSequence[cur] == map){
                        kmer_type key = signature[map] & kmer_mask;
                        auto entries_range = queryMap(map, key);
                        numResults += std::distance(entries_range.first, entries_range.second);
                        range[map] = entries_range;

                        cur++;
                    }else{
                        range[map].first = nullptr;
                        range[map].second = nullptr;
                    }                    
                }
            }   
    
            *totalNumResultsInRanges = numResults;
        };

        int getNumberOfMaps() const{
            return minhashTables.size();
        }

        int getKmerSize() const{
            return kmerSize;
        }

        int getNumResultsPerMapThreshold() const{
            return resultsPerMapThreshold;
        }

        MemoryUsage getMemoryInfo() const{
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

        void destroy(){
            minhashTables.clear();
        }

        void writeToStream(std::ostream& os) const{

            os.write(reinterpret_cast<const char*>(&kmerSize), sizeof(int));
            os.write(reinterpret_cast<const char*>(&resultsPerMapThreshold), sizeof(int));
    
            const int numTables = getNumberOfMaps();
            os.write(reinterpret_cast<const char*>(&numTables), sizeof(int));
    
            for(const auto& tableptr : minhashTables){
                tableptr->writeToStream(os);
            }
        }
    
        void loadFromStream(std::ifstream& is){
            destroy();
    
            is.read(reinterpret_cast<char*>(&kmerSize), sizeof(int));
            is.read(reinterpret_cast<char*>(&resultsPerMapThreshold), sizeof(int));
    
            int numTables = 0;
    
            is.read(reinterpret_cast<char*>(&numTables), sizeof(int));
    
            for(int i = 0; i < numTables; i++){
                HashTable table;
                table.loadFromStream(is);
                addHashTable(std::move(table));
            }
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

        void construct(
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
            std::size_t sequencepitch = getEncodedNumInts2Bit(maximumSequenceLength) * sizeof(int);

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
#ifndef UNIQUE                    
                    initialMinhashes = constructTablesWithGpuHashing(
#else                     

#ifdef MAKEUNIQUEAFTERHASHING    
                    initialMinhashes = constructTablesWithGpuHashingUniquekmers(
#else 
                    initialMinhashes = constructTablesWithGpuHashingUniquekmers2(
#endif 

#endif                        
                        currentIterNumTables, 
                        numConstructedTables,
                        readStorage.getNumberOfReads(),
                        readStorage.getSequenceLengthUpperBound(),
                        runtimeOptions,
                        readStorage
                    );

                    //check free gpu mem for transformation
                    std::size_t estRequiredFreeGpuMem = 0;
                    for(int i = 0; i < currentIterNumTables; i++){
                        std::size_t est = HashTable::estimateGpuMemoryRequiredForInit(numReads); 

                        estRequiredFreeGpuMem = std::max(estRequiredFreeGpuMem, est);
                    }
                        
                    std::size_t freeGpuMem, totalGpuMem;
                    cudaMemGetInfo(&freeGpuMem, &totalGpuMem); CUERR;

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

                    //TODO fix this. signatures are already being accounted for. determine memory required for transformation
                    for(int i = 0; i < 2 + int(minhashTables.size()); i++){
                        const std::size_t requiredMemPerTable = Minhasher::Map_t::getRequiredSizeInBytesBeforeCompaction(nReads);
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
                    bool didSaveGpudata = false;

                    //if there is more than 10% gpu memory missing, make room for it
                    //if(std::size_t(freeGpuMem * 1.1) < estRequiredFreeGpuMem){
                    {
                        //always make room
                        std::ofstream rstempostream(rstempfile, std::ios::binary);
                        savedReadstorageGpuData = std::move(readStorage.saveGpuDataAndFreeGpuMem(rstempostream, availableMemoryToSaveGpuPartitions));

                        didSaveGpudata = true;
                    }

                    
                    
                    //if all tables could be constructed at once, no need to save them to temporary file
                    if(requestedNumberOfMaps == currentIterNumTables){

                        for(int i = 0; i < currentIterNumTables; i++){
                            int globalTableId = numConstructedTables;
                            
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
                                deviceIds
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
                                deviceIds
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

        



    private:
        

        Range_t queryMap(int id, const Key_t& key) const{
            HashTable::QueryResult qr = minhashTables[id]->query(key);

            return std::make_pair(qr.valuesBegin, qr.valuesBegin + qr.numValues);
        }

        void addHashTable(HashTable&& hm){
            minhashTables.emplace_back(std::make_unique<HashTable>(std::move(hm)));
        }

        void computeReadHashesOnGpu(
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
        constructTablesWithGpuHashing(
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



        std::pair< std::vector<std::vector<kmer_type>>, std::vector<std::vector<read_number>> > 
        constructTablesWithGpuHashingUniquekmers1(
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
            const std::size_t encodedSequencePitchInInts = getEncodedNumInts2Bit(upperBoundSequenceLength);

            const auto& deviceIds = runtimeOptions.deviceIds;
            const int numThreads = runtimeOptions.threads;

            assert(deviceIds.size() > 0);

            const int deviceId = deviceIds[0];

            cudaSetDevice(deviceId); CUERR;

            const int numHashFuncs = numTables;
            const int firstHashFunc = firstTableId;
            const std::size_t signaturesRowPitchElements = 48;

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

            SimpleAllocationDevice<unsigned int, 1> d_sequenceData(encodedSequencePitchInInts * parallelReads);
            SimpleAllocationDevice<int, 0> d_lengths(parallelReads);

            SimpleAllocationPinnedHost<read_number, 0> h_indices(parallelReads);
            SimpleAllocationDevice<read_number, 0> d_indices(parallelReads);

            SimpleAllocationPinnedHost<std::uint64_t, 0> h_signatures(signaturesRowPitchElements * parallelReads);
            SimpleAllocationDevice<std::uint64_t, 0> d_signatures(signaturesRowPitchElements * parallelReads);
            SimpleAllocationDevice<int, 0> d_hashFuncIds(signaturesRowPitchElements * parallelReads);
            SimpleAllocationPinnedHost<int, 0> h_hashFuncIds(signaturesRowPitchElements * parallelReads);
            SimpleAllocationDevice<int, 0> d_signatureSizePerSequence(parallelReads);
            SimpleAllocationPinnedHost<int, 0> h_signatureSizePerSequence(parallelReads);
            SimpleAllocationDevice<std::uint64_t, 0> d_temp(signaturesRowPitchElements * parallelReads);

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

                const auto hashFuncIdsRowPitchElements = signaturesRowPitchElements;

                callUniqueMinhashSignaturesKernel_async(
                    d_temp,
                    d_signatures,
                    signaturesRowPitchElements,
                    d_hashFuncIds,
                    hashFuncIdsRowPitchElements,
                    d_signatureSizePerSequence,
                    d_sequenceData,
                    encodedSequencePitchInInts,
                    curBatchsize,
                    d_lengths,
                    getKmerSize(),
                    48,
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

                cudaMemcpyAsync(
                    h_hashFuncIds, 
                    d_hashFuncIds, 
                    hashFuncIdsRowPitchElements * sizeof(int) * curBatchsize, 
                    D2H, 
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    h_signatureSizePerSequence, 
                    d_signatureSizePerSequence, 
                    sizeof(int) * curBatchsize, 
                    D2H, 
                    stream
                ); CUERR;

                cudaStreamSynchronize(stream); CUERR;


                auto lambda = [
                    &, 
                    readIdBegin, 
                    sigs = h_signatures.get(),
                    sigsizes = h_signatureSizePerSequence.get(),
                    hids = h_hashFuncIds.get()
                ](auto begin, auto end, int threadId) {
                    std::uint64_t countlimit = 10000;
                    std::uint64_t count = 0;

                    for (read_number readId = begin; readId < end; readId++){
                        read_number localId = readId - readIdBegin;

                        const int signatureSize = sigsizes[localId];
                        for(int i = 0; i < signatureSize; i++){
                            const kmer_type kmer = kmer_mask & sigs[signaturesRowPitchElements * localId + i];
                            const int hashFuncId = hids[signaturesRowPitchElements * localId + i];
                            
                            if(firstTableId <= hashFuncId && hashFuncId < firstTableId + numHashFuncs){
                                const int localHashFuncId = hashFuncId - firstTableId;
                                kmersPerFunc[localHashFuncId][readId] = kmer;
                                readIdsPerFunc[localHashFuncId][readId] = readId;
                            }
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

        std::pair< std::vector<std::vector<kmer_type>>, std::vector<std::vector<read_number>> > 
        constructTablesWithGpuHashingUniquekmers2(
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
            const std::size_t encodedSequencePitchInInts = getEncodedNumInts2Bit(upperBoundSequenceLength);

            const auto& deviceIds = runtimeOptions.deviceIds;
            const int numThreads = runtimeOptions.threads;

            assert(deviceIds.size() > 0);

            const int deviceId = deviceIds[0];

            cudaSetDevice(deviceId); CUERR;

            const int numHashFuncs = numTables;
            const int firstHashFunc = firstTableId;
            const std::size_t signaturesRowPitchElements = 48;

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

                callMinhashSignaturesOfUniqueKmersKernel128_async(
                    d_signatures,
                    signaturesRowPitchElements,
                    d_sequenceData,
                    encodedSequencePitchInInts,
                    curBatchsize,
                    d_lengths,
                    getKmerSize(),
                    48,
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

                // for(int i = 0; i < 10; i++){
                //     for(int k = 0; k < signaturesRowPitchElements; k++){
                //         std::cerr << h_signatures[i * signaturesRowPitchElements + k] << " ";
                //     }
                //     std::cerr << "\n";
                // }


                auto lambda = [&, readIdBegin](auto begin, auto end, int threadId) {
                    std::uint64_t countlimit = 10000;
                    std::uint64_t count = 0;

                    for (read_number readId = begin; readId < end; readId++){
                        read_number localId = readId - readIdBegin;

                        for(int i = 0; i < numHashFuncs; i++){
                            const kmer_type kmer = kmer_mask & h_signatures[signaturesRowPitchElements * localId + i + firstTableId];
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

        int loadConstructedTablesFromFile(
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


        int kmerSize;
        int resultsPerMapThreshold;
        std::vector<std::unique_ptr<HashTable>> minhashTables;
    };






    
}
}



#endif
