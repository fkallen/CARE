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

#include <sequencehelpers.hpp>
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

            bool isInitialized = false;
            int deviceId;

            DeviceBuffer<std::uint64_t> d_minhashSignatures;
            PinnedBuffer<std::uint64_t> h_minhashSignatures;            

            PinnedBuffer<read_number> h_candidate_read_ids_tmp;
            DeviceBuffer<read_number> d_candidate_read_ids_tmp;

            PinnedBuffer<int> h_begin_offsets;
            DeviceBuffer<int> d_begin_offsets;
            PinnedBuffer<int> h_end_offsets;
            DeviceBuffer<int> d_end_offsets;
            PinnedBuffer<int> h_global_begin_offsets;
            DeviceBuffer<int> d_global_begin_offsets;

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
            std::vector<GpuSegmentedUnique::Handle> segmentedUniqueHandles;


            void resize(const GpuMinhasher& minhasher, std::size_t numSequences, int numThreads = 1){
                const std::size_t maximumResultSize 
                    = minhasher.getNumResultsPerMapThreshold() * minhasher.getNumberOfMaps() * numSequences;

                d_minhashSignatures.resize(minhasher.getNumberOfMaps() * numSequences);
                h_minhashSignatures.resize(minhasher.getNumberOfMaps() * numSequences);
                h_candidate_read_ids_tmp.resize(maximumResultSize);
                d_candidate_read_ids_tmp.resize(maximumResultSize);

                h_begin_offsets.resize(numSequences+1);
                d_begin_offsets.resize(numSequences+1);
                h_end_offsets.resize(numSequences+1);
                d_end_offsets.resize(numSequences+1);
                h_global_begin_offsets.resize(numSequences);
                d_global_begin_offsets.resize(numSequences);
            
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
                handlehost(h_end_offsets);
                handlehost(h_global_begin_offsets);
    
                handledevice(d_minhashSignatures);
                handledevice(d_candidate_read_ids_tmp);
                handledevice(d_begin_offsets);
                handledevice(d_end_offsets);
                handledevice(d_global_begin_offsets);

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

                for(const auto& h : segmentedUniqueHandles){
                    info += h->getMemoryInfo();
                }
    
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
                h_end_offsets.destroy();
                d_end_offsets.destroy();
                h_global_begin_offsets.destroy();
                d_global_begin_offsets.destroy();

                d_cub_temp.destroy();

                allRanges.clear();
                allRanges.shrink_to_fit();

                d_temp.destroy();
                d_signatureSizePerSequence.destroy();
                h_signatureSizePerSequence.destroy();

                d_hashFuncIds.destroy();
                h_hashFuncIds.destroy();

                segmentedUniqueHandle = nullptr;
                for(auto& h : segmentedUniqueHandles){
                    h = nullptr;
                }

                cudaSetDevice(cur); CUERR;
                isInitialized = false;
            }
        };

        static QueryHandle makeQueryHandle(){
            QueryHandle handle;
            handle.segmentedUniqueHandle = GpuSegmentedUnique::makeHandle();            

            cudaGetDevice(&handle.deviceId); CUERR;

            handle.isInitialized = true;

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

        std::array<std::uint64_t, maximum_number_of_maps> 
        hostminhashfunction(const char* sequence, int sequenceLength, int kmerLength, int numHashFuncs) const noexcept{

            const int length = sequenceLength;

            std::array<std::uint64_t, maximum_number_of_maps> minhashSignature;
            std::fill_n(minhashSignature.begin(), numHashFuncs, std::numeric_limits<std::uint64_t>::max());

            if(length < kmerLength) return minhashSignature;

            constexpr int maximum_kmer_length = max_k<kmer_type>::value;
            const kmer_type kmer_mask = std::numeric_limits<kmer_type>::max() >> ((maximum_kmer_length - kmerLength) * 2);
            const int rcshiftamount = (maximum_kmer_length - kmerLength) * 2;

            auto handlekmer = [&](auto fwd, auto rc, int numhashfunc){
                using hasher = hashers::MurmurHash<std::uint64_t>;

                const auto smallest = std::min(fwd, rc);
                const auto hashvalue = hasher::hash(smallest + numhashfunc);
                minhashSignature[numhashfunc] = std::min(minhashSignature[numhashfunc], hashvalue);
            };

            kmer_type kmer_encoded = 0;
            kmer_type rc_kmer_encoded = std::numeric_limits<kmer_type>::max();

            auto addBase = [&](char c){
                kmer_encoded <<= 2;
                rc_kmer_encoded >>= 2;
                switch(c) {
                case 'A':
                    kmer_encoded |= 0;
                    rc_kmer_encoded |= kmer_type(3) << (sizeof(kmer_type) * 8 - 2);
                    break;
                case 'C':
                    kmer_encoded |= 1;
                    rc_kmer_encoded |= kmer_type(2) << (sizeof(kmer_type) * 8 - 2);
                    break;
                case 'G':
                    kmer_encoded |= 2;
                    rc_kmer_encoded |= kmer_type(1) << (sizeof(kmer_type) * 8 - 2);
                    break;
                case 'T':
                    kmer_encoded |= 3;
                    rc_kmer_encoded |= kmer_type(0) << (sizeof(kmer_type) * 8 - 2);
                    break;
                default:break;
                }
            };

            for(int i = 0; i < kmerLength - 1; i++){
                addBase(sequence[i]);
            }

            for(int i = kmerLength - 1; i < length; i++){
                addBase(sequence[i]);

                for(int m = 0; m < numHashFuncs; m++){
                    handlekmer(kmer_encoded & kmer_mask, 
                                rc_kmer_encoded >> rcshiftamount, 
                                m);
                }
            }

            return minhashSignature;
        }

        //host version
        void getCandidates(
            QueryHandle& handle, 
            std::vector<read_number>& ids,
            const char* sequence,
            int sequenceLength
        ) const{

            // we do not consider reads which are shorter than k
            if(sequenceLength < getKmerSize()){
                ids.clear();
                return;
            }

            const std::uint64_t kmer_mask = getKmerMask();
    
            auto hashValues = hostminhashfunction(sequence, sequenceLength, getKmerSize(), getNumberOfMaps());

            std::array<Range_t, maximum_number_of_maps> allRanges;

            int totalNumResults = 0;
    
            nvtx::push_range("queryPrecalculatedSignatures", 6);
            queryPrecalculatedSignatures(
                hashValues.data(),
                allRanges.data(),
                &totalNumResults, 
                1
            );
            nvtx::pop_range();

            auto handlesEnd = std::remove_if(
                allRanges.begin(), 
                allRanges.end(), 
                [](const auto& range){return 0 == std::distance(range.first, range.second);}
            );

            const int numNonEmptyRanges = std::distance(allRanges.begin(), handlesEnd);

            ids.resize(totalNumResults);

            nvtx::push_range("k_way_set_union", 7);
            SetUnionHandle suHandle;
            auto resultEnd = k_way_set_union(suHandle, ids.data(), allRanges.data(), numNonEmptyRanges);
            nvtx::pop_range();
            const std::size_t resultSize = std::distance(ids.data(), resultEnd);
            ids.erase(ids.begin() + resultSize, ids.end());
        }



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
            assert(handle.isInitialized);

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
            if(totalNumIds == 0){
                cudaMemsetAsync(d_similarReadsPerSequence, 0, sizeof(int) * numSequences, stream);
                cudaMemsetAsync(d_similarReadsPerSequencePrefixSum, 0, sizeof(int) * (numSequences + 1), stream);
                return;
            }

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
            assert(handle.isInitialized);

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
            if(totalNumIds == 0){
                cudaMemsetAsync(d_similarReadsPerSequence, 0, sizeof(int) * numSequences, stream);
                cudaMemsetAsync(d_similarReadsPerSequencePrefixSum, 0, sizeof(int) * (numSequences + 1), stream);
                return;
            }
            
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

            // cudaStreamSynchronize(stream); CUERR;
            // static int aaa = 0;

            // if(aaa == 0){
            //     std::cerr << numSequences << "\n";
            //     std::cerr << "h_begin_offsets:\n";
            //     for(int i = 0; i < numSequences + 1; i++){
            //         std::cerr << handle.h_begin_offsets[i] << ", ";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "d_similarReadsPerSequence:\n";
            //     for(int i = 0; i < numSequences; i++){
            //         std::cerr << d_similarReadsPerSequence[i] << ", ";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "h_candidate_read_ids_tmp:\n";
            //     for(int i = 0; i < getNumResultsPerMapThreshold() * getNumberOfMaps() * numSequences; i++){
            //         std::cerr << handle.h_candidate_read_ids_tmp[i] << ", ";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "d_candidate_read_ids_tmp:\n";
            //     for(int i = 0; i < getNumResultsPerMapThreshold() * getNumberOfMaps() * numSequences; i++){
            //         std::cerr << handle.d_candidate_read_ids_tmp[i] << ", ";
            //     }
            //     std::cerr << "\n";

            //     aaa = 1;
            // }

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
        void getIdsOfSimilarReadsNormalExcludingSelfNew(
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
            assert(handle.isInitialized);

            int currentDeviceId = 0;
            cudaGetDevice(&currentDeviceId); CUERR;
            cudaSetDevice(deviceId); CUERR;

            //const std::size_t maximumResultSize = getNumResultsPerMapThreshold() * getNumberOfMaps() * numSequences;

            handle.d_minhashSignatures.resize(getNumberOfMaps() * numSequences);
            handle.h_minhashSignatures.resize(getNumberOfMaps() * numSequences);
            // handle.h_candidate_read_ids_tmp.resize(maximumResultSize);
            // handle.d_candidate_read_ids_tmp.resize(maximumResultSize);
            handle.h_begin_offsets.resize(numSequences+1);
            handle.d_begin_offsets.resize(numSequences+1);
            handle.h_end_offsets.resize(numSequences+1);
            handle.d_end_offsets.resize(numSequences+1);
            handle.h_global_begin_offsets.resize(numSequences);
            handle.d_global_begin_offsets.resize(numSequences);

            std::vector<Range_t>& allRanges = handle.allRanges;

            // const int maxNumThreads = parallelFor.getNumThreads();

            // while(handle.segmentedUniqueHandles.size() < std::size_t(maxNumThreads)){
            //     handle.segmentedUniqueHandles.emplace_back(GpuSegmentedUnique::makeHandle());
            // }

            allRanges.resize(getNumberOfMaps() * numSequences);
            // idsPerChunk.resize(maxNumThreads, 0);   
            // numAnchorsPerChunk.resize(maxNumThreads, 0);
            // idsPerChunkPrefixSum.resize(maxNumThreads, 0);
            // numAnchorsPerChunkPrefixSum.resize(maxNumThreads, 0);

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


            // std::fill(idsPerChunk.begin(), idsPerChunk.end(), 0);
            // std::fill(numAnchorsPerChunk.begin(), numAnchorsPerChunk.end(), 0);
    
            cudaStreamSynchronize(stream); CUERR; //wait for D2H transfers of signatures anchor data

            // std::vector<int> numSequencesPerPartition(maxNumThreads, 0);
            // std::vector<int> numSequencesPerPartitionPrefixSum(maxNumThreads, 0);

            // for(int i = 0; i < maxNumThreads; i++){
            //     int defaultNum = numSequences / maxNumThreads;
            //     const int remainder = numSequences % maxNumThreads;
            //     if(i < remainder){
            //         defaultNum++;
            //     }
            //     numSequencesPerPartition[i] = defaultNum;
            //     if(i != maxNumThreads-1){
            //         numSequencesPerPartitionPrefixSum[i+1] = numSequencesPerPartitionPrefixSum[i] + defaultNum;
            //     }
            // }

            // std::cerr << "numSequencesPerPartition\n";
            // for(int i = 0; i < maxNumThreads; i++){
            //     std::cerr << numSequencesPerPartition[i] << ", ";
            // }
            // std::cerr << "\n";

            // std::cerr << "numSequencesPerPartitionPrefixSum\n";
            // for(int i = 0; i < maxNumThreads; i++){
            //     std::cerr << numSequencesPerPartitionPrefixSum[i] << ", ";
            // }
            // std::cerr << "\n";

#if 1

            

            int myTotalNumberOfPossibleCandidates = 0;

            nvtx::push_range("queryPrecalculatedSignatures", 6);
            queryPrecalculatedSignatures(
                handle.h_minhashSignatures.get(),
                allRanges.data(),
                &myTotalNumberOfPossibleCandidates, 
                numSequences
            );
            nvtx::pop_range();

            if(myTotalNumberOfPossibleCandidates == 0){
                cudaMemsetAsync(d_similarReadsPerSequence, 0, sizeof(int) * numSequences, stream);
                cudaMemsetAsync(d_similarReadsPerSequencePrefixSum, 0, sizeof(int) * (numSequences + 1), stream);
                return;
            }

            constexpr int roundUpTo = 10000;
            const int roundedTotalNum = SDIV(myTotalNumberOfPossibleCandidates, roundUpTo) * roundUpTo;
            handle.h_candidate_read_ids_tmp.resize(roundedTotalNum);
            handle.d_candidate_read_ids_tmp.resize(roundedTotalNum);

            const int myNumSequences = numSequences;
            const int myNumSequencesOffset = 0;

            const int candidateIdsOffset = 0;
            read_number* hostdatabegin = handle.h_candidate_read_ids_tmp.get();
            read_number* devicedatabegin = d_similarReadIds;
            Range_t* const myRanges = allRanges.data();
            const std::uint64_t* const mySignatures = handle.h_minhashSignatures;
            int* const h_my_begin_offsets = handle.h_begin_offsets;
            int* const h_my_end_offsets = handle.h_end_offsets;
            int* const d_my_begin_offsets = handle.d_begin_offsets;
            int* const d_my_end_offsets = handle.d_end_offsets;

            //copy hits from hash tables to pinned memory
            auto* dest = hostdatabegin;    
            const int lmax = myNumSequences * getNumberOfMaps();

            for(int sequenceIndex = 0; sequenceIndex < myNumSequences; sequenceIndex++){

                h_my_begin_offsets[sequenceIndex] = std::distance(hostdatabegin, dest);
                handle.h_global_begin_offsets[myNumSequencesOffset + sequenceIndex] = std::distance(handle.h_candidate_read_ids_tmp.get(), dest);

                for(int mapIndex = 0; mapIndex < getNumberOfMaps(); mapIndex++){
                    const int k = sequenceIndex * getNumberOfMaps() + mapIndex;
                    
                    constexpr int nextprefetch = 2;

                    //prefetch first element of next range if the next range is not empty
                    if(k+nextprefetch < lmax){
                        if(myRanges[k+nextprefetch].first != myRanges[k+nextprefetch].second){
                            __builtin_prefetch(myRanges[k+nextprefetch].first, 0, 0);
                        }
                    }
                    const auto& range = myRanges[k];
                    dest = std::copy(range.first, range.second, dest);
                }

                h_my_end_offsets[sequenceIndex] = std::distance(hostdatabegin, dest);
            }

            // std::cerr << "h_my_begin_offsets partition "<< partitionId << "\n";
            // for(int i = 0; i < myNumSequences; i++){
            //     std::cerr << h_my_begin_offsets[i] << "," ;
            // }
            // std::cerr << "\n";

            // std::cerr << "h_my_end_offsets partition "<< partitionId << "\n";
            // for(int i = 0; i < myNumSequences; i++){
            //     std::cerr << h_my_end_offsets[i] << "," ;
            // }
            // std::cerr << "\n";

            cudaMemcpyAsync(
                devicedatabegin,
                hostdatabegin,
                sizeof(read_number) * myTotalNumberOfPossibleCandidates,
                H2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_my_begin_offsets,
                h_my_begin_offsets,
                sizeof(int) * myNumSequences,
                H2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_my_end_offsets,
                h_my_end_offsets,
                sizeof(int) * myNumSequences,
                H2D,
                stream
            ); CUERR;

            //copy-kernel to device for read ids and offsets
            // helpers::lambda_kernel<<<SDIV(myTotalNumberOfPossibleCandidates, 256), 256, 0, stream>>>(
            //     [=] __device__ (){
            //         const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            //         const int stride = blockDim.x * gridDim.x;

            //         for(int i = tid; i < myTotalNumberOfPossibleCandidates; i += stride){
            //             devicedatabegin[i] = hostdatabegin[i];
            //         }

            //         for(int i = tid; i < myNumSequences; i += stride){
            //             d_my_begin_offsets[i] = h_my_begin_offsets[i];
            //         }

            //         for(int i = tid; i < myNumSequences; i += stride){
            //             d_my_end_offsets[i] = h_my_end_offsets[i];
            //         }
            //     }
            // ); CUERR;

            GpuSegmentedUnique::unique(
                handle.segmentedUniqueHandle,
                devicedatabegin, //input
                myTotalNumberOfPossibleCandidates,
                handle.d_candidate_read_ids_tmp.get() + candidateIdsOffset, //output
                d_similarReadsPerSequence + myNumSequencesOffset,
                myNumSequences,
                d_my_begin_offsets, //device accessible
                d_my_end_offsets, //device accessible
                h_my_begin_offsets,
                h_my_end_offsets,
                0,
                sizeof(read_number) * 8,
                stream
            );
#else
            auto processHashes = [&, this](int partitionBegin, int partitionEnd, int threadId){
                for(int partitionId = partitionBegin; partitionId < partitionEnd; partitionId++){
                    const int myNumSequences = numSequencesPerPartition[partitionId];
                    const int myNumSequencesOffset = numSequencesPerPartitionPrefixSum[partitionId];

                    const int candidateIdsOffset = getNumResultsPerMapThreshold() * getNumberOfMaps() * myNumSequencesOffset;
                    read_number* hostdatabegin = handle.h_candidate_read_ids_tmp.get() + candidateIdsOffset;
                    read_number* devicedatabegin = d_similarReadIds + candidateIdsOffset;
                    Range_t* const myRanges = allRanges.data() + myNumSequencesOffset * getNumberOfMaps();
                    const std::uint64_t* const mySignatures = handle.h_minhashSignatures.get() + myNumSequencesOffset * getNumberOfMaps();
                    int* const h_my_begin_offsets = handle.h_begin_offsets + myNumSequencesOffset;
                    int* const h_my_end_offsets = handle.h_end_offsets + myNumSequencesOffset;
                    int* const d_my_begin_offsets = handle.d_begin_offsets + myNumSequencesOffset;
                    int* const d_my_end_offsets = handle.d_end_offsets + myNumSequencesOffset;
                    
        
                    int myTotalNumberOfPossibleCandidates = 0;

                    nvtx::push_range("queryPrecalculatedSignatures", 6);
                    queryPrecalculatedSignatures(
                        mySignatures,
                        myRanges,
                        &myTotalNumberOfPossibleCandidates, 
                        myNumSequences
                    );
                    nvtx::pop_range();

                    //copy hits from hash tables to pinned memory
                    auto* dest = hostdatabegin;    
                    const int lmax = myNumSequences * getNumberOfMaps();

                    for(int sequenceIndex = 0; sequenceIndex < myNumSequences; sequenceIndex++){

                        h_my_begin_offsets[sequenceIndex] = std::distance(hostdatabegin, dest);
                        handle.h_global_begin_offsets[myNumSequencesOffset + sequenceIndex] = std::distance(handle.h_candidate_read_ids_tmp.get(), dest);

                        for(int mapIndex = 0; mapIndex < getNumberOfMaps(); mapIndex++){
                            const int k = sequenceIndex * getNumberOfMaps() + mapIndex;
                            
                            constexpr int nextprefetch = 2;

                            //prefetch first element of next range if the next range is not empty
                            if(k+nextprefetch < lmax){
                                if(myRanges[k+nextprefetch].first != myRanges[k+nextprefetch].second){
                                    __builtin_prefetch(myRanges[k+nextprefetch].first, 0, 0);
                                }
                            }
                            const auto& range = myRanges[k];
                            dest = std::copy(range.first, range.second, dest);
                        }

                        h_my_end_offsets[sequenceIndex] = std::distance(hostdatabegin, dest);
                    }

                    // std::cerr << "h_my_begin_offsets partition "<< partitionId << "\n";
                    // for(int i = 0; i < myNumSequences; i++){
                    //     std::cerr << h_my_begin_offsets[i] << "," ;
                    // }
                    // std::cerr << "\n";

                    // std::cerr << "h_my_end_offsets partition "<< partitionId << "\n";
                    // for(int i = 0; i < myNumSequences; i++){
                    //     std::cerr << h_my_end_offsets[i] << "," ;
                    // }
                    // std::cerr << "\n";

                    cudaMemcpyAsync(
                        devicedatabegin,
                        hostdatabegin,
                        sizeof(read_number) * myTotalNumberOfPossibleCandidates,
                        H2D,
                        stream
                    ); CUERR;

                    cudaMemcpyAsync(
                        d_my_begin_offsets,
                        h_my_begin_offsets,
                        sizeof(int) * myNumSequences,
                        H2D,
                        stream
                    ); CUERR;

                    cudaMemcpyAsync(
                        d_my_end_offsets,
                        h_my_end_offsets,
                        sizeof(int) * myNumSequences,
                        H2D,
                        stream
                    ); CUERR;

                    //copy-kernel to device for read ids and offsets
                    // helpers::lambda_kernel<<<SDIV(myTotalNumberOfPossibleCandidates, 256), 256, 0, stream>>>(
                    //     [=] __device__ (){
                    //         const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    //         const int stride = blockDim.x * gridDim.x;

                    //         for(int i = tid; i < myTotalNumberOfPossibleCandidates; i += stride){
                    //             devicedatabegin[i] = hostdatabegin[i];
                    //         }

                    //         for(int i = tid; i < myNumSequences; i += stride){
                    //             d_my_begin_offsets[i] = h_my_begin_offsets[i];
                    //         }

                    //         for(int i = tid; i < myNumSequences; i += stride){
                    //             d_my_end_offsets[i] = h_my_end_offsets[i];
                    //         }
                    //     }
                    // ); CUERR;

                    GpuSegmentedUnique::unique(
                        handle.segmentedUniqueHandles[partitionId],
                        devicedatabegin, //input
                        myTotalNumberOfPossibleCandidates,
                        handle.d_candidate_read_ids_tmp.get() + candidateIdsOffset, //output
                        d_similarReadsPerSequence + myNumSequencesOffset,
                        myNumSequences,
                        d_my_begin_offsets, //device accessible
                        d_my_end_offsets, //device accessible
                        h_my_begin_offsets,
                        h_my_end_offsets,
                        0,
                        sizeof(read_number) * 8,
                        stream
                    );


                }
   
            };

    
            parallelFor(
                0, 
                maxNumThreads, 
                processHashes
            );
#endif
            // handle.h_global_begin_offsets[0] = 0;

            // std::cerr << "h_global_begin_offsets\n";
            // for(int i = 0; i < numSequences + 1; i++){
            //     std::cerr << handle.h_global_begin_offsets[i] << "," ;
            // }
            // std::cerr << "\n";

            cudaMemcpyAsync(
                handle.d_global_begin_offsets.get(),
                handle.h_global_begin_offsets.get(),
                sizeof(int) * numSequences,
                H2D,
                stream
            ); CUERR;

            //processHashes(0, maxNumThreads, 0);

            



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
                    d_begin_offsets = handle.d_global_begin_offsets.get(),
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

            // cudaStreamSynchronize(stream);

            // static int aaa = 0;

            // if(aaa == 0){
            //     std::cerr << numSequences << "\n";
            //     std::cerr << "h_begin_offsets:\n";
            //     for(int i = 0; i < numSequences + 1; i++){
            //         std::cerr << handle.h_begin_offsets[i] << ", ";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "d_similarReadsPerSequence:\n";
            //     for(int i = 0; i < numSequences; i++){
            //         std::cerr << d_similarReadsPerSequence[i] << ", ";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "h_candidate_read_ids_tmp:\n";
            //     for(int i = 0; i < getNumResultsPerMapThreshold() * getNumberOfMaps() * numSequences; i++){
            //         std::cerr << handle.h_candidate_read_ids_tmp[i] << ", ";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "d_candidate_read_ids_tmp:\n";
            //     for(int i = 0; i < getNumResultsPerMapThreshold() * getNumberOfMaps() * numSequences; i++){
            //         std::cerr << handle.d_candidate_read_ids_tmp[i] << ", ";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "d_candidate_read_ids_tmp:\n";
            //     for(int i = 0; i < getNumResultsPerMapThreshold() * getNumberOfMaps() * numSequences; i++){
            //         std::cerr << handle.d_candidate_read_ids_tmp[i] << ", ";
            //     }
            //     std::cerr << "\n";

            //     aaa = 1;
            // }



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
            assert(handle.isInitialized);
            
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
