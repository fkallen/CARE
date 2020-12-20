#ifndef CARE_FAKEFakeGpuMinhasher_CUH
#define CARE_FAKEFakeGpuMinhasher_CUH

#include <config.hpp>

#include <gpu/distributedreadstorage.hpp>
#include <gpu/cuda_unique.cuh>
#include <gpu/minhashingkernels.cuh>
#include <cpuhashtable.hpp>
#include <gpu/gpuminhasher.cuh>

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


#include <cub/cub.cuh>


namespace care{
namespace gpu{



    /*
        Minhasher which can store query results in gpu memory and uses the gpu to parallelize some portions of the code
        However, hash tables reside on the host
    */
    class FakeGpuMinhasher : public GpuMinhasher{
    public:
        using Key_t = GpuMinhasher::Key;
        using Value_t = read_number;
    private:
        using HashTable = CpuReadOnlyMultiValueHashTable<kmer_type, read_number>;
        using Range_t = std::pair<const Value_t*, const Value_t*>;

        struct QueryData{
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


            // void resize(const FakeGpuMinhasher& minhasher, std::size_t numSequences, int numThreads = 1){
            //     const std::size_t maximumResultSize 
            //         = minhasher.getNumResultsPerMapThreshold() * minhasher.getNumberOfMaps() * numSequences;

            //     d_minhashSignatures.resize(minhasher.getNumberOfMaps() * numSequences);
            //     h_minhashSignatures.resize(minhasher.getNumberOfMaps() * numSequences);
            //     h_candidate_read_ids_tmp.resize(maximumResultSize);
            //     d_candidate_read_ids_tmp.resize(maximumResultSize);

            //     h_begin_offsets.resize(numSequences+1);
            //     d_begin_offsets.resize(numSequences+1);
            //     h_end_offsets.resize(numSequences+1);
            //     d_end_offsets.resize(numSequences+1);
            //     h_global_begin_offsets.resize(numSequences);
            //     d_global_begin_offsets.resize(numSequences);
            
            //     allRanges.resize(minhasher.getNumberOfMaps() * numSequences);
            //     idsPerChunk.resize(numThreads, 0);   
            //     numAnchorsPerChunk.resize(numThreads, 0);
            //     idsPerChunkPrefixSum.resize(numThreads, 0);
            //     numAnchorsPerChunkPrefixSum.resize(numThreads, 0);

            //     d_temp.resize(minhasher.getNumberOfMaps() * numSequences);
            //     d_signatureSizePerSequence.resize(numSequences);
            //     h_signatureSizePerSequence.resize(numSequences);

            //     d_hashFuncIds.resize(minhasher.getNumberOfMaps() * numSequences);
            //     h_hashFuncIds.resize(minhasher.getNumberOfMaps() * numSequences);
            // }

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

        
    public:

        using QueryHandle = GpuMinhasher::QueryHandle;

        FakeGpuMinhasher() : FakeGpuMinhasher(16, 50){

        }

        FakeGpuMinhasher(int maxNumKeys_, int maxValuesPerKey, int k)
            : maxNumKeys(maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxValuesPerKey){

        }

        FakeGpuMinhasher(const FakeGpuMinhasher&) = delete;
        FakeGpuMinhasher(FakeGpuMinhasher&&) = default;
        FakeGpuMinhasher& operator=(const FakeGpuMinhasher&) = delete;
        FakeGpuMinhasher& operator=(FakeGpuMinhasher&&) = default;

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



        void constructFromReadStorage(
            const FileOptions &fileOptions,
            const RuntimeOptions &runtimeOptions,
            const MemoryOptions& memoryOptions,
            std::uint64_t nReads,
            const CorrectionOptions& correctionOptions,
            const DistributedReadStorage& gpuReadStorage
        );
 

        QueryHandle makeQueryHandle() const override {
            auto data = std::make_unique<QueryData>();
            data->segmentedUniqueHandle = GpuSegmentedUnique::makeHandle();
            cudaGetDevice(&data->deviceId); CUERR;
            data->isInitialized = true;

            std::lock_guard<std::mutex> lg(m);
            const int handleid = counter++;
            QueryHandle h = constructHandle(handleid);

            tempdataVector.emplace_back(std::move(data));
            return h;
        }



        void query(
            QueryHandle& handle,
            const unsigned int* d_encodedSequences,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int deviceId, 
            cudaStream_t stream,
            read_number* d_similarReadIds,
            int* d_similarReadsPerSequence,
            int* d_similarReadsPerSequencePrefixSum
        ) const override{
            query_impl(
                handle,
                nullptr,
                d_encodedSequences,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                deviceId,
                stream, 
                d_similarReadIds,
                d_similarReadsPerSequence,
                d_similarReadsPerSequencePrefixSum
            );
        }

        void queryExcludingSelf(
            QueryHandle& handle,
            const read_number* d_readIds,
            const unsigned int* d_encodedSequences,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int deviceId, 
            cudaStream_t stream,
            read_number* d_similarReadIds,
            int* d_similarReadsPerSequence,
            int* d_similarReadsPerSequencePrefixSum
        ) const override{
            query_impl(
                handle,
                d_readIds,
                d_encodedSequences,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                deviceId,
                stream, 
                d_similarReadIds,
                d_similarReadsPerSequence,
                d_similarReadsPerSequencePrefixSum
            );
        }

        void compact(cudaStream_t stream) override{
            std::cerr << "FakeGpuMinhasher::compact\n";
        }

        MemoryUsage getMemoryInfo() const noexcept override{
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

        MemoryUsage getMemoryInfo(const QueryHandle& handle) const noexcept override{
            return tempdataVector[handle.getId()]->getMemoryInfo();
        }

        int getNumResultsPerMapThreshold() const noexcept override{
            return resultsPerMapThreshold;
        }
        
        int getNumberOfMaps() const noexcept override{
            return minhashTables.size();
        }

        void destroy() override{
            minhashTables.clear();
        }

        void query_impl(
            QueryHandle& handle,
            const read_number* d_readIds,
            const unsigned int* d_encodedSequences,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int deviceId, 
            cudaStream_t stream,
            read_number* d_similarReadIds,
            int* d_similarReadsPerSequence,
            int* d_similarReadsPerSequencePrefixSum
        ) const{
            QueryData* const queryData = tempdataVector[handle.getId()].get();

            assert(queryData->isInitialized);
            if(numSequences == 0) return;

            cub::SwitchDevice sd(deviceId);

            std::size_t cubtempbytes = 0;

            cub::DeviceScan::InclusiveSum(
                nullptr,
                cubtempbytes,
                (int*) nullptr,
                (int*) nullptr,
                numSequences,
                stream
            );

            queryData->d_minhashSignatures.resize(
                std::max(
                    getNumberOfMaps() * numSequences, 
                    int(SDIV(cubtempbytes, sizeof(std::uint64_t)))
                )
            );
            queryData->h_minhashSignatures.resize(getNumberOfMaps() * numSequences);
            queryData->h_begin_offsets.resize(numSequences+1);
            queryData->d_begin_offsets.resize(numSequences+1);
            queryData->h_end_offsets.resize(numSequences+1);
            queryData->d_end_offsets.resize(numSequences+1);
            queryData->h_global_begin_offsets.resize(numSequences);
            queryData->d_global_begin_offsets.resize(numSequences);

            std::vector<Range_t>& allRanges = queryData->allRanges;

            allRanges.resize(getNumberOfMaps() * numSequences);

            const std::size_t hashValuesPitchInElements = getNumberOfMaps();
            const int firstHashFunc = 0;

            callMinhashSignaturesKernel(
                queryData->d_minhashSignatures.get(),
                hashValuesPitchInElements,
                d_encodedSequences,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                getNumberOfMaps(),
                firstHashFunc,
                stream
            );

            cudaMemcpyAsync(
                queryData->h_minhashSignatures.get(),
                queryData->d_minhashSignatures.get(),
                queryData->h_minhashSignatures.sizeInBytes(),
                H2D,
                stream
            ); CUERR;
    
            cudaStreamSynchronize(stream); CUERR; //wait for D2H transfers of signatures

            int myTotalNumberOfPossibleCandidates = 0;

            nvtx::push_range("queryPrecalculatedSignatures", 6);
            queryPrecalculatedSignatures(
                queryData->h_minhashSignatures.get(),
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
            queryData->h_candidate_read_ids_tmp.resize(roundedTotalNum);
            queryData->d_candidate_read_ids_tmp.resize(roundedTotalNum);


            //results will be in Current() buffer
            cub::DoubleBuffer<read_number> d_values_dblbuf(d_similarReadIds, queryData->d_candidate_read_ids_tmp.data());

            read_number* hostdatabegin = queryData->h_candidate_read_ids_tmp.get();

            Range_t* const myRanges = allRanges.data();
            const std::uint64_t* const mySignatures = queryData->h_minhashSignatures;
            int* const h_my_begin_offsets = queryData->h_begin_offsets;
            int* const h_my_end_offsets = queryData->h_end_offsets;
            int* const d_my_begin_offsets = queryData->d_begin_offsets;
            int* const d_my_end_offsets = queryData->d_end_offsets;

            //copy hits from hash tables to pinned memory
            auto* dest = hostdatabegin;    
            const int lmax = numSequences * getNumberOfMaps();

            for(int sequenceIndex = 0; sequenceIndex < numSequences; sequenceIndex++){

                h_my_begin_offsets[sequenceIndex] = std::distance(hostdatabegin, dest);
                queryData->h_global_begin_offsets[sequenceIndex] = std::distance(queryData->h_candidate_read_ids_tmp.get(), dest);

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
                    if(std::distance(range.first, range.second) <= getNumResultsPerMapThreshold())
                        dest = std::copy(range.first, range.second, dest);
                }

                h_my_end_offsets[sequenceIndex] = std::distance(hostdatabegin, dest);
            }

            cudaMemcpyAsync(
                d_values_dblbuf.Current(),
                hostdatabegin,
                sizeof(read_number) * myTotalNumberOfPossibleCandidates,
                H2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_my_begin_offsets,
                h_my_begin_offsets,
                sizeof(int) * numSequences,
                H2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_my_end_offsets,
                h_my_end_offsets,
                sizeof(int) * numSequences,
                H2D,
                stream
            ); CUERR;

            GpuSegmentedUnique::unique(
                queryData->segmentedUniqueHandle,
                d_values_dblbuf.Current(), //input
                myTotalNumberOfPossibleCandidates,
                d_values_dblbuf.Alternate(), //output
                d_similarReadsPerSequence,
                numSequences,
                d_my_begin_offsets, //device accessible
                d_my_end_offsets, //device accessible
                h_my_begin_offsets,
                h_my_end_offsets,
                0,
                sizeof(read_number) * 8,
                stream
            );

            cudaMemcpyAsync(
                queryData->d_global_begin_offsets.get(),
                queryData->h_global_begin_offsets.get(),
                sizeof(int) * numSequences,
                H2D,
                stream
            ); CUERR;

            if(d_readIds != nullptr){

                //remove self read ids (inplace)
                //--------------------------------------------------------------------
                callFindAndRemoveFromSegmentKernel<read_number,128,4>(
                    d_readIds,
                    d_values_dblbuf.Alternate(),
                    numSequences,
                    d_similarReadsPerSequence,
                    queryData->d_global_begin_offsets.data(),
                    stream
                );

            }

            int* d_newOffsets = d_similarReadsPerSequencePrefixSum;
            void* d_cubTemp = queryData->d_minhashSignatures.data();

            cudaMemsetAsync(d_newOffsets, 0, sizeof(int), stream); CUERR;

            cub::DeviceScan::InclusiveSum(
                d_cubTemp,
                cubtempbytes,
                d_similarReadsPerSequence,
                d_newOffsets + 1,
                numSequences,
                stream
            );

            //copy final remaining values into contiguous range
            helpers::lambda_kernel<<<numSequences, 128, 0, stream>>>(
                [
                    d_values_in = d_values_dblbuf.Alternate(),
                    d_values_out = d_values_dblbuf.Current(),
                    numSequences,
                    d_numValuesPerSequence = d_similarReadsPerSequence,
                    d_offsets = queryData->d_global_begin_offsets.data(),
                    d_newOffsets
                ] __device__ (){

                    for(int s = blockIdx.x; s < numSequences; s += gridDim.x){
                        const int numValues = d_numValuesPerSequence[s];
                        const int inOffset = d_offsets[s];
                        const int outOffset = d_newOffsets[s];

                        for(int c = threadIdx.x; c < numValues; c += blockDim.x){
                            d_values_out[outOffset + c] = d_values_in[inOffset + c];    
                        }
                    }
                }
            ); CUERR;
        }

        void queryPrecalculatedSignatures(
            const std::uint64_t* signatures, //getNumberOfMaps() elements per sequence
            FakeGpuMinhasher::Range_t* ranges, //getNumberOfMaps() elements per sequence
            int* totalNumResultsInRanges, 
            int numSequences) const;


        int getKmerSize() const{
            return kmerSize;
        }

        std::uint64_t getKmerMask() const{
            constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

            return std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - getKmerSize()) * 2);
        }


        void writeToStream(std::ostream& os) const;
    
        int loadFromStream(std::ifstream& is, int numMapsUpperLimit = std::numeric_limits<int>::max());

        int calculateResultsPerMapThreshold(int coverage);

        

        int addHashfunctions(int numExtraFunctions){
            // std::vector<std::unique_ptr<HashTable>> minhashTables;

            int added = 0;
            int cur = minhashTables.size();

            assert(!(numExtraFunctions + cur > 64));

            for(int i = 0; i < numExtraFunctions; i++){
                try{
                    auto ptr = std::make_unique<HashTable>(maxNumKeys);

                    minhashTables.emplace_back(std::move(ptr));
                    added++;
                }catch(...){

                }
            }

            return added;
        }    



private:
    

    Range_t queryMap(int id, const Key_t& key) const;

    void addHashTable(HashTable&& hm);        

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

        mutable int counter = 0;
        mutable std::mutex m{};

        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        std::vector<std::unique_ptr<HashTable>> minhashTables{};
        mutable std::vector<std::unique_ptr<QueryData>> tempdataVector{};
    };






    
}
}



#endif
