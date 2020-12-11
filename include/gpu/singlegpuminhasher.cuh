#ifndef CARE_SINGLE_GPU_MINHASHER_CUH
#define CARE_SINGLE_GPU_MINHASHER_CUH

#include <config.hpp>

#include <warpcore.cuh>

#include <gpu/distributedreadstorage.hpp>
#include <gpu/cuda_unique.cuh>
#include <cpuhashtable.hpp>
#include <gpu/gpuhashtable.cuh>
#include <gpu/kernels.hpp>


#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>

#include <sequencehelpers.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>

#include <cub/cub.cuh>

#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>

namespace care{
namespace gpu{


    class SingleGpuMinhasher{
    public:
        using Key = kmer_type;
        using Value = read_number;

        using GpuTable = GpuHashtable<Key, Value>;

        using DeviceSwitcher = cub::SwitchDevice;

        template<class T>
        using HostBuffer = helpers::SimpleAllocationPinnedHost<T, 5>;
        template<class T>
        using DeviceBuffer = helpers::SimpleAllocationDevice<T, 5>;


        SingleGpuMinhasher(int maxNumKeys_, int maxValuesPerKey, int k)
            : maxNumKeys(maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxValuesPerKey)
        {
            cudaGetDevice(&deviceId); CUERR;
        }

        int constructFromReadStorage(
            const RuntimeOptions &runtimeOptions,
            std::uint64_t nReads,
            const DistributedReadStorage& gpuReadStorage,
            int upperBoundSequenceLength,
            int maxNumHashfunctions,
            int hashFunctionOffset = 0
        ){
            
            DeviceSwitcher ds(deviceId);

            gpuHashTables.clear();


            constexpr read_number parallelReads = 1000000;
            const read_number numReads = nReads;
            const int numIters = SDIV(numReads, parallelReads);
            const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(upperBoundSequenceLength);

            const int numThreads = runtimeOptions.threads;
            ThreadPool::ParallelForHandle pforHandle;
            ThreadPool threadPool(numThreads);

            helpers::SimpleAllocationDevice<unsigned int, 1> d_sequenceData(encodedSequencePitchInInts * parallelReads);
            helpers::SimpleAllocationDevice<int, 0> d_lengths(parallelReads);

            helpers::SimpleAllocationPinnedHost<read_number, 0> h_indices(parallelReads);
            helpers::SimpleAllocationDevice<read_number, 0> d_indices(parallelReads);

            std::vector<int> usedHashFunctionNumbers;

            CudaStream stream{};

            auto sequencehandle = gpuReadStorage.makeGatherHandleSequences();

            auto showProgress = [&](auto totalCount, auto seconds){
                if(runtimeOptions.showProgress){
                    std::cout << "Hashed " << totalCount << " / " << numReads << " reads. Elapsed time: " 
                            << seconds << " seconds.\n";
                }
            };

            auto updateShowProgressInterval = [](auto duration){
                return duration * 2;
            };

            int remainingHashFunctions = maxNumHashfunctions;
            bool keepGoing = true;

            while(remainingHashFunctions > 0 && keepGoing){
                const int alreadyExistingHashFunctions = maxNumHashfunctions - remainingHashFunctions;
                int addedHashFunctions = addHashfunctions(remainingHashFunctions + 1);

                if(addedHashFunctions == 0){
                    keepGoing = false;
                    break;
                    //throw std::runtime_error("Unable to construct a single gpu hashtable. Abort.");
                }else{
                    // safety for memory
                    addedHashFunctions -= 1;
                    gpuHashTables.pop_back();

                    if(addedHashFunctions == 0){
                        keepGoing = false;
                        break;
                    }
                }

                ProgressThread<read_number> progressThread(numReads, showProgress, updateShowProgressInterval);

                std::cout << "Constructing maps: ";
                for(int i = 0; i < addedHashFunctions; i++){
                    std::cout << (alreadyExistingHashFunctions + i) << ' ';
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

                    gpuReadStorage.gatherSequenceDataToGpuBufferAsync(
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
                
                    gpuReadStorage.gatherSequenceLengthsToGpuBufferAsync(
                        d_lengths,
                        deviceId,
                        d_indices,
                        curBatchsize,
                        stream
                    );

                    insert(
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

                    progressThread.addProgress(curBatchsize);
                }

                std::cerr << "Compacting\n";
                finalize();

                progressThread.finished();

                remainingHashFunctions -= addedHashFunctions;
            }

            const int numberOfAvailableHashFunctions = maxNumHashfunctions - remainingHashFunctions;

            h_currentHashFunctionNumbers.resize(numberOfAvailableHashFunctions);
            std::copy(usedHashFunctionNumbers.begin(), usedHashFunctionNumbers.end(), h_currentHashFunctionNumbers.begin());

            return numberOfAvailableHashFunctions; 
        }

#if 0
        void constructFromReadStorage(
            const RuntimeOptions &runtimeOptions,
            std::uint64_t nReads,
            const DistributedReadStorage& gpuReadStorage,
            int upperBoundSequenceLength,
            int firstHashFunc,
            int numHashFuncs
        ){
            assert(firstHashFunc + numHashFuncs <= int(gpuHashTables.size()));

            DeviceSwitcher ds(deviceId);

            constexpr read_number parallelReads = 1000000;
            const read_number numReads = nReads;
            const int numIters = SDIV(numReads, parallelReads);
            const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(upperBoundSequenceLength);

            const int numThreads = runtimeOptions.threads;

            ThreadPool::ParallelForHandle pforHandle;

            std::cout << "Constructing maps: ";
            for(int i = 0; i < numHashFuncs; i++){
                std::cout << (firstHashFunc + i) << ' ';
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

            CudaStream stream{};

            auto sequencehandle = gpuReadStorage.makeGatherHandleSequences();

            for (int iter = 0; iter < numIters; iter++){
                read_number readIdBegin = iter * parallelReads;
                read_number readIdEnd = std::min((iter + 1) * parallelReads, numReads);

                const std::size_t curBatchsize = readIdEnd - readIdBegin;

                std::iota(h_indices.get(), h_indices.get() + curBatchsize, readIdBegin);

                cudaMemcpyAsync(d_indices, h_indices, sizeof(read_number) * curBatchsize, H2D, stream); CUERR;

                gpuReadStorage.gatherSequenceDataToGpuBufferAsync(
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
            
                gpuReadStorage.gatherSequenceLengthsToGpuBufferAsync(
                    d_lengths,
                    deviceId,
                    d_indices,
                    curBatchsize,
                    stream
                );

                insert(
                    d_sequenceData,
                    curBatchsize,
                    d_lengths,
                    encodedSequencePitchInInts,
                    d_indices,
                    firstHashFunc,
                    numHashFuncs,
                    stream
                );

                cudaStreamSynchronize(stream); CUERR;

                progressThread.addProgress(curBatchsize);
            }

            progressThread.finished();

            std::cerr << "Compacting\n";
            finalize();
        }

#endif 
        int addHashfunctions(int numExtraFunctions){
            
            DeviceSwitcher ds(deviceId);

            int added = 0;
            int cur = gpuHashTables.size();

            assert(!(numExtraFunctions + cur > 64));

            for(int i = 0; i < numExtraFunctions; i++){
                auto ptr = std::make_unique<GpuTable>(std::size_t(maxNumKeys / getLoad()),
                    getLoad(),
                    resultsPerMapThreshold);

                auto status = ptr->pop_status((cudaStream_t)0);
                cudaDeviceSynchronize(); CUERR;
                if(status.has_any_errors()){
                    std::cerr << "observed error when initialiting hash function " << (gpuHashTables.size() + 1) << " : " << i << ", " << status << "\n";
                    break;
                }else{

                    assert(!status.has_any_errors()); 
                    //TODO errorhandling

                    gpuHashTables.emplace_back(std::move(ptr));

                    added++;
                }
            }

            return added;
        }

        void insert(
            const unsigned int* d_sequenceData2Bit,
            int numSequences,
            const int* d_sequenceLengths,
            std::size_t encodedSequencePitchInInts,
            const read_number* d_readIds,
            int firstHashfunction,
            int numHashfunctions,
            const int* h_hashFunctionNumbers,
            cudaStream_t stream
        ){
            assert(firstHashfunction + numHashfunctions <= int(gpuHashTables.size()));

            DeviceSwitcher ds(deviceId);

            const std::size_t signaturesRowPitchElements = numHashfunctions;

            helpers::SimpleAllocationDevice<std::uint64_t> d_sig(numHashfunctions * numSequences);
            helpers::SimpleAllocationDevice<std::uint64_t> d_sig_trans(numHashfunctions * numSequences);
            helpers::SimpleAllocationDevice<int> d_hashFunctionNumbers(numHashfunctions);
            
            cudaMemcpyAsync(
                d_hashFunctionNumbers, 
                h_hashFunctionNumbers, 
                sizeof(int) * numHashfunctions, 
                H2D, 
                stream
            ); CUERR;

            std::uint64_t* d_signatures = d_sig.data();
            std::uint64_t* d_signatures_transposed = d_sig_trans.data();


            dim3 block(128,1,1);
            dim3 grid(SDIV(numHashfunctions * numSequences, block.x),1,1);

            callMinhashSignaturesKernel(
                d_signatures,
                signaturesRowPitchElements,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                numHashfunctions,
                d_hashFunctionNumbers,
                stream
            ); CUERR;

            helpers::call_transpose_kernel(
                d_signatures_transposed, 
                d_signatures, 
                numSequences, 
                signaturesRowPitchElements, 
                signaturesRowPitchElements,
                stream
            );

            fixKeysForGpuHashTable(d_signatures_transposed, numSequences * numHashfunctions, stream);

            for(int i = 0; i < numHashfunctions; i++){
                gpuHashTables[firstHashfunction + i]->insert(
                    d_signatures_transposed + i * numSequences,
                    d_readIds,
                    numSequences,
                    stream
                );
            }

            cudaStreamSynchronize(stream);

            for(int i = 0; i < numHashfunctions; i++){
                auto status = gpuHashTables[firstHashfunction + i]->pop_status(stream);
                cudaStreamSynchronize(stream);

                if(status.has_any_errors()){
                    std::cerr << "Error table " << (firstHashfunction + i) << " after insertion: " << status << "\n";
                }
            }
        }

        struct QueryHandleStruct{
            int deviceId;
            DeviceBuffer<std::uint64_t> d_sig;
            DeviceBuffer<std::uint64_t> d_sig_trans;
            DeviceBuffer<int> d_numValuesPerSequencePerHash;
            DeviceBuffer<int> d_numValuesPerSequencePerHashExclPSVert;
            DeviceBuffer<int> d_queryOffsetsPerSequencePerHash;
            DeviceBuffer<int> d_cubsum;
            DeviceBuffer<char> d_cub_temp;
            DeviceBuffer<Value> d_values_tmp;
            DeviceBuffer<int> d_end_offsets;
            DeviceBuffer<int> d_flags;
            DeviceBuffer<int> d_hashFunctionNumbers;

            HostBuffer<int> h_totalNumValues;
            HostBuffer<int> h_offsets;

            GpuSegmentedUnique::Handle segmentedUniqueHandle; 

            MemoryUsage getMemoryInfo() const{
                MemoryUsage mem{};

                mem.host += h_totalNumValues.capacityInBytes();
                mem.host += h_offsets.capacityInBytes();

                mem.device[deviceId] += d_sig.capacityInBytes();
                mem.device[deviceId] += d_sig_trans.capacityInBytes();
                mem.device[deviceId] += d_numValuesPerSequencePerHash.capacityInBytes();
                mem.device[deviceId] += d_numValuesPerSequencePerHashExclPSVert.capacityInBytes();
                mem.device[deviceId] += d_queryOffsetsPerSequencePerHash.capacityInBytes();
                mem.device[deviceId] += d_cubsum.capacityInBytes();
                mem.device[deviceId] += d_cub_temp.capacityInBytes();
                mem.device[deviceId] += d_values_tmp.capacityInBytes();
                mem.device[deviceId] += d_end_offsets.capacityInBytes();
                mem.device[deviceId] += d_flags.capacityInBytes();
                mem.device[deviceId] += d_hashFunctionNumbers.capacityInBytes();

                return mem;
            }
        };

        using QueryHandle = std::shared_ptr<QueryHandleStruct>;

        static QueryHandle makeQueryHandle(){
            auto ptr = std::make_shared<QueryHandleStruct>();
            cudaGetDevice(&ptr->deviceId); CUERR;
            ptr->segmentedUniqueHandle = GpuSegmentedUnique::makeHandle(); 
            return ptr;
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
        ) const{
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
        ) const{
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


        void query_impl(
            QueryHandle& queryHandle,
            const read_number* d_readIds,
            const unsigned int* d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int /*deviceId*/, 
            cudaStream_t stream,
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets //numSequences + 1
        ) const {

            DeviceSwitcher ds(deviceId);

            QueryHandleStruct& handle = *queryHandle;

            const int numHashfunctions = gpuHashTables.size();
            handle.d_hashFunctionNumbers.resize(numHashfunctions);
            cudaMemcpyAsync(
                handle.d_hashFunctionNumbers.data(), 
                h_currentHashFunctionNumbers.data(), 
                sizeof(int) * numHashfunctions, 
                H2D, 
                stream
            ); CUERR;

            const std::size_t signaturesRowPitchElements = numHashfunctions;

            std::size_t cubtempbytes = 0;

            cub::DeviceScan::InclusiveSum(
                nullptr,
                cubtempbytes,
                (int*)nullptr, 
                (int*)nullptr, 
                numSequences,
                stream
            );

            std::size_t cubtempbytes2 = 0;
            cub::DeviceReduce::Max(
                nullptr, 
                cubtempbytes2, 
                (int*)nullptr, 
                (int*)nullptr, 
                numSequences, 
                stream
            );

            cubtempbytes = std::max(cubtempbytes, cubtempbytes2);


            handle.d_sig.resize(numHashfunctions * numSequences);
            handle.d_sig_trans.resize(numHashfunctions * numSequences);
            handle.d_numValuesPerSequencePerHash.resize(numSequences * numHashfunctions);
            handle.d_numValuesPerSequencePerHashExclPSVert.resize(numSequences * numHashfunctions);
            handle.d_queryOffsetsPerSequencePerHash.resize(numSequences * numHashfunctions);
            handle.d_cubsum.resize(1 + numSequences);
            handle.d_cub_temp.resize(cubtempbytes);
            handle.h_totalNumValues.resize(1);
            handle.h_offsets.resize(1 + numSequences);

            std::uint64_t* d_signatures = handle.d_sig.data();
            std::uint64_t* d_signatures_transposed = handle.d_sig_trans.data();
            void* d_cubTemp = handle.d_cub_temp.data();

            int* d_numValuesPerSequencePerHash = handle.d_numValuesPerSequencePerHash.data();
            int* d_numValuesPerSequencePerHashExclPSVert = handle.d_numValuesPerSequencePerHashExclPSVert.data();
            int* d_queryOffsetsPerSequencePerHash = handle.d_queryOffsetsPerSequencePerHash.data();

            dim3 block(128,1,1);
            dim3 grid(SDIV(numHashfunctions * numSequences, block.x),1,1);

            callMinhashSignaturesKernel(
                d_signatures,
                signaturesRowPitchElements,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                numHashfunctions,
                handle.d_hashFunctionNumbers.data(),
                stream
            ); CUERR;

            // cudaStreamSynchronize(stream); CUERR; //DEBUG
            // for(auto h : handle.d_sig){
            //     std::cerr << h << " ";
            // }
            // std::cerr << "\n";

            helpers::call_transpose_kernel(
                d_signatures_transposed, 
                d_signatures, 
                numSequences, 
                signaturesRowPitchElements, 
                signaturesRowPitchElements,
                stream
            );

            fixKeysForGpuHashTable(d_signatures_transposed, numSequences * numHashfunctions, stream);

            //determine number of values per hashfunction per sequence
            for(int i = 0; i < numHashfunctions; i++){
                gpuHashTables[i]->numValuesPerKeyCompact(
                    d_signatures_transposed + i * numSequences,
                    numSequences,
                    d_numValuesPerSequencePerHash + i * numSequences,
                    stream
                );
            }

            //cudaMemsetAsync(d_numValuesPerSequence, 0, sizeof(int) * numSequences, stream); CUERR;

            // accumulate number of values per sequence in d_numValuesPerSequence
            // calculate vertical exclusive prefix sum
            helpers::lambda_kernel<<<1024, 256, 0, stream>>>(
                [=] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    for(int i = tid; i < numSequences; i += stride){
                        d_numValuesPerSequencePerHashExclPSVert[0 * numSequences + i] = 0;
                    }

                    for(int i = tid; i < numSequences; i += stride){
                        int vertPS = 0;
                        for(int k = 0; k < numHashfunctions; k++){
                            const int num = d_numValuesPerSequencePerHash[k * numSequences + i];

                            vertPS += num;
                            if(k < numHashfunctions - 1){
                                d_numValuesPerSequencePerHashExclPSVert[(k+1) * numSequences + i] = vertPS;
                            }else{
                                d_numValuesPerSequence[i] = vertPS;
                            }
                        }
                    }
                }
            );

            //debug
            // cudaStreamSynchronize(stream); CUERR;

            // std::cerr << "new\n";
            // for(int i = 0; i < numHashfunctions; i++){
            //     std::cerr << d_numValuesPerSequencePerHash[i] << " ";
            // }
            // std::cerr << "\n";
            // for(int i = 0; i < numHashfunctions; i++){
            //     std::cerr << d_numValuesPerSequencePerHashExclPSVert[i] << " ";
            // }
            // std::cerr << "\n";
            // for(int i = 0; i < numSequences; i++){
            //     std::cerr << d_numValuesPerSequence[i] << " ";
            // }
            // std::cerr << "\n";

            //calculate global offsets for each sequence in output array
            cudaMemsetAsync(d_offsets, 0, sizeof(int), stream); CUERR;

            cub::DeviceScan::InclusiveSum(
                d_cubTemp,
                cubtempbytes,
                d_numValuesPerSequence,
                d_offsets + 1,
                numSequences,
                stream
            );

            // compute destination offsets for each hashtable such that values of different tables 
            // for the same sequence are stored contiguous in the result array

            helpers::lambda_kernel<<<1024, 256, 0, stream>>>(
                [=] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    for(int i = tid; i < numSequences; i += stride){
                        
                        const int base = d_offsets[i];

                        //k == 0 is a copy from d_offsets
                        d_queryOffsetsPerSequencePerHash[0 * numSequences + i] = base;

                        for(int k = 1; k < numHashfunctions; k++){
                            d_queryOffsetsPerSequencePerHash[k * numSequences + i] = base + d_numValuesPerSequencePerHashExclPSVert[k * numSequences + i];
                        }
                    }
                }
            );

            int totalNumValues = 0;
            cudaMemcpyAsync(&totalNumValues, d_offsets + numSequences, sizeof(int), D2H, stream); CUERR;

            cudaStreamSynchronize(stream); CUERR;

            if(totalNumValues == 0){
                return;
            }

            cub::DeviceReduce::Max(
                d_cubTemp, 
                cubtempbytes, 
                d_numValuesPerSequence, 
                handle.d_cubsum.data(), 
                numSequences, 
                stream
            );
            int sizeOfLargestSegment = 0;
            cudaMemcpyAsync(&sizeOfLargestSegment, handle.d_cubsum.data(), sizeof(int), D2H, stream); CUERR;

            handle.d_sig.resize(SDIV( sizeof(read_number) * totalNumValues, sizeof(std::uint64_t)));
            read_number* d_values_tmp = (read_number*) handle.d_sig.data();
            //handle.d_values_tmp.resize(totalNumValues);
            handle.d_end_offsets.resize(numSequences);
            handle.d_flags.resize(totalNumValues);

            //results will be in Current() buffer
            cub::DoubleBuffer<read_number> d_values_dblbuf(d_values, d_values_tmp);

            int* d_end_offsets = handle.d_end_offsets.data();
            int* d_flags = handle.d_flags.data();

            cudaMemcpyAsync(d_end_offsets, d_offsets + 1, sizeof(int) * numSequences, D2D, stream); CUERR;

            //retrieve values

            for(int i = 0; i < numHashfunctions; i++){
                gpuHashTables[i]->retrieveCompact(
                    d_signatures_transposed + i * numSequences,
                    d_queryOffsetsPerSequencePerHash  + i * numSequences,
                    d_numValuesPerSequencePerHash + i * numSequences,
                    numSequences,
                    d_values_dblbuf.Current(),
                    stream
                );
            }

            // all values for the same key are stored in consecutive locations in d_values_tmp.
            // now, make value ranges unique

            GpuSegmentedUnique::unique(
                handle.segmentedUniqueHandle,
                d_values_dblbuf.Current(), //input values
                totalNumValues,
                d_values_dblbuf.Alternate(), //output values
                d_numValuesPerSequence, //output segment sizes
                numSequences,
                sizeOfLargestSegment,
                d_offsets, //device accessible
                d_end_offsets, //device accessible
                0,
                sizeof(read_number) * 8,
                stream
            );

            if(d_readIds != nullptr){

                // State: d_values contains unique values per sequence from all tables. num unique values per sequence are computed in d_numValuesPerSequence
                // Segment of values for sequence i begins at d_offsets[i]
                // Remove d_readIds[i] from segment i, if present. Operation is performed inplace

                callFindAndRemoveFromSegmentKernel<read_number,128,4>(
                    d_readIds,
                    d_values_dblbuf.Alternate(),
                    numSequences,
                    d_numValuesPerSequence,
                    d_offsets,
                    stream
                );

            }

            //copy values to compact array

            //repurpose
            int* d_newOffsets = handle.d_cubsum.data();

            cudaMemsetAsync(d_newOffsets, 0, sizeof(int), stream); CUERR;

            cub::DeviceScan::InclusiveSum(
                d_cubTemp,
                cubtempbytes,
                d_numValuesPerSequence,
                d_newOffsets + 1,
                numSequences,
                stream
            );

            helpers::lambda_kernel<<<numSequences, 128, 0, stream>>>(
                [
                    d_values_in = d_values_dblbuf.Alternate(),
                    d_values_out = d_values_dblbuf.Current(),
                    numSequences,
                    d_numValuesPerSequence,
                    d_offsets,
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

            cudaMemcpyAsync(d_offsets, d_newOffsets, sizeof(int) * (numSequences+1), D2D, stream); CUERR;

        }

        void compact(){
            DeviceSwitcher ds(deviceId);

            for(auto& table : gpuHashTables){
                table->compact();
            }
        }

        void finalize(){
            compact();
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage mem{};

            for(const auto& table : gpuHashTables){
                mem += table->getMemoryInfo();
            }

            return mem;
        }

        constexpr int getKmerSize() const noexcept{
            return kmerSize;
        }

        std::uint64_t getKmerMask() const{
            constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

            return std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - getKmerSize()) * 2);
        }

        constexpr float getLoad() const noexcept{
            return 0.8f;
        }

        constexpr int getNumResultsPerMapThreshold() const noexcept{
            return resultsPerMapThreshold;
        }
        
        int getNumberOfMaps() const noexcept{
            return gpuHashTables.size();
        }

        void destroy(){
            DeviceSwitcher sd(getDeviceId());
            gpuHashTables.clear();
        }

        constexpr int getDeviceId() const noexcept{
            return deviceId;
        }

        int deviceId{};
        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        HostBuffer<int> h_currentHashFunctionNumbers{};
        std::vector<std::unique_ptr<GpuTable>> gpuHashTables{};
    };


}
}




#endif