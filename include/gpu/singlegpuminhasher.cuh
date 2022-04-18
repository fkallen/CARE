#ifdef CARE_HAS_WARPCORE

#ifndef CARE_SINGLE_GPU_MINHASHER_CUH
#define CARE_SINGLE_GPU_MINHASHER_CUH

#include <config.hpp>

#include <gpu/gpureadstorage.cuh>
#include <gpu/cuda_unique.cuh>
#include <cpuhashtable.hpp>
#include <gpu/gpuhashtable.cuh>
#include <gpu/kernels.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/cubwrappers.cuh>
#include <gpu/gpusequencehasher.cuh>

#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>

#include <sequencehelpers.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>
#include <sharedmutex.hpp>

#include <cub/cub.cuh>

#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <mutex>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <gpu/rmm_utilities.cuh>

namespace care{
namespace gpu{

    class MultiGpuMinhasher; //forward declaration

    class SingleGpuMinhasher : public GpuMinhasher{
        friend class MultiGpuMinhasher;
    private:
        using GpuTable = GpuHashtable<Key, read_number>;

        using DeviceSwitcher = cub::SwitchDevice;

        template<class T>
        using HostBuffer = helpers::SimpleAllocationPinnedHost<T, 5>;
        struct QueryData{
            enum class Stage{
                None,
                NumValues,
                Retrieve
            };

            int deviceId{};
            Stage previousStage = Stage::None;
            int* d_numValuesPerSequence{};
            rmm::device_uvector<char> d_singlepersistentbuffer;

            QueryData(rmm::mr::device_memory_resource* mr)
            : d_singlepersistentbuffer(0, cudaStreamPerThread, mr)
            {
                CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));
            }

            MemoryUsage getMemoryInfo() const{
                MemoryUsage mem{};

                auto handledevice = [&](const auto& buff){
                    using ElementType = typename std::remove_reference<decltype(buff)>::type::value_type;
                    mem.device[deviceId] += buff.size() * sizeof(ElementType);
                };

                handledevice(d_singlepersistentbuffer);

                return mem;
            }
        };

    public:
        using Key = kmer_type;


        SingleGpuMinhasher(int maxNumKeys_, int maxValuesPerKey, int k)
            : maxNumKeys(maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxValuesPerKey)
        {
            CUDACHECK(cudaGetDevice(&deviceId));
        }

        std::unique_ptr<SingleGpuMinhasher> makeCopy(int targetDeviceId) const{
            DeviceSwitcher ds(targetDeviceId);

            auto result = std::make_unique<SingleGpuMinhasher>(0,0,0);
            if(!result) return nullptr;
            
            result->maxNumKeys = maxNumKeys;
            result->kmerSize = kmerSize;
            result->resultsPerMapThreshold = resultsPerMapThreshold;
            result->h_currentHashFunctionNumbers.resize(h_currentHashFunctionNumbers.size());
            std::copy(h_currentHashFunctionNumbers.begin(), h_currentHashFunctionNumbers.end(), result->h_currentHashFunctionNumbers.begin());

            std::size_t requiredTempBytes = 0;
            for(const auto& ptr : gpuHashTables){
                std::size_t bytes = ptr->getMakeCopyTempBytes();
                requiredTempBytes = std::max(requiredTempBytes, bytes);
            }

            thrust::device_vector<char> d_copytemp(requiredTempBytes);

            for(const auto& ptr : gpuHashTables){
                auto newtableptr = ptr->makeCopy(thrust::raw_pointer_cast(d_copytemp.data()), targetDeviceId);
                if(newtableptr){
                    result->gpuHashTables.push_back(std::move(newtableptr));
                }else{
                    cudaGetLastError();
                    return nullptr;
                }
            }

            std::vector<GpuTable::DeviceTableView> views;
            for(const auto& ptr : result->gpuHashTables){
                views.emplace_back(ptr->makeDeviceView());
            }

            result->d_deviceAccessibleTableViews.resize(views.size());
            CUDACHECK(cudaMemcpyAsync(
                result->d_deviceAccessibleTableViews.data(),
                views.data(),
                sizeof(GpuTable::DeviceTableView) * views.size(),
                H2D,
                cudaStreamPerThread
            ));

            CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));

            return result;
        }


        int addHashTables(int numAdditionalTables, const int* hashFunctionIds, cudaStream_t stream) override {
            
            DeviceSwitcher ds(deviceId);

            int added = 0;
            int cur = gpuHashTables.size();

            assert(!(numAdditionalTables + cur > 64));
            std::vector<int> tmpNumbers(h_currentHashFunctionNumbers.begin(), h_currentHashFunctionNumbers.end());

            for(int i = 0; i < numAdditionalTables; i++){
                auto ptr = std::make_unique<GpuTable>(std::size_t(maxNumKeys / getLoad()),
                    getLoad(),
                    resultsPerMapThreshold,
                    stream
                );

                auto status = ptr->pop_status(stream);
                CUDACHECK(cudaStreamSynchronize(stream));
                if(status.has_any_errors()){
                    if(!status.has_out_of_memory()){
                        std::cerr << "observed error when initializing hash function " << (gpuHashTables.size() + 1) << " : " << i << ", " << status << "\n";
                    }
                    break;
                }else{

                    assert(!status.has_any_errors()); 

                    gpuHashTables.emplace_back(std::move(ptr));

                    added++;
                    tmpNumbers.push_back(hashFunctionIds[i]);
                }
            }

            h_currentHashFunctionNumbers.resize(tmpNumbers.size());
            std::copy(tmpNumbers.begin(), tmpNumbers.end(), h_currentHashFunctionNumbers.begin());

            return added;
        }

        void insert(
            const unsigned int* d_sequenceData2Bit,
            int numSequences,
            const int* d_sequenceLengths,
            std::size_t encodedSequencePitchInInts,
            const read_number* d_readIds,
            const read_number* /*h_readIds*/,
            int firstHashfunction,
            int numHashfunctions,
            const int* h_hashFunctionNumbers,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr
        ) override {

            const std::size_t signaturesRowPitchElements = numHashfunctions;

            assert(firstHashfunction + numHashfunctions <= int(gpuHashTables.size()));

            DeviceSwitcher ds(deviceId);

            rmm::device_uvector<int> d_hashFunctionNumbers(numHashfunctions, stream, mr);
            
            CUDACHECK(cudaMemcpyAsync(
                d_hashFunctionNumbers.data(), 
                h_hashFunctionNumbers, 
                sizeof(int) * numHashfunctions, 
                H2D, 
                stream
            ));

            GPUSequenceHasher<kmer_type> hasher;

            auto hashResult = hasher.hash(
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                numHashfunctions,
                d_hashFunctionNumbers.data(),
                stream,
                mr
            );

            rmm::device_uvector<kmer_type> d_signatures_transposed(signaturesRowPitchElements * numSequences, stream, mr);
            helpers::call_transpose_kernel(
                d_signatures_transposed.data(),
                hashResult.d_hashvalues.data(),
                numSequences, 
                signaturesRowPitchElements, 
                signaturesRowPitchElements,
                stream
            );

            fixKeysForGpuHashTable(d_signatures_transposed.data(), numSequences * numHashfunctions, stream);

            for(int i = 0; i < numHashfunctions; i++){
                gpuHashTables[firstHashfunction + i]->insert(
                    d_signatures_transposed.data() + i * numSequences,
                    d_readIds,
                    numSequences,
                    stream
                );
            }
        }

        int checkInsertionErrors(
            int firstHashfunction,
            int numHashfunctions,
            cudaStream_t stream        
        ) override{
            int count = 0;
            for(int i = 0; i < numHashfunctions; i++){
                auto status = gpuHashTables[firstHashfunction + i]->pop_status(stream);
                CUDACHECK(cudaStreamSynchronize(stream));

                if(status.has_any_errors()){
                    count++;
                    std::cerr << "Error table " << (firstHashfunction + i) << " after insertion: " << status << "\n";
                }
            }
            return count;
        }

        MinhasherHandle makeMinhasherHandle() const override {
            auto data = std::make_unique<QueryData>(rmm::mr::get_current_device_resource());
            CUDACHECK(cudaGetDevice(&data->deviceId));

            std::unique_lock<SharedMutex> lock(sharedmutex);
            const int handleid = counter++;
            MinhasherHandle h = constructHandle(handleid);

            tempdataVector.emplace_back(std::move(data));
            return h;
        }

        void destroyHandle(MinhasherHandle& handle) const override{

            std::unique_lock<SharedMutex> lock(sharedmutex);

            const int id = handle.getId();
            assert(id < int(tempdataVector.size()));
            
            {
                cub::SwitchDevice sd(tempdataVector[id]->deviceId);
                tempdataVector[id] = nullptr;
            }
            handle = constructHandle(std::numeric_limits<int>::max());
        }

        void compact(cudaStream_t stream) override {
            DeviceSwitcher ds(deviceId);

            std::size_t required_temp_bytes = 0;

            for(auto& table : gpuHashTables){
                std::size_t temp_bytes2 = 0;
                table->compact(nullptr, temp_bytes2, stream);
                required_temp_bytes = std::max(required_temp_bytes, temp_bytes2);
            }

            std::size_t freeMem, totalMem; 
            CUDACHECK(cudaMemGetInfo(&freeMem, &totalMem));

            void* temp = nullptr;
            if(required_temp_bytes < freeMem){
                CUDACHECK(cudaMalloc(&temp, required_temp_bytes));
            }else{
                CUDACHECK(cudaMallocManaged(&temp, required_temp_bytes));
                int deviceId = 0;
                CUDACHECK(cudaGetDevice(&deviceId));
                CUDACHECK(cudaMemAdvise(temp, required_temp_bytes, cudaMemAdviseSetAccessedBy, deviceId));
            }

            for(auto& table : gpuHashTables){
                table->compact(temp, required_temp_bytes, stream);
            }

            CUDACHECK(cudaFree(temp));
        }

        MemoryUsage getMemoryInfo() const noexcept override{
            MemoryUsage mem{};

            for(const auto& table : gpuHashTables){
                mem += table->getMemoryInfo();
            }

            return mem;
        }

        MemoryUsage getMemoryInfo(const MinhasherHandle& handle) const noexcept override{
            return getQueryDataFromHandle(handle)->getMemoryInfo();
        }

        int getNumResultsPerMapThreshold() const noexcept override{
            return resultsPerMapThreshold;
        }
        
        int getNumberOfMaps() const noexcept override{
            return gpuHashTables.size();
        }

        void destroy(){
            DeviceSwitcher sd(getDeviceId());
            gpuHashTables.clear();
        }

        bool hasGpuTables() const noexcept override {
            return true;
        }

        void determineNumValues(
            MinhasherHandle& queryHandle,
            const unsigned int* d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int* d_numValuesPerSequence,
            int& totalNumValues,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr
        ) const override {

            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            std::size_t persistent_storage_bytes = 0;

            determineNumValues(
                nullptr,            
                persistent_storage_bytes,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                d_numValuesPerSequence,
                totalNumValues,
                stream,
                mr
            );

            queryData->d_singlepersistentbuffer.resize(persistent_storage_bytes, stream);

            determineNumValues(
                queryData->d_singlepersistentbuffer.data(),
                persistent_storage_bytes,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                d_numValuesPerSequence,
                totalNumValues,
                stream,
                mr
            );

            queryData->previousStage = QueryData::Stage::NumValues;
            queryData->d_numValuesPerSequence = d_numValuesPerSequence;
        }

        void retrieveValues(
            MinhasherHandle& queryHandle,
            int numSequences,
            int totalNumValues,
            read_number* d_values,
            const int* d_numValuesPerSequence,
            int* d_offsets, //numSequences + 1
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr
        ) const override {
            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            assert(queryData->previousStage == QueryData::Stage::NumValues);
            //STUPID INTERFACE EXPECTS d_numValuesPerSequence TO CONTAIN THE SAME VALUES AS RETURNED BY determineNumValues. This needs to be refactored. 
            //assert(queryData->d_numValuesPerSequence = d_numValuesPerSequence);

            std::size_t persistent_storage_bytes = queryData->d_singlepersistentbuffer.size();

            retrieveValues(
                queryData->d_singlepersistentbuffer.data(),
                persistent_storage_bytes,
                numSequences,
                totalNumValues,
                d_values,
                d_numValuesPerSequence,
                d_offsets, //numSequences + 1
                stream,
                mr
            );

            queryData->previousStage = QueryData::Stage::Retrieve;

        }


        //to make the following to functions private, their lambda kernels have to be replaced by normal kernels

        void determineNumValues(
            void* persistent_storage,            
            std::size_t& persistent_storage_bytes,
            const unsigned int* d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int* d_numValuesPerSequence,
            int& totalNumValues,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr
        ) const {

            const int numHashfunctions = gpuHashTables.size();
            const std::size_t signaturesRowPitchElements = numHashfunctions;

            void* persistent_allocations[3]{};
            std::size_t persistent_allocation_sizes[3]{};

            persistent_allocation_sizes[0] = sizeof(kmer_type) * numHashfunctions * numSequences; // d_sig_trans
            persistent_allocation_sizes[1] = sizeof(int) * numSequences * numHashfunctions; // d_numValuesPerSequencePerHash
            persistent_allocation_sizes[2] = sizeof(int) * numSequences * numHashfunctions; // d_numValuesPerSequencePerHashExclPSVert

            CUDACHECK(cub::AliasTemporaries(
                persistent_storage,
                persistent_storage_bytes,
                persistent_allocations,
                persistent_allocation_sizes
            ));

            if(persistent_storage == nullptr){
                return;
            }

            kmer_type* const d_signatures_transposed = static_cast<kmer_type*>(persistent_allocations[0]);
            int* const d_numValuesPerSequencePerHash = static_cast<int*>(persistent_allocations[1]);
            int* const d_numValuesPerSequencePerHashExclPSVert = static_cast<int*>(persistent_allocations[2]);

            rmm::device_uvector<int> d_hashFunctionNumbers(numHashfunctions, stream, mr);

            DeviceSwitcher ds(deviceId);

            CUDACHECK(cudaMemcpyAsync(
                d_hashFunctionNumbers.data(),
                h_currentHashFunctionNumbers.data(), 
                sizeof(int) * numHashfunctions, 
                H2D, 
                stream
            ));           

            GPUSequenceHasher<kmer_type> hasher;

            auto hashResult = hasher.hash(
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                getNumberOfMaps(),
                d_hashFunctionNumbers.data(),
                stream,
                mr
            );

            helpers::call_transpose_kernel(
                d_signatures_transposed, 
                hashResult.d_hashvalues.data(),
                numSequences, 
                signaturesRowPitchElements, 
                signaturesRowPitchElements,
                stream
            );

            fixKeysForGpuHashTable(d_signatures_transposed, numSequences * numHashfunctions, stream);

            //determine number of values per hashfunction per sequence
            #if 1

            {

            const int signaturesPitchInElements = numSequences;
            const int numValuesPerKeyPitchInElements = numSequences;
            constexpr int cgsize = GpuTable::DeviceTableView::cg_size();

            dim3 block(256, 1, 1);
            const int numBlocksPerTable = SDIV(numSequences, (block.x / cgsize));
            dim3 grid(numBlocksPerTable, std::min(65535, numHashfunctions), 1);

            gpuhashtablekernels::numValuesPerKeyCompactMultiTableKernel<<<grid, block, 0, stream>>>(
                d_deviceAccessibleTableViews.data(),
                numHashfunctions,
                resultsPerMapThreshold,
                d_signatures_transposed,
                signaturesPitchInElements,
                numSequences,
                d_numValuesPerSequencePerHash,
                numValuesPerKeyPitchInElements
            );

            }
            #else

            
            for(int i = 0; i < numHashfunctions; i++){
                gpuHashTables[i]->numValuesPerKeyCompact(
                    d_signatures_transposed + i * numSequences,
                    numSequences,
                    d_numValuesPerSequencePerHash + i * numSequences,
                    stream
                );
            }

            #endif

            // accumulate number of values per sequence in d_numValuesPerSequence
            // calculate vertical exclusive prefix sum
            helpers::lambda_kernel<<<SDIV(numSequences, 256), 256, 0, stream>>>(
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

            rmm::device_scalar<int> d_totalNumValues(stream, mr);

            CubCallWrapper(mr).cubReduceSum(
                d_numValuesPerSequence, 
                d_totalNumValues.data(), 
                numSequences, 
                stream
            );

            CUDACHECK(cudaMemcpyAsync(
                &totalNumValues,
                d_totalNumValues.data(),
                sizeof(int),
                D2H,
                stream
            ));
        }

        void retrieveValues(
            void* persistentbufferFromNumValues,            
            std::size_t persistent_storage_bytes,
            int numSequences,
            int totalNumValues,
            read_number* d_values,
            const int* d_numValuesPerSequence,
            int* d_offsets, //numSequences + 1
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr
        ) const {
            if(totalNumValues == 0){
                CUDACHECK(cudaMemsetAsync(d_offsets, 0, sizeof(int) * (numSequences + 1), stream));
                return;
            }

            assert(persistentbufferFromNumValues != nullptr);

            const int numHashfunctions = gpuHashTables.size();

            void* persistent_allocations[3]{};
            std::size_t persistent_allocation_sizes[3]{};

            persistent_allocation_sizes[0] = sizeof(kmer_type) * numHashfunctions * numSequences; // d_sig_trans
            persistent_allocation_sizes[1] = sizeof(int) * numSequences * numHashfunctions; // d_numValuesPerSequencePerHash
            persistent_allocation_sizes[2] = sizeof(int) * numSequences * numHashfunctions; // d_numValuesPerSequencePerHashExclPSVert

            CUDACHECK(cub::AliasTemporaries(
                persistentbufferFromNumValues,
                persistent_storage_bytes,
                persistent_allocations,
                persistent_allocation_sizes
            ));

            DeviceSwitcher ds(deviceId);

            kmer_type* const d_signatures_transposed = static_cast<kmer_type*>(persistent_allocations[0]);
            int* const d_numValuesPerSequencePerHash = static_cast<int*>(persistent_allocations[1]);
            int* const d_numValuesPerSequencePerHashExclPSVert = static_cast<int*>(persistent_allocations[2]);

            rmm::device_uvector<int> d_queryOffsetsPerSequencePerHash(numSequences * numHashfunctions, stream, mr);


            //calculate global offsets for each sequence in output array
            CUDACHECK(cudaMemsetAsync(d_offsets, 0, sizeof(int), stream));

            CubCallWrapper(mr).cubInclusiveSum(
                d_numValuesPerSequence,
                d_offsets + 1,
                numSequences,
                stream
            );

            // compute destination offsets for each hashtable such that values of different tables 
            // for the same sequence are stored contiguous in the result array

            helpers::lambda_kernel<<<SDIV(numSequences, 256), 256, 0, stream>>>(
                [
                    d_queryOffsetsPerSequencePerHash = d_queryOffsetsPerSequencePerHash.data(),
                    d_numValuesPerSequencePerHashExclPSVert,
                    numSequences,
                    numHashfunctions,
                    d_offsets
                ] __device__ (){
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

            //retrieve values

            #if 1
            {
            const int signaturesPitchInElements = numSequences;
            const int numValuesPerKeyPitchInElements = numSequences;
            const int beginOffsetsPitchInElements = numSequences;
            constexpr int cgsize = GpuTable::DeviceTableView::cg_size();

            dim3 block(256, 1, 1);
            const int numBlocksPerTable = SDIV(numSequences, (block.x / cgsize));
            dim3 grid(numBlocksPerTable, std::min(65535, numHashfunctions), 1);

            gpuhashtablekernels::retrieveCompactKernel<<<grid, block, 0, stream>>>(
                d_deviceAccessibleTableViews.data(),
                numHashfunctions,
                d_signatures_transposed,
                signaturesPitchInElements,
                d_queryOffsetsPerSequencePerHash.data(),
                beginOffsetsPitchInElements,
                d_numValuesPerSequencePerHash,
                numValuesPerKeyPitchInElements,
                resultsPerMapThreshold,
                numSequences,
                d_values
            );
            }
            #else

            for(int i = 0; i < numHashfunctions; i++){
                gpuHashTables[i]->retrieveCompact(
                    d_signatures_transposed + i * numSequences,
                    d_queryOffsetsPerSequencePerHash.data()  + i * numSequences,
                    d_numValuesPerSequencePerHash + i * numSequences,
                    numSequences,
                    d_values,
                    stream
                );
            }

            #endif
        }

        int getKmerSize() const noexcept override{
            return kmerSize;
        }


        constexpr int getDeviceId() const noexcept{
            return deviceId;
        }

        void setThreadPool(ThreadPool* /*tp*/) override {}

        void setHostMemoryLimitForConstruction(std::size_t /*bytes*/) override{

        }

        void setDeviceMemoryLimitsForConstruction(const std::vector<std::size_t>&) override {

        }

        void constructionIsFinished(cudaStream_t stream) override {
            auto numberOfAvailableHashFunctions = h_currentHashFunctionNumbers.size();
            std::vector<GpuTable::DeviceTableView> views;
            for(const auto& ptr : gpuHashTables){
                views.emplace_back(ptr->makeDeviceView());
            }

            d_deviceAccessibleTableViews.resize(numberOfAvailableHashFunctions);
            CUDACHECK(cudaMemcpyAsync(
                d_deviceAccessibleTableViews.data(),
                views.data(),
                sizeof(GpuTable::DeviceTableView) * numberOfAvailableHashFunctions,
                H2D,
                stream
            ));

            CUDACHECK(cudaStreamSynchronize(stream));
        }

        void writeToStream(std::ostream& /*os*/) const override{
            std::cerr << "SingleGpuMinhasher::writeToStream not supported\n";
        }

        int loadFromStream(std::ifstream& /*is*/, int /*numMapsUpperLimit*/) override{
            std::cerr << "SingleGpuMinhasher::loadFromStream not supported\n";
            return 0;
        } 

        bool canWriteToStream() const noexcept override { return false; };
        bool canLoadFromStream() const noexcept override { return false; };

private:

        void finalize(cudaStream_t stream = 0){
            compact(stream);
        }

        std::uint64_t getKmerMask() const{
            constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

            return std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - getKmerSize()) * 2);
        }

        constexpr float getLoad() const noexcept{
            return 0.8f;
        }

        QueryData* getQueryDataFromHandle(const MinhasherHandle& queryHandle) const{
            std::shared_lock<SharedMutex> lock(sharedmutex);

            return tempdataVector[queryHandle.getId()].get();
        }

        mutable int counter = 0;
        mutable SharedMutex sharedmutex{};
        //mutable std::shared_mutex sharedmutex{};

        int deviceId{};
        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        HostBuffer<int> h_currentHashFunctionNumbers{};
        std::vector<std::unique_ptr<GpuTable>> gpuHashTables{};
        helpers::SimpleAllocationDevice<GpuTable::DeviceTableView, 0> d_deviceAccessibleTableViews{};
        mutable std::vector<std::unique_ptr<QueryData>> tempdataVector{};
    };


}
}




#endif

#endif //#ifdef CARE_HAS_WARPCORE