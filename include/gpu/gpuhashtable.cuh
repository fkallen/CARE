#ifndef CARE_GPUHASHTABLE_CUH
#define CARE_GPUHASHTABLE_CUH

#include <warpcore.cuh>
#include <hpc_helpers.cuh>

#include <memorymanagement.hpp>

#include <memory>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cassert>
#include <future>

#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

namespace care{
namespace gpu{

    namespace gpuhashtablekernels{

        template<class T, class U>
        __global__
        void assignmentKernel(T* __restrict__ output, const U* __restrict__ input, int N){
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const int stride = blockDim.x * gridDim.x;

            for(int i = tid; i < N; i += stride){
                output[i] = input[i];
            }
        }

        template<class T, class IsValidKey>
        __global__
        void fixTableKeysKernel(T* __restrict__ keys, int numKeys, IsValidKey isValid){
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const int stride = blockDim.x * gridDim.x;

            for(int i = tid; i < numKeys; i += stride){
                T key = keys[i];
                int changed = 0;
                while(!isValid(key)){
                    key++;
                    changed = 1;
                }
                if(changed == 1){
                    keys[i] = key;
                }
            }
        }

        // template<class CompactTableView, class Key, class Value, class Offset>
        // __global__
        // void retrieveCompactKernel(
        //     CompactTableView table,
        //     const Key* __restrict__ querykeys,
        //     const int numKeys,
        //     Value* __restrict__ outValues,
        //     int valueOffset, // values for key i begin at valueOffset * i
        //     Offset* __restrict__ numValuesPerKey
        // ){
        //     const int tid = threadIdx.x + blockDim.x * blockIdx.x;
        //     const int stride = blockDim.x * gridDim.x;

        //     constexpr int tilesize = CompactTableView::cg_size();

        //     assert(stride % tilesize == 0);

        //     auto tile = cg::tiled_partition<tilesize>(cg::this_thread_block());
        //     const int tileId = tid / tilesize;
        //     const int numTiles = stride / tilesize;

        //     for(int k = tileId; k < numKeys; k += numTiles){
        //         const Key key = querykeys[k];

        //         const int num = table.retrieve(tile, key, outValues + size_t(k) * valueOffset);
        //         if(tile.thread_rank() == 0){
        //             numValuesPerKey[k] = num;
        //         }    
        //     }
        // }

        template<class CompactTableView, class Key, class Value, class Offset>
        __global__
        void retrieveCompactKernel(
            CompactTableView table,
            const Key* __restrict__ querykeys,
            const Offset* __restrict__ beginOffsets,
            const int* __restrict__ numValuesPerKey,
            const int maxValuesPerKey,
            const int numKeys,
            Value* __restrict__ outValues
        ){
            const int tid = threadIdx.x + blockDim.x * blockIdx.x;
            const int stride = blockDim.x * gridDim.x;

            constexpr int tilesize = CompactTableView::cg_size();

            assert(stride % tilesize == 0);

            auto tile = cg::tiled_partition<tilesize>(cg::this_thread_block());
            const int tileId = tid / tilesize;
            const int numTiles = stride / tilesize;

            for(int k = tileId; k < numKeys; k += numTiles){
                const Key key = querykeys[k];
                const auto beginOffset = beginOffsets[k];
                const int num = numValuesPerKey[k];

                if(num != 0){
                    table.retrieve(tile, key, outValues + beginOffset);
                }
            }
        }


        

        template<class CompactTableView, class Key, class Offset>
        __global__
        void numValuePerKeyCompactKernel(
            const CompactTableView table,
            int maxValuesPerKey,
            const Key* const __restrict__ querykeys,
            const int numKeys,
            Offset* const __restrict__ numValuesPerKey
        ){
            const int tid = threadIdx.x + blockDim.x * blockIdx.x;
            const int stride = blockDim.x * gridDim.x;

            constexpr int tilesize = CompactTableView::cg_size();

            assert(stride % tilesize == 0);

            auto tile = cg::tiled_partition<tilesize>(cg::this_thread_block());
            const int tileId = tid / tilesize;
            const int numTiles = stride / tilesize;

            for(int k = tileId; k < numKeys; k += numTiles){
                const Key key = querykeys[k];

                const int num = table.numValues(tile, key);
                if(tile.thread_rank() == 0){
                    numValuesPerKey[k] = num > maxValuesPerKey ? 0 : num;
                }    
            }
        }
    }



    template<class Key, class Value>
    class GpuHashtable{
    public:
        using MultiValueHashTable =  warpcore::MultiValueHashTable<
                Key,
                Value,
                warpcore::defaults::empty_key<Key>(),
                warpcore::defaults::tombstone_key<Key>(),
                warpcore::defaults::probing_scheme_t<Key, 8>,
                warpcore::defaults::table_storage_t<Key, Value>,
                warpcore::defaults::temp_memory_bytes()>;

        using StatusHandler = warpcore::status_handlers::ReturnStatus;

        using Index = typename MultiValueHashTable::index_type;

        using CompactKeyIndexTable = warpcore::SingleValueHashTable<
                Key,
                int,
                warpcore::defaults::empty_key<Key>(),
                warpcore::defaults::tombstone_key<Key>(),
                warpcore::defaults::probing_scheme_t<Key, 8>,
                warpcore::defaults::table_storage_t<Key, int>,
                warpcore::defaults::temp_memory_bytes()>;

        // constexpr Key empty_key() noexcept{
        //     return MultiValueHashTable::empty_key();
        // }

        // constexpr Key tombstone_key() noexcept{
        //     return MultiValueHashTable::tombstone_key();
        // }


        GpuHashtable(std::size_t pairs_, float load_, std::size_t maxValuesPerKey_)
            : maxPairs(pairs_), load(load_), maxValuesPerKey(maxValuesPerKey_){

            if(maxPairs > std::size_t(std::numeric_limits<int>::max())){
                assert(maxPairs <= std::size_t(std::numeric_limits<int>::max())); //CompactKeyIndexTable uses int
            }

            cudaGetDevice(&deviceId);

            const std::size_t capacity = maxPairs / load;
            gpuMvTable = std::move(
                //use maxValuesPerKey + 1 for hashtable. when querying, remove all values of keys with numValues == (maxValuesPerKey + 1)
                std::make_unique<MultiValueHashTable>(
                    capacity, warpcore::defaults::seed<Key>(), (maxValuesPerKey + 1)
                )
            );

        }

        HOSTDEVICEQUALIFIER
        static constexpr bool isValidKey(Key key){
            return MultiValueHashTable::is_valid_key(key);
        }

        warpcore::Status pop_status(cudaStream_t stream){
            if(isCompact){
                return gpuKeyIndexTable->pop_status(stream);
            }else{
                return gpuMvTable->pop_status(stream);
            }
        }


        void insert(
            const Key* d_keys, 
            const Value* d_values, 
            Index N, 
            cudaStream_t stream, 
            StatusHandler::base_type* d_statusarray = nullptr
        ){
            if(N == 0) return;
            assert(!isCompact);


            assert(d_keys != nullptr);
            assert(d_values != nullptr);
            assert(numKeys + N <= maxPairs);    
            assert(numValues + N <= maxPairs);

            gpuMvTable->insert(
                d_keys,
                d_values,
                N,
                stream,
                warpcore::defaults::probing_length(),
                d_statusarray
            );

            numKeys += N;
            numValues += N;
        }

//DEBUGGING
        void retrieve(
            const Key* d_keys, 
            Index N,
            Index* d_begin_offsets_out,
            Index* d_end_offsets_out,
            Value* d_values,
            cudaStream_t stream,
            StatusHandler::base_type* d_statusarray = nullptr
        ) const {
            Index num_out = 0; //TODO pinned memory?

            retrieve(
                d_keys,
                N,
                d_begin_offsets_out,
                d_end_offsets_out,
                d_values,
                num_out,
                stream,
                warpcore::defaults::probing_length(),
                d_statusarray
            );
        }
//DEBUGGING
        void retrieve(
            const Key* d_keys, 
            Index N,
            Index* d_begin_offsets_out,
            Index* d_end_offsets_out,
            Value* d_values,
            Index& num_out,
            cudaStream_t stream,
            StatusHandler::base_type* d_statusarray = nullptr
        ) const {
            gpuMvTable->retrieve(
                d_keys,
                N,
                d_begin_offsets_out,
                d_end_offsets_out,
                d_values,
                num_out,
                stream,
                warpcore::defaults::probing_length(),
                d_statusarray
            );
        }

        

        
        // template<class Offset>
        // void retrieveCompact(
        //     const Key* d_keys, 
        //     Index N,
        //     Offset* d_numValuesPerKey,
        //     Value* d_values,
        //     cudaStream_t stream,
        //     int valueOffset
        // ) const {
        //     assert(isCompact);

        //     CompactTableView table{*gpuKeyIndexTable, d_compactOffsets.data(), d_compactValues.data()};

        //     gpuhashtablekernels::retrieveCompactKernel<<<1024, 256, 0, stream>>>(
        //         table,
        //         d_keys,
        //         N,
        //         d_values,
        //         valueOffset, // values for key i begin at valueOffset * i
        //         d_numValuesPerKey
        //     );
        // }

        template<class Offset>
        void retrieveCompact(
            const Key* d_keys, 
            const Offset* d_beginOffsets,
            const Offset* d_numValuesPerKey,
            Index N,
            Value* d_values,
            cudaStream_t stream
        ) const {
            assert(isCompact);

            CompactTableView table{*gpuKeyIndexTable, d_compactOffsets.data(), d_compactValues.data()};

            gpuhashtablekernels::retrieveCompactKernel<<<1024, 256, 0, stream>>>(
                table,
                d_keys,
                d_beginOffsets,
                d_numValuesPerKey,
                maxValuesPerKey,
                N,
                d_values
            );
        }


        template<class Offset>
        void numValuesPerKeyCompact(
            const Key* d_keys, 
            Index N,
            Offset* d_numValuesPerKey,
            cudaStream_t stream
        ) const {
            assert(isCompact);

            CompactTableView table{*gpuKeyIndexTable, d_compactOffsets.data(), d_compactValues.data()};

            gpuhashtablekernels::numValuePerKeyCompactKernel<<<1024, 256, 0, stream>>>(
                table,
                maxValuesPerKey,
                d_keys,
                N,
                d_numValuesPerKey
            );
        }

        MemoryUsage getMemoryInfo() const{
            const std::size_t capacity = maxPairs / load;

            MemoryUsage result;
            if(!isCompact){
                //TODO: Get correct numbers directly from table
                result.device[deviceId] = sizeof(Key) * capacity;
                result.device[deviceId] += sizeof(Value) * capacity;
            }else{
                result.device[deviceId] = (sizeof(Key) + sizeof(int)) * (numKeys / load); //singlevalue hashtable
                result.device[deviceId] += sizeof(int) * numKeys; //offsets
                result.device[deviceId] += sizeof(Value) * numValues; //values
            }

            return result;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr bool is_valid_key(const Key key) noexcept{
            return MultiValueHashTable::is_valid_key(key);
        }

        std::size_t getMaxNumPairs() const noexcept{
            return maxPairs;
        }

        std::size_t getNumKeys() const noexcept{
            return numKeys;
        }

        std::size_t getNumValues() const noexcept{
            return numValues;
        }

        std::size_t getNumUniqueKeys(cudaStream_t stream = 0) const noexcept{
            return gpuMvTable->num_keys(stream);
        }

        void compact(
            void* d_temp,
            std::size_t& temp_bytes,
            cudaStream_t stream = 0
        ){
            Index numUniqueKeys = gpuMvTable->num_keys(stream);
            Index numValuesInTable = gpuMvTable->num_values(stream);

            const std::size_t batchsize = 100000;
            const std::size_t iters =  SDIV(numUniqueKeys, batchsize);

            void* temp_allocations[4];
            std::size_t temp_allocation_sizes[4];
            
            temp_allocation_sizes[0] = sizeof(Key) * numUniqueKeys; // h_uniqueKeys
            temp_allocation_sizes[1] = sizeof(Index) * (numUniqueKeys+1); // h_compactOffsetTmp
            temp_allocation_sizes[2] = sizeof(int) * batchsize; // h_ids
            temp_allocation_sizes[3] = sizeof(Value) * numValuesInTable; // h_compactValues
            
            cudaError_t cubstatus = cub::AliasTemporaries(
                d_temp,
                temp_bytes,
                temp_allocations,
                temp_allocation_sizes
            );
            assert(cubstatus == cudaSuccess);

            if(d_temp == nullptr){
                return;
            }

            if(isCompact) return;

            Key* const d_tmp_uniqueKeys = static_cast<Key*>(temp_allocations[0]);
            Index* const d_tmp_compactOffset = static_cast<Index*>(temp_allocations[1]);
            int* const d_tmp_ids = static_cast<int*>(temp_allocations[2]);
            Value* const d_tmp_compactValues = static_cast<Value*>(temp_allocations[3]);
   
            gpuMvTable->retrieve_all_keys(
                d_tmp_uniqueKeys,
                numUniqueKeys,
                stream
            ); CUERR;

            retrieve(
                d_tmp_uniqueKeys,
                numUniqueKeys,
                d_tmp_compactOffset,
                d_tmp_compactOffset + 1,
                d_tmp_compactValues,
                numValuesInTable,
                stream
            );

            //clear table
            gpuMvTable.reset();
            numKeys = 0;
            numValues = 0;

            //construct new table
            gpuKeyIndexTable = std::move(
                std::make_unique<CompactKeyIndexTable>(
                    numUniqueKeys / load
                )
            );

            d_compactOffsets.resize(numUniqueKeys + 1);
            d_compactValues.resize(numValuesInTable);

            //copy offsets to gpu, convert from Index to int
            gpuhashtablekernels::assignmentKernel
            <<<SDIV(numUniqueKeys+1, 256), 256, 0, stream>>>(
                d_compactOffsets.data(), 
                d_tmp_compactOffset, 
                numUniqueKeys+1
            );
            CUERR;
            
            cudaMemcpyAsync(
                d_compactValues, 
                d_tmp_compactValues, 
                d_compactValues.sizeInBytes(), 
                D2D, 
                stream
            ); CUERR;

            std::vector<int> ids(batchsize);

            for(std::size_t i = 0; i < iters; i++){
                const std::size_t begin = i * batchsize;
                const std::size_t end = std::min((i+1) * batchsize, numUniqueKeys);
                const std::size_t num = end - begin;

                std::iota(ids.data(), ids.data() + num, int(begin));
                cudaStreamSynchronize(stream); CUERR;

                cudaMemcpyAsync(d_tmp_ids, ids.data(), sizeof(int) * num, H2D, stream); CUERR;

                gpuKeyIndexTable->insert(
                    d_tmp_uniqueKeys + begin,
                    d_tmp_ids,
                    num,
                    stream,
                    warpcore::defaults::probing_length(),
                    nullptr
                );                
            }

            numKeys = numUniqueKeys;
            numValues = numValuesInTable;
            
            isCompact = true;

            cudaStreamSynchronize(stream); CUERR;
        }

        template<class Offset>
        void compactIntoHostBuffers(Key* out_uniqueKeys, Value* out_values, Offset* out_offsets) const{

            Index numUniqueKeys = gpuMvTable->num_keys();

            helpers::SimpleAllocationPinnedHost<Key, 0> h_uniqueKeys(numUniqueKeys);
            helpers::SimpleAllocationPinnedHost<Index, 0> h_compactOffsetTmp(numUniqueKeys+1);
            helpers::SimpleAllocationDevice<Key, 0> d_uniqueKeys(numUniqueKeys);
            helpers::SimpleAllocationDevice<Index, 0> d_compactOffsetTmp(numUniqueKeys+1);

            gpuMvTable->retrieve_all_keys(
                d_uniqueKeys,
                numUniqueKeys
            ); CUERR;

            Index numRetrievedValues = 0;

            gpuMvTable->retrieve(
                d_uniqueKeys.data(),
                numUniqueKeys,
                d_compactOffsetTmp.data(),
                d_compactOffsetTmp.data() + 1,
                nullptr,
                numRetrievedValues,
                (cudaStream_t)0,
                warpcore::defaults::probing_length()
            );

            helpers::SimpleAllocationPinnedHost<Value, 0> h_compactValues(numRetrievedValues); 
            helpers::SimpleAllocationDevice<Value, 0> d_compactValues(numRetrievedValues); 

            gpuMvTable->retrieve(
                d_uniqueKeys.data(),
                numUniqueKeys,
                d_compactOffsetTmp.data(),
                d_compactOffsetTmp.data() + 1,
                d_compactValues.data(),
                numRetrievedValues,
                (cudaStream_t)0,
                warpcore::defaults::probing_length()
            );

            cudaMemcpyAsync(h_uniqueKeys.data(), d_uniqueKeys.data(), sizeof(Key) * numUniqueKeys, D2H, (cudaStream_t)0); CUERR;
            cudaMemcpyAsync(h_compactOffsetTmp.data(), d_compactOffsetTmp.data(), sizeof(Index) * (1+numUniqueKeys), D2H, (cudaStream_t)0); CUERR;
            cudaMemcpyAsync(h_compactValues.data(), d_compactValues.data(), sizeof(Value) * numRetrievedValues, D2H, (cudaStream_t)0); CUERR;
            cudaStreamSynchronize((cudaStream_t)0); CUERR;
            //cudaDeviceSynchronize(); CUERR;
            std::copy(h_uniqueKeys.begin(), h_uniqueKeys.end(), out_uniqueKeys);
            std::copy(h_compactValues.begin(), h_compactValues.end(), out_values);
            std::copy(h_compactOffsetTmp.begin(), h_compactOffsetTmp.end(), out_offsets);
            //std::cerr << "numUniqueKeys = " << numUniqueKeys << ", numRetrievedValues = " << numRetrievedValues << "\n";

            // CudaEvent event0{cudaEventDisableTiming};
            // CudaEvent event1{cudaEventDisableTiming};
            // CudaEvent event2{cudaEventDisableTiming};
            // CudaStream stream0;
            // CudaStream stream1;

            // h_compactOffsetTmp[0] = 0;
            
            // gpuMvTable->retrieve_all_keys(
            //     d_uniqueKeys,
            //     numUniqueKeys
            // ); CUERR;

            // cudaMemcpyAsync(h_uniqueKeys.data(), d_uniqueKeys.data(), sizeof(Key) * numUniqueKeys, D2H, stream1); CUERR;
            // event0.record(stream1);

            //  Index numRetrievedValues = 0;
            // gpuMvTable->retrieve(
            //     d_uniqueKeys.data(),
            //     numUniqueKeys,
            //     d_compactOffsetTmp.data(),
            //     d_compactOffsetTmp.data() + 1,
            //     nullptr,
            //     numRetrievedValues,
            //     stream0,
            //     warpcore::defaults::probing_length()
            // );

            
            // // gpuMvTable->num_values(
            // //     d_uniqueKeys.data(),
            // //     numUniqueKeys,
            // //     numRetrievedValues,
            // //     d_compactOffsetTmp + 1,
            // //     stream0,
            // //     warpcore::defaults::probing_length()
            // // );

            // // cudaStreamSynchronize(0); CUERR;
            // // std::size_t required_temp_bytes = 0;
            // // cub::DeviceScan::InclusiveSum(
            // //     nullptr,
            // //     required_temp_bytes,
            // //     d_compactOffsetTmp + 1,
            // //     d_compactOffsetTmp + 1,
            // //     numUniqueKeys,
            // //     stream0
            // // );
            // helpers::SimpleAllocationPinnedHost<Value, 0> h_compactValues(numRetrievedValues); 
            // helpers::SimpleAllocationDevice<Value, 0> d_compactValues(numRetrievedValues); 
            // //helpers::SimpleAllocationDevice<char, 0> cubtemp(required_temp_bytes);

            // // cub::DeviceScan::InclusiveSum(
            // //     cubtemp,
            // //     required_temp_bytes,
            // //     d_compactOffsetTmp + 1,
            // //     d_compactOffsetTmp + 1,
            // //     numUniqueKeys,
            // //     stream0
            // // );
            // event1.record(stream0);
            // stream1.waitEvent(event1, 0);
            // cudaMemcpyAsync(h_compactOffsetTmp.data(), d_compactOffsetTmp.data(), sizeof(Offset) * (1+numUniqueKeys), D2H, stream1); CUERR;
            // event1.record(stream1);

            // // gpuMvTable->retrieve(
            // //     d_uniqueKeys.data(),
            // //     numUniqueKeys,
            // //     d_compactOffsetTmp.data(),
            // //     d_compactValues.data(),
            // //     stream0,
            // //     warpcore::defaults::probing_length()
            // // );

            // gpuMvTable->retrieve(
            //     d_uniqueKeys.data(),
            //     numUniqueKeys,
            //     d_compactOffsetTmp.data(),
            //     d_compactOffsetTmp.data() + 1,
            //     d_compactValues.data(),
            //     numRetrievedValues,
            //     stream0,
            //     warpcore::defaults::probing_length()
            // );

            // cudaMemcpyAsync(h_compactValues.data(), d_compactValues.data(), sizeof(Value) * numRetrievedValues, D2H, stream0); CUERR;

            // event2.record(stream0);

            // // retrieve(
            // //     h_uniqueKeys.data(),
            // //     numUniqueKeys,
            // //     h_compactOffsetTmp.data(),
            // //     h_compactOffsetTmp.data() + 1,
            // //     h_compactValues.data(),
            // //     numRetrievedValues,
            // //     (cudaStream_t)0
            // // );
            // auto copyToOutput = [](auto event, const auto src, auto dst){
            //     event->synchronize(); CUERR;
            //     std::copy(src->begin(), src->end(), dst);
            // };

            // auto future0 = std::async(std::launch::async,
            //     copyToOutput,
            //     &event0, &h_uniqueKeys, out_uniqueKeys
            // );

            // auto future1 = std::async(std::launch::async,
            //     copyToOutput,
            //     &event1, &h_compactOffsetTmp, out_offsets
            // );

            // copyToOutput(&event2, &h_compactValues, out_values);

            // future0.wait();
            // future1.wait();

            // std::copy(h_uniqueKeys.begin(), h_uniqueKeys.end(), out_uniqueKeys);
            // std::copy(h_compactValues.begin(), h_compactValues.end(), out_values);
            // std::copy(h_compactOffsetTmp.begin(), h_compactOffsetTmp.end(), out_offsets);
        }

        struct CompactTableView{
            CompactKeyIndexTable core;
            const int* offsets;
            const Value* values;

            DEVICEQUALIFIER
            int retrieve(
                cg::thread_block_tile<CompactKeyIndexTable::cg_size()> group,
                Key key,
                Value* outValues
            ) const noexcept{

                int keyIndex = 0;
                auto status = core.retrieve(
                    key,
                    keyIndex,
                    group
                );
                
                const int begin = offsets[keyIndex];
                const int end = offsets[keyIndex+1];
                const int num = end - begin;

                for(int p = group.thread_rank(); p < num; p += group.size()){
                    outValues[p] = values[begin + p];
                }

                return num;
            }

            DEVICEQUALIFIER
            int numValues(
                cg::thread_block_tile<CompactKeyIndexTable::cg_size()> group,
                Key key
            ) const noexcept{

                int keyIndex = 0;
                auto status = core.retrieve(
                    key,
                    keyIndex,
                    group
                );
                
                const int begin = offsets[keyIndex];
                const int end = offsets[keyIndex+1];
                const int num = end - begin;

                return num;
            }
        
            HOSTDEVICEQUALIFIER
            static constexpr int cg_size() noexcept{
                return CompactKeyIndexTable::cg_size();
            }
        };

        bool isCompact = false;
        int deviceId{};
        float load{};
        std::size_t numKeys{};
        std::size_t numValues{};
        std::size_t maxPairs{};
        std::size_t maxValuesPerKey{};
        CudaEvent event{};
        std::unique_ptr<MultiValueHashTable> gpuMvTable;
        std::unique_ptr<CompactKeyIndexTable> gpuKeyIndexTable;
        helpers::SimpleAllocationDevice<int, 0> d_compactOffsets;
        helpers::SimpleAllocationDevice<Value, 0> d_compactValues;
    };

    template<class T>
    struct GpuHashtableKeyCheck{
        __host__ __device__
        bool operator()(T key) const{
            return GpuHashtable<T, int>::isValidKey(key);
        }
    };

    template<class Key>
    void fixKeysForGpuHashTable(
        Key* d_keys,
        int numKeys,
        cudaStream_t stream
    ){
        dim3 block(128);
        dim3 grid(SDIV(numKeys, block.x));

        GpuHashtableKeyCheck<Key> isValidKey;

        gpuhashtablekernels::fixTableKeysKernel<<<grid, block, 0, stream>>>(d_keys, numKeys, isValidKey); CUERR;
    }



} //namespace gpu
} //namespace care


#endif