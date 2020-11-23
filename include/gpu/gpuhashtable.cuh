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

#include <cooperative_groups.h>

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

        template<class CompactTableView, class Key, class Value, class Offset>
        __global__
        void retrieveCompactKernel(
            CompactTableView table,
            const Key* __restrict__ querykeys,
            const int numKeys,
            Value* __restrict__ outValues,
            int valueOffset, // values for key i begin at valueOffset * i
            Offset* __restrict__ numValuesPerKey
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

                const int num = table.retrieve(tile, key, outValues + size_t(k) * valueOffset);
                if(tile.thread_rank() == 0){
                    numValuesPerKey[k] = num;
                }    
            }
        }

        template<class CompactTableView, class Key, class Value, class Offset>
        __global__
        void retrieveCompactKernel(
            CompactTableView table,
            const Key* __restrict__ querykeys,
            const Offset* __restrict__ beginOffsets,
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

                table.retrieve(tile, key, outValues + beginOffset);
            }
        }


        

        template<class CompactTableView, class Key, class Offset>
        __global__
        void numValuePerKeyCompactKernel(
            const CompactTableView table,
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
                    numValuesPerKey[k] = num;
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

            if(maxPairs > std::numeric_limits<int>::max()){
                assert(maxPairs <= std::numeric_limits<int>::max()); //CompactKeyIndexTable uses int
            }

            cudaGetDevice(&deviceId);

            const std::size_t capacity = maxPairs / load;
            gpuMvTable = std::move(
                std::make_unique<MultiValueHashTable>(
                    capacity, warpcore::defaults::seed<Key>(), maxValuesPerKey
                )
            );
        }

        HOSTDEVICEQUALIFIER
        static constexpr bool isValidKey(Key key){
            return MultiValueHashTable::is_valid_key(key);
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

        void retrieve(
            const Key* d_keys, 
            Index N,
            Index* d_begin_offsets_out,
            Index* d_end_offsets_out,
            Value* d_values,
            cudaStream_t stream,
            StatusHandler::base_type* d_statusarray = nullptr
        ){
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

        void retrieve(
            const Key* d_keys, 
            Index N,
            Index* d_begin_offsets_out,
            Index* d_end_offsets_out,
            Value* d_values,
            Index& num_out,
            cudaStream_t stream,
            StatusHandler::base_type* d_statusarray = nullptr
        ){
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

        

        
        template<class Offset>
        void retrieveCompact(
            const Key* d_keys, 
            Index N,
            Offset* d_numValuesPerKey,
            Value* d_values,
            cudaStream_t stream,
            int valueOffset
        ) const {
            assert(isCompact);

            CompactTableView table{*gpuKeyIndexTable, d_compactOffsets.data(), d_compactValues.data()};

            gpuhashtablekernels::retrieveCompactKernel<<<1024, 256, 0, stream>>>(
                table,
                d_keys,
                N,
                d_values,
                valueOffset, // values for key i begin at valueOffset * i
                d_numValuesPerKey
            );
        }

        template<class Offset>
        void retrieveCompact(
            const Key* d_keys, 
            const Offset* d_beginOffsets,
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

        void compact(){
            if(isCompact) return;

            Index numUniqueKeys = gpuMvTable->num_keys();

            helpers::SimpleAllocationPinnedHost<Key, 0> h_uniqueKeys(numUniqueKeys);
            helpers::SimpleAllocationPinnedHost<Index, 0> h_compactOffsetTmp(numUniqueKeys+1);
            
            gpuMvTable->retrieve_all_keys(
                h_uniqueKeys,
                numUniqueKeys
            ); CUERR;

            Index numRetrievedValues = 0;

            retrieve(
                h_uniqueKeys.data(),
                numUniqueKeys,
                h_compactOffsetTmp.data(),
                h_compactOffsetTmp.data() + 1,
                nullptr,
                numRetrievedValues,
                (cudaStream_t)0
            );
            cudaStreamSynchronize(0); CUERR;

            helpers::SimpleAllocationPinnedHost<Value, 0> h_compactValues(numRetrievedValues);
            

            retrieve(
                h_uniqueKeys.data(),
                numUniqueKeys,
                h_compactOffsetTmp.data(),
                h_compactOffsetTmp.data() + 1,
                h_compactValues.data(),
                numRetrievedValues,
                (cudaStream_t)0
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
            d_compactValues.resize(numRetrievedValues);

            //copy offsets to gpu, convert from Index to int
            gpuhashtablekernels::assignmentKernel<<<SDIV(numUniqueKeys+1, 256), 256>>>(
                d_compactOffsets.data(), 
                h_compactOffsetTmp.data(), 
                numUniqueKeys+1
            );
            cudaMemcpyAsync(d_compactValues, h_compactValues, d_compactValues.sizeInBytes(), H2D, 0); CUERR;


            const std::size_t batchsize = 100000;
            const std::size_t iters =  SDIV(numUniqueKeys, batchsize);
            helpers::SimpleAllocationPinnedHost<int> ids(batchsize);

            for(std::size_t i = 0; i < iters; i++){
                const std::size_t begin = i * batchsize;
                const std::size_t end = std::min((i+1) * batchsize, numUniqueKeys);
                const std::size_t num = end - begin;

                std::iota(ids.begin(), ids.end(), int(begin));
                gpuKeyIndexTable->insert(
                    h_uniqueKeys + begin,
                    ids.data(),
                    num,
                    (cudaStream_t)0,
                    warpcore::defaults::probing_length(),
                    nullptr
                );
                cudaStreamSynchronize(0); CUERR;
            }

            numKeys = numUniqueKeys;
            numValues = numRetrievedValues;
            
            isCompact = true;
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




} //namespace gpu
} //namespace care


#endif