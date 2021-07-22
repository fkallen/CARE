#ifndef MY_CUB_WRAPPERS_CUH
#define MY_CUB_WRAPPERS_CUH

#include <gpu/cubvector.cuh>

#include <cub/cub.cuh>

#include <cassert>

namespace care{

    class CubCallWrapper{
    private:
        cub::CachingDeviceAllocator* cubAllocator{};
    public:
        CubCallWrapper(cub::CachingDeviceAllocator& alloc) : cubAllocator(&alloc){ }

        template<typename InputIteratorT , typename OutputIteratorT >
        void cubExclusiveSum(
            InputIteratorT d_in,
            OutputIteratorT d_out,
            int num_items,
            cudaStream_t stream = 0,
            bool debug_synchronous = false
        ) const {
            std::size_t bytes = 0;
            cudaError_t status = cudaSuccess;

            status = cub::DeviceScan::ExclusiveSum(
                nullptr,
                bytes,
                d_in, 
                d_out, 
                num_items, 
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);

            CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

            status = cub::DeviceScan::ExclusiveSum(
                temp.data(),
                bytes,
                d_in, 
                d_out, 
                num_items, 
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);
        }

        template<typename InputIteratorT , typename OutputIteratorT >
        void cubInclusiveSum(
            InputIteratorT d_in,
            OutputIteratorT d_out,
            int num_items,
            cudaStream_t stream = 0,
            bool debug_synchronous = false
        ) const {
            std::size_t bytes = 0;
            cudaError_t status = cudaSuccess;

            status = cub::DeviceScan::InclusiveSum(
                nullptr,
                bytes,
                d_in, 
                d_out, 
                num_items, 
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);

            CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

            status = cub::DeviceScan::InclusiveSum(
                temp.data(),
                bytes,
                d_in, 
                d_out, 
                num_items, 
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);
        }

        template<typename InputIteratorT , typename OutputIteratorT , typename ScanOpT >
        void cubInclusiveScan(
            InputIteratorT d_in,
            OutputIteratorT d_out,
            ScanOpT scan_op,
            int num_items,
            cudaStream_t stream = 0,
            bool debug_synchronous = false 
        ) const {
            std::size_t bytes = 0;
            cudaError_t status = cudaSuccess;

            status = cub::DeviceScan::InclusiveScan(
                nullptr,
                bytes,
                d_in, 
                d_out, 
                scan_op, 
                num_items, 
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);

            CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

            status = cub::DeviceScan::InclusiveScan(
                temp.data(),
                bytes,
                d_in, 
                d_out, 
                scan_op, 
                num_items, 
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);
        }

        template<typename InputIteratorT , typename OutputIteratorT >
        void cubReduceSum(
            InputIteratorT d_in,
            OutputIteratorT d_out,
            int num_items,
            cudaStream_t stream = 0,
            bool debug_synchronous = false 
        ) const {
            std::size_t bytes = 0;
            cudaError_t status = cudaSuccess;

            status = cub::DeviceReduce::Sum(
                nullptr,
                bytes,
                d_in, 
                d_out, 
                num_items, 
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);

            CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

            status = cub::DeviceReduce::Sum(
                temp.data(),
                bytes,
                d_in, 
                d_out, 
                num_items, 
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);
        }

        template<typename InputIteratorT , typename FlagIterator , typename OutputIteratorT , typename NumSelectedIteratorT >
        void cubSelectFlagged(
            InputIteratorT d_in,
            FlagIterator d_flags,
            OutputIteratorT d_out,
            NumSelectedIteratorT d_num_selected_out,
            int num_items,
            cudaStream_t stream = 0,
            bool debug_synchronous = false 
        ) const {
            std::size_t bytes = 0;
            cudaError_t status = cudaSuccess;

            status = cub::DeviceSelect::Flagged(
                nullptr, 
                bytes, 
                d_in, 
                d_flags, 
                d_out, 
                d_num_selected_out, 
                num_items, 
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);

            CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

            status = cub::DeviceSelect::Flagged(
                temp.data(), 
                bytes, 
                d_in, 
                d_flags, 
                d_out, 
                d_num_selected_out, 
                num_items, 
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);
        }

        template<typename InputIteratorT , typename OutputIteratorT , typename OffsetIteratorT >
        void cubSegmentedReduceSum(
            InputIteratorT d_in,
            OutputIteratorT d_out,
            int num_segments,
            OffsetIteratorT	d_begin_offsets,
            OffsetIteratorT d_end_offsets,
            cudaStream_t stream = 0,
            bool debug_synchronous = false 
        ) const {
            std::size_t bytes = 0;
            cudaError_t status = cudaSuccess;

            status = cub::DeviceSegmentedReduce::Sum(
                nullptr, 
                bytes, 
                d_in, 
                d_out, 
                num_segments, 
                d_begin_offsets, 
                d_end_offsets,
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);

            CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

            status = cub::DeviceSegmentedReduce::Sum(
                temp.data(), 
                bytes, 
                d_in, 
                d_out, 
                num_segments, 
                d_begin_offsets, 
                d_end_offsets,
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);
        }

        template<typename KeysInputIteratorT, typename UniqueOutputIteratorT, typename ValuesInputIteratorT, typename AggregatesOutputIteratorT, typename NumRunsOutputIteratorT, typename ReductionOpT>
        void cubReduceByKey(
            KeysInputIteratorT d_keys_in,
            UniqueOutputIteratorT d_unique_out,
            ValuesInputIteratorT d_values_in,
            AggregatesOutputIteratorT d_aggregates_out,
            NumRunsOutputIteratorT d_num_runs_out,
            ReductionOpT reduction_op,
            int num_items,
            cudaStream_t stream = 0,
            bool debug_synchronous = false 
        ) const {
            std::size_t bytes = 0;
            cudaError_t status = cudaSuccess;

            status = cub::DeviceReduce::ReduceByKey(
                nullptr, 
                bytes, 
                d_keys_in, 
                d_unique_out,
                d_values_in,
                d_aggregates_out,
                d_num_runs_out,
                reduction_op,
                num_items,
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);

            CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

            status = cub::DeviceReduce::ReduceByKey(
                temp.data(), 
                bytes, 
                d_keys_in, 
                d_unique_out,
                d_values_in,
                d_aggregates_out,
                d_num_runs_out,
                reduction_op,
                num_items,
                stream,
                debug_synchronous
            );
            assert(status == cudaSuccess);
        }
    };

}




#endif