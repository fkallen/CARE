#ifndef CARE_MINHASH_QUERY_FILTER_CUH
#define CARE_MINHASH_QUERY_FILTER_CUH

#include <config.hpp>
#include <hpc_helpers.cuh>

#include <gpu/kernels.hpp>
#include <gpu/cubwrappers.cuh>
#include <gpu/cuda_unique.cuh>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

namespace care{ 
namespace gpu{


    struct GpuMinhashQueryFilter{
        static void keepDistinctAndNotMatching(
            const read_number* d_dontMatchPerSegment,
            cub::DoubleBuffer<read_number>& d_items,
            cub::DoubleBuffer<int>& d_numItemsPerSegment,
            cub::DoubleBuffer<int>& d_numItemsPerSegmentPrefixSum, //numSegments + 1
            int numSegments,
            int numItems,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
        ){
            if(numItems <= 0) return;
            if(numSegments <= 0) return;

            GpuSegmentedUnique::unique(
                d_items.Current(),
                numItems,
                d_items.Alternate(),
                d_numItemsPerSegment.Alternate(),
                numSegments,
                d_numItemsPerSegmentPrefixSum.Current(),
                d_numItemsPerSegmentPrefixSum.Current() + 1,
                stream,
                mr
            );

            if(d_dontMatchPerSegment != nullptr){
                //remove self read ids (inplace)
                //--------------------------------------------------------------------
                callFindAndRemoveFromSegmentKernel<read_number,128,4>(
                    d_dontMatchPerSegment,
                    d_items.Alternate(),
                    numSegments,
                    d_numItemsPerSegment.Alternate(),
                    d_numItemsPerSegmentPrefixSum.Current(),
                    stream
                );
            }

            CubCallWrapper(mr).cubInclusiveSum(
                d_numItemsPerSegment.Alternate(),
                d_numItemsPerSegmentPrefixSum.Alternate() + 1,
                numSegments,
                stream
            );
            CUDACHECK(cudaMemsetAsync(d_numItemsPerSegmentPrefixSum.Alternate(), 0, sizeof(int), stream));

            //copy final remaining values into contiguous range
            helpers::lambda_kernel<<<numSegments, 128, 0, stream>>>(
                [
                    d_items_in = d_items.Alternate(),
                    d_items_out = d_items.Current(),
                    numSegments,
                    d_numItemsPerSegment = d_numItemsPerSegment.Alternate(),
                    d_offsets = d_numItemsPerSegmentPrefixSum.Current(),
                    d_newOffsets = d_numItemsPerSegmentPrefixSum.Alternate()
                ] __device__ (){

                    for(int s = blockIdx.x; s < numSegments; s += gridDim.x){
                        const int numValues = d_numItemsPerSegment[s];
                        const int inOffset = d_offsets[s];
                        const int outOffset = d_newOffsets[s];

                        for(int c = threadIdx.x; c < numValues; c += blockDim.x){
                            d_items_out[outOffset + c] = d_items_in[inOffset + c];    
                        }
                    }
                }
            ); CUDACHECKASYNC;

            d_numItemsPerSegment.selector++;
            d_numItemsPerSegmentPrefixSum.selector++;

            // helpers::lambda_kernel<<<1,1,0, stream>>>([
            //     d_offsets  = d_numItemsPerSegmentPrefixSum.Current(),
            //     numSegments
            // ] __device__ (){
            //     printf("final offsets before unique\n");
            //     for(int i = 0; i < numSegments+1; i++){
            //         printf("%d ", d_offsets[i]);
            //     }
            //     printf("\n");
            // }); CUDACHECKASYNC;
            // CUDACHECK(cudaDeviceSynchronize());

            // helpers::lambda_kernel<<<1,1,0, stream>>>([
            //     d_numValuesPerSequence = d_numItemsPerSegment.Current(),
            //     numSegments
            // ] __device__ (){
            //     printf("final numValuesPerSequence\n");
            //     for(int i = 0; i < numSegments; i++){
            //         printf("%d ", d_numValuesPerSequence[i]);
            //     }
            //     printf("\n");
            // }); CUDACHECKASYNC;
            // CUDACHECK(cudaDeviceSynchronize());

            // helpers::lambda_kernel<<<1,1,0, stream>>>([
            //     d_values_out = d_items.Current()
            // ] __device__ (){
            //     printf("final values\n");
            //     for(int r = 0; r < 20; r++){
            //         for(int c = 0; c < 16; c++){
            //             printf("%d ", d_values_out[r * 16 + c]);
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }); CUDACHECKASYNC;
            // CUDACHECK(cudaDeviceSynchronize());
        }

        struct DeviceTempStorage{
            DeviceTempStorage(std::vector<int> deviceIds_)
                : DeviceTempStorage(deviceIds_, [&](){
                    return std::vector<cudaStream_t>(deviceIds_.size(), cudaStreamPerThread);
                }())
            {
                const int numGpus = deviceIds_.size();
                for(int g = 0; g < numGpus; g++){
                    cub::SwitchDevice sd(deviceIds_[g]);
                    CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));
                }
            }
            DeviceTempStorage(std::vector<int> deviceIds_, const std::vector<cudaStream_t>& streams)
                : temp{std::move(deviceIds_), streams}
                {}

            DeviceTempStorage(const DeviceTempStorage&) = delete;
            DeviceTempStorage(DeviceTempStorage&&) = default;
            DeviceTempStorage& operator=(const DeviceTempStorage&) = delete;
            DeviceTempStorage& operator=(DeviceTempStorage&&) = default;

            GpuSegmentedUnique::DeviceTempStorage temp;
        };

        static void keepDistinctAndNotMatching(
            const std::vector<const read_number*>& vec_d_dontMatchPerSegment,
            std::vector<cub::DoubleBuffer<read_number>>& vec_d_items,
            std::vector<cub::DoubleBuffer<int>>& vec_d_numItemsPerSegment,
            std::vector<cub::DoubleBuffer<int>>& vec_d_numItemsPerSegmentPrefixSum, //numSegments + 1
            const std::vector<int>& vec_numSegments,
            const std::vector<int>& vec_numItems,
            const std::vector<cudaStream_t>& streams,
            const std::vector<int>& deviceIds,
            int* h_tempstorage, // 128 * deviceIds.size()
            DeviceTempStorage& deviceTempStorage
        ){
            nvtx::ScopedRange sr_("keepDistinctAndNotMatching",4);
            const int numGpus = deviceIds.size();

            std::vector<read_number*> vec_d_items_current;
            std::vector<read_number*> vec_d_items_alternate;
            std::vector<int*> vec_d_numItemsPerSegment_alternate;
            std::vector<const int*> vec_d_numItemsPerSegmentPrefixSum_current;
            std::vector<const int*> vec_d_numItemsPerSegmentPrefixSum_current_plus1;

            for(int g = 0; g < numGpus; g++){
                vec_d_items_current.push_back(vec_d_items[g].Current());
                vec_d_items_alternate.push_back(vec_d_items[g].Alternate());
                vec_d_numItemsPerSegment_alternate.push_back(vec_d_numItemsPerSegment[g].Alternate());
                vec_d_numItemsPerSegmentPrefixSum_current.push_back(vec_d_numItemsPerSegmentPrefixSum[g].Current());
                vec_d_numItemsPerSegmentPrefixSum_current_plus1.push_back(vec_d_numItemsPerSegmentPrefixSum[g].Current() + 1);
            }

            GpuSegmentedUnique::unique(
                vec_d_items_current,
                vec_numItems,
                vec_d_items_alternate,
                vec_d_numItemsPerSegment_alternate,
                vec_numSegments,
                vec_d_numItemsPerSegmentPrefixSum_current,
                vec_d_numItemsPerSegmentPrefixSum_current_plus1,
                streams,
                deviceIds,
                h_tempstorage,
                deviceTempStorage.temp
            );


            for(int g = 0; g < numGpus; g++){
                if(vec_numItems[g] > 0 && vec_numSegments[g] > 0){
                    cub::SwitchDevice sd{deviceIds[g]};
                    if(vec_d_dontMatchPerSegment[g] != nullptr){
                        //remove self read ids (inplace)
                        //--------------------------------------------------------------------
                        callFindAndRemoveFromSegmentKernel<read_number,128,4>(
                            vec_d_dontMatchPerSegment[g],
                            vec_d_items[g].Alternate(),
                            vec_numSegments[g],
                            vec_d_numItemsPerSegment[g].Alternate(),
                            vec_d_numItemsPerSegmentPrefixSum[g].Current(),
                            streams[g]
                        );
                    }
                    size_t cubBytes = 0;
                    CUDACHECK(cub::DeviceScan::InclusiveSum(
                        nullptr,
                        cubBytes,
                        vec_d_numItemsPerSegment[g].Alternate(),
                        vec_d_numItemsPerSegmentPrefixSum[g].Alternate() + 1,
                        vec_numSegments[g],
                        streams[g]
                    ));
                    resizeUninitialized(deviceTempStorage.temp.vec_d_buffers[g], cubBytes, streams[g]);
                    CUDACHECK(cub::DeviceScan::InclusiveSum(
                        deviceTempStorage.temp.vec_d_buffers[g].data(),
                        cubBytes,
                        vec_d_numItemsPerSegment[g].Alternate(),
                        vec_d_numItemsPerSegmentPrefixSum[g].Alternate() + 1,
                        vec_numSegments[g],
                        streams[g]
                    ));

                    CUDACHECK(cudaMemsetAsync(
                        vec_d_numItemsPerSegmentPrefixSum[g].Alternate(), 
                        0, 
                        sizeof(int), 
                        streams[g]
                    ));

                    //copy final remaining values into contiguous range
                    helpers::lambda_kernel<<<vec_numSegments[g], 128, 0, streams[g]>>>(
                        [
                            d_items_in = vec_d_items[g].Alternate(),
                            d_items_out = vec_d_items[g].Current(),
                            numSegments = vec_numSegments[g],
                            d_numItemsPerSegment = vec_d_numItemsPerSegment[g].Alternate(),
                            d_offsets = vec_d_numItemsPerSegmentPrefixSum[g].Current(),
                            d_newOffsets = vec_d_numItemsPerSegmentPrefixSum[g].Alternate()
                        ] __device__ (){

                            for(int s = blockIdx.x; s < numSegments; s += gridDim.x){
                                const int numValues = d_numItemsPerSegment[s];
                                const int inOffset = d_offsets[s];
                                const int outOffset = d_newOffsets[s];

                                for(int c = threadIdx.x; c < numValues; c += blockDim.x){
                                    d_items_out[outOffset + c] = d_items_in[inOffset + c];    
                                }
                            }
                        }
                    ); CUDACHECKASYNC;

                    vec_d_numItemsPerSegment[g].selector++;
                    vec_d_numItemsPerSegmentPrefixSum[g].selector++;
                }
            }



            
        }
    
        static void keepDistinct(
            cub::DoubleBuffer<read_number>& d_items,
            cub::DoubleBuffer<int>& d_numItemsPerSegment,
            cub::DoubleBuffer<int>& d_numItemsPerSegmentPrefixSum, //numSegments + 1
            int numSegments,
            int numItems,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
        ){
            keepDistinctAndNotMatching(
                nullptr,
                d_items,
                d_numItemsPerSegment,
                d_numItemsPerSegmentPrefixSum,
                numSegments,
                numItems,
                stream,
                mr
            );
        }
    };



}}




#endif