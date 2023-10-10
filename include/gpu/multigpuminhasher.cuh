#ifdef CARE_HAS_WARPCORE


#ifndef CARE_MULTI_GPU_MINHASHER_CUH
#define CARE_MULTI_GPU_MINHASHER_CUH

#include <config.hpp>


#include <gpu/gpureadstorage.cuh>
#include <gpu/cuda_unique.cuh>
#include <gpu/singlegpuminhasher.cuh>
#include <gpu/gpuminhasher.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/cubwrappers.cuh>
#include <gpu/multigputransfers.cuh>

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
#include <cassert>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <gpu/rmm_utilities.cuh>
namespace care{
namespace gpu{

    namespace multigpuminhasherkernels{
        template<int blocksize, int itemsPerThread>
        __global__
        void aggregatePartitionResultsSingleBlockKernel(
            const int* __restrict__ numResultsPerSequencePerPartition,
            int* __restrict__ numResultsPerSequence,
            int numSequences,
            int numPartitions,
            int* __restrict__ maxNumResultsPerSequence,
            int* __restrict__ offsets
        ){
            assert(gridDim.x * gridDim.y * gridDim.z == 1);

            struct BlockPrefixCallbackOp{
                // Running prefix
                int running_total;
                // Constructor
                __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
                // Callback operator to be entered by the first warp of threads in the block.
                // Thread-0 is responsible for returning a value for seeding the block-wide scan.
                __device__ int operator()(int block_aggregate)
                {
                    int old_prefix = running_total;
                    running_total += block_aggregate;
                    return old_prefix;
                }
            };

            using BlockReduce = cub::BlockReduce<int, blocksize>;
            using BlockScan = cub::BlockScan<int, blocksize>;

            __shared__ typename BlockReduce::TempStorage smem_reduce;
            __shared__ typename BlockScan::TempStorage smem_scan;

            constexpr int itemsPerIteration = blocksize * itemsPerThread;

            const int numIterations = SDIV(numSequences, itemsPerIteration);
            int myMax = 0;
            BlockPrefixCallbackOp prefix_op(0);

            //full iterations
            for(int iteration = 0; iteration < numIterations; iteration++){
                const int s = iteration * itemsPerIteration + threadIdx.x;
                int sum = 0;

                if(s < numSequences){
                    for(int r = 0; r < numPartitions; r++){
                        sum += numResultsPerSequencePerPartition[r * numSequences + s];
                    }

                    numResultsPerSequence[s] = sum;
                    myMax = max(myMax, sum);
                }

                BlockScan(smem_scan).ExclusiveSum(sum, sum, prefix_op);

                if(s < numSequences){
                    offsets[s] = sum;
                }
            }

            myMax = BlockReduce(smem_reduce).Reduce(myMax, cub::Max{});

            if(threadIdx.x == 0){
                *maxNumResultsPerSequence = myMax;
                offsets[numSequences] = prefix_op.running_total;
            }
        };

        template<class Dummy = void>
        __global__
        void aggregateNumValuesPartitionResultsKernel(
            const int* __restrict__ numResultsPerSequencePerPartition,
            int* __restrict__ numResultsPerSequence,
            int numSequences,
            int numPartitions
        ){
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const int stride = blockDim.x * gridDim.x;

            for(int s = tid; s < numSequences; s += stride){
                int sum = 0;
                
                for(int r = 0; r < numPartitions; r++){
                    sum += numResultsPerSequencePerPartition[r * numSequences + s];
                }

                numResultsPerSequence[s] = sum;
            }

        };
    
        template<class Dummy = void>
        __global__
        void copyToInterleavedKernel(
            const read_number* __restrict__ inputdata,
            const int* __restrict__ inputoffsets,
            const int* __restrict__ inputsegmentsizes,
            const int* __restrict__ inputnumpergpuPS,
            const int* __restrict__ outputbeginoffsets,
            read_number* __restrict__ outputdata,
            int numSequences,
            int partitions
        ){                
            for(int i = blockIdx.x; i < numSequences; i += gridDim.x){
                const int beginoffset = outputbeginoffsets[i];

                int runningOffset = 0;

                for(int r = 0; r < partitions; r++){
                    const int segmentsize = inputsegmentsizes[r * numSequences + i];
                    const int inputOffset = inputoffsets[r * (numSequences+1) + i];
                    const int gpuOffset = inputnumpergpuPS[r];

                    const read_number* myinput = inputdata + gpuOffset + inputOffset;

                    for(int k = threadIdx.x; k < segmentsize; k += blockDim.x){
                        outputdata[beginoffset + runningOffset + k] = myinput[k];
                    }

                    runningOffset += segmentsize;
                }
            }
        }

        template<class Dummy = void>
        __global__
        void copyToInterleavedKernel_fixoffsets(
            const read_number* __restrict__ inputdata,
            const int* __restrict__ inputoffsets,
            const int* __restrict__ inputsegmentsizes,
            const int* __restrict__ inputnumpergpuPS,
            const int* __restrict__ outputbeginoffsets,
            read_number* __restrict__ outputdata,
            int numSequences,
            int partitions
        ){                
            for(int i = blockIdx.x; i < numSequences; i += gridDim.x){
                const int beginoffset = outputbeginoffsets[i];

                int runningOffset = 0;

                for(int r = 0; r < partitions; r++){
                    const int segmentsize = inputsegmentsizes[r * numSequences + i];
                    const int inputOffset = inputoffsets[r * (numSequences+1) + i] 
                        - inputoffsets[r * (numSequences+1) + 0];
                    const int gpuOffset = inputnumpergpuPS[r];

                    const read_number* myinput = inputdata + gpuOffset + inputOffset;

                    // if(threadIdx.x == 0){
                    //     printf("i %d r %d, segmentsize %d inputOffset %d, gpuOffset %d, runningOffset %d, beginoffset %d, myinput begins at %d\n",
                    //         i, r, segmentsize, inputOffset, gpuOffset, runningOffset, beginoffset, gpuOffset + inputOffset);
                    // }

                    for(int k = threadIdx.x; k < segmentsize; k += blockDim.x){
                        outputdata[beginoffset + runningOffset + k] = myinput[k];
                    }

                    runningOffset += segmentsize;
                }
            }
        }

        template<class Dummy = void>
        __global__ 
        void copyResultsToDestinationKernel(
            const read_number* input,
            int* inputoffsets,
            read_number* output,
            const int* outputoffsets,
            const int* d_numValuesPerSequence,
            int numSequences
        ){
            for(int s = blockIdx.x; s < numSequences; s += gridDim.x){
                const int inputoffset = inputoffsets[s];
                const int outputoffset = outputoffsets[s];
                const int segmentsize = d_numValuesPerSequence[s];

                for(int i = threadIdx.x; i < segmentsize; i += blockDim.x){
                    output[outputoffset + i] = input[inputoffset + i];
                }

                __syncthreads();
                //update d_offsets within this kernel to avoid additional api call
                if(threadIdx.x == 0){
                    inputoffsets[s] = outputoffset;
                }
            }

            //last entry of offsets (total number) is not used for copying. no need for sync
            if(blockIdx.x == 0 && threadIdx.x == 0){
                inputoffsets[numSequences] = outputoffsets[numSequences];
            }

        }

        //copy N elements from source to each destination buffer
        //destinations must be distinct
        template<class T>
        __global__ 
        void scatterFullArrayKernel(
            const T* __restrict__ src,
            int N,
            T* const __restrict__  * __restrict__  dsts,
            int numDestinations
        ){
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const int stride = blockDim.x * gridDim.x;
            for(int d = blockIdx.y; d < numDestinations; d += gridDim.y){
                T* const dst = dsts[d];
                for(int i = tid; i < N; i += stride){
                    dst[i] = src[i];
                }
            }
        }

        //copy the i-th segment of size numElementsPerDestination to the i-th destination
        template<class T>
        __global__ 
        void scatterSegmentedArrayKernel(
            const T* __restrict__ src,
            const int* __restrict__ beginOffsets,
            const int* __restrict__ endOffsets,
            T* __restrict__  * __restrict__  dsts,
            int numDestinations
        ){
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const int stride = blockDim.x * gridDim.x;
            for(int d = blockIdx.y; d < numDestinations; d += gridDim.y){
                const int begin = beginOffsets[d];
                const int end = endOffsets[d];
                const int N = end - begin;

                T* const dst = dsts[d];
                for(int i = tid; i < N; i += stride){
                    dst[i] = src[begin + i];
                }
            }
        }

    }


    class MultiGpuMinhasher : public GpuMinhasher, public GpuMinhasherWithMultiQuery{
    public:
        //using Key = GpuMinhasher::Key;
        using Key = SingleGpuMinhasher::Key;
        using Value = SingleGpuMinhasher::Value;

        enum class Layout {FirstFit, EvenShare};
    private:
        using DeviceSwitcher = cub::SwitchDevice;
        
        template<class T>
        using HostBuffer = helpers::SimpleAllocationPinnedHost<T>;
        struct QueryData{
            enum class Stage{
                None,
                NumValues,
                Retrieve
            };

            Stage previousStage = Stage::None;


            CudaEvent callerEvent{cudaEventDisableTiming};
            HostBuffer<char> pinnedData{};
            std::vector<CudaStream> streams{}; // 1 for each data gpu, created on that data gpu
            std::vector<CudaEvent> events{}; // 1 for each data gpu, created on that data gpu
            std::vector<CudaEvent> events2{}; // 1 for each data gpu, created on that data gpu
            std::vector<std::unique_ptr<MinhasherHandle>> singlegpuMinhasherHandles{};
            std::vector<int> dataDeviceIds{};

            int numDataGpus{};
            std::vector<int> callerDeviceIds{};
            std::vector<std::vector<CudaStream>> vec_callerStreamsPerDataGpu{}; // numDataGpus streams for each caller, created on caller
            std::vector<std::vector<CudaEvent>> vec_callerEventsPerDataGpu{}; // numDataGpus events for each caller, created on caller
            
            std::vector<CudaEvent> vec_callerEvents{}; // 1 event for each caller, created on caller
            std::vector<rmm::device_uvector<int>> vec_d_numValuesPerSequence{}; // 1 for each caller gpu

            std::vector<rmm::device_uvector<char>> vec_d_singlegpuminhasherBuffers{}; // 1 for each data gpu

            std::vector<rmm::device_uvector<char>> vec_d_callerTempBuffers; // 1 for each caller gpu
            std::vector<rmm::device_uvector<char>> vec_d_targetTempBuffers; // 1 for each data gpu

            int* pinned_totalNumValuesPerTarget{};
            int* pinned_numValuesPerTargetPerCaller{};
            int* pinned_numValuesPerTarget_verticalPS{};
            int* pinned_numSequencesPerCallerPS{};
            int* pinned_totalNumValuesPerCaller{};
            void** pinned_bufferpointers{};
            void** pinned_bufferpointers2{};
            void** pinned_bufferpointers3{};
            int* pinned_singledata{};

            bool callerGpusAreEqualToDataGpus = false;

            ~QueryData(){
                for(int d = 0; d < numDataGpus; d++){
                    if(d < int(vec_d_singlegpuminhasherBuffers.size())){
                        cub::SwitchDevice sd(dataDeviceIds[d]);
                        vec_d_singlegpuminhasherBuffers[d].release();
                        vec_d_targetTempBuffers[d].release();
                    }
                } 
                for(int g = 0; g < int(callerDeviceIds.size()); g++){
                    if(g < int(vec_d_numValuesPerSequence.size())){
                        cub::SwitchDevice sd(callerDeviceIds[g]);
                        vec_d_numValuesPerSequence[g].release();
                        vec_d_callerTempBuffers[g].release();
                    }
                } 
            }

            void setCallerDeviceIds(const std::vector<int>& ids){
                if(callerDeviceIds != ids){
                    for(int g = 0; g < int(callerDeviceIds.size()); g++){
                        cub::SwitchDevice sd(callerDeviceIds[g]);
                        vec_d_callerTempBuffers[g].release();
                    }
                    vec_d_callerTempBuffers.clear(); 
                    vec_callerStreamsPerDataGpu.clear();
                    vec_callerEventsPerDataGpu.clear();
                    vec_callerEvents.clear();
                    callerDeviceIds = ids;

                    std::vector<int> vecA = dataDeviceIds;
                    std::vector<int> vecB = callerDeviceIds;
                    std::sort(vecA.begin(), vecA.end());
                    std::sort(vecB.begin(), vecB.end());
                    callerGpusAreEqualToDataGpus = vecA == vecB;

                    const int numCallerGpus = ids.size();

                    vec_callerStreamsPerDataGpu.resize(numCallerGpus);
                    vec_callerEventsPerDataGpu.resize(numCallerGpus);

                    for(int g = 0; g < numCallerGpus; g++){
                        cub::SwitchDevice sd(ids[g]);
                        vec_callerEvents.emplace_back(cudaEventDisableTiming);

                        vec_d_callerTempBuffers.emplace_back(0, cudaStreamPerThread);

                        CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));

                        for(int d = 0; d < numDataGpus; d++){
                            vec_callerStreamsPerDataGpu[g].emplace_back();
                            vec_callerEventsPerDataGpu[g].emplace_back(cudaEventDisableTiming);
                        }
                    }
                }
            }

            MemoryUsage getMemoryInfo() const{
                MemoryUsage mem{};

                return mem;
            }
        };

    public: 

        MultiGpuMinhasher(
            Layout layout_, 
            int maxNumKeys_, 
            int maxValuesPerKey, 
            int k, 
            float loadfactor_, 
            std::vector<int> deviceIds_,
            bool directPeerAccess_
        )
            : directPeerAccess(directPeerAccess_),
            layout(layout_), 
            maxNumKeys(maxNumKeys_), 
            kmerSize(k), 
            resultsPerMapThreshold(maxValuesPerKey), 
            loadfactor(loadfactor_), 
            deviceIds(deviceIds_)
        {
            for(auto deviceId : deviceIds){
                cub::SwitchDevice sd{deviceId};
                auto mh = std::make_unique<SingleGpuMinhasher>(maxNumKeys, resultsPerMapThreshold, k, loadfactor_);
                sgpuMinhashers.emplace_back(std::move(mh));

                hashFunctionIdsPerGpu.emplace_back();
            }
        }

        int addHashTables(int numAdditionalTables, const int* hashFunctionIds, cudaStream_t stream) override{

            std::vector<int> hashFunctionIdsTmp(hashFunctionIds, hashFunctionIds + numAdditionalTables);

            CudaEvent event;
            event.record(stream);

            const int numDevices = deviceIds.size();
            std::vector<CudaEvent> events;
            for(int g = 0; g < numDevices; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                events.emplace_back(cudaEventDisableTiming);

                CUDACHECK(cudaStreamWaitEvent(cudaStreamPerThread, event, 0));
            }

            int remainingTables = numAdditionalTables;

            if(layout == Layout::EvenShare){
                while(remainingTables > 0){
                    int numZeroAdded = 0;

                    for(int g = 0; g < numDevices; g++){
                        if(remainingTables > 0){
                            cub::SwitchDevice sd{deviceIds[g]};
                            int addedTables = sgpuMinhashers[g]->addHashTables(1, hashFunctionIdsTmp.data(), cudaStreamPerThread);

                            for(int x = 0; x < addedTables; x++){
                                hashTableLocations.push_back(g);
                            }
                            hashFunctionIdsPerGpu[g].insert(hashFunctionIdsPerGpu[g].end(), hashFunctionIdsTmp.begin(), hashFunctionIdsTmp.begin() + addedTables);

                            hashFunctionIdsTmp.erase(hashFunctionIdsTmp.begin(), hashFunctionIdsTmp.begin() + addedTables);
                            remainingTables -= addedTables;

                            if(addedTables == 0){
                                numZeroAdded++;
                            }
                        }
                    }

                    if(numZeroAdded == numDevices){
                        break;
                    }
                }
            }else{
                assert(layout == Layout::FirstFit);

                for(int g = 0; g < numDevices; g++){
                    if(remainingTables > 0){
                        cub::SwitchDevice sd{deviceIds[g]};
                        int addedTables = sgpuMinhashers[g]->addHashTables(remainingTables, hashFunctionIdsTmp.data(), cudaStreamPerThread);

                        for(int x = 0; x < addedTables; x++){
                            hashTableLocations.push_back(g);
                        }
                        hashFunctionIdsPerGpu[g].insert(hashFunctionIdsPerGpu[g].end(), hashFunctionIdsTmp.begin(), hashFunctionIdsTmp.begin() + addedTables);

                        hashFunctionIdsTmp.erase(hashFunctionIdsTmp.begin(), hashFunctionIdsTmp.begin() + addedTables);
                        remainingTables -= addedTables;

                        if(addedTables > 0){
                            break;
                        }
                    }
                }
            }

            for(int g = 0; g < numDevices; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                events[g].record(cudaStreamPerThread);
            }

            for(int g = 0; g < numDevices; g++){
                CUDACHECK(cudaStreamWaitEvent(stream, events[g], 0));
            }

            return numAdditionalTables - remainingTables;
        }

        void insert(
            const unsigned int* d_sequenceData2Bit,
            int numSequences,
            const int* d_sequenceLengths,
            std::size_t encodedSequencePitchInInts,
            const read_number* d_readIds,
            const read_number* h_readIds,
            int firstHashfunction,
            int numHashfunctions,
            const int* h_hashFunctionIds,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* /*mr*/
        ) override {
            assert(firstHashfunction + numHashfunctions <= hashTableLocations.size());
            if(numHashfunctions == 0) return;
            if(numSequences == 0) return;

            int oldDeviceId = 0;
            CUDACHECK(cudaGetDevice(&oldDeviceId));

            CudaEvent event;
            event.record(stream);
            std::vector<CudaEvent> events;

            for(int g = 0; g < int(deviceIds.size()); g++){
                cub::SwitchDevice sd{deviceIds[g]};
                events.emplace_back(cudaEventDisableTiming);

                CUDACHECK(cudaStreamWaitEvent(cudaStreamPerThread, event, 0));
            }

            std::vector<int> numHashfunctionsPerTargetGpu(deviceIds.size(), 0);
            std::vector<int> firstHashfunctionPerTargetGpu(deviceIds.size(), 0);
            std::vector<std::vector<int>> hashFunctionIdsPerTargetGpu(deviceIds.size());

            for(int g = 0; g < int(deviceIds.size()); g++){
                int countBefore = 0;
                for(int i = 0; i < firstHashfunction + numHashfunctions; i++){
                    if(hashTableLocations[i] == g){
                        if(i < firstHashfunction){
                            countBefore++;
                        }else{
                            numHashfunctionsPerTargetGpu[g]++;
                            hashFunctionIdsPerTargetGpu[g].push_back(h_hashFunctionIds[i - firstHashfunction]);
                        }
                    }
                }
                firstHashfunctionPerTargetGpu[g] = countBefore;
                
                assert(numHashfunctionsPerTargetGpu[g] == int(hashFunctionIdsPerTargetGpu[g].size()));
            }
            
            std::vector<rmm::device_uvector<unsigned int>> vec_d_sequenceData2Bit_target;
            std::vector<rmm::device_uvector<int>> vec_d_sequenceLengths_target;
            std::vector<rmm::device_uvector<read_number>> vec_d_readIds_target;

            //broadcast to all gpus, excluding current gpu
            for(int g = 0; g < int(deviceIds.size()); g++){
                if(numHashfunctionsPerTargetGpu[g] > 0){
                    if(deviceIds[g] != oldDeviceId){
                        cub::SwitchDevice sd{deviceIds[g]};

                        //copy input data to target gpu
                        auto* targetmr = rmm::mr::get_current_device_resource();
                        rmm::device_uvector<unsigned int> d_sequenceData2Bit_target(encodedSequencePitchInInts * numSequences, cudaStreamPerThread, targetmr);
                        rmm::device_uvector<int> d_sequenceLengths_target(numSequences, cudaStreamPerThread, targetmr);
                        rmm::device_uvector<read_number> d_readIds_target(numSequences, cudaStreamPerThread, targetmr);

                        CUDACHECK(cudaMemcpyPeerAsync(
                            d_sequenceData2Bit_target.data(),
                            deviceIds[g],
                            d_sequenceData2Bit,
                            oldDeviceId,
                            sizeof(unsigned int) * encodedSequencePitchInInts * numSequences,
                            cudaStreamPerThread
                        ));

                        CUDACHECK(cudaMemcpyPeerAsync(
                            d_sequenceLengths_target.data(),
                            deviceIds[g],
                            d_sequenceLengths,
                            oldDeviceId,
                            sizeof(int) * numSequences,
                            cudaStreamPerThread
                        ));

                        CUDACHECK(cudaMemcpyPeerAsync(
                            d_readIds_target.data(),
                            deviceIds[g],
                            d_readIds,
                            oldDeviceId,
                            sizeof(read_number) * numSequences,
                            cudaStreamPerThread
                        ));

                        vec_d_sequenceData2Bit_target.push_back(std::move(d_sequenceData2Bit_target));
                        vec_d_sequenceLengths_target.push_back(std::move(d_sequenceLengths_target));
                        vec_d_readIds_target.push_back(std::move(d_readIds_target));
                    }else{
                        vec_d_sequenceData2Bit_target.emplace_back(0, cudaStreamPerThread);
                        vec_d_sequenceLengths_target.emplace_back(0, cudaStreamPerThread);
                        vec_d_readIds_target.emplace_back(0, cudaStreamPerThread);
                    }
                }
            }

            //insert on each gpu
            for(int g = 0; g < int(deviceIds.size()); g++){
                if(numHashfunctionsPerTargetGpu[g] > 0){
                    cub::SwitchDevice sd{deviceIds[g]};

                    const unsigned int* d_seq = d_sequenceData2Bit;
                    const int* d_len = d_sequenceLengths;
                    const read_number* d_ids = d_readIds;

                    if(deviceIds[g] != oldDeviceId){
                        d_seq = vec_d_sequenceData2Bit_target[g].data();
                        d_len = vec_d_sequenceLengths_target[g].data();
                        d_ids = vec_d_readIds_target[g].data();
                    }

                    sgpuMinhashers[g]->insert(
                        d_seq,
                        numSequences,
                        d_len,
                        encodedSequencePitchInInts,
                        d_ids,
                        h_readIds,
                        firstHashfunctionPerTargetGpu[g],
                        numHashfunctionsPerTargetGpu[g],
                        hashFunctionIdsPerTargetGpu[g].data(),
                        cudaStreamPerThread,
                        rmm::mr::get_current_device_resource()
                    );
                    if(deviceIds[g] != oldDeviceId){
                        vec_d_sequenceData2Bit_target[g].release();
                        vec_d_sequenceLengths_target[g].release();
                        vec_d_readIds_target[g].release();
                    }
                    events[g].record(cudaStreamPerThread);
                }
            }

            for(int g = 0; g < int(deviceIds.size()); g++){
                if(numHashfunctionsPerTargetGpu[g] > 0){
                    CUDACHECK(cudaStreamWaitEvent(stream, events[g], 0));
                }
            }
        }

        int checkInsertionErrors(
            int firstHashfunction,
            int numHashfunctions,
            cudaStream_t stream        
        ) override{
            CudaEvent event;
            event.record(stream);

            std::vector<int> numHashfunctionsPerTargetGpu(deviceIds.size(), 0);
            std::vector<int> firstHashfunctionPerTargetGpu(deviceIds.size(), 0);

            for(int g = 0; g < int(deviceIds.size()); g++){
                int countBefore = 0;
                for(int i = 0; i < firstHashfunction + numHashfunctions; i++){
                    if(hashTableLocations[i] == g){
                        if(i < firstHashfunction){
                            countBefore++;
                        }else{
                            numHashfunctionsPerTargetGpu[g]++;
                        }
                    }
                }
                firstHashfunctionPerTargetGpu[g] = countBefore;
            }

            int count = 0;

            for(int g = 0; g < int(deviceIds.size()); g++){
                cub::SwitchDevice sd{deviceIds[g]};

                CUDACHECK(cudaStreamWaitEvent(cudaStreamPerThread, event, 0));
                count += sgpuMinhashers[g]->checkInsertionErrors(
                    firstHashfunctionPerTargetGpu[g],
                    numHashfunctionsPerTargetGpu[g],
                    cudaStreamPerThread
                );
            }

            return count;
        }

        // bool tryReplication(){
        //     if(sgpuMinhashers.size() == 1 && usableDeviceIds.size() < deviceIds.size()){
        //         //all hashtables fit into one gpu. try to replace the hash tables on all gpus

        //         std::vector<std::unique_ptr<SingleGpuMinhasher>> replicas{};
        //         bool ok = false;
        //         try{
        //             nvtx::push_range("replicate single gpu minhasher", 0);

        //             for(std::size_t i = 1; i < deviceIds.size(); i++){
        //                 const int targetDeviceId = deviceIds[i];
        //                 helpers::CpuTimer rtimer("make singlegpu minhasher replica");
        //                 replicas.emplace_back(sgpuMinhashers[0]->makeCopy(targetDeviceId));
        //                 rtimer.print();
        //             }
        //             ok = std::all_of(replicas.begin(), replicas.end(), [](const auto& uniqueptr){ return bool(uniqueptr); });

        //             nvtx::pop_range();
        //         }catch(...){
        //             cudaGetLastError();
        //             std::cerr << "error replicating single gpu minhasher. Skipping.\n";
        //         }
        //         if(ok){                    
        //             sgpuMinhashers.insert(sgpuMinhashers.end(), std::make_move_iterator(replicas.begin()), std::make_move_iterator(replicas.end()));

        //             HostBuffer<int> h_currentHashFunctionNumbers(vec_h_currentHashFunctionIds[0].size());
        //             std::copy(vec_h_currentHashFunctionIds[0].begin(), vec_h_currentHashFunctionIds[0].end(), h_currentHashFunctionNumbers.begin());
        //             vec_h_currentHashFunctionIds.push_back(std::move(h_currentHashFunctionNumbers));

        //             usableDeviceIds = deviceIds;

        //             isReplicatedSingleGpu = true;
        //         }

        //         return ok;
        //     }else{
        //         return false;
        //     }
        // }



        MinhasherHandle makeMinhasherHandle() const override{
            auto ptr = createQueryData();
            auto ptr2 = createQueryData();

            std::unique_lock<SharedMutex> lock(sharedmutex);
            const int handleid = counter++;
            MinhasherHandle h = constructHandle(handleid);

            tempdataVector.emplace_back(std::move(ptr));
            multi_tempdataVector.emplace_back(std::move(ptr2));

            return h;
        }

        void destroyHandle(MinhasherHandle& handle) const override{            

            std::unique_lock<SharedMutex> lock(sharedmutex);

            const int id = handle.getId();
            assert(id < int(tempdataVector.size()));
            assert(id < int(multi_tempdataVector.size()));

            const int numMinhashers = sgpuMinhashers.size();
            for(int i = 0; i < numMinhashers; i++){
                sgpuMinhashers[i]->destroyHandle(*tempdataVector[id]->singlegpuMinhasherHandles[i]);
                sgpuMinhashers[i]->destroyHandle(*multi_tempdataVector[id]->singlegpuMinhasherHandles[i]);
            }
            
            {
                tempdataVector[id] = nullptr;
                multi_tempdataVector[id] = nullptr;
            }
            handle = constructHandle(std::numeric_limits<int>::max());
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
        ) const override{
            QueryData* const queryData = getQueryDataFromHandle(queryHandle);
            queryData->previousStage = QueryData::Stage::NumValues;

            if(numSequences == 0){
                // CUDACHECK(cudaMemsetAsync(d_numValuesPerSequence, 0, sizeof(int) * numSequences, stream));
                totalNumValues = 0;
                return;
            }

            int oldDeviceId = 0;
            CUDACHECK(cudaGetDevice(&oldDeviceId));
            
            rmm::device_uvector<int> d_numValuesPerSequencePerGpu(numSequences * deviceIds.size(), stream, mr);

            CUDACHECK(cudaEventRecord(queryData->callerEvent, stream));

            for(int g = 0; g < int(deviceIds.size()); g++){
                cub::SwitchDevice sd{deviceIds[g]};

                CUDACHECK(cudaStreamWaitEvent(queryData->streams[g], queryData->callerEvent, 0));
            }

            std::vector<rmm::device_uvector<unsigned int>> vec_d_sequenceData2Bit_target;
            std::vector<rmm::device_uvector<int>> vec_d_sequenceLengths_target;

            //broadcast to other gpus
            for(int g = 0; g < int(deviceIds.size()); g++){
                if(deviceIds[g] != oldDeviceId){
                    cub::SwitchDevice sd{deviceIds[g]};

                    auto* targetmr = rmm::mr::get_current_device_resource();
                    rmm::device_uvector<unsigned int> d_sequenceData2Bit_target(encodedSequencePitchInInts * numSequences, queryData->streams[g].getStream(), targetmr);
                    rmm::device_uvector<int> d_sequenceLengths_target(numSequences, queryData->streams[g].getStream(), targetmr);

                    CUDACHECK(cudaMemcpyPeerAsync(
                        d_sequenceData2Bit_target.data(),
                        deviceIds[g],
                        d_sequenceData2Bit,
                        oldDeviceId,
                        sizeof(unsigned int) * encodedSequencePitchInInts * numSequences,
                        queryData->streams[g]
                    ));

                    CUDACHECK(cudaMemcpyPeerAsync(
                        d_sequenceLengths_target.data(),
                        deviceIds[g],
                        d_sequenceLengths,
                        oldDeviceId,
                        sizeof(int) * numSequences,
                        queryData->streams[g]
                    ));

                    vec_d_sequenceData2Bit_target.push_back(std::move(d_sequenceData2Bit_target));
                    vec_d_sequenceLengths_target.push_back(std::move(d_sequenceLengths_target));
                }else{
                    vec_d_sequenceData2Bit_target.emplace_back(0, queryData->streams[g].getStream());
                    vec_d_sequenceLengths_target.emplace_back(0, queryData->streams[g].getStream());
                }
            }

            //determine num values on each gpu, and collect results in d_numValuesPerSequencePerGpu
            for(int g = 0; g < int(deviceIds.size()); g++){
                cub::SwitchDevice sd{deviceIds[g]};

                const unsigned int* d_seq = d_sequenceData2Bit;
                const int* d_len = d_sequenceLengths;
                if(deviceIds[g] != oldDeviceId){
                    d_seq = vec_d_sequenceData2Bit_target[g].data();
                    d_len = vec_d_sequenceLengths_target[g].data();
                }

                int& myTotalNumValues = queryData->pinned_singledata[g];
                auto* targetmr = rmm::mr::get_current_device_resource();
                rmm::device_uvector<int> d_numValuesPerSequence_target(numSequences, queryData->streams[g].getStream(), targetmr);

                sgpuMinhashers[g]->determineNumValues(
                    *queryData->singlegpuMinhasherHandles[g].get(),
                    d_seq,
                    encodedSequencePitchInInts,
                    d_len,
                    numSequences,
                    d_numValuesPerSequence_target.data(),
                    myTotalNumValues,
                    queryData->streams[g],
                    targetmr
                );

                vec_d_sequenceData2Bit_target[g].release();
                vec_d_sequenceLengths_target[g].release();

                queryData->vec_d_numValuesPerSequence.push_back(std::move(d_numValuesPerSequence_target));
            }

            //gather num values to d_numValuesPerSequencePerGpu
            for(int g = 0; g < int(deviceIds.size()); g++){
                cub::SwitchDevice sd{deviceIds[g]};

                CUDACHECK(cudaMemcpyPeerAsync(
                    d_numValuesPerSequencePerGpu.data() + numSequences * g,
                    oldDeviceId,
                    queryData->vec_d_numValuesPerSequence[g].data(),
                    deviceIds[g],
                    sizeof(int) * numSequences,
                    queryData->streams[g]
                ));
            }

            //join streams to wait for pinned_singledata and d_numValuesPerSequencePerGpu
            for(int g = 0; g < int(deviceIds.size()); g++){
                cub::SwitchDevice sd{deviceIds[g]};
                CUDACHECK(cudaStreamSynchronize(queryData->streams[g])); 
            }

            dim3 block = 128;
            dim3 grid = SDIV(numSequences, block.x);
            multigpuminhasherkernels::aggregateNumValuesPartitionResultsKernel
                    <<<grid, block, 0, stream>>>(
                d_numValuesPerSequencePerGpu.data(),
                d_numValuesPerSequence,
                numSequences,
                deviceIds.size()
            );
            CUDACHECKASYNC;

            totalNumValues = 0;
            for(int g = 0; g < int(deviceIds.size()); g++){
                totalNumValues += queryData->pinned_singledata[g];
            }
        }

        void multi_determineNumValues(
            MinhasherHandle& queryHandle,
            const std::vector<const unsigned int*>& vec_d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const std::vector<const int*>& vec_d_sequenceLengths,
            const std::vector<int>& vec_numSequences,
            const std::vector<int*>& vec_d_numValuesPerSequence,
            const std::vector<int*>& vec_totalNumValues,
            const std::vector<cudaStream_t>& callerStreams,
            const std::vector<int>& callerDeviceIds,
            const std::vector<rmm::mr::device_memory_resource*>& mrs
        ) const override{
            if(directPeerAccess){
                multi_determineNumValues_directPeerAccess(
                    queryHandle,
                    vec_d_sequenceData2Bit,
                    encodedSequencePitchInInts,
                    vec_d_sequenceLengths,
                    vec_numSequences,
                    vec_d_numValuesPerSequence,
                    vec_totalNumValues,
                    callerStreams,
                    callerDeviceIds,
                    mrs
                );
            }else{
                multi_determineNumValues_peerCopy(
                    queryHandle,
                    vec_d_sequenceData2Bit,
                    encodedSequencePitchInInts,
                    vec_d_sequenceLengths,
                    vec_numSequences,
                    vec_d_numValuesPerSequence,
                    vec_totalNumValues,
                    callerStreams,
                    callerDeviceIds,
                    mrs
                );
            }
            
        }

        // collective execution, implements collective singlegpuminhasher access
        void multi_determineNumValues_peerCopy(
            MinhasherHandle& queryHandle,
            const std::vector<const unsigned int*>& vec_d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const std::vector<const int*>& vec_d_sequenceLengths,
            const std::vector<int>& vec_numSequences,
            const std::vector<int*>& vec_d_numValuesPerSequence,
            const std::vector<int*>& vec_totalNumValues,
            const std::vector<cudaStream_t>& callerStreams,
            const std::vector<int>& callerDeviceIds,
            const std::vector<rmm::mr::device_memory_resource*>& /*mrs*/
        ) const {
            nvtx::ScopedRange sr_("multi_determineNumValues_peerCopy", 3);
            int oldDeviceId = 0;
            CUDACHECK(cudaGetDevice(&oldDeviceId));

            const int numDataGpus = deviceIds.size();
            const int numCallerGpus = callerDeviceIds.size();

            QueryData* queryData = getMultiQueryDataFromHandle(queryHandle);
            queryData->previousStage = QueryData::Stage::NumValues;
            queryData->setCallerDeviceIds(callerDeviceIds);

            int* const h_numSequencesPerCallerPS = queryData->pinned_numSequencesPerCallerPS;
            h_numSequencesPerCallerPS[0] = 0;
            for(int g = 0; g < numCallerGpus; g++){
                h_numSequencesPerCallerPS[g+1] = h_numSequencesPerCallerPS[g] + vec_numSequences[g];
            }
            const int totalNumSequencesForAllGpus = h_numSequencesPerCallerPS[numCallerGpus];

            for(int g = 0; g < numCallerGpus; g++){
                *vec_totalNumValues[g] = 0;
            }

            if(totalNumSequencesForAllGpus == 0){
                return;
            }


            std::vector<unsigned int*> vec_d_sequenceData2Bit_target;
            std::vector<int*> vec_d_sequenceLengths_target;
            std::vector<char*> vec_cubtemp_target;
            std::vector<size_t> vec_cubtempbytes_target;
            std::vector<kmer_type*> vec_d_signatures_target;
            std::vector<bool*> vec_d_hashvalues_isValid_target;
            

            //for each gpu which stores hash tables, allocate buffer to collect input sequences of all callers
            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                void* allocations[5]{};
                std::size_t allocation_sizes[5]{};
                std::size_t storage_bytes = 0;

                allocation_sizes[0] = sizeof(unsigned int) * encodedSequencePitchInInts * totalNumSequencesForAllGpus; // d_sequenceData2Bit_target
                allocation_sizes[1] = sizeof(int) * totalNumSequencesForAllGpus; // d_sequenceLengths_target

                size_t cubbytes1 = 0;
                cub::DeviceReduce::Sum(
                    nullptr,
                    cubbytes1,
                    (int*)nullptr, 
                    (int*)nullptr, 
                    totalNumSequencesForAllGpus, 
                    queryData->streams[d]
                );

                size_t cubbytes2 = 0;
                cub::DeviceSegmentedReduce::Sum(
                    nullptr,
                    cubbytes2,
                    (int*)nullptr, 
                    (int*)nullptr, 
                    totalNumSequencesForAllGpus, 
                    (int*)nullptr, 
                    (int*)nullptr, 
                    queryData->streams[d]
                );
                const size_t cubBytes = std::max(cubbytes1, cubbytes2);
                allocation_sizes[2] = cubBytes;

                const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();
                allocation_sizes[3] = sizeof(kmer_type) * totalNumSequencesForAllGpus * numHashFunctions;
                allocation_sizes[4] = sizeof(bool) * totalNumSequencesForAllGpus * numHashFunctions;


                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));
                resizeUninitialized(queryData->vec_d_targetTempBuffers[d], storage_bytes, queryData->streams[d].getStream());
                CUDACHECK(cub::AliasTemporaries(
                    queryData->vec_d_targetTempBuffers[d].data(),
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                vec_d_sequenceData2Bit_target.push_back(static_cast<unsigned int*>(allocations[0]));
                vec_d_sequenceLengths_target.push_back(static_cast<int*>(allocations[1]));
                vec_cubtemp_target.push_back(static_cast<char*>(allocations[2]));
                vec_cubtempbytes_target.push_back(cubBytes);
                vec_d_signatures_target.push_back(static_cast<kmer_type*>(allocations[3]));
                vec_d_hashvalues_isValid_target.push_back(static_cast<bool*>(allocations[4]));
            }

            // wait for async allocation
            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                CUDACHECK(cudaStreamSynchronize(queryData->streams[d]));
            }


            {
                std::vector<std::vector<const void*>> srcBuffers(numCallerGpus, std::vector<const void*>(numDataGpus));
                std::vector<std::vector<void*>> dstBuffers(numCallerGpus, std::vector<void*>(numDataGpus));
                std::vector<std::vector<size_t>> transferSizesBytes(numCallerGpus, std::vector<size_t>(numDataGpus));
                //broadcast sequences
                for(int g = 0; g < numCallerGpus; g++){
                    for(int d = 0; d < numDataGpus; d++){
                        srcBuffers[g][d] = vec_d_sequenceData2Bit[g];
                        transferSizesBytes[g][d] = sizeof(unsigned int) * encodedSequencePitchInInts * vec_numSequences[g];
                        dstBuffers[g][d] = vec_d_sequenceData2Bit_target[d] + encodedSequencePitchInInts * h_numSequencesPerCallerPS[g];
                    }
                }
                multigpu_transfer(
                    callerDeviceIds,
                    srcBuffers,
                    transferSizesBytes,
                    callerStreams,
                    deviceIds,
                    dstBuffers
                );

                //broadcast sequence lengths
                for(int g = 0; g < numCallerGpus; g++){
                    for(int d = 0; d < numDataGpus; d++){
                        srcBuffers[g][d] =  vec_d_sequenceLengths[g];
                        transferSizesBytes[g][d] = sizeof(int) * vec_numSequences[g];
                        dstBuffers[g][d] = vec_d_sequenceLengths_target[d] + h_numSequencesPerCallerPS[g];
                    }
                }

                multigpu_transfer(
                    callerDeviceIds,
                    srcBuffers,
                    transferSizesBytes,
                    callerStreams,
                    deviceIds,
                    dstBuffers
                );
            }

            //wait for transfers
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                CUDACHECK(cudaStreamSynchronize(callerStreams[g]));
            }

            //determine num values on each hash table gpu, and collect results in d_numValuesPerSequence_target

            std::vector<kmer_type*> vec_d_signatures_transposed(numDataGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerHash(numDataGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerHashExclPSVert(numDataGpus, nullptr);

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();

                void* persistent_allocations[3]{};
                std::size_t persistent_allocation_sizes[3]{};
                std::size_t persistent_storage_bytes = 0;

                persistent_allocation_sizes[0] = sizeof(kmer_type) * totalNumSequencesForAllGpus * numHashFunctions; // d_sig_trans
                persistent_allocation_sizes[1] = sizeof(int) * totalNumSequencesForAllGpus * numHashFunctions; // d_numValuesPerSequencePerHash
                persistent_allocation_sizes[2] = sizeof(int) * totalNumSequencesForAllGpus * numHashFunctions; // d_numValuesPerSequencePerHashExclPSVert

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    persistent_storage_bytes,
                    persistent_allocations,
                    persistent_allocation_sizes
                ));
                resizeUninitialized(queryData->vec_d_singlegpuminhasherBuffers[d], persistent_storage_bytes, queryData->streams[d].getStream());
                CUDACHECK(cub::AliasTemporaries(
                    queryData->vec_d_singlegpuminhasherBuffers[d].data(),
                    persistent_storage_bytes,
                    persistent_allocations,
                    persistent_allocation_sizes
                ));
                vec_d_signatures_transposed[d] = static_cast<kmer_type*>(persistent_allocations[0]);
                vec_d_numValuesPerSequencePerHash[d] = static_cast<int*>(persistent_allocations[1]);
                vec_d_numValuesPerSequencePerHashExclPSVert[d] = static_cast<int*>(persistent_allocations[2]);

                dim3 block(128,1,1);
                dim3 grid(SDIV(numHashFunctions * totalNumSequencesForAllGpus, block.x),1,1);

                gpusequencehasher::minhashSignatures3264Kernel<<<grid, block, 0, queryData->streams[d]>>>(
                    vec_d_signatures_target[d],
                    numHashFunctions,
                    vec_d_hashvalues_isValid_target[d],
                    vec_d_sequenceData2Bit_target[d],
                    encodedSequencePitchInInts,
                    totalNumSequencesForAllGpus,
                    vec_d_sequenceLengths_target[d],
                    getKmerSize(),
                    numHashFunctions,
                    sgpuMinhashers[d]->d_currentHashFunctionNumbers.data()
                ); CUDACHECKASYNC;
            }

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();
                const std::size_t signaturesRowPitchElements = numHashFunctions;

                helpers::call_transpose_kernel(
                    vec_d_signatures_transposed[d], 
                    vec_d_signatures_target[d],
                    totalNumSequencesForAllGpus, 
                    signaturesRowPitchElements, 
                    signaturesRowPitchElements,
                    queryData->streams[d]
                );
            }

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();
                fixKeysForGpuHashTable<Key, Value>(
                    vec_d_signatures_transposed[d], 
                    totalNumSequencesForAllGpus * numHashFunctions, 
                    queryData->streams[d]
                );
            }

            for(int d = 0; d < numDataGpus; d++){
                if(totalNumSequencesForAllGpus > 0){
                    CUDACHECK(cudaSetDevice(deviceIds[d]));
                    const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();

                    const int signaturesPitchInElements = totalNumSequencesForAllGpus;
                    const int numValuesPerKeyPitchInElements = totalNumSequencesForAllGpus;
                    constexpr int cgsize = SingleGpuMinhasher::GpuTable::DeviceTableView::cg_size();

                    dim3 block(256, 1, 1);
                    const int numBlocksPerTable = SDIV(totalNumSequencesForAllGpus, (block.x / cgsize));
                    dim3 grid(numBlocksPerTable, std::min(65535, numHashFunctions), 1);

                    gpuhashtablekernels::numValuesPerKeyCompactMultiTableKernel<<<grid, block, 0, queryData->streams[d]>>>(
                        sgpuMinhashers[d]->d_deviceAccessibleTableViews.data(),
                        numHashFunctions,
                        resultsPerMapThreshold,
                        vec_d_signatures_transposed[d],
                        signaturesPitchInElements,
                        totalNumSequencesForAllGpus,
                        vec_d_numValuesPerSequencePerHash[d],
                        numValuesPerKeyPitchInElements
                    );
                    CUDACHECKASYNC
                }
            }

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();

                resizeUninitialized(queryData->vec_d_numValuesPerSequence[d], totalNumSequencesForAllGpus, queryData->streams[d].getStream());

                if(totalNumSequencesForAllGpus > 0){

                    sgpuminhasherkernels::accumulateNumValuesPerSequenceKernel<<<SDIV(totalNumSequencesForAllGpus, 256), 256, 0, queryData->streams[d]>>>(
                        vec_d_numValuesPerSequencePerHash[d],
                        vec_d_numValuesPerSequencePerHashExclPSVert[d],
                        queryData->vec_d_numValuesPerSequence[d].data(),
                        totalNumSequencesForAllGpus,
                        numHashFunctions
                    );
                    CUDACHECKASYNC;
                }
            }


            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));

                cub::DeviceReduce::Sum(
                    vec_cubtemp_target[d],
                    vec_cubtempbytes_target[d],
                    queryData->vec_d_numValuesPerSequence[d].data(),
                    queryData->pinned_totalNumValuesPerTarget + d, //write to pinned memory
                    totalNumSequencesForAllGpus, 
                    queryData->streams[d]
                );    
            }




            //send num values per sequence back to caller gpus
            std::vector<int*> vec_d_numValuesPerSequencePerDataGpu(numCallerGpus, nullptr);
            std::vector<void*> vec_d_cubReduceTemp(numCallerGpus, nullptr);
            std::vector<size_t> vec_d_cubReduceTempBytes(numCallerGpus, 0);

            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                {
                    void* allocations[2]{};
                    std::size_t allocation_sizes[2]{};
                    std::size_t storage_bytes = 0;

                    allocation_sizes[0] = sizeof(int) * vec_numSequences[g] * numDataGpus; // d_numValuesPerSequencePerDataGpu
                    cub::DeviceReduce::Sum(
                        nullptr,
                        allocation_sizes[1],
                        (int*)nullptr, 
                        (int*)nullptr, 
                        vec_numSequences[g], 
                        callerStreams[g]
                    );

                    CUDACHECK(cub::AliasTemporaries(
                        nullptr,
                        storage_bytes,
                        allocations,
                        allocation_sizes
                    ));
                    resizeUninitialized(queryData->vec_d_callerTempBuffers[g], storage_bytes, callerStreams[g]);
                    CUDACHECK(cub::AliasTemporaries(
                        queryData->vec_d_callerTempBuffers[g].data(),
                        storage_bytes,
                        allocations,
                        allocation_sizes
                    ));

                    vec_d_numValuesPerSequencePerDataGpu[g] = reinterpret_cast<int*>(allocations[0]);
                    vec_d_cubReduceTemp[g] = allocations[1];
                    vec_d_cubReduceTempBytes[g] = allocation_sizes[1];
                }
            }
            //wait for async allocation
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                CUDACHECK(cudaStreamSynchronize(callerStreams[g]));
            }

            {
                std::vector<std::vector<const void*>> srcBuffers(numDataGpus, std::vector<const void*>(numCallerGpus));
                std::vector<std::vector<void*>> dstBuffers(numDataGpus, std::vector<void*>(numCallerGpus));
                std::vector<std::vector<size_t>> transferSizesBytes(numDataGpus, std::vector<size_t>(numCallerGpus));
                std::vector<cudaStream_t> srcStreams(numDataGpus);
                for(int d = 0; d < numDataGpus; d++){
                    srcStreams[d] = queryData->streams[d];
                    for(int g = 0; g < numCallerGpus; g++){
                        srcBuffers[d][g] = queryData->vec_d_numValuesPerSequence[d].data() + h_numSequencesPerCallerPS[g];
                        transferSizesBytes[d][g] = sizeof(int) * vec_numSequences[g];
                        dstBuffers[d][g] = vec_d_numValuesPerSequencePerDataGpu[g] + d * vec_numSequences[g];
                    }
                }
                multigpu_transfer(
                    deviceIds,
                    srcBuffers,
                    transferSizesBytes,
                    srcStreams,
                    callerDeviceIds,
                    dstBuffers
                );
            }

            int* const h_numValuesPerTargetPerCaller = queryData->pinned_numValuesPerTargetPerCaller;

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                cub::DeviceSegmentedReduce::Sum(
                    vec_cubtemp_target[d],
                    vec_cubtempbytes_target[d],
                    queryData->vec_d_numValuesPerSequence[d].data(),
                    h_numValuesPerTargetPerCaller + d * numCallerGpus, //write directly to pinned memory
                    numCallerGpus,
                    h_numSequencesPerCallerPS, //read from pinned memory
                    h_numSequencesPerCallerPS + 1,
                    queryData->streams[d]
                );

                // CubCallWrapper(targetmr).cubSegmentedReduceSum(
                //     queryData->vec_d_numValuesPerSequence[d].data(),
                //     h_numValuesPerTargetPerCaller + d * numCallerGpus, //write directly to pinned memory
                //     numCallerGpus,
                //     h_numSequencesPerCallerPS, //read from pinned memory
                //     h_numSequencesPerCallerPS + 1,
                //     queryData->streams[d]
                // );
            }

            //wait for transfers
            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                CUDACHECK(cudaStreamSynchronize(queryData->streams[d]));
            }
           

            //compute sum of the num results per data gpu to obtain the final numValuesPerSequence
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                if(vec_numSequences[g] > 0){
                    dim3 block = 128;
                    dim3 grid = SDIV(vec_numSequences[g], block.x);
                    multigpuminhasherkernels::aggregateNumValuesPartitionResultsKernel
                            <<<grid, block, 0, callerStreams[g]>>>(
                        vec_d_numValuesPerSequencePerDataGpu[g],
                        vec_d_numValuesPerSequence[g],
                        vec_numSequences[g],
                        numDataGpus
                    );
                    CUDACHECKASYNC;
                }
            }

            //compute total num results for each caller
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));

                if(vec_numSequences[g] > 0){
                    cub::DeviceReduce::Sum(
                        vec_d_cubReduceTemp[g],
                        vec_d_cubReduceTempBytes[g],
                        vec_d_numValuesPerSequence[g], 
                        queryData->pinned_totalNumValuesPerCaller + g,
                        vec_numSequences[g], 
                        callerStreams[g]
                    );

                    // CubCallWrapper(mrs[g]).cubReduceSum(
                    //     vec_d_numValuesPerSequence[g], 
                    //     queryData->pinned_totalNumValuesPerCaller + g,
                    //     vec_numSequences[g], 
                    //     callerStreams[g]
                    // );
                }else{
                    queryData->pinned_totalNumValuesPerCaller[g] = 0;
                }
            }

            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                //wait for pinned_totalNumValuesPerCaller
                CUDACHECK(cudaStreamSynchronize(callerStreams[g]));

                *vec_totalNumValues[g] = queryData->pinned_totalNumValuesPerCaller[g];
            }

            // {
            //     int N = 0;
            //     CUDACHECK(cudaGetDeviceCount(&N));
            //     for(int i = 0; i < N; i++){
            //         CUDACHECK(cudaSetDevice(i));
            //         CUDACHECK(cudaDeviceSynchronize());
            //     }
            // }


            CUDACHECK(cudaSetDevice(oldDeviceId));
        }

        void multi_determineNumValues_directPeerAccess(
            MinhasherHandle& queryHandle,
            const std::vector<const unsigned int*>& vec_d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const std::vector<const int*>& vec_d_sequenceLengths,
            const std::vector<int>& vec_numSequences,
            const std::vector<int*>& vec_d_numValuesPerSequence,
            const std::vector<int*>& vec_totalNumValues,
            const std::vector<cudaStream_t>& callerStreams,
            const std::vector<int>& callerDeviceIds,
            const std::vector<rmm::mr::device_memory_resource*>& /*mrs*/
        ) const {
            nvtx::ScopedRange sr_("multi_determineNumValues_directPeerAccess", 3);
            int oldDeviceId = 0;
            CUDACHECK(cudaGetDevice(&oldDeviceId));

            const int numDataGpus = deviceIds.size();
            const int numCallerGpus = callerDeviceIds.size();

            QueryData* queryData = getMultiQueryDataFromHandle(queryHandle);
            queryData->previousStage = QueryData::Stage::NumValues;
            queryData->setCallerDeviceIds(callerDeviceIds);

            int* const h_numSequencesPerCallerPS = queryData->pinned_numSequencesPerCallerPS;
            h_numSequencesPerCallerPS[0] = 0;
            for(int g = 0; g < numCallerGpus; g++){
                h_numSequencesPerCallerPS[g+1] = h_numSequencesPerCallerPS[g] + vec_numSequences[g];
            }
            const int totalNumSequencesForAllGpus = h_numSequencesPerCallerPS[numCallerGpus];

            for(int g = 0; g < numCallerGpus; g++){
                *vec_totalNumValues[g] = 0;
            }

            if(totalNumSequencesForAllGpus == 0){
                return;
            }



            std::vector<unsigned int*> vec_d_sequenceData2Bit_target;
            std::vector<int*> vec_d_sequenceLengths_target;
            std::vector<char*> vec_cubtemp_target;
            std::vector<size_t> vec_cubtempbytes_target;
            std::vector<kmer_type*> vec_d_signatures_target;
            std::vector<bool*> vec_d_hashvalues_isValid_target;

            //for each gpu which stores hash tables, allocate buffer to collect input sequences of all callers
            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                void* allocations[5]{};
                std::size_t allocation_sizes[5]{};
                std::size_t storage_bytes = 0;

                allocation_sizes[0] = sizeof(unsigned int) * encodedSequencePitchInInts * totalNumSequencesForAllGpus; // d_sequenceData2Bit_target
                allocation_sizes[1] = sizeof(int) * totalNumSequencesForAllGpus; // d_sequenceLengths_target

                size_t cubbytes1 = 0;
                cub::DeviceReduce::Sum(
                    nullptr,
                    cubbytes1,
                    (int*)nullptr, 
                    (int*)nullptr, 
                    totalNumSequencesForAllGpus, 
                    queryData->streams[d]
                );

                size_t cubbytes2 = 0;
                cub::DeviceSegmentedReduce::Sum(
                    nullptr,
                    cubbytes2,
                    (int*)nullptr, 
                    (int*)nullptr, 
                    totalNumSequencesForAllGpus, 
                    (int*)nullptr, 
                    (int*)nullptr, 
                    queryData->streams[d]
                );
                const size_t cubBytes = std::max(cubbytes1, cubbytes2);
                allocation_sizes[2] = cubBytes;

                const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();
                allocation_sizes[3] = sizeof(kmer_type) * totalNumSequencesForAllGpus * numHashFunctions;
                allocation_sizes[4] = sizeof(bool) * totalNumSequencesForAllGpus * numHashFunctions;


                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));
                resizeUninitialized(queryData->vec_d_targetTempBuffers[d], storage_bytes, queryData->streams[d].getStream());

                CUDACHECK(cub::AliasTemporaries(
                    queryData->vec_d_targetTempBuffers[d].data(),
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                vec_d_sequenceData2Bit_target.push_back(static_cast<unsigned int*>(allocations[0]));
                vec_d_sequenceLengths_target.push_back(static_cast<int*>(allocations[1]));
                vec_cubtemp_target.push_back(static_cast<char*>(allocations[2]));
                vec_cubtempbytes_target.push_back(cubBytes);
                vec_d_signatures_target.push_back(static_cast<kmer_type*>(allocations[3]));
                vec_d_hashvalues_isValid_target.push_back(static_cast<bool*>(allocations[4]));
            }

            // wait for async allocation
            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                CUDACHECK(cudaStreamSynchronize(queryData->streams[d]));
            }

            //send sequences
            for(int g = 0; g < numCallerGpus; g++){
                if(vec_numSequences[g] > 0){
                    CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                    unsigned int** const h_dstSequencesPointers = reinterpret_cast<unsigned int**>(queryData->pinned_bufferpointers + g * numDataGpus);
                    for(int d = 0; d < numDataGpus; d++){
                        h_dstSequencesPointers[(g+d) % numDataGpus] = vec_d_sequenceData2Bit_target[d] + encodedSequencePitchInInts * h_numSequencesPerCallerPS[g];
                    }

                    dim3 grid1(SDIV(encodedSequencePitchInInts * vec_numSequences[g], 128), numDataGpus);
                    multigpuminhasherkernels::scatterFullArrayKernel<<<grid1, 128, 0, callerStreams[g]>>>(
                        vec_d_sequenceData2Bit[g],
                        encodedSequencePitchInInts * vec_numSequences[g],
                        h_dstSequencesPointers,
                        numDataGpus
                    );
                    CUDACHECKASYNC;
                }
            }

            //send sequence lengths
            for(int g = 0; g < numCallerGpus; g++){
                if(vec_numSequences[g] > 0){
                    CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                    int** const h_dstLengthPointers = reinterpret_cast<int**>(queryData->pinned_bufferpointers2 + g * numDataGpus);
                    for(int d = 0; d < numDataGpus; d++){
                        h_dstLengthPointers[(g+d) % numDataGpus] = vec_d_sequenceLengths_target[d] + h_numSequencesPerCallerPS[g];
                    }

                    dim3 grid2(SDIV(vec_numSequences[g], 128), numDataGpus);
                    multigpuminhasherkernels::scatterFullArrayKernel<<<grid2, 128, 0, callerStreams[g]>>>(
                        vec_d_sequenceLengths[g],
                        vec_numSequences[g],
                        h_dstLengthPointers,
                        numDataGpus
                    );
                    CUDACHECKASYNC;
                }
            }

            std::vector<int*> vec_d_numValuesPerSequencePerDataGpu(numCallerGpus, nullptr);
            std::vector<void*> vec_d_cubReduceTemp(numCallerGpus, nullptr);
            std::vector<size_t> vec_d_cubReduceTempBytes(numCallerGpus, 0);

            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                {
                    void* allocations[2]{};
                    std::size_t allocation_sizes[2]{};
                    std::size_t storage_bytes = 0;

                    allocation_sizes[0] = sizeof(int) * vec_numSequences[g] * numDataGpus; // d_numValuesPerSequencePerDataGpu
                    cub::DeviceReduce::Sum(
                        nullptr,
                        allocation_sizes[1],
                        (int*)nullptr, 
                        (int*)nullptr, 
                        vec_numSequences[g], 
                        callerStreams[g]
                    );

                    CUDACHECK(cub::AliasTemporaries(
                        nullptr,
                        storage_bytes,
                        allocations,
                        allocation_sizes
                    ));
                    resizeUninitialized(queryData->vec_d_callerTempBuffers[g], storage_bytes, callerStreams[g]);
                    CUDACHECK(cub::AliasTemporaries(
                        queryData->vec_d_callerTempBuffers[g].data(),
                        storage_bytes,
                        allocations,
                        allocation_sizes
                    ));

                    vec_d_numValuesPerSequencePerDataGpu[g] = reinterpret_cast<int*>(allocations[0]);
                    vec_d_cubReduceTemp[g] = allocations[1];
                    vec_d_cubReduceTempBytes[g] = allocation_sizes[1];
                }
            }


            //wait for transfers and allocations
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                CUDACHECK(cudaStreamSynchronize(callerStreams[g]));
            }


            //determine num values on each hash table gpu, and collect results in d_numValuesPerSequence_target

            std::vector<kmer_type*> vec_d_signatures_transposed(numDataGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerHash(numDataGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerHashExclPSVert(numDataGpus, nullptr);

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();

                void* persistent_allocations[3]{};
                std::size_t persistent_allocation_sizes[3]{};
                std::size_t persistent_storage_bytes = 0;

                persistent_allocation_sizes[0] = sizeof(kmer_type) * totalNumSequencesForAllGpus * numHashFunctions; // d_sig_trans
                persistent_allocation_sizes[1] = sizeof(int) * totalNumSequencesForAllGpus * numHashFunctions; // d_numValuesPerSequencePerHash
                persistent_allocation_sizes[2] = sizeof(int) * totalNumSequencesForAllGpus * numHashFunctions; // d_numValuesPerSequencePerHashExclPSVert

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    persistent_storage_bytes,
                    persistent_allocations,
                    persistent_allocation_sizes
                ));
                resizeUninitialized(queryData->vec_d_singlegpuminhasherBuffers[d], persistent_storage_bytes, queryData->streams[d].getStream());
                CUDACHECK(cub::AliasTemporaries(
                    queryData->vec_d_singlegpuminhasherBuffers[d].data(),
                    persistent_storage_bytes,
                    persistent_allocations,
                    persistent_allocation_sizes
                ));
                vec_d_signatures_transposed[d] = static_cast<kmer_type*>(persistent_allocations[0]);
                vec_d_numValuesPerSequencePerHash[d] = static_cast<int*>(persistent_allocations[1]);
                vec_d_numValuesPerSequencePerHashExclPSVert[d] = static_cast<int*>(persistent_allocations[2]);


                dim3 block(128,1,1);
                dim3 grid(SDIV(numHashFunctions * totalNumSequencesForAllGpus, block.x),1,1);

                gpusequencehasher::minhashSignatures3264Kernel<<<grid, block, 0, queryData->streams[d]>>>(
                    vec_d_signatures_target[d],
                    numHashFunctions,
                    vec_d_hashvalues_isValid_target[d],
                    vec_d_sequenceData2Bit_target[d],
                    encodedSequencePitchInInts,
                    totalNumSequencesForAllGpus,
                    vec_d_sequenceLengths_target[d],
                    getKmerSize(),
                    numHashFunctions,
                    sgpuMinhashers[d]->d_currentHashFunctionNumbers.data()
                ); CUDACHECKASYNC;
            }
            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();
                const std::size_t signaturesRowPitchElements = numHashFunctions;

                helpers::call_transpose_kernel(
                    vec_d_signatures_transposed[d], 
                    vec_d_signatures_target[d],
                    totalNumSequencesForAllGpus, 
                    signaturesRowPitchElements, 
                    signaturesRowPitchElements,
                    queryData->streams[d]
                );
            }

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();
                fixKeysForGpuHashTable<Key, Value>(
                    vec_d_signatures_transposed[d], 
                    totalNumSequencesForAllGpus * numHashFunctions, 
                    queryData->streams[d]
                );
            }

            for(int d = 0; d < numDataGpus; d++){
                if(totalNumSequencesForAllGpus > 0){
                    CUDACHECK(cudaSetDevice(deviceIds[d]));
                    const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();

                    const int signaturesPitchInElements = totalNumSequencesForAllGpus;
                    const int numValuesPerKeyPitchInElements = totalNumSequencesForAllGpus;
                    constexpr int cgsize = SingleGpuMinhasher::GpuTable::DeviceTableView::cg_size();

                    dim3 block(256, 1, 1);
                    const int numBlocksPerTable = SDIV(totalNumSequencesForAllGpus, (block.x / cgsize));
                    dim3 grid(numBlocksPerTable, std::min(65535, numHashFunctions), 1);

                    gpuhashtablekernels::numValuesPerKeyCompactMultiTableKernel<<<grid, block, 0, queryData->streams[d]>>>(
                        sgpuMinhashers[d]->d_deviceAccessibleTableViews.data(),
                        numHashFunctions,
                        resultsPerMapThreshold,
                        vec_d_signatures_transposed[d],
                        signaturesPitchInElements,
                        totalNumSequencesForAllGpus,
                        vec_d_numValuesPerSequencePerHash[d],
                        numValuesPerKeyPitchInElements
                    );
                    CUDACHECKASYNC
                }
            }

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();
                resizeUninitialized(queryData->vec_d_numValuesPerSequence[d], totalNumSequencesForAllGpus, queryData->streams[d].getStream());

                if(totalNumSequencesForAllGpus > 0){

                    sgpuminhasherkernels::accumulateNumValuesPerSequenceKernel<<<SDIV(totalNumSequencesForAllGpus, 256), 256, 0, queryData->streams[d]>>>(
                        vec_d_numValuesPerSequencePerHash[d],
                        vec_d_numValuesPerSequencePerHashExclPSVert[d],
                        queryData->vec_d_numValuesPerSequence[d].data(),
                        totalNumSequencesForAllGpus,
                        numHashFunctions
                    );
                    CUDACHECKASYNC;
                }
            }

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                // CubCallWrapper(rmm::mr::get_current_device_resource()).cubReduceSum(
                //     queryData->vec_d_numValuesPerSequence[d].data(),
                //     queryData->pinned_totalNumValuesPerTarget + d, //write to pinned memory
                //     totalNumSequencesForAllGpus, 
                //     queryData->streams[d]
                // );
                cub::DeviceReduce::Sum(
                    vec_cubtemp_target[d],
                    vec_cubtempbytes_target[d],
                    queryData->vec_d_numValuesPerSequence[d].data(),
                    queryData->pinned_totalNumValuesPerTarget + d, //write to pinned memory
                    totalNumSequencesForAllGpus, 
                    queryData->streams[d]
                );    
            }

            //send num values per sequence back to caller gpus

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                int** const h_dstPointers = reinterpret_cast<int**>(queryData->pinned_bufferpointers + d * numCallerGpus);
                for(int g = 0; g < numCallerGpus; g++){
                    h_dstPointers[g] = vec_d_numValuesPerSequencePerDataGpu[g] + d * vec_numSequences[g];
                }

                int* const h_offsets = reinterpret_cast<int*>(queryData->pinned_bufferpointers2 + d * numCallerGpus);
                for(int g = 0; g < numCallerGpus+1; g++){
                    h_offsets[g] = h_numSequencesPerCallerPS[g];
                }

                dim3 grid1(SDIV(vec_numSequences[d], 128), numCallerGpus);

                multigpuminhasherkernels::scatterSegmentedArrayKernel<<<grid1, 128, 0,  queryData->streams[d]>>>(
                    queryData->vec_d_numValuesPerSequence[d].data(),
                    h_offsets,
                    h_offsets + 1,
                    h_dstPointers,
                    numCallerGpus
                );
            }
            int* const h_numValuesPerTargetPerCaller = queryData->pinned_numValuesPerTargetPerCaller;

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                cub::DeviceSegmentedReduce::Sum(
                    vec_cubtemp_target[d],
                    vec_cubtempbytes_target[d],
                    queryData->vec_d_numValuesPerSequence[d].data(),
                    h_numValuesPerTargetPerCaller + d * numCallerGpus, //write directly to pinned memory
                    numCallerGpus,
                    h_numSequencesPerCallerPS, //read from pinned memory
                    h_numSequencesPerCallerPS + 1,
                    queryData->streams[d]
                );

                // CubCallWrapper(targetmr).cubSegmentedReduceSum(
                //     queryData->vec_d_numValuesPerSequence[d].data(),
                //     h_numValuesPerTargetPerCaller + d * numCallerGpus, //write directly to pinned memory
                //     numCallerGpus,
                //     h_numSequencesPerCallerPS, //read from pinned memory
                //     h_numSequencesPerCallerPS + 1,
                //     queryData->streams[d]
                // );
            }

            //wait for transfers
            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                CUDACHECK(cudaStreamSynchronize(queryData->streams[d]));
            }
           

            //compute sum of the num results per data gpu to obtain the final numValuesPerSequence
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                if(vec_numSequences[g] > 0){
                    dim3 block = 128;
                    dim3 grid = SDIV(vec_numSequences[g], block.x);
                    multigpuminhasherkernels::aggregateNumValuesPartitionResultsKernel
                            <<<grid, block, 0, callerStreams[g]>>>(
                        vec_d_numValuesPerSequencePerDataGpu[g],
                        vec_d_numValuesPerSequence[g],
                        vec_numSequences[g],
                        numDataGpus
                    );
                    CUDACHECKASYNC;
                }
            }

            //compute total num results for each caller
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));

                if(vec_numSequences[g] > 0){
                    cub::DeviceReduce::Sum(
                        vec_d_cubReduceTemp[g],
                        vec_d_cubReduceTempBytes[g],
                        vec_d_numValuesPerSequence[g], 
                        queryData->pinned_totalNumValuesPerCaller + g,
                        vec_numSequences[g], 
                        callerStreams[g]
                    );
                }else{
                    queryData->pinned_totalNumValuesPerCaller[g] = 0;
                }
            }

            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                //wait for pinned_totalNumValuesPerCaller
                CUDACHECK(cudaStreamSynchronize(callerStreams[g]));

                *vec_totalNumValues[g] = queryData->pinned_totalNumValuesPerCaller[g];
            }


            CUDACHECK(cudaSetDevice(oldDeviceId));
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
        ) const override{
            QueryData* const queryData = getQueryDataFromHandle(queryHandle);
            assert(queryData->previousStage == QueryData::Stage::NumValues);
            queryData->previousStage = QueryData::Stage::Retrieve;

            if(numSequences == 0){
                CUDACHECK(cudaMemsetAsync(d_offsets, 0, sizeof(int) * (numSequences + 1), stream));
                return;
            }

            int oldDeviceId = 0;
            CUDACHECK(cudaGetDevice(&oldDeviceId));

            rmm::device_uvector<read_number> d_allValues(totalNumValues, stream, mr);
            rmm::device_uvector<int> d_numValuesPerSequencePerGpu(numSequences * deviceIds.size(), stream, mr);
            rmm::device_uvector<int> d_offsetsPerSequencePerGpu((numSequences+1) * deviceIds.size(), stream, mr);

            CUDACHECK(cudaEventRecord(queryData->callerEvent, stream));

            CubCallWrapper(mr).cubInclusiveSum(
                d_numValuesPerSequence,
                d_offsets + 1,
                numSequences,
                stream
            );
            CUDACHECK(cudaMemsetAsync(d_offsets, 0, sizeof(int), stream));

            std::vector<rmm::device_uvector<read_number>> vec_d_values_target;
            std::vector<rmm::device_uvector<int>> vec_d_offsets_target;

            for(int g = 0; g < int(deviceIds.size()); g++){
                cub::SwitchDevice sd{deviceIds[g]};

                CUDACHECK(cudaStreamWaitEvent(queryData->streams[g], queryData->callerEvent, 0));

                const int totalNumValuesTarget = queryData->pinned_singledata[g];

                auto* targetmr = rmm::mr::get_current_device_resource();
                rmm::device_uvector<read_number> d_values_target(totalNumValuesTarget, queryData->streams[g].getStream(), targetmr);
                rmm::device_uvector<int> d_offsets_target(numSequences + 1, queryData->streams[g].getStream(), targetmr);

                sgpuMinhashers[g]->retrieveValues(
                    *queryData->singlegpuMinhasherHandles[g].get(),
                    numSequences,
                    totalNumValuesTarget,
                    d_values_target.data(),
                    queryData->vec_d_numValuesPerSequence[g].data(),
                    d_offsets_target.data(), //numSequences + 1
                    queryData->streams[g],
                    targetmr
                );

                vec_d_values_target.push_back(std::move(d_values_target));
                vec_d_offsets_target.push_back(std::move(d_offsets_target));
            }

            int* h_gatherOffsets = queryData->pinned_singledata + deviceIds.size();
            std::exclusive_scan(
                queryData->pinned_singledata, 
                queryData->pinned_singledata + deviceIds.size(), 
                h_gatherOffsets, 
                0
            );

            //gather results from targets
            for(int g = 0; g < int(deviceIds.size()); g++){
                cub::SwitchDevice sd{deviceIds[g]};
                const int totalNumValuesTarget = queryData->pinned_singledata[g];

                CUDACHECK(cudaMemcpyPeerAsync(
                    d_allValues.data() + h_gatherOffsets[g],
                    oldDeviceId,
                    vec_d_values_target[g].data(),
                    deviceIds[g],
                    sizeof(read_number) * totalNumValuesTarget,
                    queryData->streams[g]
                ));
                vec_d_values_target[g].release();
            }

            for(int g = 0; g < int(deviceIds.size()); g++){
                cub::SwitchDevice sd{deviceIds[g]};

                CUDACHECK(cudaMemcpyPeerAsync(
                    d_numValuesPerSequencePerGpu.data() + numSequences * g,
                    oldDeviceId,
                    queryData->vec_d_numValuesPerSequence[g].data(),
                    deviceIds[g],
                    sizeof(int) * numSequences,
                    queryData->streams[g]
                ));
                queryData->vec_d_numValuesPerSequence[g].release();
            }

            for(int g = 0; g < int(deviceIds.size()); g++){
                cub::SwitchDevice sd{deviceIds[g]};

                CUDACHECK(cudaMemcpyPeerAsync(
                    d_offsetsPerSequencePerGpu.data() + (numSequences+1) * g,
                    oldDeviceId,
                    vec_d_offsets_target[g].data(),
                    deviceIds[g],
                    sizeof(int) * (numSequences+1),
                    queryData->streams[g]
                ));
                vec_d_offsets_target[g].release();
            }

            rmm::device_uvector<int> d_gatherOffsets(deviceIds.size(), stream, mr);
            CUDACHECK(cudaMemcpyAsync(
                d_gatherOffsets.data(), 
                h_gatherOffsets, 
                sizeof(int) * deviceIds.size(), 
                H2D, 
                stream
            ));

            //join per-gpu streams to caller stream to wait for gathered results
            for(int g = 0; g < int(deviceIds.size()); g++){
                cub::SwitchDevice sd{deviceIds[g]};
                CUDACHECK(cudaEventRecord(queryData->events[g], queryData->streams[g]));
            }
            for(int g = 0; g < int(deviceIds.size()); g++){
                CUDACHECK(cudaStreamWaitEvent(stream, queryData->events[g]));
            }

            //copy values to output array, interleave results for same sequence
            multigpuminhasherkernels::copyToInterleavedKernel<<<numSequences, 128, 0, stream>>>(
                d_allValues.data(),
                d_offsetsPerSequencePerGpu.data(),
                d_numValuesPerSequencePerGpu.data(),
                d_gatherOffsets.data(),
                d_offsets,
                d_values,
                numSequences,
                deviceIds.size()
            ); CUDACHECKASYNC


            queryData->vec_d_numValuesPerSequence.clear();
        }

        void multi_retrieveValues(
            MinhasherHandle& queryHandle,
            const std::vector<int>& vec_numSequences,
            const std::vector<const int*>& vec_totalNumValues,
            const std::vector<read_number*>& vec_d_values,
            const std::vector<const int*>& vec_d_numValuesPerSequence,
            const std::vector<int*> vec_d_offsets, //numSequences + 1
            const std::vector<cudaStream_t>& streams,
            const std::vector<int>& callerDeviceIds,
            const std::vector<rmm::mr::device_memory_resource*>& mrs
        ) const override{
            if(directPeerAccess){
                multi_retrieveValues_directPeerAccess(
                    queryHandle,
                    vec_numSequences,
                    vec_totalNumValues,
                    vec_d_values,
                    vec_d_numValuesPerSequence,
                    vec_d_offsets,
                    streams,
                    callerDeviceIds,
                    mrs
                );
            }else{
                multi_retrieveValues_peerCopy(
                    queryHandle,
                    vec_numSequences,
                    vec_totalNumValues,
                    vec_d_values,
                    vec_d_numValuesPerSequence,
                    vec_d_offsets,
                    streams,
                    callerDeviceIds,
                    mrs
                );
            }
        }

        // collective execution, implements collective singlegpuminhasher access
        void multi_retrieveValues_peerCopy(
            MinhasherHandle& queryHandle,
            const std::vector<int>& vec_numSequences,
            const std::vector<const int*>& vec_totalNumValues,
            const std::vector<read_number*>& vec_d_values,
            const std::vector<const int*>& vec_d_numValuesPerSequence,
            const std::vector<int*> vec_d_offsets, //numSequences + 1
            const std::vector<cudaStream_t>& streams,
            const std::vector<int> callerDeviceIds,
            const std::vector<rmm::mr::device_memory_resource*>& /*mrs*/
        ) const {
            nvtx::ScopedRange sr_("multi_retrieveValues_peerCopy", 3);

            int oldDeviceId = 0;
            CUDACHECK(cudaGetDevice(&oldDeviceId));

            QueryData* const queryData = getMultiQueryDataFromHandle(queryHandle);
            assert(queryData->previousStage == QueryData::Stage::NumValues);
            assert(queryData->callerDeviceIds == callerDeviceIds);
            queryData->previousStage = QueryData::Stage::Retrieve;

            const int numDataGpus = deviceIds.size();
            const int numCallerGpus = callerDeviceIds.size();

            for(int g = 0; g < numCallerGpus; g++){
                if(vec_numSequences[g] == 0){
                    CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                    CUDACHECK(cudaMemsetAsync(vec_d_offsets[g], 0, sizeof(int) * (vec_numSequences[g] + 1), streams[g]));
                }
            }

            const int* const h_numSequencesPerCallerPS = queryData->pinned_numSequencesPerCallerPS;
            const int totalNumSequencesForAllGpus = h_numSequencesPerCallerPS[numCallerGpus];
            if(totalNumSequencesForAllGpus == 0){
                return;
            }

            //compute num values per caller per target

            int* h_numValuesPerTargetPerCaller = queryData->pinned_numValuesPerTargetPerCaller;

            // for(int d = 0; d < numDataGpus; d++){
            //     CUDACHECK(cudaSetDevice(deviceIds[d]));
            //     auto* targetmr = rmm::mr::get_current_device_resource();

            //     CubCallWrapper(targetmr).cubSegmentedReduceSum(
            //         queryData->vec_d_numValuesPerSequence[d].data(),
            //         h_numValuesPerTargetPerCaller + d * numCallerGpus, //write directly to pinned memory
            //         numCallerGpus,
            //         h_numSequencesPerCallerPS, //read from pinned memory
            //         h_numSequencesPerCallerPS + 1,
            //         queryData->streams[d]
            //     );
            //     CUDACHECK(cudaEventRecord(queryData->events[d], queryData->streams[d]));
            // }

            std::vector<read_number*> vec_d_allValues(numCallerGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerGpu(numCallerGpus, nullptr);
            std::vector<int*> vec_d_offsetsPerSequencePerGpu(numCallerGpus, nullptr);

            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));

                void* allocations[4]{};
                std::size_t allocation_sizes[4]{};
                std::size_t storage_bytes = 0;

                constexpr int roundUpTo = 10000;
                const int roundedTotalNumValues = SDIV((*vec_totalNumValues[g]), roundUpTo) * roundUpTo;

                allocation_sizes[0] = sizeof(read_number) * roundedTotalNumValues;
                allocation_sizes[1] = sizeof(int) * vec_numSequences[g] * numDataGpus;
                allocation_sizes[2] = sizeof(int) * (vec_numSequences[g]+1) * numDataGpus;

                CUDACHECK(cub::DeviceScan::InclusiveSum(
                    nullptr, 
                    allocation_sizes[3], 
                    vec_d_numValuesPerSequence[g], 
                    vec_d_offsets[g] + 1, 
                    vec_numSequences[g],
                    streams[g]
                ));

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                resizeUninitialized(queryData->vec_d_callerTempBuffers[g], storage_bytes, streams[g]);
                CUDACHECK(cub::AliasTemporaries(
                    queryData->vec_d_callerTempBuffers[g].data(),
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                vec_d_allValues[g] = reinterpret_cast<read_number*>(allocations[0]);
                vec_d_numValuesPerSequencePerGpu[g] = reinterpret_cast<int*>(allocations[1]);
                vec_d_offsetsPerSequencePerGpu[g] = reinterpret_cast<int*>(allocations[2]);

                //wait for the allocation so it can be used later for transfer in different stream
                CUDACHECK(cudaStreamSynchronize(streams[g]));

                CUDACHECK(cudaMemsetAsync(vec_d_offsets[g], 0, sizeof(int), streams[g]));
                CUDACHECK(cub::DeviceScan::InclusiveSum(
                    allocations[3], 
                    allocation_sizes[3], 
                    vec_d_numValuesPerSequence[g], 
                    vec_d_offsets[g] + 1, 
                    vec_numSequences[g],
                    streams[g]
                ));

            }
            
            std::vector<kmer_type*> vec_d_signatures_transposed(numDataGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerHash(numDataGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerHashExclPSVert(numDataGpus, nullptr);

            std::vector<read_number*> vec_d_values_target(numDataGpus, nullptr);
            std::vector<int*> vec_d_offsets_target(numDataGpus, nullptr);
            std::vector<int*> vec_d_queryOffsetsPerSequencePerHash(numDataGpus, nullptr);

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();

                auto* targetmr = rmm::mr::get_current_device_resource();

                {
                    void* persistent_allocations[3]{};
                    std::size_t persistent_allocation_sizes[3]{};
                    std::size_t persistent_storage_bytes = queryData->vec_d_singlegpuminhasherBuffers[d].size();

                    persistent_allocation_sizes[0] = sizeof(kmer_type) * totalNumSequencesForAllGpus * numHashFunctions; // d_sig_trans
                    persistent_allocation_sizes[1] = sizeof(int) * totalNumSequencesForAllGpus * numHashFunctions; // d_numValuesPerSequencePerHash
                    persistent_allocation_sizes[2] = sizeof(int) * totalNumSequencesForAllGpus * numHashFunctions; // d_numValuesPerSequencePerHashExclPSVert

                    CUDACHECK(cub::AliasTemporaries(
                        queryData->vec_d_singlegpuminhasherBuffers[d].data(),
                        persistent_storage_bytes,
                        persistent_allocations,
                        persistent_allocation_sizes
                    ));
                    vec_d_signatures_transposed[d] = static_cast<kmer_type*>(persistent_allocations[0]);
                    vec_d_numValuesPerSequencePerHash[d] = static_cast<int*>(persistent_allocations[1]);
                    vec_d_numValuesPerSequencePerHashExclPSVert[d] = static_cast<int*>(persistent_allocations[2]);
                }


                void* allocations[4]{};
                std::size_t allocation_sizes[4]{};
                std::size_t storage_bytes = 0;

                const int totalNumValuesTarget = queryData->pinned_totalNumValuesPerTarget[d];
                constexpr int roundUpTo = 10000;
                const int roundedTotalNumValuesTarget = SDIV(totalNumValuesTarget, roundUpTo) * roundUpTo;

                allocation_sizes[0] = sizeof(read_number) * roundedTotalNumValuesTarget; // d_values_target
                allocation_sizes[1] = sizeof(int) * (totalNumSequencesForAllGpus + 1); // d_offsets_target
                allocation_sizes[2] = sizeof(int) * totalNumSequencesForAllGpus * numHashFunctions; // d_queryOffsetsPerSequencePerHash
                CUDACHECK(cub::DeviceScan::InclusiveSum(
                    nullptr, 
                    allocation_sizes[3], 
                    queryData->vec_d_numValuesPerSequence[d].data(),
                    vec_d_offsets_target[d] + 1,
                    totalNumSequencesForAllGpus,
                    queryData->streams[d]
                ));

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));
                resizeUninitialized(queryData->vec_d_targetTempBuffers[d], storage_bytes, queryData->streams[d].getStream());
                CUDACHECK(cub::AliasTemporaries(
                    queryData->vec_d_targetTempBuffers[d].data(),
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                vec_d_values_target[d] = static_cast<read_number*>(allocations[0]);
                vec_d_offsets_target[d] = static_cast<int*>(allocations[1]);
                vec_d_queryOffsetsPerSequencePerHash[d] = static_cast<int*>(allocations[2]);


                CUDACHECK(cudaMemsetAsync(vec_d_offsets_target[d], 0, sizeof(int), queryData->streams[d]));

                CUDACHECK(cub::DeviceScan::InclusiveSum(
                    allocations[3], 
                    allocation_sizes[3], 
                    queryData->vec_d_numValuesPerSequence[d].data(),
                    vec_d_offsets_target[d] + 1,
                    totalNumSequencesForAllGpus,
                    queryData->streams[d]
                ));
            }

            for(int d = 0; d < numDataGpus; d++){
                if(totalNumSequencesForAllGpus > 0){
                    CUDACHECK(cudaSetDevice(deviceIds[d]));
                    const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();

                    sgpuminhasherkernels::computeQueryDestinationOffsetsKernel<<<SDIV(totalNumSequencesForAllGpus, 256), 256, 0, queryData->streams[d]>>>(
                        vec_d_queryOffsetsPerSequencePerHash[d],
                        vec_d_numValuesPerSequencePerHashExclPSVert[d],
                        totalNumSequencesForAllGpus,
                        numHashFunctions,
                        vec_d_offsets_target[d]
                    );
                    CUDACHECKASYNC
                }
            }

            for(int d = 0; d < numDataGpus; d++){
                if(totalNumSequencesForAllGpus > 0){
                    CUDACHECK(cudaSetDevice(deviceIds[d]));
                    const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();

                    const int signaturesPitchInElements = totalNumSequencesForAllGpus;
                    const int numValuesPerKeyPitchInElements = totalNumSequencesForAllGpus;
                    const int beginOffsetsPitchInElements = totalNumSequencesForAllGpus;
                    constexpr int cgsize = SingleGpuMinhasher::GpuTable::DeviceTableView::cg_size();

                    dim3 block(256, 1, 1);
                    const int numBlocksPerTable = SDIV(totalNumSequencesForAllGpus, (block.x / cgsize));
                    dim3 grid(numBlocksPerTable, std::min(65535, numHashFunctions), 1);

                    gpuhashtablekernels::retrieveCompactKernel<<<grid, block, 0, queryData->streams[d]>>>(
                        sgpuMinhashers[d]->d_deviceAccessibleTableViews.data(),
                        numHashFunctions,
                        vec_d_signatures_transposed[d],
                        signaturesPitchInElements,
                        vec_d_queryOffsetsPerSequencePerHash[d],
                        beginOffsetsPitchInElements,
                        vec_d_numValuesPerSequencePerHash[d],
                        numValuesPerKeyPitchInElements,
                        resultsPerMapThreshold,
                        totalNumSequencesForAllGpus,
                        vec_d_values_target[d]
                    );
                    CUDACHECKASYNC
                    //CUDACHECK(cudaEventRecord(queryData->events2[d], queryData->streams[d]));
                }
            }




            //wait for h_numValuesPerTargetPerCaller
            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                CUDACHECK(cudaEventSynchronize(queryData->events[d]));
            }

            std::vector<std::vector<int>> numValuesPerTargetPerCaller_horizontalPS(numDataGpus, std::vector<int>(numCallerGpus+1, 0));
            std::vector<std::vector<int>> numValuesPerTargetPerCaller_verticalPS(numDataGpus+1, std::vector<int>(numCallerGpus, 0));

            for(int d = 0; d < numDataGpus; d++){
                for(int g = 0; g < numCallerGpus; g++){
                    numValuesPerTargetPerCaller_horizontalPS[d][g+1] 
                        = numValuesPerTargetPerCaller_horizontalPS[d][g] + h_numValuesPerTargetPerCaller[d * numCallerGpus + g];
                }
            }
            for(int g = 0; g < numCallerGpus; g++){
                for(int d = 0; d < numDataGpus; d++){
                    numValuesPerTargetPerCaller_verticalPS[d+1][g] 
                        = numValuesPerTargetPerCaller_verticalPS[d][g] + h_numValuesPerTargetPerCaller[d * numCallerGpus + g];
                }
            }
            for(int g = 0; g < numCallerGpus; g++){
                assert(*vec_totalNumValues[g] == numValuesPerTargetPerCaller_verticalPS[numDataGpus][g]);
            }

            // std::cout << "retrieve numValuesPerTargetPerCaller\n";
            // for(int d = 0; d < numDataGpus; d++){
            //     for(int g = 0; g < numCallerGpus; g++){
            //         std::cout << h_numValuesPerTargetPerCaller[d * numCallerGpus + g] << " ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "retrieve horizontal ps\n";
            // for(int d = 0; d < numDataGpus; d++){
            //     for(int g = 0; g < numCallerGpus+1; g++){
            //         std::cout << numValuesPerTargetPerCaller_horizontalPS[d][g] << " ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "retrieve vertical ps\n";
            // for(int d = 0; d < numDataGpus+1; d++){
            //     for(int g = 0; g < numCallerGpus; g++){
            //         std::cout << numValuesPerTargetPerCaller_verticalPS[d][g] << " ";
            //     }
            //     std::cout << "\n";
            // }


            {
                std::vector<std::vector<const void*>> srcBuffers(numDataGpus, std::vector<const void*>(numCallerGpus));
                std::vector<std::vector<void*>> dstBuffers(numDataGpus, std::vector<void*>(numCallerGpus));
                std::vector<std::vector<size_t>> transferSizesBytes(numDataGpus, std::vector<size_t>(numCallerGpus));
                std::vector<cudaStream_t> srcStreams(numDataGpus);
                //send values
                for(int d = 0; d < numDataGpus; d++){
                    srcStreams[d] = queryData->streams[d];
                    for(int g = 0; g < numCallerGpus; g++){
                        srcBuffers[d][g] = vec_d_values_target[d] + numValuesPerTargetPerCaller_horizontalPS[d][g];
                        transferSizesBytes[d][g] = sizeof(read_number) * h_numValuesPerTargetPerCaller[d * numCallerGpus + g];
                        dstBuffers[d][g] = vec_d_allValues[g] + numValuesPerTargetPerCaller_verticalPS[d][g];
                    }
                }
                multigpu_transfer(
                    deviceIds,
                    srcBuffers,
                    transferSizesBytes,
                    srcStreams,
                    callerDeviceIds,
                    dstBuffers
                );

                //send numValuesPerSequence
                for(int d = 0; d < numDataGpus; d++){
                    srcStreams[d] = queryData->streams[d];
                    for(int g = 0; g < numCallerGpus; g++){
                        srcBuffers[d][g] = queryData->vec_d_numValuesPerSequence[d].data() + h_numSequencesPerCallerPS[g];
                        transferSizesBytes[d][g] = sizeof(int) * (vec_numSequences[g]);
                        dstBuffers[d][g] = vec_d_numValuesPerSequencePerGpu[g] + vec_numSequences[g] * d;
                    }
                }
                multigpu_transfer(
                    deviceIds,
                    srcBuffers,
                    transferSizesBytes,
                    srcStreams,
                    callerDeviceIds,
                    dstBuffers
                );

                //send value offsets
                for(int d = 0; d < numDataGpus; d++){
                    srcStreams[d] = queryData->streams[d];
                    for(int g = 0; g < numCallerGpus; g++){
                        srcBuffers[d][g] = vec_d_offsets_target[d] + h_numSequencesPerCallerPS[g];
                        transferSizesBytes[d][g] = sizeof(int) * (vec_numSequences[g]+1);
                        dstBuffers[d][g] = vec_d_offsetsPerSequencePerGpu[g] + (vec_numSequences[g]+1) * d;
                    }
                }
                multigpu_transfer(
                    deviceIds,
                    srcBuffers,
                    transferSizesBytes,
                    srcStreams,
                    callerDeviceIds,
                    dstBuffers
                );
            }

            //wait for transfers
            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                CUDACHECK(cudaStreamSynchronize(queryData->streams[d]));
            }

            // for(int g = 0; g < numCallerGpus; g++){
            //     CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
            //     for(int d = 0; d < numDataGpus; d++){
            //         CUDACHECK(cudaStreamWaitEvent(queryData->vec_callerStreamsPerDataGpu[g][d], queryData->events2[d], 0));
            //     }
            // }

            //gather results from targets
            // for(int d = 0; d < numDataGpus; d++){
            //     for(int g = 0; g < numCallerGpus; g++){
            //         if(vec_numSequences[g] > 0){ 
            //             CUDACHECK(cudaSetDevice(callerDeviceIds[g]));

            //             CUDACHECK(cudaMemcpyPeerAsync(
            //                 vec_d_allValues[g] + numValuesPerTargetPerCaller_verticalPS[d][g],
            //                 callerDeviceIds[g],
            //                 vec_d_values_target[d] + numValuesPerTargetPerCaller_horizontalPS[d][g],
            //                 deviceIds[d],
            //                 sizeof(read_number) * h_numValuesPerTargetPerCaller[d * numCallerGpus + g],
            //                 queryData->vec_callerStreamsPerDataGpu[g][d]
            //             ));
            //         }
            //     }
            // }
            // for(int d = 0; d < numDataGpus; d++){
            //     for(int g = 0; g < numCallerGpus; g++){
            //         if(vec_numSequences[g] > 0){
            //             CUDACHECK(cudaSetDevice(callerDeviceIds[g]));

            //             CUDACHECK(cudaMemcpyPeerAsync(
            //                 vec_d_numValuesPerSequencePerGpu[g] + vec_numSequences[g] * d,
            //                 callerDeviceIds[g],
            //                 queryData->vec_d_numValuesPerSequence[d].data() + h_numSequencesPerCallerPS[g],
            //                 deviceIds[d],
            //                 sizeof(int) * (vec_numSequences[g]),
            //                 queryData->vec_callerStreamsPerDataGpu[g][d]
            //             ));
            //         }
            //     }
            // }
            // for(int d = 0; d < numDataGpus; d++){
            //     for(int g = 0; g < numCallerGpus; g++){
            //         if(vec_numSequences[g] > 0){
            //             CUDACHECK(cudaSetDevice(callerDeviceIds[g]));

            //             CUDACHECK(cudaMemcpyPeerAsync(
            //                 vec_d_offsetsPerSequencePerGpu[g] + (vec_numSequences[g]+1) * d,
            //                 callerDeviceIds[g],
            //                 vec_d_offsets_target[d] + h_numSequencesPerCallerPS[g],
            //                 deviceIds[d],
            //                 sizeof(int) * (vec_numSequences[g]+1),
            //                 queryData->vec_callerStreamsPerDataGpu[g][d]
            //             ));
            //         }
            //     }
            // }

            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                // for(int d = 0; d < numDataGpus; d++){
                //     CUDACHECK(cudaEventRecord(queryData->vec_callerEventsPerDataGpu[g][d], queryData->vec_callerStreamsPerDataGpu[g][d]));
                //     CUDACHECK(cudaStreamWaitEvent(streams[g], queryData->vec_callerEventsPerDataGpu[g][d], 0));
                // }


                int* const h_numValuesPerTarget_verticalPS = queryData->pinned_numValuesPerTarget_verticalPS + g * numDataGpus;

                for(int d = 0; d < numDataGpus; d++){
                    h_numValuesPerTarget_verticalPS[d] = numValuesPerTargetPerCaller_verticalPS[d][g];
                }

                //copy values to output array, interleave results for same sequence
                //offsets which do not start at 0 are normalized to 0.
                multigpuminhasherkernels::copyToInterleavedKernel_fixoffsets<<<vec_numSequences[g], 128, 0, streams[g]>>>(
                    vec_d_allValues[g],
                    vec_d_offsetsPerSequencePerGpu[g],
                    vec_d_numValuesPerSequencePerGpu[g],
                    //d_numValuesPerTarget_verticalPS.data(),
                    h_numValuesPerTarget_verticalPS, //read from pinned memory
                    vec_d_offsets[g],
                    vec_d_values[g],
                    vec_numSequences[g],
                    numDataGpus
                ); CUDACHECKASYNC

            }

            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                //wait until all work is done on caller gpu g . then release memory
                CUDACHECK(cudaStreamSynchronize(streams[g]));
            }

            //work on all callers is done. all transfers from data gpus are complete


            // {
            //     int N = 0;
            //     CUDACHECK(cudaGetDeviceCount(&N));
            //     for(int i = 0; i < N; i++){
            //         CUDACHECK(cudaSetDevice(i));
            //         CUDACHECK(cudaDeviceSynchronize());
            //     }
            // }

            CUDACHECK(cudaSetDevice(oldDeviceId));
        }

        // collective execution, implements collective singlegpuminhasher access
        void multi_retrieveValues_directPeerAccess(
            MinhasherHandle& queryHandle,
            const std::vector<int>& vec_numSequences,
            const std::vector<const int*>& vec_totalNumValues,
            const std::vector<read_number*>& vec_d_values,
            const std::vector<const int*>& vec_d_numValuesPerSequence,
            const std::vector<int*> vec_d_offsets, //numSequences + 1
            const std::vector<cudaStream_t>& streams,
            const std::vector<int> callerDeviceIds,
            const std::vector<rmm::mr::device_memory_resource*>& /*mrs*/
        ) const {
            nvtx::ScopedRange sr_("multi_retrieveValues_directPeerAccess", 3);

            int oldDeviceId = 0;
            CUDACHECK(cudaGetDevice(&oldDeviceId));

            QueryData* const queryData = getMultiQueryDataFromHandle(queryHandle);
            assert(queryData->previousStage == QueryData::Stage::NumValues);
            assert(queryData->callerDeviceIds == callerDeviceIds);
            queryData->previousStage = QueryData::Stage::Retrieve;

            const int numDataGpus = deviceIds.size();
            const int numCallerGpus = callerDeviceIds.size();

            for(int g = 0; g < numCallerGpus; g++){
                if(vec_numSequences[g] == 0){
                    CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                    CUDACHECK(cudaMemsetAsync(vec_d_offsets[g], 0, sizeof(int) * (vec_numSequences[g] + 1), streams[g]));
                }
            }

            const int* const h_numSequencesPerCallerPS = queryData->pinned_numSequencesPerCallerPS;
            const int totalNumSequencesForAllGpus = h_numSequencesPerCallerPS[numCallerGpus];
            if(totalNumSequencesForAllGpus == 0){
                return;
            }

            //compute num values per caller per target

            int* h_numValuesPerTargetPerCaller = queryData->pinned_numValuesPerTargetPerCaller;


            std::vector<read_number*> vec_d_allValues(numCallerGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerGpu(numCallerGpus, nullptr);
            std::vector<int*> vec_d_offsetsPerSequencePerGpu(numCallerGpus, nullptr);

            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));

                void* allocations[4]{};
                std::size_t allocation_sizes[4]{};
                std::size_t storage_bytes = 0;

                constexpr int roundUpTo = 10000;
                const int roundedTotalNumValues = SDIV((*vec_totalNumValues[g]), roundUpTo) * roundUpTo;

                allocation_sizes[0] = sizeof(read_number) * roundedTotalNumValues;
                allocation_sizes[1] = sizeof(int) * vec_numSequences[g] * numDataGpus;
                allocation_sizes[2] = sizeof(int) * (vec_numSequences[g]+1) * numDataGpus;

                CUDACHECK(cub::DeviceScan::InclusiveSum(
                    nullptr, 
                    allocation_sizes[3], 
                    vec_d_numValuesPerSequence[g], 
                    vec_d_offsets[g] + 1, 
                    vec_numSequences[g],
                    streams[g]
                ));

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                resizeUninitialized(queryData->vec_d_callerTempBuffers[g], storage_bytes, streams[g]);
                CUDACHECK(cub::AliasTemporaries(
                    queryData->vec_d_callerTempBuffers[g].data(),
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                vec_d_allValues[g] = reinterpret_cast<read_number*>(allocations[0]);
                vec_d_numValuesPerSequencePerGpu[g] = reinterpret_cast<int*>(allocations[1]);
                vec_d_offsetsPerSequencePerGpu[g] = reinterpret_cast<int*>(allocations[2]);

                //wait for the allocation so it can be used later for transfer in different stream
                CUDACHECK(cudaStreamSynchronize(streams[g]));

                CUDACHECK(cudaMemsetAsync(vec_d_offsets[g], 0, sizeof(int), streams[g]));
                CUDACHECK(cub::DeviceScan::InclusiveSum(
                    allocations[3], 
                    allocation_sizes[3], 
                    vec_d_numValuesPerSequence[g], 
                    vec_d_offsets[g] + 1, 
                    vec_numSequences[g],
                    streams[g]
                ));

            }
            
            std::vector<kmer_type*> vec_d_signatures_transposed(numDataGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerHash(numDataGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerHashExclPSVert(numDataGpus, nullptr);

            std::vector<read_number*> vec_d_values_target(numDataGpus, nullptr);
            std::vector<int*> vec_d_offsets_target(numDataGpus, nullptr);
            std::vector<int*> vec_d_queryOffsetsPerSequencePerHash(numDataGpus, nullptr);

            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();

                auto* targetmr = rmm::mr::get_current_device_resource();

                {
                    void* persistent_allocations[3]{};
                    std::size_t persistent_allocation_sizes[3]{};
                    std::size_t persistent_storage_bytes = queryData->vec_d_singlegpuminhasherBuffers[d].size();

                    persistent_allocation_sizes[0] = sizeof(kmer_type) * totalNumSequencesForAllGpus * numHashFunctions; // d_sig_trans
                    persistent_allocation_sizes[1] = sizeof(int) * totalNumSequencesForAllGpus * numHashFunctions; // d_numValuesPerSequencePerHash
                    persistent_allocation_sizes[2] = sizeof(int) * totalNumSequencesForAllGpus * numHashFunctions; // d_numValuesPerSequencePerHashExclPSVert

                    CUDACHECK(cub::AliasTemporaries(
                        queryData->vec_d_singlegpuminhasherBuffers[d].data(),
                        persistent_storage_bytes,
                        persistent_allocations,
                        persistent_allocation_sizes
                    ));
                    vec_d_signatures_transposed[d] = static_cast<kmer_type*>(persistent_allocations[0]);
                    vec_d_numValuesPerSequencePerHash[d] = static_cast<int*>(persistent_allocations[1]);
                    vec_d_numValuesPerSequencePerHashExclPSVert[d] = static_cast<int*>(persistent_allocations[2]);
                }


                void* allocations[4]{};
                std::size_t allocation_sizes[4]{};
                std::size_t storage_bytes = 0;

                const int totalNumValuesTarget = queryData->pinned_totalNumValuesPerTarget[d];
                constexpr int roundUpTo = 10000;
                const int roundedTotalNumValuesTarget = SDIV(totalNumValuesTarget, roundUpTo) * roundUpTo;

                allocation_sizes[0] = sizeof(read_number) * roundedTotalNumValuesTarget; // d_values_target
                allocation_sizes[1] = sizeof(int) * (totalNumSequencesForAllGpus + 1); // d_offsets_target
                allocation_sizes[2] = sizeof(int) * totalNumSequencesForAllGpus * numHashFunctions; // d_queryOffsetsPerSequencePerHash
                CUDACHECK(cub::DeviceScan::InclusiveSum(
                    nullptr, 
                    allocation_sizes[3], 
                    queryData->vec_d_numValuesPerSequence[d].data(),
                    vec_d_offsets_target[d] + 1,
                    totalNumSequencesForAllGpus,
                    queryData->streams[d]
                ));

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));
                resizeUninitialized(queryData->vec_d_targetTempBuffers[d], storage_bytes, queryData->streams[d].getStream());
                CUDACHECK(cub::AliasTemporaries(
                    queryData->vec_d_targetTempBuffers[d].data(),
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                vec_d_values_target[d] = static_cast<read_number*>(allocations[0]);
                vec_d_offsets_target[d] = static_cast<int*>(allocations[1]);
                vec_d_queryOffsetsPerSequencePerHash[d] = static_cast<int*>(allocations[2]);


                CUDACHECK(cudaMemsetAsync(vec_d_offsets_target[d], 0, sizeof(int), queryData->streams[d]));

                CUDACHECK(cub::DeviceScan::InclusiveSum(
                    allocations[3], 
                    allocation_sizes[3], 
                    queryData->vec_d_numValuesPerSequence[d].data(),
                    vec_d_offsets_target[d] + 1,
                    totalNumSequencesForAllGpus,
                    queryData->streams[d]
                ));
            }

            for(int d = 0; d < numDataGpus; d++){
                if(totalNumSequencesForAllGpus > 0){
                    CUDACHECK(cudaSetDevice(deviceIds[d]));
                    const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();

                    sgpuminhasherkernels::computeQueryDestinationOffsetsKernel<<<SDIV(totalNumSequencesForAllGpus, 256), 256, 0, queryData->streams[d]>>>(
                        vec_d_queryOffsetsPerSequencePerHash[d],
                        vec_d_numValuesPerSequencePerHashExclPSVert[d],
                        totalNumSequencesForAllGpus,
                        numHashFunctions,
                        vec_d_offsets_target[d]
                    );
                    CUDACHECKASYNC
                }
            }

            for(int d = 0; d < numDataGpus; d++){
                if(totalNumSequencesForAllGpus > 0){
                    CUDACHECK(cudaSetDevice(deviceIds[d]));
                    const int numHashFunctions = sgpuMinhashers[d]->getNumberOfMaps();

                    const int signaturesPitchInElements = totalNumSequencesForAllGpus;
                    const int numValuesPerKeyPitchInElements = totalNumSequencesForAllGpus;
                    const int beginOffsetsPitchInElements = totalNumSequencesForAllGpus;
                    constexpr int cgsize = SingleGpuMinhasher::GpuTable::DeviceTableView::cg_size();

                    dim3 block(256, 1, 1);
                    const int numBlocksPerTable = SDIV(totalNumSequencesForAllGpus, (block.x / cgsize));
                    dim3 grid(numBlocksPerTable, std::min(65535, numHashFunctions), 1);

                    gpuhashtablekernels::retrieveCompactKernel<<<grid, block, 0, queryData->streams[d]>>>(
                        sgpuMinhashers[d]->d_deviceAccessibleTableViews.data(),
                        numHashFunctions,
                        vec_d_signatures_transposed[d],
                        signaturesPitchInElements,
                        vec_d_queryOffsetsPerSequencePerHash[d],
                        beginOffsetsPitchInElements,
                        vec_d_numValuesPerSequencePerHash[d],
                        numValuesPerKeyPitchInElements,
                        resultsPerMapThreshold,
                        totalNumSequencesForAllGpus,
                        vec_d_values_target[d]
                    );
                    CUDACHECKASYNC
                }
            }




            //wait for h_numValuesPerTargetPerCaller
            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                CUDACHECK(cudaEventSynchronize(queryData->events[d]));
            }

            std::vector<std::vector<int>> numValuesPerTargetPerCaller_horizontalPS(numDataGpus, std::vector<int>(numCallerGpus+1, 0));
            std::vector<std::vector<int>> numValuesPerTargetPerCaller_verticalPS(numDataGpus+1, std::vector<int>(numCallerGpus, 0));

            for(int d = 0; d < numDataGpus; d++){
                for(int g = 0; g < numCallerGpus; g++){
                    numValuesPerTargetPerCaller_horizontalPS[d][g+1] 
                        = numValuesPerTargetPerCaller_horizontalPS[d][g] + h_numValuesPerTargetPerCaller[d * numCallerGpus + g];
                }
            }
            for(int g = 0; g < numCallerGpus; g++){
                for(int d = 0; d < numDataGpus; d++){
                    numValuesPerTargetPerCaller_verticalPS[d+1][g] 
                        = numValuesPerTargetPerCaller_verticalPS[d][g] + h_numValuesPerTargetPerCaller[d * numCallerGpus + g];
                }
            }
            for(int g = 0; g < numCallerGpus; g++){
                assert(*vec_totalNumValues[g] == numValuesPerTargetPerCaller_verticalPS[numDataGpus][g]);
            }

            // std::cout << "retrieve numValuesPerTargetPerCaller\n";
            // for(int d = 0; d < numDataGpus; d++){
            //     for(int g = 0; g < numCallerGpus; g++){
            //         std::cout << h_numValuesPerTargetPerCaller[d * numCallerGpus + g] << " ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "retrieve horizontal ps\n";
            // for(int d = 0; d < numDataGpus; d++){
            //     for(int g = 0; g < numCallerGpus+1; g++){
            //         std::cout << numValuesPerTargetPerCaller_horizontalPS[d][g] << " ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "retrieve vertical ps\n";
            // for(int d = 0; d < numDataGpus+1; d++){
            //     for(int g = 0; g < numCallerGpus; g++){
            //         std::cout << numValuesPerTargetPerCaller_verticalPS[d][g] << " ";
            //     }
            //     std::cout << "\n";
            // }

            //send numValuesPerSequence
            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                int** const h_dstPointers = reinterpret_cast<int**>(queryData->pinned_bufferpointers + 0 * numCallerGpus * numDataGpus + d * numCallerGpus);
                for(int g = 0; g < numCallerGpus; g++){
                    h_dstPointers[g] = vec_d_numValuesPerSequencePerGpu[g] + vec_numSequences[g] * d;
                }

                int* const h_offsets = reinterpret_cast<int*>(queryData->pinned_bufferpointers2 + 0 * numCallerGpus * numDataGpus + d * numCallerGpus);
                for(int g = 0; g < numCallerGpus+1; g++){
                    h_offsets[g] = h_numSequencesPerCallerPS[g];
                }

                dim3 grid1(SDIV(vec_numSequences[d], 128), numCallerGpus);

                multigpuminhasherkernels::scatterSegmentedArrayKernel<<<grid1, 128, 0,  queryData->streams[d]>>>(
                    queryData->vec_d_numValuesPerSequence[d].data(),
                    h_offsets,
                    h_offsets + 1,
                    h_dstPointers,
                    numCallerGpus
                );
            }
            //send value offsets
            for(int d = 0; d < numDataGpus; d++){
                if(vec_numSequences[d] > 0){
                    CUDACHECK(cudaSetDevice(deviceIds[d]));
                    int** const h_dstPointers = reinterpret_cast<int**>(queryData->pinned_bufferpointers + 1 * numCallerGpus * numDataGpus + d * numCallerGpus);
                    for(int g = 0; g < numCallerGpus; g++){
                        h_dstPointers[g] = vec_d_offsetsPerSequencePerGpu[g] + (vec_numSequences[g]+1) * d;
                    }

                    int* const h_beginoffsets = reinterpret_cast<int*>(queryData->pinned_bufferpointers2 + 1 * numCallerGpus * numDataGpus + d * numCallerGpus);
                    for(int g = 0; g < numCallerGpus; g++){
                        h_beginoffsets[g] = h_numSequencesPerCallerPS[g];
                    }
                    int* const h_endoffsets = reinterpret_cast<int*>(queryData->pinned_bufferpointers3 + 1 * numCallerGpus * numDataGpus + d * numCallerGpus);
                    for(int g = 0; g < numCallerGpus; g++){
                        h_endoffsets[g] = h_numSequencesPerCallerPS[g+1] + 1;
                    }

                    dim3 grid1(SDIV(vec_numSequences[d], 128), numCallerGpus);

                    multigpuminhasherkernels::scatterSegmentedArrayKernel<<<grid1, 128, 0,  queryData->streams[d]>>>(
                        vec_d_offsets_target[d],
                        h_beginoffsets,
                        h_endoffsets,
                        h_dstPointers,
                        numCallerGpus
                    );
                }
            }


            {
                std::vector<std::vector<const void*>> srcBuffers(numDataGpus, std::vector<const void*>(numCallerGpus));
                std::vector<std::vector<void*>> dstBuffers(numDataGpus, std::vector<void*>(numCallerGpus));
                std::vector<std::vector<size_t>> transferSizesBytes(numDataGpus, std::vector<size_t>(numCallerGpus));
                std::vector<cudaStream_t> srcStreams(numDataGpus);
                //send values
                for(int d = 0; d < numDataGpus; d++){
                    srcStreams[d] = queryData->streams[d];
                    for(int g = 0; g < numCallerGpus; g++){
                        srcBuffers[d][g] = vec_d_values_target[d] + numValuesPerTargetPerCaller_horizontalPS[d][g];
                        transferSizesBytes[d][g] = sizeof(read_number) * h_numValuesPerTargetPerCaller[d * numCallerGpus + g];
                        dstBuffers[d][g] = vec_d_allValues[g] + numValuesPerTargetPerCaller_verticalPS[d][g];
                    }
                }
                multigpu_transfer(
                    deviceIds,
                    srcBuffers,
                    transferSizesBytes,
                    srcStreams,
                    callerDeviceIds,
                    dstBuffers
                );

                //send numValuesPerSequence
                // for(int d = 0; d < numDataGpus; d++){
                //     srcStreams[d] = queryData->streams[d];
                //     for(int g = 0; g < numCallerGpus; g++){
                //         srcBuffers[d][g] = queryData->vec_d_numValuesPerSequence[d].data() + h_numSequencesPerCallerPS[g];
                //         transferSizesBytes[d][g] = sizeof(int) * (vec_numSequences[g]);
                //         dstBuffers[d][g] = vec_d_numValuesPerSequencePerGpu[g] + vec_numSequences[g] * d;
                //     }
                // }
                // multigpu_transfer(
                //     deviceIds,
                //     srcBuffers,
                //     transferSizesBytes,
                //     srcStreams,
                //     callerDeviceIds,
                //     dstBuffers
                // );

                //send value offsets
                // for(int d = 0; d < numDataGpus; d++){
                //     srcStreams[d] = queryData->streams[d];
                //     for(int g = 0; g < numCallerGpus; g++){
                //         srcBuffers[d][g] = vec_d_offsets_target[d] + h_numSequencesPerCallerPS[g];
                //         transferSizesBytes[d][g] = sizeof(int) * (vec_numSequences[g]+1);
                //         dstBuffers[d][g] = vec_d_offsetsPerSequencePerGpu[g] + (vec_numSequences[g]+1) * d;
                //     }
                // }
                // multigpu_transfer(
                //     deviceIds,
                //     srcBuffers,
                //     transferSizesBytes,
                //     srcStreams,
                //     callerDeviceIds,
                //     dstBuffers
                // );
            }

            //wait for transfers
            for(int d = 0; d < numDataGpus; d++){
                CUDACHECK(cudaSetDevice(deviceIds[d]));
                CUDACHECK(cudaStreamSynchronize(queryData->streams[d]));
            }

            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));

                int* const h_numValuesPerTarget_verticalPS = queryData->pinned_numValuesPerTarget_verticalPS + g * numDataGpus;

                for(int d = 0; d < numDataGpus; d++){
                    h_numValuesPerTarget_verticalPS[d] = numValuesPerTargetPerCaller_verticalPS[d][g];
                }

                //copy values to output array, interleave results for same sequence
                //offsets which do not start at 0 are normalized to 0.
                multigpuminhasherkernels::copyToInterleavedKernel_fixoffsets<<<vec_numSequences[g], 128, 0, streams[g]>>>(
                    vec_d_allValues[g],
                    vec_d_offsetsPerSequencePerGpu[g],
                    vec_d_numValuesPerSequencePerGpu[g],
                    //d_numValuesPerTarget_verticalPS.data(),
                    h_numValuesPerTarget_verticalPS, //read from pinned memory
                    vec_d_offsets[g],
                    vec_d_values[g],
                    vec_numSequences[g],
                    numDataGpus
                ); CUDACHECKASYNC

            }

            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                //wait until all work is done on caller gpu g .
                CUDACHECK(cudaStreamSynchronize(streams[g]));
            }

            CUDACHECK(cudaSetDevice(oldDeviceId));
        }


        void compact(cudaStream_t stream) override {
            CudaEvent event;
            event.record(stream);

            // for(int g = 0; g < int(deviceIds.size()); g++){
            //     cub::SwitchDevice sd{deviceIds[g]};
            //     CUDACHECK(cudaStreamWaitEvent(cudaStreamPerThread, event, 0));

            //     sgpuMinhashers[g]->compact(cudaStreamPerThread);

            //     CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));
            // } 

            const int numGpus = deviceIds.size();

            std::vector<size_t> requiredTempBytesPerGpu(numGpus);
            auto usesManagedMemory = std::make_unique<bool[]>(numGpus);

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                assert(deviceIds[g] == sgpuMinhashers[g]->deviceId);
                CUDACHECK(cudaStreamWaitEvent(cudaStreamPerThread, event, 0));

                std::size_t required_temp_bytes = 0;

                for(auto& table : sgpuMinhashers[g]->gpuHashTables){
                    std::size_t temp_bytes2 = 0;
                    table->compact(nullptr, temp_bytes2, cudaStreamPerThread);
                    required_temp_bytes = std::max(required_temp_bytes, temp_bytes2);
                }
                requiredTempBytesPerGpu[g] = required_temp_bytes;
            }


            size_t hostMemoryAvailable = hostMemoryLimit;
            int processed = 0;
            while(processed != numGpus){
                std::vector<void*> tempsVector;
                for(int g = processed; g < numGpus; g++){
                    cub::SwitchDevice sd{deviceIds[g]};
                    std::size_t freeMem, totalMem; 
                    CUDACHECK(cudaMemGetInfo(&freeMem, &totalMem));

                    size_t required_temp_bytes = requiredTempBytesPerGpu[g];
                    void* temp = nullptr;
                    if(required_temp_bytes < freeMem){
                        CUDACHECK(cudaMalloc(&temp, required_temp_bytes));
                        usesManagedMemory[g] = false;
                    }else{
                        if(required_temp_bytes < hostMemoryAvailable){
                            CUDACHECK(cudaMallocManaged(&temp, required_temp_bytes));
                            int deviceId = 0;
                            CUDACHECK(cudaGetDevice(&deviceId));
                            CUDACHECK(cudaMemAdvise(temp, required_temp_bytes, cudaMemAdviseSetAccessedBy, deviceId));
                            hostMemoryAvailable -= required_temp_bytes;
                            //std::cout << "g = " << g << ", managed\n";
                            usesManagedMemory[g] = true;
                        }else{
                            break;
                        }
                    }
                    assert(required_temp_bytes == 0 || temp != nullptr);
                    tempsVector.push_back(temp);
                }

                const int numToProcess = tempsVector.size();
                if(numToProcess == 0){
                    throw std::runtime_error("Out of cpu memory during hashtable construction");
                }
                #pragma omp parallel for
                for(int i = 0; i < numToProcess; i++){
                    cub::SwitchDevice sd(deviceIds[processed + i]);
                    const int numGpuTables = sgpuMinhashers[processed + i]->gpuHashTables.size();
                    for(int t = 0; t < numGpuTables; t++){
                        auto& table = sgpuMinhashers[processed + i]->gpuHashTables[t];
                        table->compact(tempsVector[i], requiredTempBytesPerGpu[processed + i], cudaStreamPerThread);

                        if(t + 1 < numGpuTables && usesManagedMemory[processed + i]){
                            std::size_t freeMem, totalMem; 
                            CUDACHECK(cudaMemGetInfo(&freeMem, &totalMem));
                            if(requiredTempBytesPerGpu[processed + i] < freeMem){
                            //    CUDACHECK(cudaFree(tempsVector[i])); //dealloc managed
                            //    CUDACHECK(cudaMalloc(&tempsVector[i], requiredTempBytesPerGpu[processed + i])); //alloc normal
                            //    std::cout << "switched from managed to normal\n";

                                CUDACHECK(cudaMemPrefetchAsync(
                                    tempsVector[i], 
                                    requiredTempBytesPerGpu[processed + i], 
                                    deviceIds[processed + i], 
                                    cudaStreamPerThread
                                ));

                                usesManagedMemory[processed + i] = false;
                            }
                        }
                    }
                }

                for(int i = 0; i < numToProcess; i++){
                    cub::SwitchDevice sd(deviceIds[processed + i]);
                    CUDACHECK(cudaFree(tempsVector[i]));
                }
                processed += numToProcess;
                hostMemoryAvailable = hostMemoryLimit;
            }

 
        }

        void constructionIsFinished(cudaStream_t stream) override{
            CudaEvent event;
            event.record(stream);

            std::vector<int> deviceIdsTmp;
            std::vector<std::unique_ptr<SingleGpuMinhasher>> sgpuMinhashersTmp;
            std::vector<std::vector<int>> hashFunctionIdsPerGpuTmp;

            for(int g = 0; g < int(deviceIds.size()); g++){
                cub::SwitchDevice sd{deviceIds[g]};
                CUDACHECK(cudaStreamWaitEvent(cudaStreamPerThread, event, 0));

                sgpuMinhashers[g]->constructionIsFinished(cudaStreamPerThread);

                CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));

                //only keep single-gpu minhashers which are used
                if(sgpuMinhashers[g]->getNumberOfMaps() > 0){
                    deviceIdsTmp.push_back(deviceIds[g]);
                    sgpuMinhashersTmp.push_back(std::move(sgpuMinhashers[g]));
                    hashFunctionIdsPerGpuTmp.push_back(std::move(hashFunctionIdsPerGpu[g]));
                }
            }
            
            std::swap(deviceIds, deviceIdsTmp);
            std::swap(sgpuMinhashers, sgpuMinhashersTmp);
            std::swap(hashFunctionIdsPerGpu, hashFunctionIdsPerGpuTmp);

            // std::cerr << "hashTableLocations:\n";
            // for(int i = 0; i < getNumberOfMaps(); i++){
            //     std::cerr << hashTableLocations[i] << " ";
            // }
            // std::cerr << "\n";

            // for(int g = 0; g < int(deviceIds.size()); g++){
            //     std::cerr << "hashFunctionIdsPerGpu " << g << " (id " << deviceIds[g] << ")" << "\n";
            //     for(auto x : hashFunctionIdsPerGpu[g]){
            //         std::cerr << x << " ";
            //     }
            //     std::cerr << "\n";
            // }

            // for(int g = 0; g < int(deviceIds.size()); g++){
            //     std::cerr << "actual stored hashFunctionIdsPerGpu " << g << " (id " << deviceIds[g] << ")" << "\n";
            //     const int num = sgpuMinhashers[g]->h_currentHashFunctionNumbers.size();
            //     for(int i = 0; i < num; i++){
            //         const int x = sgpuMinhashers[g]->h_currentHashFunctionNumbers[i];
            //         std::cerr << x << " ";
            //     }
            //     std::cerr << "\n";
            // }
        }

        MemoryUsage getMemoryInfo() const noexcept override{
            MemoryUsage mem{};

            for(const auto& minhasher : sgpuMinhashers){
                mem += minhasher->getMemoryInfo();
            }

            return mem;
        }

        MemoryUsage getMemoryInfo(const MinhasherHandle& handle) const noexcept override{
            return tempdataVector[handle.getId()]->getMemoryInfo()
                + multi_tempdataVector[handle.getId()]->getMemoryInfo();
        }

        int getNumResultsPerMapThreshold() const noexcept override{
            return resultsPerMapThreshold;
        }
        
        int getNumberOfMaps() const noexcept override{
            return hashTableLocations.size();
        }

        int getKmerSize() const noexcept override{
            return kmerSize;
        }

        void destroy(){
            for(auto& minhasher : sgpuMinhashers){
                DeviceSwitcher sd(minhasher->getDeviceId());
                minhasher->destroy();
            }
        }

        bool hasGpuTables() const noexcept override {
            return true;
        }

        void setThreadPool(ThreadPool* /*tp*/) override {}

        void setHostMemoryLimitForConstruction(std::size_t bytes) override{
            hostMemoryLimit = bytes;
        }

        void setDeviceMemoryLimitsForConstruction(const std::vector<std::size_t>&) override {

        }

        void writeToStream(std::ostream& /*os*/) const override{
            std::cerr << "MultiGpuMinhasher::writeToStream not supported\n";
        }

        int loadFromStream(std::ifstream& /*is*/, int /*numMapsUpperLimit*/) override{
            std::cerr << "MultiGpuMinhasher::loadFromStream not supported\n";
            return 0;
        } 

        bool canWriteToStream() const noexcept override { return false; };
        bool canLoadFromStream() const noexcept override { return false; };

private:        

        std::uint64_t getKmerMask() const{
            constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

            return std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - getKmerSize()) * 2);
        }

        constexpr float getLoad() const noexcept{
            return 0.8f;
        }

        std::unique_ptr<QueryData> createQueryData() const{
            auto ptr = std::make_unique<QueryData>();
            const int numMinhashers = sgpuMinhashers.size();

            ptr->numDataGpus = numMinhashers;

            for(int i = 0; i < numMinhashers; i++){
                DeviceSwitcher ds(sgpuMinhashers[i]->getDeviceId());

                ptr->streams.emplace_back();
                ptr->events.emplace_back(cudaEventDisableTiming);
                ptr->events2.emplace_back(cudaEventDisableTiming);

                ptr->dataDeviceIds.push_back(sgpuMinhashers[i]->getDeviceId());
                ptr->singlegpuMinhasherHandles.emplace_back(std::make_unique<MinhasherHandle>(sgpuMinhashers[i]->makeMinhasherHandle()));
                ptr->vec_d_singlegpuminhasherBuffers.emplace_back(0, cudaStreamPerThread);
                ptr->vec_d_numValuesPerSequence.emplace_back(0, cudaStreamPerThread);
                ptr->vec_d_targetTempBuffers.emplace_back(0, cudaStreamPerThread);
                CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));
            }

            {
                int numGpus = 0;
                CUDACHECK(cudaGetDeviceCount(&numGpus));

                size_t allocation_sizes[9];
                allocation_sizes[0] = sizeof(int) * numGpus; // pinned_totalNumValuesPerTarget
                allocation_sizes[1] = sizeof(int) * numGpus * numGpus; // pinned_numValuesPerTargetPerCaller
                allocation_sizes[2] = sizeof(int) * numGpus * numGpus; // pinned_numValuesPerTarget_verticalPS
                allocation_sizes[3] = sizeof(int) * (numGpus + 1); // pinned_numSequencesPerCallerPS
                allocation_sizes[4] = sizeof(int) * numGpus; // pinned_totalNumValuesPerCaller
                const int foo = ((numGpus+1) * (numGpus+1)) + 1;
                allocation_sizes[5] = sizeof(void*) * 3*foo; // pinned_bufferpointers
                allocation_sizes[6] = sizeof(void*) * 3*foo; // pinned_bufferpointers2
                allocation_sizes[7] = sizeof(void*) * 3*foo; // pinned_bufferpointers3
                allocation_sizes[8] = sizeof(int) * (2 * numGpus); // pinned_singledata


                void* allocations[9]{};

                size_t temp_storage_bytes = 0;

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    temp_storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                ptr->pinnedData.resize(temp_storage_bytes);

                CUDACHECK(cub::AliasTemporaries(
                    ptr->pinnedData.data(),
                    temp_storage_bytes,
                    allocations,
                    allocation_sizes
                ));
                


                ptr->pinned_totalNumValuesPerTarget = static_cast<int*>(allocations[0]);
                ptr->pinned_numValuesPerTargetPerCaller = static_cast<int*>(allocations[1]);
                ptr->pinned_numValuesPerTarget_verticalPS = static_cast<int*>(allocations[2]);
                ptr->pinned_numSequencesPerCallerPS = static_cast<int*>(allocations[3]);
                ptr->pinned_totalNumValuesPerCaller = static_cast<int*>(allocations[4]);
                ptr->pinned_bufferpointers = static_cast<void**>(allocations[5]);
                ptr->pinned_bufferpointers2 = static_cast<void**>(allocations[6]);
                ptr->pinned_bufferpointers3 = static_cast<void**>(allocations[7]);
                ptr->pinned_singledata = static_cast<int*>(allocations[8]);
            }


            return ptr;
        }

        QueryData* getQueryDataFromHandle(const MinhasherHandle& queryHandle) const{
            std::shared_lock<SharedMutex> lock(sharedmutex);

            return tempdataVector[queryHandle.getId()].get();
        }

        QueryData* getMultiQueryDataFromHandle(const MinhasherHandle& queryHandle) const{
            std::shared_lock<SharedMutex> lock(sharedmutex);

            return multi_tempdataVector[queryHandle.getId()].get();
        }

        int getNumberOfOccupiedDevices() const{
            int n = 0;
            for(const auto& vec : hashFunctionIdsPerGpu){
                if(vec.size() > 0){
                    n++;
                }
            }
            return n;
        }        

        mutable int counter = 0;
        mutable SharedMutex sharedmutex{};

        bool directPeerAccess{};
        Layout layout{};
        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        float loadfactor{};
        size_t hostMemoryLimit{};
        std::vector<int> deviceIds;
        std::vector<std::unique_ptr<SingleGpuMinhasher>> sgpuMinhashers{};
        mutable std::vector<std::unique_ptr<QueryData>> tempdataVector{};
        mutable std::vector<std::unique_ptr<QueryData>> multi_tempdataVector{};

        std::vector<std::vector<int>> hashFunctionIdsPerGpu{};
        std::vector<int> hashTableLocations{};
    };












    class ReplicatedSingleGpuMinhasher : public GpuMinhasher, public GpuMinhasherWithMultiQuery{
    public:
        using Key = SingleGpuMinhasher::Key;
        using Value = SingleGpuMinhasher::Value;
    private:
        static constexpr int cgsize = SingleGpuMinhasher::GpuTable::DeviceTableView::cg_size();
        using DeviceSwitcher = cub::SwitchDevice;

        template<class T>
        using HostBuffer = helpers::SimpleAllocationPinnedHost<T, 5>;

        struct QueryData{
            enum class Stage{
                None,
                NumValues,
                Retrieve
            };

            Stage previousStage = Stage::None;

            std::vector<std::unique_ptr<MinhasherHandle>> singlegpuMinhasherHandles;

            std::vector<int> dataDeviceIds{};
            //std::vector<rmm::device_uvector<int>> vec_d_numValuesPerSequence{}; // 1 for each caller gpu
            std::vector<rmm::device_uvector<char>> vec_d_persistentBuffers{}; // 1 for each caller gpu

            std::vector<int> callerDeviceIds;

            HostBuffer<char> pinnedData{};
            int* pinned_totalNumValues{};

            void setCallerDeviceIds(const std::vector<int>& newCallerDeviceIds){
                if(callerDeviceIds != newCallerDeviceIds){
                    for(int g = 0; g < int(callerDeviceIds.size()); g++){
                        cub::SwitchDevice sd(callerDeviceIds[g]);
                        vec_d_persistentBuffers[g].release();
                    }

                    callerDeviceIds = newCallerDeviceIds;

                    vec_d_persistentBuffers.clear();
                    for(int g = 0; g < int(callerDeviceIds.size()); g++){
                        cub::SwitchDevice sd(callerDeviceIds[g]);
                        vec_d_persistentBuffers.emplace_back(0, cudaStreamPerThread);
                        CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));
                    }
                }
            }

            MemoryUsage getMemoryInfo() const{
                MemoryUsage mem{};

                return mem;
            }

            ~QueryData(){
                // for(int d = 0; d < int(dataDeviceIds.size()); d++){
                //     if(d < int(vec_d_singlegpuminhasherBuffers.size())){
                //         cub::SwitchDevice sd(dataDeviceIds[d]);
                //         vec_d_singlegpuminhasherBuffers[d].release();
                //     }
                // } 
                for(int g = 0; g < int(callerDeviceIds.size()); g++){
                    if(g < int(vec_d_persistentBuffers.size())){
                        cub::SwitchDevice sd(callerDeviceIds[g]);
                        //vec_d_numValuesPerSequence[g].release();
                        vec_d_persistentBuffers[g].release();
                    }
                }   
            }
        };

    public: 

        //base must be located on one of deviceIds gpus. it is replicated to all gpus of deviceIds
        ReplicatedSingleGpuMinhasher(std::unique_ptr<SingleGpuMinhasher>&& base, std::vector<int> deviceIds_)
            : deviceIds(deviceIds_)
        {
            const int baseDeviceId = base->getDeviceId();
            auto iter = std::find(deviceIds.begin(), deviceIds.end(), baseDeviceId);
            assert(iter != deviceIds.end());
            const int baseIndex = std::distance(deviceIds.begin(), iter);

            sgpuMinhashers.resize(deviceIds.size());
            sgpuMinhashers[baseIndex] = std::move(base);

            for(size_t i = 0; i < deviceIds.size(); i++){
                const int targetDeviceId = deviceIds[i];
                if(targetDeviceId != baseDeviceId){
                    sgpuMinhashers[i] = sgpuMinhashers[baseIndex]->makeCopy(targetDeviceId);
                }
            }
        }

        int addHashTables(int, const int*, cudaStream_t) override{
            return 0;
        }

        void insert(
            const unsigned int*,
            int,
            const int*,
            std::size_t,
            const read_number*,
            const read_number*,
            int,
            int,
            const int*,
            cudaStream_t,
            rmm::mr::device_memory_resource*
        ) override {
            
        }

        int checkInsertionErrors(
            int,
            int,
            cudaStream_t        
        ) override{
            return 0;
        }

        MinhasherHandle makeMinhasherHandle() const override{
            auto ptr = createQueryData();

            std::unique_lock<SharedMutex> lock(sharedmutex);
            const int handleid = counter++;
            MinhasherHandle h = constructHandle(handleid);

            tempdataVector.emplace_back(std::move(ptr));

            return h;
        }

        void destroyHandle(MinhasherHandle& handle) const override{            

            std::unique_lock<SharedMutex> lock(sharedmutex);

            const int id = handle.getId();
            assert(id < int(tempdataVector.size()));

            const int numMinhashers = sgpuMinhashers.size();
            for(int i = 0; i < numMinhashers; i++){
                sgpuMinhashers[i]->destroyHandle(*tempdataVector[id]->singlegpuMinhasherHandles[i]);
            }
            
            {
                tempdataVector[id] = nullptr;
            }
            handle = constructHandle(std::numeric_limits<int>::max());
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
        ) const override{
            int currentDeviceId = 0;
            CUDACHECK(cudaGetDevice(&currentDeviceId));
            auto deviceIdIter = std::find(deviceIds.begin(), deviceIds.end(), currentDeviceId);
            assert(deviceIdIter != deviceIds.end());
            const int sgpuIndex = std::distance(deviceIds.begin(), deviceIdIter);

            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            sgpuMinhashers[sgpuIndex]->determineNumValues(
                *queryData->singlegpuMinhasherHandles[sgpuIndex].get(),
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                d_numValuesPerSequence,
                totalNumValues,
                stream,
                mr
            );
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
        ) const override{
            int currentDeviceId = 0;
            CUDACHECK(cudaGetDevice(&currentDeviceId));
            auto deviceIdIter = std::find(deviceIds.begin(), deviceIds.end(), currentDeviceId);
            assert(deviceIdIter != deviceIds.end());
            const int sgpuIndex = std::distance(deviceIds.begin(), deviceIdIter);

            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            sgpuMinhashers[sgpuIndex]->retrieveValues(
                *queryData->singlegpuMinhasherHandles[sgpuIndex].get(),
                numSequences,
                totalNumValues,
                d_values,
                d_numValuesPerSequence,
                d_offsets,
                stream,
                mr
            );
        }

        void multi_determineNumValues(
            MinhasherHandle& queryHandle,
            const std::vector<const unsigned int*>& vec_d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const std::vector<const int*>& vec_d_sequenceLengths,
            const std::vector<int>& vec_numSequences,
            const std::vector<int*>& vec_d_numValuesPerSequence,
            const std::vector<int*>& vec_totalNumValues,
            const std::vector<cudaStream_t>& callerStreams,
            const std::vector<int>& callerDeviceIds,
            const std::vector<rmm::mr::device_memory_resource*>& mrs
        ) const override{
            nvtx::ScopedRange sr_("multi_determineNumValues", 3);
            int currentDeviceId = 0;
            CUDACHECK(cudaGetDevice(&currentDeviceId));

            const int numDataGpus = deviceIds.size();
            const int numCallerGpus = callerDeviceIds.size();

            QueryData* queryData = getQueryDataFromHandle(queryHandle);
            queryData->previousStage = QueryData::Stage::NumValues;
            queryData->setCallerDeviceIds(callerDeviceIds);

            std::vector<int> sgpuIndices(numCallerGpus);
            for(int g = 0; g < numCallerGpus; g++){
                auto deviceIdIter = std::find(deviceIds.begin(), deviceIds.end(), callerDeviceIds[g]);
                assert(deviceIdIter != deviceIds.end());
                const int sgpuIndex = std::distance(deviceIds.begin(), deviceIdIter);
                sgpuIndices[g] = sgpuIndex;
            }

            const int totalNumSequencesForAllGpus = std::reduce(vec_numSequences.begin(), vec_numSequences.end());

            for(int g = 0; g < numCallerGpus; g++){
                *vec_totalNumValues[g] = 0;
            }

            if(totalNumSequencesForAllGpus == 0){
                return;
            }

            std::vector<rmm::device_uvector<char>> vec_d_tempBuffers;
            std::vector<char*> vec_cubtemp(numCallerGpus, nullptr);
            std::vector<size_t> vec_cubtempbytes(numCallerGpus, 0);
            std::vector<kmer_type*> vec_d_signatures(numCallerGpus, nullptr);
            std::vector<bool*> vec_d_hashvalues_isValid(numCallerGpus, nullptr);
            std::vector<kmer_type*> vec_d_signatures_transposed(numCallerGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerHash(numCallerGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerHashExclPSVert(numCallerGpus, nullptr);

            // allocate and hash
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                const int numSequences = vec_numSequences[g];
                const int sgpuIndex = sgpuIndices[g];

                void* allocations[3]{};
                std::size_t allocation_sizes[3]{};
                std::size_t storage_bytes = 0;

                size_t cubbytes1 = 0;
                cub::DeviceReduce::Sum(
                    nullptr,
                    cubbytes1,
                    (int*)nullptr, 
                    (int*)nullptr, 
                    numSequences, 
                    callerStreams[g]
                );

                const size_t cubBytes = cubbytes1;
                allocation_sizes[0] = cubBytes;

                const int numHashFunctions = sgpuMinhashers[sgpuIndex]->getNumberOfMaps();
                allocation_sizes[1] = sizeof(kmer_type) * numSequences * numHashFunctions;
                allocation_sizes[2] = sizeof(bool) * numSequences * numHashFunctions;


                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));
                vec_d_tempBuffers.emplace_back(storage_bytes, callerStreams[g], mrs[g]);
                CUDACHECK(cub::AliasTemporaries(
                    vec_d_tempBuffers.back().data(),
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                vec_cubtemp[g] = (static_cast<char*>(allocations[0]));
                vec_cubtempbytes[g] = (cubBytes);
                vec_d_signatures[g] = (static_cast<kmer_type*>(allocations[1]));
                vec_d_hashvalues_isValid[g] = (static_cast<bool*>(allocations[2]));




                void* persistent_allocations[3]{};
                std::size_t persistent_allocation_sizes[3]{};
                std::size_t persistent_storage_bytes = 0;

                persistent_allocation_sizes[0] = sizeof(kmer_type) * numSequences * numHashFunctions; // d_sig_trans
                persistent_allocation_sizes[1] = sizeof(int) * numSequences * numHashFunctions; // d_numValuesPerSequencePerHash
                persistent_allocation_sizes[2] = sizeof(int) * numSequences * numHashFunctions; // d_numValuesPerSequencePerHashExclPSVert

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    persistent_storage_bytes,
                    persistent_allocations,
                    persistent_allocation_sizes
                ));
                queryData->vec_d_persistentBuffers[g].resize(persistent_storage_bytes, callerStreams[g]);
                CUDACHECK(cub::AliasTemporaries(
                    queryData->vec_d_persistentBuffers[g].data(),
                    persistent_storage_bytes,
                    persistent_allocations,
                    persistent_allocation_sizes
                ));
                vec_d_signatures_transposed[g] = static_cast<kmer_type*>(persistent_allocations[0]);
                vec_d_numValuesPerSequencePerHash[g] = static_cast<int*>(persistent_allocations[1]);
                vec_d_numValuesPerSequencePerHashExclPSVert[g] = static_cast<int*>(persistent_allocations[2]);

                if(numSequences > 0){
                    dim3 block(128,1,1);
                    dim3 grid(SDIV(numHashFunctions * numSequences, block.x),1,1);

                    gpusequencehasher::minhashSignatures3264Kernel<<<grid, block, 0, callerStreams[g]>>>(
                        vec_d_signatures[g],
                        numHashFunctions,
                        vec_d_hashvalues_isValid[g],
                        vec_d_sequenceData2Bit[g],
                        encodedSequencePitchInInts,
                        numSequences,
                        vec_d_sequenceLengths[g],
                        getKmerSize(),
                        numHashFunctions,
                        sgpuMinhashers[sgpuIndex]->d_currentHashFunctionNumbers.data()
                    ); CUDACHECKASYNC;
                }
            }

            //transpose
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                const int numSequences = vec_numSequences[g];
                const int sgpuIndex = sgpuIndices[g];
                if(numSequences > 0){

                    const int numHashFunctions = sgpuMinhashers[sgpuIndex]->getNumberOfMaps();
                    const std::size_t signaturesRowPitchElements = numHashFunctions;

                    helpers::call_transpose_kernel(
                        vec_d_signatures_transposed[g], 
                        vec_d_signatures[g],
                        numSequences, 
                        signaturesRowPitchElements, 
                        signaturesRowPitchElements,
                        callerStreams[g]
                    );
                }
            }

            //fix keys
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                const int numSequences = vec_numSequences[g];
                const int sgpuIndex = sgpuIndices[g];
                if(numSequences > 0){

                    const int numHashFunctions = sgpuMinhashers[sgpuIndex]->getNumberOfMaps();
                    fixKeysForGpuHashTable<Key, Value>(
                        vec_d_signatures[g], 
                        numSequences * numHashFunctions, 
                        callerStreams[g]
                    );
                }
            }

            //query num values
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                const int numSequences = vec_numSequences[g];
                const int sgpuIndex = sgpuIndices[g];
                if(numSequences > 0){
                    const int numHashFunctions = sgpuMinhashers[sgpuIndex]->getNumberOfMaps();

                    const int signaturesPitchInElements = numSequences;
                    const int numValuesPerKeyPitchInElements = numSequences;

                    dim3 block(256, 1, 1);
                    const int numBlocksPerTable = SDIV(numSequences, (block.x / cgsize));
                    dim3 grid(numBlocksPerTable, std::min(65535, numHashFunctions), 1);

                    gpuhashtablekernels::numValuesPerKeyCompactMultiTableKernel<<<grid, block, 0, callerStreams[g]>>>(
                        sgpuMinhashers[sgpuIndex]->d_deviceAccessibleTableViews.data(),
                        numHashFunctions,
                        getNumResultsPerMapThreshold(),
                        vec_d_signatures_transposed[g],
                        signaturesPitchInElements,
                        numSequences,
                        vec_d_numValuesPerSequencePerHash[g],
                        numValuesPerKeyPitchInElements
                    );
                    CUDACHECKASYNC
                }
            }

            //compute num values per sequence
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                const int numSequences = vec_numSequences[g];
                const int sgpuIndex = sgpuIndices[g];

                const int numHashFunctions = sgpuMinhashers[sgpuIndex]->getNumberOfMaps();

                //rmm::device_uvector<int> d_numValuesPerSequence(numSequences, callerStreams[g]);

                if(numSequences > 0){

                    sgpuminhasherkernels::accumulateNumValuesPerSequenceKernel<<<SDIV(numSequences, 256), 256, 0, callerStreams[g]>>>(
                        vec_d_numValuesPerSequencePerHash[g],
                        vec_d_numValuesPerSequencePerHashExclPSVert[g],
                        vec_d_numValuesPerSequence[g],
                        numSequences,
                        numHashFunctions
                    );
                    CUDACHECKASYNC;
                }

                //queryData->vec_d_numValuesPerSequence.push_back(std::move(d_numValuesPerSequence));
            }

            //compute total num values 
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                const int numSequences = vec_numSequences[g];
                const int sgpuIndex = sgpuIndices[g];

                if(numSequences > 0){

                    const int numHashFunctions = sgpuMinhashers[sgpuIndex]->getNumberOfMaps();
                    cub::DeviceReduce::Sum(
                        vec_cubtemp[g],
                        vec_cubtempbytes[g],
                        vec_d_numValuesPerSequence[g],
                        queryData->pinned_totalNumValues + g, //write to pinned memory
                        numSequences, 
                        callerStreams[g]
                    );   
                }
            }

            //dealloc
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                vec_d_tempBuffers[g].release();
            }

            //sync
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                CUDACHECK(cudaStreamSynchronize(callerStreams[g]));

                *vec_totalNumValues[g] = queryData->pinned_totalNumValues[g];
            }



            CUDACHECK(cudaSetDevice(currentDeviceId));
        }

        void multi_retrieveValues(
            MinhasherHandle& queryHandle,
            const std::vector<int>& vec_numSequences,
            const std::vector<const int*>& /*vec_totalNumValues*/,
            const std::vector<read_number*>& vec_d_values,
            const std::vector<const int*>& vec_d_numValuesPerSequence,
            const std::vector<int*> vec_d_offsets, //numSequences + 1
            const std::vector<cudaStream_t>& callerStreams,
            const std::vector<int>& callerDeviceIds,
            const std::vector<rmm::mr::device_memory_resource*>& /*mrs*/
        ) const {
            nvtx::ScopedRange sr_("multi_retrieveValues", 3);

            int currentDeviceId = 0;
            CUDACHECK(cudaGetDevice(&currentDeviceId));

            QueryData* const queryData = getQueryDataFromHandle(queryHandle);
            assert(queryData->previousStage == QueryData::Stage::NumValues);
            assert(queryData->callerDeviceIds == callerDeviceIds);
            queryData->previousStage = QueryData::Stage::Retrieve;

            const int numDataGpus = deviceIds.size();
            const int numCallerGpus = callerDeviceIds.size();

            std::vector<int> sgpuIndices(numCallerGpus);
            for(int g = 0; g < numCallerGpus; g++){
                auto deviceIdIter = std::find(deviceIds.begin(), deviceIds.end(), callerDeviceIds[g]);
                assert(deviceIdIter != deviceIds.end());
                const int sgpuIndex = std::distance(deviceIds.begin(), deviceIdIter);
                sgpuIndices[g] = sgpuIndex;
            }

            for(int g = 0; g < numCallerGpus; g++){
                if(vec_numSequences[g] == 0){
                    CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                    CUDACHECK(cudaMemsetAsync(vec_d_offsets[g], 0, sizeof(int) * (vec_numSequences[g] + 1), callerStreams[g]));
                }
            }

            const int totalNumSequencesForAllGpus = std::reduce(vec_numSequences.begin(), vec_numSequences.end());
            if(totalNumSequencesForAllGpus == 0){
                return;
            }

            std::vector<rmm::device_uvector<char>> vec_d_tempBuffers;
            std::vector<char*> vec_cubtemp(numCallerGpus, nullptr);
            std::vector<size_t> vec_cubtempbytes(numCallerGpus, 0);
            std::vector<int*> vec_d_queryOffsetsPerSequencePerHash(numCallerGpus, nullptr);

            std::vector<kmer_type*> vec_d_signatures_transposed(numCallerGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerHash(numCallerGpus, nullptr);
            std::vector<int*> vec_d_numValuesPerSequencePerHashExclPSVert(numCallerGpus, nullptr);

            // allocate and compute d_offsets
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                const int numSequences = vec_numSequences[g];
                const int sgpuIndex = sgpuIndices[g];

                const int numHashFunctions = sgpuMinhashers[sgpuIndex]->getNumberOfMaps();

                void* allocations[2]{};
                std::size_t allocation_sizes[2]{};
                std::size_t storage_bytes = 0;

                size_t cubbytes1 = 0;
                cub::DeviceScan::InclusiveSum(
                    nullptr,
                    cubbytes1,
                    (int*)nullptr, 
                    (int*)nullptr, 
                    numSequences, 
                    callerStreams[g]
                );

                const size_t cubBytes = cubbytes1;
                allocation_sizes[0] = cubBytes;
                allocation_sizes[1] = sizeof(int) * numSequences * numHashFunctions;


                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));
                vec_d_tempBuffers.emplace_back(storage_bytes, callerStreams[g]);
                CUDACHECK(cub::AliasTemporaries(
                    vec_d_tempBuffers.back().data(),
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                vec_cubtemp[g] = (static_cast<char*>(allocations[0]));
                vec_cubtempbytes[g] = (cubBytes);
                vec_d_queryOffsetsPerSequencePerHash[g] = (static_cast<int*>(allocations[1]));


                void* persistent_allocations[3]{};
                std::size_t persistent_allocation_sizes[3]{};
                std::size_t persistent_storage_bytes = queryData->vec_d_persistentBuffers[g].size();

                persistent_allocation_sizes[0] = sizeof(kmer_type) * numSequences * numHashFunctions; // d_sig_trans
                persistent_allocation_sizes[1] = sizeof(int) * numSequences * numHashFunctions; // d_numValuesPerSequencePerHash
                persistent_allocation_sizes[2] = sizeof(int) * numSequences * numHashFunctions; // d_numValuesPerSequencePerHashExclPSVert

                CUDACHECK(cub::AliasTemporaries(
                    queryData->vec_d_persistentBuffers[g].data(),
                    persistent_storage_bytes,
                    persistent_allocations,
                    persistent_allocation_sizes
                ));
                vec_d_signatures_transposed[g] = static_cast<kmer_type*>(persistent_allocations[0]);
                vec_d_numValuesPerSequencePerHash[g] = static_cast<int*>(persistent_allocations[1]);
                vec_d_numValuesPerSequencePerHashExclPSVert[g] = static_cast<int*>(persistent_allocations[2]);

                if(numSequences > 0){
                    CUDACHECK(cudaMemsetAsync(
                        vec_d_offsets[g], 
                        0, 
                        sizeof(int), 
                        callerStreams[g]
                    ));
                    cub::DeviceScan::InclusiveSum(
                        vec_cubtemp[g],
                        vec_cubtempbytes[g],
                        vec_d_numValuesPerSequence[g], 
                        vec_d_offsets[g] + 1, 
                        numSequences, 
                        callerStreams[g]
                    );
                }
            }

            //computeQueryDestinationOffsets
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                const int numSequences = vec_numSequences[g];
                const int sgpuIndex = sgpuIndices[g];

                const int numHashFunctions = sgpuMinhashers[sgpuIndex]->getNumberOfMaps();
                if(numSequences > 0){
                    sgpuminhasherkernels::computeQueryDestinationOffsetsKernel<<<SDIV(numSequences, 256), 256, 0, callerStreams[g]>>>(
                        vec_d_queryOffsetsPerSequencePerHash[g],
                        vec_d_numValuesPerSequencePerHashExclPSVert[g],
                        numSequences,
                        numHashFunctions,
                        vec_d_offsets[g]
                    );
                    CUDACHECKASYNC
                }
            }

            //query values
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                const int numSequences = vec_numSequences[g];
                const int sgpuIndex = sgpuIndices[g];

                const int numHashFunctions = sgpuMinhashers[sgpuIndex]->getNumberOfMaps();
                if(numSequences > 0){
                    const int signaturesPitchInElements = numSequences;
                    const int numValuesPerKeyPitchInElements = numSequences;
                    const int beginOffsetsPitchInElements = numSequences;

                    dim3 block(256, 1, 1);
                    const int numBlocksPerTable = SDIV(numSequences, (block.x / cgsize));
                    dim3 grid(numBlocksPerTable, std::min(65535, numHashFunctions), 1);

                    gpuhashtablekernels::retrieveCompactKernel<<<grid, block, 0, callerStreams[g]>>>(
                        sgpuMinhashers[sgpuIndex]->d_deviceAccessibleTableViews.data(),
                        numHashFunctions,
                        vec_d_signatures_transposed[g],
                        signaturesPitchInElements,
                        vec_d_queryOffsetsPerSequencePerHash[g],
                        beginOffsetsPitchInElements,
                        vec_d_numValuesPerSequencePerHash[g],
                        numValuesPerKeyPitchInElements,
                        getNumResultsPerMapThreshold(),
                        numSequences,
                        vec_d_values[g]
                    );
                    CUDACHECKASYNC
                }
            }

            //deallocate
            for(int g = 0; g < numCallerGpus; g++){
                CUDACHECK(cudaSetDevice(callerDeviceIds[g]));
                vec_d_tempBuffers[g].release();
            }


            CUDACHECK(cudaSetDevice(currentDeviceId));
        }


        void compact(cudaStream_t) override {      
        }

        void constructionIsFinished(cudaStream_t) override{            
        }

        MemoryUsage getMemoryInfo() const noexcept override{
            MemoryUsage mem{};

            for(const auto& minhasher : sgpuMinhashers){
                mem += minhasher->getMemoryInfo();
            }

            return mem;
        }

        MemoryUsage getMemoryInfo(const MinhasherHandle& handle) const noexcept override{
            return tempdataVector[handle.getId()]->getMemoryInfo();
        }

        int getNumResultsPerMapThreshold() const noexcept override{
            return sgpuMinhashers[0]->getNumResultsPerMapThreshold();
        }
        
        int getNumberOfMaps() const noexcept override{
            return sgpuMinhashers[0]->getNumberOfMaps();
        }

        int getKmerSize() const noexcept override{
            return sgpuMinhashers[0]->getKmerSize();
        }

        void destroy(){
            for(auto& minhasher : sgpuMinhashers){
                DeviceSwitcher sd(minhasher->getDeviceId());
                minhasher->destroy();
            }
        }

        bool hasGpuTables() const noexcept override {
            return true;
        }

        void setThreadPool(ThreadPool* /*tp*/) override {}

        void setHostMemoryLimitForConstruction(std::size_t /*bytes*/) override{

        }

        void setDeviceMemoryLimitsForConstruction(const std::vector<std::size_t>&) override {

        }

        void writeToStream(std::ostream& /*os*/) const override{
            std::cerr << "ReplicatedSingleGpuMinhasher::writeToStream not supported\n";
        }

        int loadFromStream(std::ifstream& /*is*/, int /*numMapsUpperLimit*/) override{
            std::cerr << "ReplicatedSingleGpuMinhasher::loadFromStream not supported\n";
            return 0;
        } 

        bool canWriteToStream() const noexcept override { return false; };
        bool canLoadFromStream() const noexcept override { return false; };

private:

        std::unique_ptr<QueryData> createQueryData() const{
            auto ptr = std::make_unique<QueryData>();
            const int numMinhashers = sgpuMinhashers.size();

            int numGpus = 0;
            CUDACHECK(cudaGetDeviceCount(&numGpus));

            size_t allocation_sizes[1];
            allocation_sizes[0] = sizeof(int) * numGpus; // pinned_totalNumValues

            void* allocations[1]{};

            size_t temp_storage_bytes = 0;

            CUDACHECK(cub::AliasTemporaries(
                nullptr,
                temp_storage_bytes,
                allocations,
                allocation_sizes
            ));

            ptr->pinnedData.resize(temp_storage_bytes);

            CUDACHECK(cub::AliasTemporaries(
                ptr->pinnedData.data(),
                temp_storage_bytes,
                allocations,
                allocation_sizes
            ));
            
            ptr->pinned_totalNumValues = static_cast<int*>(allocations[0]);

            for(int i = 0; i < numMinhashers; i++){
                DeviceSwitcher ds(sgpuMinhashers[i]->getDeviceId());

                ptr->dataDeviceIds.push_back(sgpuMinhashers[i]->getDeviceId());
                ptr->singlegpuMinhasherHandles.emplace_back(std::make_unique<MinhasherHandle>(sgpuMinhashers[i]->makeMinhasherHandle()));
                //ptr->vec_d_singlegpuminhasherBuffers.emplace_back(0, cudaStreamPerThread);
                CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));
            }
            return ptr;
        }

        std::uint64_t getKmerMask() const{
            constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

            return std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - getKmerSize()) * 2);
        }

        QueryData* getQueryDataFromHandle(const MinhasherHandle& queryHandle) const{
            std::shared_lock<SharedMutex> lock(sharedmutex);

            return tempdataVector[queryHandle.getId()].get();
        }       

        mutable int counter = 0;
        mutable SharedMutex sharedmutex{};

        std::vector<int> deviceIds;
        std::vector<std::unique_ptr<SingleGpuMinhasher>> sgpuMinhashers{};
        mutable std::vector<std::unique_ptr<QueryData>> tempdataVector{};
    };


}
}




#endif

#endif //#ifdef CARE_HAS_WARPCORE

