

#include <gpu/correct_gpu.hpp>
#include <gpu/distributedreadstorage.hpp>
#include <gpu/nvtxtimelinemarkers.hpp>
#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/minhashkernels.hpp>
#include <gpu/gpumsa.cuh>

#include <correctionresultprocessing.hpp>
#include <memorymanagement.hpp>
#include <config.hpp>
#include <qualityscoreweights.hpp>
#include <sequence.hpp>

//#include <minhasher.hpp>
#include <options.hpp>
#include <rangegenerator.hpp>
#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>
#include <filehelpers.hpp>
#include <concurrencyhelpers.hpp>

#include <hpc_helpers.cuh>

#include <cuda_profiler_api.h>

#include <memory>
#include <sstream>
#include <cstdlib>
#include <thread>
#include <vector>
#include <string>
#include <condition_variable>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <queue>
#include <unordered_map>

#include <omp.h>

#include <cub/cub.cuh>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;



#define USE_MSA_MINIMIZATION

#ifdef USE_MSA_MINIMIZATION
    constexpr int max_num_minimizations = 5;
#else     
    constexpr int max_num_minimizations = 0;
#endif   



//#define DO_PROFILE

#ifdef DO_PROFILE
    constexpr size_t num_reads_to_profile = 100000;
#endif


//#define USE_CUDA_GRAPH


namespace care{
namespace gpu{

    //using Minhasher_t = Minhasher;
    using Minhasher_t = GpuMinhasher;

    template<int gridsize, int blocksize>
    __global__
    void setAnchorIndicesOfCandidateskernel(
        int* __restrict__ d_anchorIndicesOfCandidates,
        const int* __restrict__ numAnchorsPtr,
        const int* __restrict__ d_candidates_per_subject,
        const int* __restrict__ d_candidates_per_subject_prefixsum
    ){
        for(int anchorIndex = blockIdx.x; anchorIndex < *numAnchorsPtr; anchorIndex += gridsize){
            const int offset = d_candidates_per_subject_prefixsum[anchorIndex];
            const int numCandidatesOfAnchor = d_candidates_per_subject[anchorIndex];
            int* const beginptr = &d_anchorIndicesOfCandidates[offset];

            for(int localindex = threadIdx.x; localindex < numCandidatesOfAnchor; localindex += blocksize){
                beginptr[localindex] = anchorIndex;
            }
        }
    }


    template<int blocksize, class Flags>
    __global__
    void selectIndicesOfFlagsOnlyOneBlock(
        int* __restrict__ selectedIndices,
        int* __restrict__ numSelectedIndices,
        const Flags flags,
        const int* __restrict__ numFlags
    ){
        constexpr int ITEMS_PER_THREAD = 4;

        using BlockScan = cub::BlockScan<int, blocksize>;

        __shared__ typename BlockScan::TempStorage temp_storage;

        int aggregate = 0;
        const int iters = SDIV(*numFlags, blocksize * ITEMS_PER_THREAD);
        const int threadoffset = ITEMS_PER_THREAD * threadIdx.x;

        for(int iter = 0; iter < iters; iter++){

            int data[ITEMS_PER_THREAD];
            int prefixsum[ITEMS_PER_THREAD];

            const int iteroffset = blocksize * ITEMS_PER_THREAD * iter;

            #pragma unroll
            for(int k = 0; k < ITEMS_PER_THREAD; k++){
                if(iteroffset + threadoffset + k < *numFlags){
                    data[k] = flags[iteroffset + threadoffset + k];
                }else{
                    data[k] = 0;
                }
            }

            int block_aggregate = 0;
            BlockScan(temp_storage).ExclusiveSum(data, prefixsum, block_aggregate);

            #pragma unroll
            for(int k = 0; k < ITEMS_PER_THREAD; k++){
                if(data[k] == 1){
                    selectedIndices[prefixsum[k]] = iteroffset + threadoffset + k;
                }
            }

            aggregate += block_aggregate;

            __syncthreads();
        }

        if(threadIdx.x == 0){
            *numSelectedIndices = aggregate;
        }

    }


    //read status bitmask
    constexpr std::uint8_t readCorrectedAsHQAnchor = 1;
    constexpr std::uint8_t readCouldNotBeCorrectedAsAnchor = 2;

    constexpr int primary_stream_index = 0;
    constexpr int secondary_stream_index = 1;
    constexpr int preparation_stream_index = 2;
    constexpr int nStreamsPerBatch = 3;

    constexpr int correction_finished_event_index = 0;
    constexpr int result_transfer_finished_event_index = 1;
    constexpr int alignment_data_transfer_h2d_finished_event_index = 2;
    constexpr int nEventsPerBatch = 3;

    constexpr int doNotUseEditsValue = -1;





    struct CudaGraph{
        bool valid = false;
        cudaGraphExec_t execgraph = nullptr;

        CudaGraph() = default;
        
        CudaGraph(const CudaGraph&) = delete;
        CudaGraph& operator=(const CudaGraph&) = delete;

        CudaGraph(CudaGraph&& rhs){
            *this = std::move(rhs);
        }

        CudaGraph& operator=(CudaGraph&& rhs){
            destroy();

            valid = std::exchange(rhs.valid, false);
            execgraph = std::exchange(rhs.execgraph, nullptr);

            return *this;
        }

        ~CudaGraph(){
            destroy();
        }

        void destroy(){
            if(execgraph != nullptr){
                cudaGraphExecDestroy(execgraph); CUERR;
            }
            execgraph = nullptr;
            valid = false;
        }

        template<class Func>
        void capture(Func&& func){
            if(execgraph != nullptr){
                cudaGraphExecDestroy(execgraph); CUERR;
            }

            cudaStream_t stream;
            cudaStreamCreate(&stream); CUERR;
            
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed); CUERR;

            func(stream);

            cudaGraph_t graph;
            cudaStreamEndCapture(stream, &graph); CUERR;
            
            cudaGraphExec_t execGraph;
            cudaGraphNode_t errorNode;
            auto logBuffer = std::make_unique<char[]>(1025);
            std::fill_n(logBuffer.get(), 1025, 0);
            cudaError_t status = cudaGraphInstantiate(&execGraph, graph, &errorNode, logBuffer.get(), 1025);
            if(status != cudaSuccess){
                if(logBuffer[1024] != '\0'){
                    std::cerr << "cudaGraphInstantiate: truncated error message: ";
                    std::copy_n(logBuffer.get(), 1025, std::ostream_iterator<char>(std::cerr, ""));
                    std::cerr << "\n";
                }else{
                    std::cerr << "cudaGraphInstantiate: error message: ";
                    std::cerr << logBuffer.get();
                    std::cerr << "\n";
                }
                CUERR;
            }            

            cudaGraphDestroy(graph); CUERR;

            execgraph = execGraph;
            valid = true;

            cudaStreamDestroy(stream); CUERR;
        }

        void execute(cudaStream_t stream){
            assert(valid);
            
            cudaGraphLaunch(execgraph, stream); CUERR;
        }
    };






    struct NextIterationData{
        static constexpr int overprovisioningPercent = 0;

        template<class T>
        using DeviceBuffer = SimpleAllocationDevice<T, overprovisioningPercent>;
        
        template<class T>
        using PinnedBuffer = SimpleAllocationPinnedHost<T, overprovisioningPercent>;

        PinnedBuffer<unsigned int> h_subject_sequences_data;
        PinnedBuffer<int> h_subject_sequences_lengths;
        PinnedBuffer<read_number> h_subject_read_ids;
        PinnedBuffer<read_number> h_candidate_read_ids;
        PinnedBuffer<int> h_numAnchors;
        PinnedBuffer<int> h_numCandidates;
        DeviceBuffer<unsigned int> d_subject_sequences_data;
        DeviceBuffer<int> d_subject_sequences_lengths;
        DeviceBuffer<read_number> d_subject_read_ids;
        DeviceBuffer<read_number> d_candidate_read_ids;
        DeviceBuffer<int> d_candidates_per_subject;
        DeviceBuffer<int> d_candidates_per_subject_tmp;
        DeviceBuffer<int> d_candidates_per_subject_prefixsum;
        DeviceBuffer<int> d_numAnchors;
        DeviceBuffer<int> d_numCandidates;

        //private buffers
        PinnedBuffer<read_number> h_leftoverAnchorReadIds;
        PinnedBuffer<int> h_numLeftoverAnchors;
        PinnedBuffer<int> h_numLeftoverCandidates;
        DeviceBuffer<int> d_numLeftoverAnchors;
        DeviceBuffer<int> d_leftoverAnchorLengths;
        DeviceBuffer<read_number> d_leftoverAnchorReadIds;
        DeviceBuffer<int> d_numLeftoverCandidates;
        DeviceBuffer<read_number> d_leftoverCandidateReadIds;
        DeviceBuffer<int> d_leftoverCandidatesPerAnchors;
        DeviceBuffer<unsigned int> d_leftoverAnchorSequences;

        DeviceBuffer<char> d_cubTemp;

        int n_new_subjects = -1;
        int numCandidatesLimit = 0;
        int batchsize = 0;
        int resultsPerMap = 0;
        std::size_t encodedSequencePitchInInts = 0;

        const GpuMinhasher* minhasher;
        GpuMinhasher::QueryHandle minhasherQueryHandle;

        const DistributedReadStorage* readStorage;
        DistributedReadStorage::GatherHandleSequences subjectSequenceGatherHandle;

        cudaStream_t stream;
        cudaEvent_t event;
        int deviceId;

        ThreadPool* threadPool;
        ThreadPool::ParallelForHandle pforHandle;

        cpu::RangeGenerator<read_number>* readIdGenerator;

        MergeRangesGpuHandle<read_number> mergeRangesGpuHandle;

        SyncFlag syncFlag;

        struct Pointers{
            void* h_subject_sequences_data{};
            void* h_candidate_read_ids{};
            bool operator<(const Pointers rhs) const{
                if(h_subject_sequences_data < rhs.h_subject_sequences_data){
                    return true;
                }
                if(h_subject_sequences_data > rhs.h_subject_sequences_data){
                    return false;
                }
                if(h_candidate_read_ids < rhs.h_candidate_read_ids){
                    return true;
                }
                return false;
            }
        };

        std::map<Pointers, CudaGraph> leftoverCalculationExecutionGraphs;

        void init(   
            const GpuMinhasher& minhasher, 
            const DistributedReadStorage* readStorage, 
            ThreadPool* threadPool,
            cpu::RangeGenerator<read_number>* readIdGenerator,
            int deviceId,
            int batchsize,
            int numCandidatesLimit,
            size_t encodedSequencePitchInInts,
            int maxNumThreads,
            int numMinhashMaps,
            int resultsPerMap
        ){
            NextIterationData& nextData = *this;

            nextData.minhasher = &minhasher;
            nextData.readStorage = readStorage;
            nextData.threadPool = threadPool;
            nextData.deviceId = deviceId;
            nextData.numCandidatesLimit = numCandidatesLimit;
            nextData.encodedSequencePitchInInts = encodedSequencePitchInInts;
            nextData.batchsize = batchsize;
            nextData.resultsPerMap = resultsPerMap;
            nextData.readIdGenerator = readIdGenerator;

            nextData.subjectSequenceGatherHandle = readStorage->makeGatherHandleSequences();
    
            cudaSetDevice(deviceId); CUERR;
            cudaStreamCreate(&nextData.stream); CUERR;
            cudaEventCreate(&nextData.event); CUERR;
    
            nextData.mergeRangesGpuHandle = makeMergeRangesGpuHandle<read_number>();
            nextData.minhasherQueryHandle = GpuMinhasher::makeQueryHandle();

            nextData.minhasherQueryHandle.resize(minhasher, batchsize, maxNumThreads);
    
            nextData.d_numLeftoverAnchors.resize(1);
            nextData.d_numLeftoverCandidates.resize(1);
            nextData.h_numLeftoverAnchors.resize(1);
            nextData.h_numLeftoverCandidates.resize(1);
            nextData.h_numAnchors.resize(1);
            nextData.h_numCandidates.resize(1);
            nextData.d_numAnchors.resize(1);
            nextData.d_numCandidates.resize(1);
    
            cudaMemsetAsync(nextData.d_numLeftoverAnchors.get(), 0, sizeof(int), nextData.stream); CUERR;
            cudaMemsetAsync(nextData.d_numLeftoverCandidates.get(), 0, sizeof(int), nextData.stream); CUERR;
    
            nextData.h_numLeftoverAnchors[0] = 0;
            nextData.h_numLeftoverCandidates[0] = 0;
        
            nextData.h_subject_sequences_data.resize(encodedSequencePitchInInts * batchsize);
            nextData.d_subject_sequences_data.resize(encodedSequencePitchInInts * batchsize);
            nextData.h_subject_sequences_lengths.resize(batchsize);
            nextData.d_subject_sequences_lengths.resize(batchsize);
            nextData.h_subject_read_ids.resize(batchsize);
            nextData.d_subject_read_ids.resize(batchsize);
                
            const int maxNumIds = resultsPerMap * numMinhashMaps * batchsize;
    
            nextData.h_candidate_read_ids.resize(maxNumIds);
            nextData.d_candidate_read_ids.resize(maxNumIds + numCandidatesLimit);
            nextData.d_candidates_per_subject.resize(2*batchsize);
            nextData.d_candidates_per_subject_tmp.resize(2*batchsize);
            nextData.d_candidates_per_subject_prefixsum.resize(batchsize+1);

            cudaMemsetAsync(nextData.d_candidates_per_subject_prefixsum.get(), 0, sizeof(int), nextData.stream); CUERR;
    
            nextData.h_leftoverAnchorReadIds.resize(batchsize);
            nextData.d_leftoverAnchorReadIds.resize(batchsize);
            nextData.d_leftoverAnchorLengths.resize(batchsize);
            nextData.d_leftoverCandidateReadIds.resize(maxNumIds + numCandidatesLimit);
            nextData.d_leftoverCandidatesPerAnchors.resize(batchsize);
            nextData.d_leftoverAnchorSequences.resize(encodedSequencePitchInInts * batchsize);

            std::size_t cubBytes = 0;

            cub::DeviceScan::InclusiveSum(
                nullptr, 
                cubBytes,
                (int*)nullptr,
                (int*)nullptr,
                batchsize,
                nextData.stream
            ); CUERR;

            nextData.d_cubTemp.resize(cubBytes);
    
            cudaStreamSynchronize(nextData.stream);
        }
    
        void destroy(){
            NextIterationData& nextData = *this;

            cudaSetDevice(nextData.deviceId); CUERR;
            cudaStreamDestroy(nextData.stream); CUERR;
            cudaEventDestroy(nextData.event); CUERR;
    
            nextData.h_subject_sequences_data.destroy();
            nextData.h_subject_sequences_lengths.destroy();
            nextData.h_subject_read_ids.destroy();
            nextData.h_candidate_read_ids.destroy();
    
            nextData.d_subject_sequences_data.destroy();
            nextData.d_subject_sequences_lengths.destroy();
            nextData.d_subject_read_ids.destroy();
            nextData.d_candidate_read_ids.destroy();
            nextData.d_candidates_per_subject.destroy();
            nextData.d_candidates_per_subject_tmp.destroy();
            nextData.d_candidates_per_subject_prefixsum.destroy();
    
            nextData.d_numLeftoverAnchors.destroy();
            nextData.d_leftoverAnchorLengths.destroy();
            nextData.d_leftoverAnchorReadIds.destroy();
            nextData.d_numLeftoverCandidates.destroy();
            nextData.d_leftoverCandidateReadIds.destroy();
            nextData.d_leftoverCandidatesPerAnchors.destroy();
            nextData.h_numLeftoverAnchors.destroy();
            nextData.h_numLeftoverCandidates.destroy();
            nextData.d_leftoverAnchorSequences.destroy();
    
            nextData.h_leftoverAnchorReadIds.destroy();
            nextData.d_cubTemp.destroy();
    
            destroyMergeRangesGpuHandle(nextData.mergeRangesGpuHandle);
            GpuMinhasher::destroyQueryHandle(nextData.minhasherQueryHandle);
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

            handlehost(h_subject_sequences_data);
            handlehost(h_subject_sequences_lengths);
            handlehost(h_subject_read_ids);
            handlehost(h_candidate_read_ids);
            handlehost(h_numAnchors);
            handlehost(h_numCandidates);
            handlehost(h_leftoverAnchorReadIds);
            handlehost(h_numLeftoverAnchors);
            handlehost(h_numLeftoverCandidates);

            handledevice(d_subject_sequences_data);
            handledevice(d_subject_sequences_lengths);
            handledevice(d_subject_read_ids);
            handledevice(d_candidate_read_ids);
            handledevice(d_candidates_per_subject);
            handledevice(d_candidates_per_subject_tmp);
            handledevice(d_candidates_per_subject_prefixsum);
            handledevice(d_numAnchors);
            handledevice(d_numCandidates);
            handledevice(d_numLeftoverAnchors);
            handledevice(d_leftoverAnchorLengths);
            handledevice(d_leftoverAnchorReadIds);
            handledevice(d_numLeftoverCandidates);
            handledevice(d_leftoverCandidateReadIds);
            handledevice(d_leftoverCandidatesPerAnchors);
            handledevice(d_leftoverAnchorSequences);

            handledevice(d_cubTemp);

            auto queryHandleInfo = minhasherQueryHandle.getMemoryInfo();

            info.host += queryHandleInfo.host;

            for(auto pair : queryHandleInfo.device){
                info.device[pair.first] += pair.second;
            }

            return info;
        }

        void leftoverCalculation(cudaStream_t syncStream){
            auto& nextData = *this;

            std::size_t cubTempBytes = nextData.d_cubTemp.capacityInBytes();
            void* cubTemp = nextData.d_cubTemp.get();
            //d_candidates_per_subject_prefixsum[0] is 0
            cub::DeviceScan::InclusiveSum(
                cubTemp, 
                cubTempBytes,
                nextData.d_leftoverCandidatesPerAnchors.get(),
                nextData.d_candidates_per_subject_prefixsum.get() + 1,
                nextData.batchsize,
                syncStream
            ); CUERR;
    
            //find new numbers of leftover candidates and anchors

            generic_kernel<<<1, 1, 0, syncStream>>>(
                [
                    d_candidates_per_subject_prefixsum = nextData.d_candidates_per_subject_prefixsum.get(),
                    d_numAnchors = nextData.d_numAnchors.get(),
                    d_numCandidates = nextData.d_numCandidates.get(),
                    d_numLeftoverAnchors = nextData.d_numLeftoverAnchors.get(),
                    d_numLeftoverCandidates = nextData.d_numLeftoverCandidates.get(),
                    numCandidatesLimit = nextData.numCandidatesLimit
                ]__device__(){
                    const int numAnchors = *d_numAnchors; // leftover + new anchors

                    const int totalNumCandidates = d_candidates_per_subject_prefixsum[numAnchors];

                    if(totalNumCandidates - numCandidatesLimit > 0){

                        //find the first anchor index which is left over
                        auto iter = thrust::lower_bound(
                            thrust::seq,
                            d_candidates_per_subject_prefixsum,
                            d_candidates_per_subject_prefixsum + numAnchors + 1,
                            numCandidatesLimit
                        );

                        const int index = thrust::distance(d_candidates_per_subject_prefixsum, iter) - 1;
    
                        const int newNumLeftoverAnchors = numAnchors - index;
                        *d_numLeftoverAnchors = newNumLeftoverAnchors;
                        *d_numAnchors = numAnchors - newNumLeftoverAnchors;

                        if(index < numAnchors){

                            const int newNumLeftoverCandidates = totalNumCandidates - d_candidates_per_subject_prefixsum[index];
                            
                            *d_numLeftoverCandidates = newNumLeftoverCandidates;
                            *d_numCandidates = totalNumCandidates - newNumLeftoverCandidates;
                        }else{
                            *d_numLeftoverCandidates = 0;
                            *d_numCandidates = totalNumCandidates - 0;
                        }
                    }else{
                        *d_numLeftoverAnchors = 0;
                        *d_numLeftoverCandidates = 0;
                        *d_numAnchors = numAnchors - 0;
                        *d_numCandidates = totalNumCandidates - 0;
                    }
                }
            ); CUERR;
 
                
            //copy all data from leftover buffers to output buffers
            generic_kernel<<<240, 256, 0, syncStream>>>(
                [
                    d_numAnchors = nextData.d_numAnchors.get(),
                    d_numCandidates = nextData.d_numCandidates.get(),
                    d_numLeftoverAnchors = nextData.d_numLeftoverAnchors.get(),
                    d_numLeftoverCandidates = nextData.d_numLeftoverCandidates.get(),
                    d_candidates_per_subject = nextData.d_candidates_per_subject.get(),
                    d_leftoverCandidatesPerAnchors = nextData.d_leftoverCandidatesPerAnchors.get(),
        
                    d_leftoverAnchorSequences = nextData.d_leftoverAnchorSequences.get(),
                    d_leftoverAnchorLengths = nextData.d_leftoverAnchorLengths.get(),
                    d_leftoverAnchorReadIds = nextData.d_leftoverAnchorReadIds.get(),
                    d_leftoverCandidateReadIds = nextData.d_leftoverCandidateReadIds.get(),
                    
                    d_subject_read_ids = nextData.d_subject_read_ids.get(),
                    d_subject_sequences_lengths = nextData.d_subject_sequences_lengths.get(),
                    d_candidate_read_ids = nextData.d_candidate_read_ids.get(),
                    d_subject_sequences_data = nextData.d_subject_sequences_data.get(),
                    encodedSequencePitchInInts = nextData.encodedSequencePitchInInts
                ]__device__(){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;
    
                    const int numAnchors = *d_numAnchors;
                    const int numCandidates = *d_numCandidates;
                    const int numLeftoverAnchors = *d_numLeftoverAnchors;
                    const int numLeftoverCandidates = *d_numLeftoverCandidates;

                    const int anchorsToCopy = numAnchors + numLeftoverAnchors;
                    const int candidatesToCopy = numCandidates + numLeftoverCandidates;
    
    
                    for(int i = tid; i < anchorsToCopy; i += stride){
                        d_subject_read_ids[i] = d_leftoverAnchorReadIds[i];
                        d_subject_sequences_lengths[i] = d_leftoverAnchorLengths[i];
                        d_candidates_per_subject[i] = d_leftoverCandidatesPerAnchors[i];
                    }
    
                    for(int i = tid; i < anchorsToCopy * encodedSequencePitchInInts; i += stride){
                        d_subject_sequences_data[i] = d_leftoverAnchorSequences[i];
                    }

                    for(int i = tid; i < candidatesToCopy; i += stride){
                        d_candidate_read_ids[i] = d_leftoverCandidateReadIds[i];
                    }
                }
            ); CUERR;

            //copy new leftover data from output buffers to the front of leftover buffers
            generic_kernel<<<240, 256, 0, syncStream>>>(
                [
                    d_numAnchors = nextData.d_numAnchors.get(),
                    d_numCandidates = nextData.d_numCandidates.get(),
                    d_numLeftoverAnchors = nextData.d_numLeftoverAnchors.get(),
                    d_numLeftoverCandidates = nextData.d_numLeftoverCandidates.get(),
                    d_candidates_per_subject = nextData.d_candidates_per_subject.get(),
                    d_leftoverCandidatesPerAnchors = nextData.d_leftoverCandidatesPerAnchors.get(),
        
                    d_leftoverAnchorSequences = nextData.d_leftoverAnchorSequences.get(),
                    d_leftoverAnchorLengths = nextData.d_leftoverAnchorLengths.get(),
                    d_leftoverAnchorReadIds = nextData.d_leftoverAnchorReadIds.get(),
                    d_leftoverCandidateReadIds = nextData.d_leftoverCandidateReadIds.get(),
                    
                    d_subject_read_ids = nextData.d_subject_read_ids.get(),
                    d_subject_sequences_lengths = nextData.d_subject_sequences_lengths.get(),
                    d_candidate_read_ids = nextData.d_candidate_read_ids.get(),
                    d_subject_sequences_data = nextData.d_subject_sequences_data.get(),
                    encodedSequencePitchInInts = nextData.encodedSequencePitchInInts
                ]__device__(){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;
    
                    const int numAnchors = *d_numAnchors;
                    const int numCandidates = *d_numCandidates;
                    const int numLeftoverAnchors = *d_numLeftoverAnchors;
                    const int numLeftoverCandidates = *d_numLeftoverCandidates;    
    
                    for(int i = tid; i < numLeftoverAnchors; i += stride){
                        d_leftoverAnchorReadIds[i] = d_subject_read_ids[numAnchors + i];
                        d_leftoverAnchorLengths[i] = d_subject_sequences_lengths[numAnchors + i];
                        d_leftoverCandidatesPerAnchors[i] = d_candidates_per_subject[numAnchors + i];
                    }
    
                    for(int i = tid; i < numLeftoverAnchors * encodedSequencePitchInInts; i += stride){
                        d_leftoverAnchorSequences[i] 
                            = d_subject_sequences_data[numAnchors * encodedSequencePitchInInts + i];
                    }

                    for(int i = tid; i < numLeftoverCandidates; i += stride){
                        d_leftoverCandidateReadIds[i] = d_candidate_read_ids[numCandidates + i];
                    }
                }
            ); CUERR;

            //copy data from device to host
            generic_kernel<<<240, 256, 0, syncStream>>>(
                [
                    h_numAnchors = nextData.h_numAnchors.get(),
                    d_numAnchors = nextData.d_numAnchors.get(),
                    h_numCandidates = nextData.h_numCandidates.get(),
                    d_numCandidates = nextData.d_numCandidates.get(),
                    h_numLeftoverAnchors = nextData.h_numLeftoverAnchors.get(),
                    d_numLeftoverAnchors = nextData.d_numLeftoverAnchors.get(),
                    h_numLeftoverCandidates = nextData.h_numLeftoverCandidates.get(),
                    d_numLeftoverCandidates = nextData.d_numLeftoverCandidates.get(),
                    h_candidate_read_ids = nextData.h_candidate_read_ids.get(),
                    d_candidate_read_ids = nextData.d_candidate_read_ids.get(),
                    h_leftoverAnchorReadIds = nextData.h_leftoverAnchorReadIds.get(),
                    d_leftoverAnchorReadIds = nextData.d_leftoverAnchorReadIds.get(),
                    h_subject_read_ids = nextData.h_subject_read_ids.get(),
                    d_subject_read_ids = nextData.d_subject_read_ids.get()
                ]__device__(){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    auto copy = [&](auto* dest, auto* src, size_t numbytes){
                        using CopyType = int4;

                        const size_t iters = numbytes / sizeof(CopyType);
    
                        for(size_t index = tid; index < iters; index += stride){
                            ((CopyType*)dest)[index] = ((const CopyType*)src)[index];
                        }

                        const size_t remainder = numbytes - sizeof(CopyType) * iters;
                        for(size_t index = tid; index < remainder; index += stride){
                            ((char*)dest)[sizeof(CopyType) * iters + index] 
                                = ((const char*)src)[sizeof(CopyType) * iters + index];
                        }
                    };

                    if(tid == 0){
                        *h_numAnchors = *d_numAnchors;
                        *h_numCandidates = *d_numCandidates;
                        *h_numLeftoverAnchors = *d_numLeftoverAnchors;
                        *h_numLeftoverCandidates = *d_numLeftoverCandidates;
                    }

                    copy(
                        h_candidate_read_ids, 
                        d_candidate_read_ids, 
                        *d_numCandidates * sizeof(read_number)
                    );

                    copy(
                        h_leftoverAnchorReadIds, 
                        d_leftoverAnchorReadIds, 
                        *d_numLeftoverAnchors * sizeof(read_number)
                    );

                    copy(
                        h_subject_read_ids, 
                        d_subject_read_ids, 
                        *d_numAnchors * sizeof(read_number)
                    );
                }
            ); CUERR;
        }

        void executeLeftoverCalculation(cudaStream_t syncStream){

#ifndef USE_CUDA_GRAPH

            leftoverCalculation(syncStream);

#else 

            Pointers pointers;
            pointers.h_subject_sequences_data = h_subject_sequences_data.get();
            pointers.h_candidate_read_ids = h_candidate_read_ids.get();

            auto& graph = leftoverCalculationExecutionGraphs[pointers];

            if(graph.valid){
                graph.execute(syncStream);
            }else{
                graph.capture(
                    [this](cudaStream_t stream){
                        leftoverCalculation(stream);
                    }
                );
                assert(graph.valid);
                graph.execute(syncStream);
            }

#endif
        }

        void prepareNewDataForCorrection(cudaStream_t syncStream){
            NextIterationData& nextData = *this;
       
            cudaSetDevice(nextData.deviceId); CUERR;
    
            const size_t encodedSequencePitchInInts = nextData.encodedSequencePitchInInts;
            const int numMinhashMaps = nextData.minhasher->getNumberOfMaps();
    
            //data of new anchors is appended to leftover data
    
            const int numLeftoverAnchors = *nextData.h_numLeftoverAnchors.get();
            read_number* const readIdsBegin = nextData.h_leftoverAnchorReadIds.get();
            read_number* const readIdsEnd = nextData.readIdGenerator->next_n_into_buffer(
                nextData.batchsize - numLeftoverAnchors, 
                readIdsBegin + numLeftoverAnchors
            );
            nextData.n_new_subjects = std::distance(readIdsBegin + numLeftoverAnchors, readIdsEnd);
    
            nextData.h_numAnchors[0] = nextData.n_new_subjects + numLeftoverAnchors;
    
            if(nextData.h_numAnchors[0] == 0){
                return;
            };
    
            cudaMemcpyAsync(
                nextData.d_numAnchors.get(),
                nextData.h_numAnchors.get(),
                sizeof(int),
                H2D,
                syncStream
            ); CUERR;
    
            cudaMemcpyAsync(
                nextData.d_leftoverAnchorReadIds.get() + numLeftoverAnchors,
                nextData.h_leftoverAnchorReadIds.get() + numLeftoverAnchors,
                sizeof(read_number) * nextData.n_new_subjects,
                H2D,
                syncStream
            ); CUERR;
    
            // std::cerr << "gather anchors\n";
            // get sequence data and length of new anchors.
            nextData.readStorage->gatherSequenceDataToGpuBufferAsync(
                nextData.threadPool,
                nextData.subjectSequenceGatherHandle,
                nextData.d_leftoverAnchorSequences.get() + numLeftoverAnchors * encodedSequencePitchInInts,
                encodedSequencePitchInInts,
                nextData.h_leftoverAnchorReadIds.get() + numLeftoverAnchors,
                nextData.d_leftoverAnchorReadIds.get() + numLeftoverAnchors,
                nextData.n_new_subjects,
                nextData.deviceId,
                syncStream
            );
    
            nextData.readStorage->gatherSequenceLengthsToGpuBufferAsync(
                nextData.d_leftoverAnchorLengths.get() + numLeftoverAnchors,
                nextData.deviceId,
                nextData.d_leftoverAnchorReadIds.get() + numLeftoverAnchors,
                nextData.n_new_subjects,            
                syncStream
            );
    
            //minhash the retrieved anchors to find candidate ids
    
            const int numLeftoverCandidates = nextData.h_numLeftoverCandidates[0];
    
            ParallelForLoopExecutor parallelFor(nextData.threadPool, &nextData.pforHandle);
    
            nextData.minhasher->getIdsOfSimilarReads(
                nextData.minhasherQueryHandle,
                nextData.d_leftoverAnchorReadIds.get() + numLeftoverAnchors,
                nextData.h_leftoverAnchorReadIds.get() + numLeftoverAnchors,
                nextData.d_leftoverAnchorSequences.get() + numLeftoverAnchors * encodedSequencePitchInInts,
                encodedSequencePitchInInts,
                nextData.d_leftoverAnchorLengths.get() + numLeftoverAnchors,
                nextData.n_new_subjects,
                nextData.deviceId, 
                syncStream,
                parallelFor,
                nextData.d_leftoverCandidateReadIds.get() + numLeftoverCandidates,
                nextData.d_leftoverCandidatesPerAnchors.get() + numLeftoverAnchors,
                nextData.d_candidates_per_subject_prefixsum.get() + numLeftoverAnchors
            );
    
            nvtx::push_range("leftover_calculation", 3);
          
            executeLeftoverCalculation(syncStream);

            nvtx::pop_range();
    
            cudaStreamSynchronize(syncStream); CUERR;
        }
    };

    struct UnprocessedCorrectionResults{
        static constexpr int overprovisioningPercent = 0;
        
        template<class T>
        using PinnedBuffer = SimpleAllocationPinnedHost<T, overprovisioningPercent>;

        int n_subjects;
        int n_queries;
        int decodedSequencePitchInBytes;
        int encodedSequencePitchInInts;
        int maxNumEditsPerSequence;
        std::size_t editsPitchInBytes;

        PinnedBuffer<read_number> h_subject_read_ids;
        PinnedBuffer<bool> h_subject_is_corrected;
        PinnedBuffer<AnchorHighQualityFlag> h_is_high_quality_subject;
        PinnedBuffer<int> h_num_corrected_candidates_per_anchor;
        PinnedBuffer<int> h_num_corrected_candidates_per_anchor_prefixsum;
        PinnedBuffer<int> h_indices_of_corrected_candidates;
        PinnedBuffer<read_number> h_candidate_read_ids;
        PinnedBuffer<char> h_corrected_subjects;
        PinnedBuffer<char> h_corrected_candidates;
        PinnedBuffer<int> h_subject_sequences_lengths;
        PinnedBuffer<int> h_candidate_sequences_lengths;
        PinnedBuffer<int> h_alignment_shifts;

        PinnedBuffer<TempCorrectedSequence::EncodedEdit> h_editsPerCorrectedSubject;
        PinnedBuffer<int> h_numEditsPerCorrectedSubject;

        PinnedBuffer<TempCorrectedSequence::EncodedEdit> h_editsPerCorrectedCandidate;
        PinnedBuffer<int> h_numEditsPerCorrectedCandidate;


        void init(
                int batchsize, 
                int maxCandidates, 
                int maxNumIdsFromMinhashing,
                size_t decodedSequencePitchInBytes, 
                int maxNumEditsPerSequence,
                std::size_t editsPitchInBytes
        ){
            const std::size_t numEditsAnchors = SDIV(editsPitchInBytes * batchsize, sizeof(TempCorrectedSequence::EncodedEdit));
            const std::size_t numEditsCandidates = SDIV(editsPitchInBytes * maxCandidates, sizeof(TempCorrectedSequence::EncodedEdit));

            h_subject_read_ids.resize(batchsize);
            h_subject_is_corrected.resize(batchsize);
            h_is_high_quality_subject.resize(batchsize);
            h_num_corrected_candidates_per_anchor.resize(batchsize);
            h_num_corrected_candidates_per_anchor_prefixsum.resize(batchsize);
            h_indices_of_corrected_candidates.resize(maxCandidates);
            h_candidate_read_ids.resize(maxNumIdsFromMinhashing);
            h_corrected_subjects.resize(batchsize * decodedSequencePitchInBytes);
            h_corrected_candidates.resize(maxCandidates * decodedSequencePitchInBytes);
            h_subject_sequences_lengths.resize(batchsize);
            h_candidate_sequences_lengths.resize(maxCandidates);
            h_alignment_shifts.resize(maxCandidates);
            h_editsPerCorrectedSubject.resize(numEditsAnchors);
            h_numEditsPerCorrectedSubject.resize(batchsize);
            h_editsPerCorrectedCandidate.resize(numEditsCandidates);
            h_numEditsPerCorrectedCandidate.resize(maxCandidates);
        }

        void destroy(){
            h_subject_read_ids.destroy();
            h_subject_is_corrected.destroy();
            h_is_high_quality_subject.destroy();
            h_num_corrected_candidates_per_anchor.destroy();
            h_num_corrected_candidates_per_anchor_prefixsum.destroy();
            h_indices_of_corrected_candidates.destroy();
            h_candidate_read_ids.destroy();
            h_corrected_subjects.destroy();
            h_corrected_candidates.destroy();
            h_subject_sequences_lengths.destroy();
            h_candidate_sequences_lengths.destroy();
            h_alignment_shifts.destroy();
            h_editsPerCorrectedSubject.destroy();
            h_numEditsPerCorrectedSubject.destroy();
            h_editsPerCorrectedCandidate.destroy();
            h_numEditsPerCorrectedCandidate.destroy();
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info;
            info.host = 0;

            auto handlehost = [&](const auto& buff){
                info.host += buff.capacityInBytes();
            };

            handlehost(h_subject_read_ids);
            handlehost(h_subject_is_corrected);
            handlehost(h_is_high_quality_subject);
            handlehost(h_num_corrected_candidates_per_anchor);
            handlehost(h_num_corrected_candidates_per_anchor_prefixsum);
            handlehost(h_indices_of_corrected_candidates);
            handlehost(h_candidate_read_ids);
            handlehost(h_corrected_subjects);
            handlehost(h_corrected_candidates);
            handlehost(h_subject_sequences_lengths);
            handlehost(h_candidate_sequences_lengths);
            handlehost(h_alignment_shifts);
            handlehost(h_editsPerCorrectedSubject);
            handlehost(h_numEditsPerCorrectedSubject);
            handlehost(h_editsPerCorrectedCandidate);
            handlehost(h_numEditsPerCorrectedCandidate);

            return info;
        }
    };

    struct OutputData{
        int numAnchors = 0;
        int numCandidates = 0;

        std::vector<TempCorrectedSequence> anchorCorrections;
        std::vector<EncodedTempCorrectedSequence> encodedAnchorCorrections;
        std::vector<TempCorrectedSequence> candidateCorrections;
        std::vector<EncodedTempCorrectedSequence> encodedCandidateCorrections;

        std::vector<int> subjectIndicesToProcess;
        std::vector<std::pair<int,int>> candidateIndicesToProcess;

        UnprocessedCorrectionResults rawResults;

        void resizeAnchors(int newNumAnchors){
            if(newNumAnchors > numAnchors){
                anchorCorrections.resize(newNumAnchors);
                encodedAnchorCorrections.resize(newNumAnchors);
            }

            numAnchors = newNumAnchors;
        }

        void resizeCandidates(int newNumCandidates){
            if(newNumCandidates > numCandidates){
                candidateCorrections.resize(newNumCandidates);
                encodedCandidateCorrections.resize(newNumCandidates);
            }

            numCandidates = newNumCandidates;
        }

        int getNumAnchors() const{
            return numAnchors;
        }

        int getNumCandidates() const{
            return numCandidates;
        }
    };



    struct Batch {
        static constexpr int overprovisioningPercent = 0;

        template<class T>
        using DeviceBuffer = SimpleAllocationDevice<T, overprovisioningPercent>;
        
        template<class T>
        using PinnedBuffer = SimpleAllocationPinnedHost<T, overprovisioningPercent>;

        Batch() = default;
        Batch(const Batch&) = delete;
        Batch(Batch&&) = default;
        Batch& operator=(const Batch&) = delete;
        Batch& operator=(Batch&&) = default;


        bool isFirstIteration = true;
        int threadsInThreadPool = 1;
        int id = -1;
        int deviceId = 0;
        int min_overlap = -1;
        int msa_max_column_count = -1;        
        int numCandidatesLimit = 0;        
        int encodedSequencePitchInInts;
        int decodedSequencePitchInBytes;
        int qualityPitchInBytes;
        int maxNumEditsPerSequence;
        size_t editsPitchInBytes = 0;  
        size_t msaColumnPitchInElements = 0;

        BackgroundThread* outputThread;
        BackgroundThread* backgroundWorker;
        BackgroundThread* unpackWorker;
        ThreadPool* threadPool;

        PinnedBuffer<int> h_numAnchors;
        PinnedBuffer<int> h_numCandidates;
        PinnedBuffer<unsigned int> h_subject_sequences_data;
        PinnedBuffer<unsigned int> h_candidate_sequences_data;
        PinnedBuffer<unsigned int> h_transposedCandidateSequencesData;
        PinnedBuffer<int> h_subject_sequences_lengths;
        PinnedBuffer<int> h_candidate_sequences_lengths;
        PinnedBuffer<int> h_candidates_per_subject;
        PinnedBuffer<int> h_candidates_per_subject_prefixsum;
        PinnedBuffer<read_number> h_subject_read_ids;
        PinnedBuffer<read_number> h_candidate_read_ids;
        PinnedBuffer<int> h_anchorIndicesOfCandidates; // candidate i belongs to anchor anchorIndicesOfCandidates[i]
        PinnedBuffer<int> h_indices;
        PinnedBuffer<int> h_indices_per_subject;
        PinnedBuffer<int> h_num_indices;
        PinnedBuffer<int> h_indices_of_corrected_subjects;
        PinnedBuffer<int> h_num_indices_of_corrected_subjects;
        PinnedBuffer<TempCorrectedSequence::EncodedEdit> h_editsPerCorrectedSubject;
        PinnedBuffer<int> h_numEditsPerCorrectedSubject;
        PinnedBuffer<TempCorrectedSequence::EncodedEdit> h_editsPerCorrectedCandidate;
        PinnedBuffer<int> h_numEditsPerCorrectedCandidate;
        PinnedBuffer<bool> h_anchorContainsN;
        PinnedBuffer<bool> h_candidateContainsN;
        PinnedBuffer<char> h_subject_qualities;
        PinnedBuffer<char> h_candidate_qualities;    
        PinnedBuffer<char> h_corrected_subjects;
        PinnedBuffer<char> h_corrected_candidates;
        PinnedBuffer<int> h_num_corrected_candidates_per_anchor;
        PinnedBuffer<int> h_num_corrected_candidates_per_anchor_prefixsum;
        PinnedBuffer<int> h_num_total_corrected_candidates;
        PinnedBuffer<bool> h_subject_is_corrected;
        PinnedBuffer<int> h_indices_of_corrected_candidates;
        PinnedBuffer<int> h_num_uncorrected_positions_per_subject;
        PinnedBuffer<int> h_uncorrected_positions_per_subject;    
        PinnedBuffer<AnchorHighQualityFlag> h_is_high_quality_subject;
        PinnedBuffer<int> h_high_quality_subject_indices;
        PinnedBuffer<int> h_num_high_quality_subject_indices;    
        PinnedBuffer<int> h_alignment_scores;
        PinnedBuffer<int> h_alignment_overlaps;
        PinnedBuffer<int> h_alignment_shifts;
        PinnedBuffer<int> h_alignment_nOps;
        PinnedBuffer<bool> h_alignment_isValid;
        PinnedBuffer<BestAlignment_t> h_alignment_best_alignment_flags;    
        PinnedBuffer<char> h_consensus;
        PinnedBuffer<float> h_support;
        PinnedBuffer<int> h_coverage;
        PinnedBuffer<float> h_origWeights;
        PinnedBuffer<int> h_origCoverages;
        PinnedBuffer<MSAColumnProperties> h_msa_column_properties;
        PinnedBuffer<int> h_counts;
        PinnedBuffer<float> h_weights;    
        DeviceBuffer<char> d_tempstorage;
        DeviceBuffer<int> d_numAnchors;
        DeviceBuffer<int> d_numCandidates;
        DeviceBuffer<unsigned int> d_subject_sequences_data;
        DeviceBuffer<unsigned int> d_candidate_sequences_data;
        DeviceBuffer<unsigned int> d_transposedCandidateSequencesData;
        DeviceBuffer<int> d_subject_sequences_lengths;
        DeviceBuffer<int> d_candidate_sequences_lengths;
        DeviceBuffer<int> d_candidates_per_subject;
        DeviceBuffer<int> d_candidates_per_subject_prefixsum;
        DeviceBuffer<read_number> d_subject_read_ids;
        DeviceBuffer<read_number> d_candidate_read_ids;
        DeviceBuffer<int> d_anchorIndicesOfCandidates; // candidate i belongs to anchor anchorIndicesOfCandidates[i]
        DeviceBuffer<int> d_indices;
        DeviceBuffer<int> d_indices_per_subject;
        DeviceBuffer<int> d_num_indices;
        DeviceBuffer<int> d_indices_tmp;
        DeviceBuffer<int> d_indices_per_subject_tmp;
        DeviceBuffer<int> d_num_indices_tmp;
        DeviceBuffer<int> d_indices_of_corrected_subjects;
        DeviceBuffer<int> d_num_indices_of_corrected_subjects;
        DeviceBuffer<TempCorrectedSequence::EncodedEdit> d_editsPerCorrectedSubject;
        DeviceBuffer<int> d_numEditsPerCorrectedSubject;
        DeviceBuffer<TempCorrectedSequence::EncodedEdit> d_editsPerCorrectedCandidate;
        DeviceBuffer<int> d_numEditsPerCorrectedCandidate;
        DeviceBuffer<bool> d_anchorContainsN;
        DeviceBuffer<bool> d_candidateContainsN;   
        DeviceBuffer<char> d_subject_qualities;
        DeviceBuffer<char> d_candidate_qualities;
        DeviceBuffer<char> d_candidate_qualities_transposed;        
        DeviceBuffer<char> d_corrected_subjects;
        DeviceBuffer<char> d_corrected_candidates;
        DeviceBuffer<int> d_num_corrected_candidates_per_anchor;
        DeviceBuffer<int> d_num_corrected_candidates_per_anchor_prefixsum;
        DeviceBuffer<int> d_num_total_corrected_candidates;
        DeviceBuffer<bool> d_subject_is_corrected;
        DeviceBuffer<int> d_indices_of_corrected_candidates;
        DeviceBuffer<int> d_num_uncorrected_positions_per_subject;
        DeviceBuffer<int> d_uncorrected_positions_per_subject;    
        DeviceBuffer<AnchorHighQualityFlag> d_is_high_quality_subject;
        DeviceBuffer<int> d_high_quality_subject_indices;
        DeviceBuffer<int> d_num_high_quality_subject_indices;      
        DeviceBuffer<int> d_alignment_scores;
        DeviceBuffer<int> d_alignment_overlaps;
        DeviceBuffer<int> d_alignment_shifts;
        DeviceBuffer<int> d_alignment_nOps;
        DeviceBuffer<bool> d_alignment_isValid;
        DeviceBuffer<BestAlignment_t> d_alignment_best_alignment_flags;    
        DeviceBuffer<bool> d_canExecute; 
        DeviceBuffer<char> d_consensus;
        DeviceBuffer<float> d_support;
        DeviceBuffer<int> d_coverage;
        DeviceBuffer<float> d_origWeights;
        DeviceBuffer<int> d_origCoverages;
        DeviceBuffer<MSAColumnProperties> d_msa_column_properties;
        DeviceBuffer<int> d_counts;
        DeviceBuffer<float> d_weights;


        struct Pointers{
            void* h_subject_sequences_data{};
            void* h_candidate_read_ids{};
            bool operator<(const Pointers& rhs) const{
                if(h_subject_sequences_data < rhs.h_subject_sequences_data){
                    return true;
                }
                if(h_subject_sequences_data > rhs.h_subject_sequences_data){
                    return false;
                }
                if(h_candidate_read_ids < rhs.h_candidate_read_ids){
                    return true;
                }
                return false;
            }
        };

        std::map<Pointers, CudaGraph> executionGraphs;

        //std::array<CudaGraph, 2> executionGraphs{};
		std::array<cudaStream_t, nStreamsPerBatch> streams;
		std::array<cudaEvent_t, nEventsPerBatch> events;
        ThreadPool::ParallelForHandle pforHandle;    
		KernelLaunchHandle kernelLaunchHandle;        
        DistributedReadStorage::GatherHandleSequences subjectSequenceGatherHandle;
        DistributedReadStorage::GatherHandleSequences candidateSequenceGatherHandle;
        DistributedReadStorage::GatherHandleQualities subjectQualitiesGatherHandle;
        DistributedReadStorage::GatherHandleQualities candidateQualitiesGatherHandle;

        cpu::RangeGenerator<read_number>* readIdGenerator;
		const Minhasher_t* minhasher;
        const DistributedReadStorage* readStorage;
		CorrectionOptions correctionOptions;
        GoodAlignmentProperties goodAlignmentProperties;
        SequenceFileProperties sequenceFileProperties;
        RuntimeOptions runtimeOptions;
        FileOptions fileOptions;
		std::atomic_uint8_t* correctionStatusFlagsPerRead;
        std::function<void(const TempCorrectedSequence*, const EncodedTempCorrectedSequence*)> saveCorrectedSequence;

        WaitableData<OutputData> waitableOutputData;
        NextIterationData nextIterationData;

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

            handlehost(h_numAnchors);
            handlehost(h_numCandidates);
            handlehost(h_subject_sequences_data);
            handlehost(h_candidate_sequences_data);
            handlehost(h_transposedCandidateSequencesData);
            handlehost(h_subject_sequences_lengths);
            handlehost(h_candidate_sequences_lengths);
            handlehost(h_candidates_per_subject);
            handlehost(h_candidates_per_subject_prefixsum);
            handlehost(h_subject_read_ids);
            handlehost(h_candidate_read_ids);
            handlehost(h_anchorIndicesOfCandidates);
            handlehost(h_indices);
            handlehost(h_indices_per_subject);
            handlehost(h_num_indices);
            handlehost(h_indices_of_corrected_subjects);
            handlehost(h_num_indices_of_corrected_subjects);
            handlehost(h_editsPerCorrectedSubject);
            handlehost(h_numEditsPerCorrectedSubject);
            handlehost(h_editsPerCorrectedCandidate);
            handlehost(h_numEditsPerCorrectedCandidate);
            handlehost(h_anchorContainsN);
            handlehost(h_candidateContainsN);
            handlehost(h_subject_qualities);
            handlehost(h_candidate_qualities);    
            handlehost(h_corrected_subjects);
            handlehost(h_corrected_candidates);
            handlehost(h_num_corrected_candidates_per_anchor);
            handlehost(h_num_corrected_candidates_per_anchor_prefixsum);
            handlehost(h_num_total_corrected_candidates);
            handlehost(h_subject_is_corrected);
            handlehost(h_indices_of_corrected_candidates);
            handlehost(h_num_uncorrected_positions_per_subject);
            handlehost(h_uncorrected_positions_per_subject);    
            handlehost(h_is_high_quality_subject);
            handlehost(h_high_quality_subject_indices);
            handlehost(h_num_high_quality_subject_indices);    
            handlehost(h_alignment_scores);
            handlehost(h_alignment_overlaps);
            handlehost(h_alignment_shifts);
            handlehost(h_alignment_nOps);
            handlehost(h_alignment_isValid);
            handlehost(h_alignment_best_alignment_flags);    
            handlehost(h_consensus);
            handlehost(h_support);
            handlehost(h_coverage);
            handlehost(h_origWeights);
            handlehost(h_origCoverages);
            handlehost(h_msa_column_properties);
            handlehost(h_counts);
            handlehost(h_weights); 

            handledevice(d_tempstorage);
            handledevice(d_numAnchors);
            handledevice(d_numCandidates);
            handledevice(d_subject_sequences_data);
            handledevice(d_candidate_sequences_data);
            handledevice(d_transposedCandidateSequencesData);
            handledevice(d_subject_sequences_lengths);
            handledevice(d_candidate_sequences_lengths);
            handledevice(d_candidates_per_subject);
            handledevice(d_candidates_per_subject_prefixsum);
            handledevice(d_subject_read_ids);
            handledevice(d_candidate_read_ids);
            handledevice(d_anchorIndicesOfCandidates); 
            handledevice(d_indices);
            handledevice(d_indices_per_subject);
            handledevice(d_num_indices);
            handledevice(d_indices_tmp);
            handledevice(d_indices_per_subject_tmp);
            handledevice(d_num_indices_tmp);
            handledevice(d_indices_of_corrected_subjects);
            handledevice(d_num_indices_of_corrected_subjects);
            handledevice(d_editsPerCorrectedSubject);
            handledevice(d_numEditsPerCorrectedSubject);
            handledevice(d_editsPerCorrectedCandidate);
            handledevice(d_numEditsPerCorrectedCandidate);
            handledevice(d_anchorContainsN);
            handledevice(d_candidateContainsN);   
            handledevice(d_subject_qualities);
            handledevice(d_candidate_qualities);
            handledevice(d_candidate_qualities_transposed);        
            handledevice(d_corrected_subjects);
            handledevice(d_corrected_candidates);
            handledevice(d_num_corrected_candidates_per_anchor);
            handledevice(d_num_corrected_candidates_per_anchor_prefixsum);
            handledevice(d_num_total_corrected_candidates);
            handledevice(d_subject_is_corrected);
            handledevice(d_indices_of_corrected_candidates);
            handledevice(d_num_uncorrected_positions_per_subject);
            handledevice(d_uncorrected_positions_per_subject);    
            handledevice(d_is_high_quality_subject);
            handledevice(d_high_quality_subject_indices);
            handledevice(d_num_high_quality_subject_indices);      
            handledevice(d_alignment_scores);
            handledevice(d_alignment_overlaps);
            handledevice(d_alignment_shifts);
            handledevice(d_alignment_nOps);
            handledevice(d_alignment_isValid);
            handledevice(d_alignment_best_alignment_flags);    
            handledevice(d_canExecute); 
            handledevice(d_consensus);
            handledevice(d_support);
            handledevice(d_coverage);
            handledevice(d_origWeights);
            handledevice(d_origCoverages);
            handledevice(d_msa_column_properties);
            handledevice(d_counts);
            handledevice(d_weights);

            return info;
        }

		void reset(){
            *h_numAnchors = 0;
            *h_numCandidates = 0;
        }

        void resize(
            int batchsize,
            const CorrectionOptions& correctionOptions,
            const GoodAlignmentProperties& goodAlignmentProperties,
            const SequenceFileProperties& sequenceFileProperties,
            int numMinhashMaps
        ){

            Batch& batchData = *this;

            auto& streams = batchData.streams;
    
            const auto sequence_pitch = batchData.decodedSequencePitchInBytes;
            const auto maxCandidates = batchData.numCandidatesLimit;
            const auto encodedSeqPitchInts = batchData.encodedSequencePitchInInts;
            const auto qualPitchBytes = batchData.qualityPitchInBytes;
            const auto msaColumnPitchInElements = batchData.msaColumnPitchInElements;
                    
            const int resultsPerMap = calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage);
            const int maxNumIds = resultsPerMap * numMinhashMaps * batchsize;
    
            h_subject_sequences_data.resize(encodedSeqPitchInts * batchsize);
            h_subject_sequences_lengths.resize(batchsize);
            h_subject_read_ids.resize(batchsize);
            h_candidate_read_ids.resize(maxNumIds);
            d_subject_sequences_data.resize(encodedSeqPitchInts * batchsize);
            d_subject_sequences_lengths.resize(batchsize);
            d_subject_read_ids.resize(batchsize);
            d_candidate_read_ids.resize(maxNumIds + maxCandidates);
            d_candidates_per_subject.resize(2*batchsize);
            d_candidates_per_subject_prefixsum.resize(batchsize+1);
            cudaMemsetAsync(d_candidates_per_subject_prefixsum.get(), 0, sizeof(int), streams[primary_stream_index]); CUERR;
    
            h_candidates_per_subject_prefixsum.resize(batchsize+1);

            h_numAnchors.resize(1);
            h_numCandidates.resize(1);
            d_numAnchors.resize(1);
            d_numCandidates.resize(1);
    
    
            h_candidate_sequences_data.resize(maxCandidates * encodedSeqPitchInts);
            h_transposedCandidateSequencesData.resize(maxCandidates * encodedSeqPitchInts);
            h_subject_sequences_lengths.resize(batchsize);
            h_candidate_sequences_lengths.resize(maxCandidates);
            h_anchorIndicesOfCandidates.resize(maxCandidates);
    
            d_subject_sequences_data.resize(batchsize * encodedSeqPitchInts);
            d_candidate_sequences_data.resize(maxCandidates * encodedSeqPitchInts);
            d_transposedCandidateSequencesData.resize(maxCandidates * encodedSeqPitchInts);
            d_subject_sequences_lengths.resize(batchsize);
            d_candidate_sequences_lengths.resize(maxCandidates);
            d_anchorIndicesOfCandidates.resize(maxCandidates);
    
            //alignment output
    
            h_alignment_shifts.resize(2*maxCandidates);
            d_alignment_overlaps.resize(maxCandidates);
            d_alignment_shifts.resize(maxCandidates);
            d_alignment_nOps.resize(maxCandidates);
            d_alignment_isValid.resize(maxCandidates);
            d_alignment_best_alignment_flags.resize(maxCandidates);
            h_alignment_best_alignment_flags.resize(maxCandidates);
    
            // candidate indices
    
            h_indices.resize(maxCandidates);
            h_indices_per_subject.resize(batchsize);
            h_num_indices.resize(1);
    
            d_indices.resize(maxCandidates);
            d_indices_per_subject.resize(batchsize);
            d_num_indices.resize(1);
            d_indices_tmp.resize(maxCandidates);
            d_indices_per_subject_tmp.resize(batchsize);
            d_num_indices_tmp.resize(1);
    
            h_indices_of_corrected_subjects.resize(batchsize);
            h_num_indices_of_corrected_subjects.resize(1);
            d_indices_of_corrected_subjects.resize(batchsize);
            d_num_indices_of_corrected_subjects.resize(1);

            std::size_t numEditsAnchors = SDIV(editsPitchInBytes * batchsize, sizeof(TempCorrectedSequence::EncodedEdit));
            std::size_t numEditsCandidates = SDIV(editsPitchInBytes * maxCandidates, sizeof(TempCorrectedSequence::EncodedEdit));
    
            h_editsPerCorrectedSubject.resize(numEditsAnchors);
            h_numEditsPerCorrectedSubject.resize(batchsize);
            h_anchorContainsN.resize(batchsize);
    
            d_editsPerCorrectedSubject.resize(numEditsAnchors);
            d_numEditsPerCorrectedSubject.resize(batchsize);
            d_anchorContainsN.resize(batchsize);
    
            h_editsPerCorrectedCandidate.resize(numEditsCandidates);
            h_numEditsPerCorrectedCandidate.resize(maxCandidates);
            h_candidateContainsN.resize(maxCandidates);
    
            d_editsPerCorrectedCandidate.resize(numEditsCandidates);
            d_numEditsPerCorrectedCandidate.resize(maxCandidates);
            d_candidateContainsN.resize(maxCandidates);
    
            //qualitiy scores
            if(correctionOptions.useQualityScores) {
                h_subject_qualities.resize(batchsize * qualPitchBytes);
                h_candidate_qualities.resize(maxCandidates * qualPitchBytes);
    
                d_subject_qualities.resize(batchsize * qualPitchBytes);
                d_candidate_qualities.resize(maxCandidates * qualPitchBytes);
                d_candidate_qualities_transposed.resize(maxCandidates * qualPitchBytes);            
            }
    
    
            //correction results
    
            h_corrected_subjects.resize(batchsize * sequence_pitch);
            h_corrected_candidates.resize(maxCandidates * sequence_pitch);
            h_num_corrected_candidates_per_anchor.resize(batchsize);
            h_num_corrected_candidates_per_anchor_prefixsum.resize(batchsize);
            h_num_total_corrected_candidates.resize(1);
            h_subject_is_corrected.resize(batchsize);
            h_indices_of_corrected_candidates.resize(maxCandidates);
            h_num_uncorrected_positions_per_subject.resize(batchsize);
            h_uncorrected_positions_per_subject.resize(batchsize * sequenceFileProperties.maxSequenceLength);
            
            d_corrected_subjects.resize(batchsize * sequence_pitch);
            d_corrected_candidates.resize(maxCandidates * sequence_pitch);
            d_num_corrected_candidates_per_anchor.resize(batchsize);
            d_num_corrected_candidates_per_anchor_prefixsum.resize(batchsize);
            d_num_total_corrected_candidates.resize(1);
            d_subject_is_corrected.resize(batchsize);
            d_indices_of_corrected_candidates.resize(maxCandidates);
            d_num_uncorrected_positions_per_subject.resize(batchsize);
            d_uncorrected_positions_per_subject.resize(batchsize * sequenceFileProperties.maxSequenceLength);
    
            h_is_high_quality_subject.resize(batchsize);
            h_high_quality_subject_indices.resize(batchsize);
            h_num_high_quality_subject_indices.resize(1);
    
            d_is_high_quality_subject.resize(batchsize);
            d_high_quality_subject_indices.resize(batchsize);
            d_num_high_quality_subject_indices.resize(1);
    
            //multiple sequence alignment
    
            h_consensus.resize(batchsize * msaColumnPitchInElements);
            h_support.resize(batchsize * msaColumnPitchInElements);
            h_coverage.resize(batchsize * msaColumnPitchInElements);
            h_origWeights.resize(batchsize * msaColumnPitchInElements);
            h_origCoverages.resize(batchsize * msaColumnPitchInElements);
            h_msa_column_properties.resize(batchsize);
            h_counts.resize(batchsize * 4 * msaColumnPitchInElements);
            h_weights.resize(batchsize * 4 * msaColumnPitchInElements);
    
            d_consensus.resize(batchsize * msaColumnPitchInElements);
            d_support.resize(batchsize * msaColumnPitchInElements);
            d_coverage.resize(batchsize * msaColumnPitchInElements);
            d_origWeights.resize(batchsize * msaColumnPitchInElements);
            d_origCoverages.resize(batchsize * msaColumnPitchInElements);
            d_msa_column_properties.resize(batchsize);
            d_counts.resize(batchsize * 4 * msaColumnPitchInElements);
            d_weights.resize(batchsize * 4 * msaColumnPitchInElements);
    
    
            d_canExecute.resize(1);
    
               
            
            
            std::size_t flagTemp = sizeof(bool) * maxCandidates;
            std::size_t popcountShdTempBytes = 0; 
            
            const bool removeAmbiguousAnchors = correctionOptions.excludeAmbiguousReads;
            const bool removeAmbiguousCandidates = correctionOptions.excludeAmbiguousReads;
    
            call_popcount_shifted_hamming_distance_kernel_async(
                nullptr,
                popcountShdTempBytes,
                d_alignment_overlaps.get(),
                d_alignment_shifts.get(),
                d_alignment_nOps.get(),
                d_alignment_isValid.get(),
                d_alignment_best_alignment_flags.get(),
                d_subject_sequences_data.get(),
                d_candidate_sequences_data.get(),
                d_subject_sequences_lengths.get(),
                d_candidate_sequences_lengths.get(),
                d_candidates_per_subject_prefixsum.get(),
                d_candidates_per_subject.get(),
                d_anchorIndicesOfCandidates.get(),
                d_numAnchors.get(),
                d_numCandidates.get(),
                d_anchorContainsN.get(),
                removeAmbiguousAnchors,
                d_candidateContainsN.get(),
                removeAmbiguousCandidates,
                batchsize,
                maxCandidates,
                sequenceFileProperties.maxSequenceLength,
                encodedSequencePitchInInts,
                goodAlignmentProperties.min_overlap,
                goodAlignmentProperties.maxErrorRate,
                goodAlignmentProperties.min_overlap_ratio,
                correctionOptions.estimatedErrorrate,                
                streams[primary_stream_index],
                kernelLaunchHandle
            );
            
            // this buffer will also serve as temp storage for cub. The required memory for cub 
            // is less than popcountShdTempBytes.
            popcountShdTempBytes = std::max(flagTemp, popcountShdTempBytes);
            d_tempstorage.resize(popcountShdTempBytes);
        }

        void destroy(){
            h_subject_sequences_data.destroy();
            h_candidate_sequences_data.destroy();
            h_transposedCandidateSequencesData.destroy();
            h_subject_sequences_lengths.destroy();
            h_candidate_sequences_lengths.destroy();
            h_candidates_per_subject.destroy();
            h_candidates_per_subject_prefixsum.destroy();
            h_subject_read_ids.destroy();
            h_candidate_read_ids.destroy();
    
            d_subject_sequences_data.destroy();
            d_candidate_sequences_data.destroy();
            d_transposedCandidateSequencesData.destroy();
            d_subject_sequences_lengths.destroy();
            d_candidate_sequences_lengths.destroy();
            d_candidates_per_subject.destroy();
            d_candidates_per_subject_prefixsum.destroy();
            d_subject_read_ids.destroy();
            d_candidate_read_ids.destroy();
    
            h_subject_qualities.destroy();
            h_candidate_qualities.destroy();
    
            d_subject_qualities.destroy();
            d_candidate_qualities.destroy();
            d_candidate_qualities_transposed.destroy();
    
            h_consensus.destroy();
            h_support.destroy();
            h_coverage.destroy();
            h_origWeights.destroy();
            h_origCoverages.destroy();
            h_msa_column_properties.destroy();
            h_counts.destroy();
            h_weights.destroy();
    
            d_consensus.destroy();
            d_support.destroy();
            d_coverage.destroy();
            d_origWeights.destroy();
            d_origCoverages.destroy();
            d_msa_column_properties.destroy();
            d_counts.destroy();
            d_weights.destroy();
    
            h_alignment_scores.destroy();
            h_alignment_overlaps.destroy();
            h_alignment_shifts.destroy();
            h_alignment_nOps.destroy();
            h_alignment_isValid.destroy();
            h_alignment_best_alignment_flags.destroy();
    
            d_alignment_scores.destroy();
            d_alignment_overlaps.destroy();
            d_alignment_shifts.destroy();
            d_alignment_nOps.destroy();
            d_alignment_isValid.destroy();
            d_alignment_best_alignment_flags.destroy();
    
            h_corrected_subjects.destroy();
            h_corrected_candidates.destroy();
            h_num_corrected_candidates_per_anchor.destroy();
            h_num_corrected_candidates_per_anchor_prefixsum.destroy();
            h_subject_is_corrected.destroy();
            h_indices_of_corrected_candidates.destroy();
            h_is_high_quality_subject.destroy();
            h_high_quality_subject_indices.destroy();
            h_num_high_quality_subject_indices.destroy();
            h_num_uncorrected_positions_per_subject.destroy();
            h_uncorrected_positions_per_subject.destroy();
    
            d_corrected_subjects.destroy();
            d_corrected_candidates.destroy();
            d_num_corrected_candidates_per_anchor.destroy();
            d_num_corrected_candidates_per_anchor_prefixsum.destroy();
            d_subject_is_corrected.destroy();
            d_indices_of_corrected_candidates.destroy();
            d_is_high_quality_subject.destroy();
            d_high_quality_subject_indices.destroy();
            d_num_high_quality_subject_indices.destroy();
            d_num_uncorrected_positions_per_subject.destroy();
            d_uncorrected_positions_per_subject.destroy();
    
            h_indices.destroy();
            h_indices_per_subject.destroy();
            h_num_indices.destroy();
    
            d_indices.destroy();
            d_indices_per_subject.destroy();
            d_num_indices.destroy();
    
            d_indices_tmp.destroy();
            d_indices_per_subject_tmp.destroy();
            d_num_indices_tmp.destroy();
    
            d_indices_of_corrected_subjects.destroy();
            d_num_indices_of_corrected_subjects.destroy();
    
    
            h_editsPerCorrectedSubject.destroy();
            h_numEditsPerCorrectedSubject.destroy();
            h_editsPerCorrectedCandidate.destroy();
            h_numEditsPerCorrectedCandidate.destroy();
            h_anchorContainsN.destroy();
            h_candidateContainsN.destroy();
    
            d_editsPerCorrectedSubject.destroy();
            d_numEditsPerCorrectedSubject.destroy();
            d_editsPerCorrectedCandidate.destroy();
            d_numEditsPerCorrectedCandidate.destroy();
            d_anchorContainsN.destroy();
            d_candidateContainsN.destroy();
    
            d_canExecute.destroy();
            
            d_tempstorage.destroy();
            d_numAnchors.destroy();
            d_numCandidates.destroy();
            h_numAnchors.destroy();
            h_numCandidates.destroy();
        }

        void updateFromIterationData(NextIterationData& data){
            std::swap(h_subject_sequences_data, data.h_subject_sequences_data);
            std::swap(h_subject_sequences_lengths, data.h_subject_sequences_lengths);
            std::swap(h_subject_read_ids, data.h_subject_read_ids);
            std::swap(h_candidate_read_ids, data.h_candidate_read_ids);        

            std::swap(d_subject_sequences_data, data.d_subject_sequences_data);
            std::swap(d_subject_sequences_lengths, data.d_subject_sequences_lengths);
            std::swap(d_subject_read_ids, data.d_subject_read_ids);
            std::swap(d_candidate_read_ids, data.d_candidate_read_ids);
            std::swap(d_candidates_per_subject, data.d_candidates_per_subject);
            std::swap(d_candidates_per_subject_prefixsum, data.d_candidates_per_subject_prefixsum);

            std::swap(h_numAnchors, data.h_numAnchors);
            std::swap(d_numAnchors, data.d_numAnchors);
            std::swap(h_numCandidates, data.h_numCandidates);
            std::swap(d_numCandidates, data.d_numCandidates);

            data.h_numAnchors[0] = 0;
            data.h_numCandidates[0] = 0;
        }


        void moveResultsToOutputData(OutputData& outputData){
            auto& rawResults = outputData.rawResults;

            rawResults.n_subjects = *h_numAnchors;
            rawResults.n_queries = *h_numCandidates;
            rawResults.encodedSequencePitchInInts = encodedSequencePitchInInts;
            rawResults.decodedSequencePitchInBytes = decodedSequencePitchInBytes;
            rawResults.maxNumEditsPerSequence = maxNumEditsPerSequence;
            rawResults.editsPitchInBytes = editsPitchInBytes;

            std::swap(h_subject_read_ids, rawResults.h_subject_read_ids);
            std::swap(h_subject_is_corrected, rawResults.h_subject_is_corrected);
            std::swap(h_is_high_quality_subject, rawResults.h_is_high_quality_subject);
            std::swap(h_num_corrected_candidates_per_anchor, rawResults.h_num_corrected_candidates_per_anchor);
            std::swap(h_num_corrected_candidates_per_anchor_prefixsum, rawResults.h_num_corrected_candidates_per_anchor_prefixsum);
            std::swap(h_indices_of_corrected_candidates, rawResults.h_indices_of_corrected_candidates);
            std::swap(h_candidate_read_ids, rawResults.h_candidate_read_ids);
            std::swap(h_corrected_subjects, rawResults.h_corrected_subjects);
            std::swap(h_corrected_candidates, rawResults.h_corrected_candidates);
            std::swap(h_subject_sequences_lengths, rawResults.h_subject_sequences_lengths);
            std::swap(h_candidate_sequences_lengths, rawResults.h_candidate_sequences_lengths);
            std::swap(h_alignment_shifts, rawResults.h_alignment_shifts);

            std::swap(h_editsPerCorrectedSubject, rawResults.h_editsPerCorrectedSubject);
            std::swap(h_numEditsPerCorrectedSubject, rawResults.h_numEditsPerCorrectedSubject);

            std::swap(h_editsPerCorrectedCandidate, rawResults.h_editsPerCorrectedCandidate);
            std::swap(h_numEditsPerCorrectedCandidate, rawResults.h_numEditsPerCorrectedCandidate);

        }


        void writebatchdebug(){
            cudaDeviceSynchronize(); CUERR;

            auto& batch = *this;

            std::ofstream os("batch.bin", std::ios::binary);
            os.write(reinterpret_cast<const char*>(&batch.goodAlignmentProperties), sizeof(GoodAlignmentProperties));
            os.write(reinterpret_cast<const char*>(&batch.correctionOptions), sizeof(CorrectionOptions));
            os.write(reinterpret_cast<const char*>(&batch.encodedSequencePitchInInts), sizeof(int));
            os.write(reinterpret_cast<const char*>(&batch.qualityPitchInBytes), sizeof(int));
            os.write(reinterpret_cast<const char*>(&batch.msaColumnPitchInElements), sizeof(std::size_t));
    
            auto writeallocation = [&](const auto& alloc){
                std::size_t bytes = alloc.sizeInBytes();
                std::size_t elements = alloc.size();
                os.write(reinterpret_cast<const char*>(&elements), sizeof(std::size_t));
                os.write(reinterpret_cast<const char*>(alloc.get()), bytes);
            };
    
            writeallocation(batch.d_numAnchors);
            writeallocation(batch.d_numCandidates);
            writeallocation(batch.d_subject_sequences_data);
            writeallocation(batch.d_subject_sequences_lengths);
            writeallocation(batch.d_candidate_sequences_data);
            writeallocation(batch.d_candidate_sequences_lengths);
            writeallocation(batch.d_subject_qualities);
            writeallocation(batch.d_candidate_qualities);
            writeallocation(batch.d_candidates_per_subject_prefixsum);
            writeallocation(batch.d_alignment_overlaps);
            writeallocation(batch.d_alignment_shifts);
            writeallocation(batch.d_alignment_nOps);
            writeallocation(batch.d_alignment_best_alignment_flags);
            writeallocation(batch.d_indices);
            writeallocation(batch.d_indices_per_subject);
        }

	};


    void getCandidateAlignments(Batch& batch);
    void buildMultipleSequenceAlignment(Batch& batch);
    void removeCandidatesOfDifferentRegionFromMSA(Batch& batch);
    void correctSubjects(Batch& batch);
    void correctCandidates(Batch& batch);
    


    void buildGraphViaCapture(Batch& batch){
        cudaSetDevice(batch.deviceId); CUERR;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        Batch::Pointers pointers;
        pointers.h_subject_sequences_data = batch.h_subject_sequences_data.get();
        pointers.h_candidate_read_ids = batch.h_candidate_read_ids.get();

        auto& cudagraph = batch.executionGraphs[pointers];

        if(!cudagraph.valid){
            // std::cerr << "rebuild graph\n";

            if(cudagraph.execgraph != nullptr){
                cudaGraphExecDestroy(cudagraph.execgraph); CUERR;
            }
            
            //std::cerr << "correct_gpu buildGraphViaCapture start batch id " << batch.id << "\n";
            cudaStreamBeginCapture(streams[primary_stream_index], cudaStreamCaptureModeRelaxed); CUERR;
            //fork to capture secondary stream
            cudaEventRecord(events[0], streams[primary_stream_index]); CUERR;
            cudaStreamWaitEvent(streams[secondary_stream_index], events[0], 0); CUERR;


            getCandidateAlignments(batch);
            buildMultipleSequenceAlignment(batch);
            #ifdef USE_MSA_MINIMIZATION
            removeCandidatesOfDifferentRegionFromMSA(batch);
            #endif
            correctSubjects(batch);
            if(batch.correctionOptions.correctCandidates){
                correctCandidates(batch);                
            }

            //join forked stream for valid capture
            cudaEventRecord(events[0], streams[secondary_stream_index]); CUERR;
            cudaStreamWaitEvent(streams[primary_stream_index], events[0], 0); CUERR;

            cudaGraph_t graph;
            cudaStreamEndCapture(streams[primary_stream_index], &graph); CUERR;
            //std::cerr << "correct_gpu buildGraphViaCapture stop batch id " << batch.id << "\n";
            
            cudaGraphExec_t execGraph;
            cudaGraphNode_t errorNode;
            auto logBuffer = std::make_unique<char[]>(1025);
            std::fill_n(logBuffer.get(), 1025, 0);
            cudaError_t status = cudaGraphInstantiate(&execGraph, graph, &errorNode, logBuffer.get(), 1025);
            if(status != cudaSuccess){
                if(logBuffer[1024] != '\0'){
                    std::cerr << "cudaGraphInstantiate: truncated error message: ";
                    std::copy_n(logBuffer.get(), 1025, std::ostream_iterator<char>(std::cerr, ""));
                    std::cerr << "\n";
                }else{
                    std::cerr << "cudaGraphInstantiate: error message: ";
                    std::cerr << logBuffer.get();
                    std::cerr << "\n";
                }
                CUERR;
            }            

            cudaGraphDestroy(graph); CUERR;

            cudagraph.execgraph = execGraph;

            cudagraph.valid = true;
        }
    }

    void executeGraph(Batch& batch){
        cudaSetDevice(batch.deviceId); CUERR;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;

        buildGraphViaCapture(batch);

        Batch::Pointers pointers;
        pointers.h_subject_sequences_data = batch.h_subject_sequences_data.get();
        pointers.h_candidate_read_ids = batch.h_candidate_read_ids.get();

        auto& graph = batch.executionGraphs[pointers];

        assert(graph.valid);
        cudaGraphLaunch(graph.execgraph, streams[primary_stream_index]); CUERR;
    }







    void getNextBatchForCorrection(Batch& batchData){
        Batch* batchptr = &batchData;

        auto getDataForNextIteration = [batchptr](){
            nvtx::push_range("prepareNewDataForCorrection",1);

            batchptr->nextIterationData.prepareNewDataForCorrection(
                batchptr->streams[preparation_stream_index]
            );

            cudaStreamSynchronize(batchptr->streams[preparation_stream_index]); CUERR;

            if(batchptr->nextIterationData.h_numAnchors[0] > 0){                
                batchptr->nextIterationData.syncFlag.signal();
            }else{
                batchptr->nextIterationData.h_numCandidates[0] = 0;
                batchptr->nextIterationData.syncFlag.signal();
            }
            nvtx::pop_range();
        };

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batchData.streams;

#if 0
        batchData.nextIterationData.syncFlag.setBusy();
        getDataForNextIteration();
        batchData.nextIterationData.syncFlag.wait();
        batchData.updateFromIterationData(batchData.nextIterationData); 

#else
        if(batchData.isFirstIteration){
            batchData.nextIterationData.syncFlag.setBusy();

            getDataForNextIteration();        
         
            batchData.isFirstIteration = false;
        }else{
            batchData.nextIterationData.syncFlag.wait(); //wait until data is available
        }

        batchData.updateFromIterationData(batchData.nextIterationData);        
            
        batchData.nextIterationData.syncFlag.setBusy();

        
        std::array<cudaEvent_t, nEventsPerBatch>& events = batchData.events;

        cudaEventRecord(events[0], streams[preparation_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[primary_stream_index], events[0], 0); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], events[0], 0); CUERR;

#if 1   
        //asynchronously prepare data for next iteration
        batchData.backgroundWorker->enqueue(
            getDataForNextIteration
        );
#else  
        getDataForNextIteration();
#endif

#endif


        {
            bool* d_canExecutePtr = batchData.d_canExecute.get();
            int* d_numTotalCorrectedCandidatePtr = batchData.d_num_total_corrected_candidates.get();

            generic_kernel<<<1,1,0,streams[primary_stream_index]>>>(
                [=] __device__ (){
                    *d_canExecutePtr = true;
                    *d_numTotalCorrectedCandidatePtr = 0;
                }
            ); CUERR;
        }

    }



    void getCandidateSequenceData(Batch& batch, const DistributedReadStorage& readStorage){

        cudaSetDevice(batch.deviceId); CUERR;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        const auto batchsize = batch.correctionOptions.batchsize;
        const auto maxCandidates = batch.numCandidatesLimit;

        cudaMemcpyAsync(
            batch.h_subject_sequences_lengths,
            batch.d_subject_sequences_lengths, //filled by nextiteration data
            sizeof(int) * batchsize,
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        readStorage.readsContainN_async(
            batch.deviceId,
            batch.d_anchorContainsN.get(), 
            batch.d_subject_read_ids.get(), 
            batch.d_numAnchors,
            batchsize, 
            streams[primary_stream_index]
        );

        readStorage.readsContainN_async(
            batch.deviceId,
            batch.d_candidateContainsN.get(), 
            batch.d_candidate_read_ids.get(), 
            batch.d_numCandidates,
            maxCandidates, 
            streams[primary_stream_index]
        );  

        readStorage.gatherSequenceLengthsToGpuBufferAsync(
                                        batch.d_candidate_sequences_lengths.get(),
                                        batch.deviceId,
                                        batch.d_candidate_read_ids.get(),
                                        batch.h_numCandidates[0],            
                                        streams[primary_stream_index]);
        // std::cerr << "gather candidates\n";
        readStorage.gatherSequenceDataToGpuBufferAsync(
            batch.threadPool,
            batch.candidateSequenceGatherHandle,
            batch.d_candidate_sequences_data.get(),
            batch.encodedSequencePitchInInts,
            batch.h_candidate_read_ids,
            batch.d_candidate_read_ids,
            batch.h_numCandidates[0],
            batch.deviceId,
            streams[primary_stream_index]);

        call_transpose_kernel(
            batch.d_transposedCandidateSequencesData.get(), 
            batch.d_candidate_sequences_data.get(), 
            batch.h_numCandidates[0], 
            batch.encodedSequencePitchInInts, 
            batch.encodedSequencePitchInInts, 
            streams[primary_stream_index]
        );


        cudaEventRecord(events[alignment_data_transfer_h2d_finished_event_index], streams[primary_stream_index]); CUERR;

        cudaStreamWaitEvent(streams[secondary_stream_index], events[alignment_data_transfer_h2d_finished_event_index], 0) ;


        cudaMemcpyAsync(batch.h_candidate_sequences_lengths,
                        batch.d_candidate_sequences_lengths,
                        sizeof(int) * maxCandidates,
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

    }

    void getQualities(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

		std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;

        const auto* gpuReadStorage = batch.readStorage;

		if(batch.correctionOptions.useQualityScores) {
            // std::cerr << "gather anchor qual\n";
            gpuReadStorage->gatherQualitiesToGpuBufferAsync(
                batch.threadPool,
                batch.subjectQualitiesGatherHandle,
                batch.d_subject_qualities,
                batch.qualityPitchInBytes,
                batch.h_subject_read_ids,
                batch.d_subject_read_ids,
                batch.h_numAnchors[0],
                batch.deviceId,
                streams[primary_stream_index]);

            // std::cerr << "gather candidate qual\n";
            gpuReadStorage->gatherQualitiesToGpuBufferAsync(
                batch.threadPool,
                batch.candidateQualitiesGatherHandle,
                batch.d_candidate_qualities,
                batch.qualityPitchInBytes,
                batch.h_candidate_read_ids.get(),
                batch.d_candidate_read_ids.get(),
                batch.h_numCandidates[0],
                batch.deviceId,
                streams[primary_stream_index]);
        }

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
	}


	void getCandidateAlignments(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;

        const auto batchsize = batch.correctionOptions.batchsize;
        const auto maxCandidates = batch.numCandidatesLimit;        

        {
            const int* numAnchorsPtr = batch.d_numAnchors.get();
            int* d_anchorIndicesOfCandidates = batch.d_anchorIndicesOfCandidates.get();
            int* d_candidates_per_subject = batch.d_candidates_per_subject.get();
            int* d_candidates_per_subject_prefixsum = batch.d_candidates_per_subject_prefixsum.get();
            
            setAnchorIndicesOfCandidateskernel<1024, 128>
                    <<<1024, 128, 0, streams[primary_stream_index]>>>(
                batch.d_anchorIndicesOfCandidates.get(),
                batch.d_numAnchors.get(),
                batch.d_candidates_per_subject.get(),
                batch.d_candidates_per_subject_prefixsum.get()
            ); CUERR;
        }

        std::size_t tempBytes = batch.d_tempstorage.sizeInBytes();

        const bool removeAmbiguousAnchors = batch.correctionOptions.excludeAmbiguousReads;
        const bool removeAmbiguousCandidates = batch.correctionOptions.excludeAmbiguousReads;

        call_popcount_shifted_hamming_distance_kernel_async(
            batch.d_tempstorage.get(),
            tempBytes,
            batch.d_alignment_overlaps.get(),
            batch.d_alignment_shifts.get(),
            batch.d_alignment_nOps.get(),
            batch.d_alignment_isValid.get(),
            batch.d_alignment_best_alignment_flags.get(),
            batch.d_subject_sequences_data.get(),
            batch.d_candidate_sequences_data.get(),
            batch.d_subject_sequences_lengths.get(),
            batch.d_candidate_sequences_lengths.get(),
            batch.d_candidates_per_subject_prefixsum.get(),
            batch.d_candidates_per_subject.get(),
            batch.d_anchorIndicesOfCandidates.get(),
            batch.d_numAnchors.get(),
            batch.d_numCandidates.get(),
            batch.d_anchorContainsN.get(),
            removeAmbiguousAnchors,
            batch.d_candidateContainsN.get(),
            removeAmbiguousCandidates,
            batchsize,
            maxCandidates,
            batch.sequenceFileProperties.maxSequenceLength,
            batch.encodedSequencePitchInInts,
            batch.goodAlignmentProperties.min_overlap,
            batch.goodAlignmentProperties.maxErrorRate,
            batch.goodAlignmentProperties.min_overlap_ratio,
            batch.correctionOptions.estimatedErrorrate,
            //batch.maxSubjectLength,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );

        call_cuda_filter_alignments_by_mismatchratio_kernel_async(
            batch.d_alignment_best_alignment_flags.get(),
            batch.d_alignment_nOps.get(),
            batch.d_alignment_overlaps.get(),
            batch.d_candidates_per_subject_prefixsum.get(),
            batch.d_numAnchors.get(),
            batch.d_numCandidates.get(),
            batchsize,
            maxCandidates,
            batch.correctionOptions.estimatedErrorrate,
            batch.correctionOptions.estimatedCoverage * batch.correctionOptions.m_coverage,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );

        callSelectIndicesOfGoodCandidatesKernelAsync(
            batch.d_indices.get(),
            batch.d_indices_per_subject.get(),
            batch.d_num_indices.get(),
            batch.d_alignment_best_alignment_flags.get(),
            batch.d_candidates_per_subject.get(),
            batch.d_candidates_per_subject_prefixsum.get(),
            batch.d_anchorIndicesOfCandidates.get(),
            batch.d_numAnchors.get(),
            batch.d_numCandidates.get(),
            batchsize,
            maxCandidates,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );


        // cudaMemcpyAsync(batch.h_num_indices,
        //                 batch.d_num_indices,
        //                 sizeof(int),
        //                 D2H,
        //                 streams[primary_stream_index]); CUERR;       

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

        //std::cerr << "After alignment: " << *batch.h_num_indices << " / " << *batch.h_numCandidates << "\n";
	}

    void buildMultipleSequenceAlignment(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;

        const auto batchsize = batch.correctionOptions.batchsize;
        const auto maxCandidates = batch.numCandidatesLimit;

        GPUMultiMSA multiMSA;

        multiMSA.numMSAs = batchsize;
        multiMSA.columnPitchInElements = batch.msaColumnPitchInElements;
        multiMSA.counts = batch.d_counts.get();
        multiMSA.weights = batch.d_weights.get();
        multiMSA.coverages = batch.d_coverage.get();
        multiMSA.consensus = batch.d_consensus.get();
        multiMSA.support = batch.d_support.get();
        multiMSA.origWeights = batch.d_origWeights.get();
        multiMSA.origCoverages = batch.d_origCoverages.get();
        multiMSA.columnProperties = batch.d_msa_column_properties.get();

        callConstructMultipleSequenceAlignmentsKernel_async(
            multiMSA,
            batch.d_alignment_overlaps.get(),
            batch.d_alignment_shifts.get(),
            batch.d_alignment_nOps.get(),
            batch.d_alignment_best_alignment_flags.get(),
            batch.d_subject_sequences_lengths.get(),
            batch.d_candidate_sequences_lengths.get(),
            batch.d_indices,
            batch.d_indices_per_subject,
            batch.d_candidates_per_subject_prefixsum,
            batch.d_subject_sequences_data.get(),
            batch.d_candidate_sequences_data.get(),
            batch.d_subject_qualities.get(),
            batch.d_candidate_qualities.get(),
            batch.d_numAnchors.get(),
            batch.goodAlignmentProperties.maxErrorRate,
            batchsize,
            maxCandidates,
            batch.correctionOptions.useQualityScores,
            batch.encodedSequencePitchInInts,
            batch.qualityPitchInBytes,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );

        //At this point the msa is built

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
	}





    void removeCandidatesOfDifferentRegionFromMSA(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;

        const auto batchsize = batch.correctionOptions.batchsize;
        const auto maxCandidates = batch.numCandidatesLimit;            

        std::array<int*,2> d_indices_dblbuf{
            batch.d_indices.get(), 
            batch.d_indices_tmp.get()
        };
        std::array<int*,2> d_indices_per_subject_dblbuf{
            batch.d_indices_per_subject.get(), 
            batch.d_indices_per_subject_tmp.get()
        };
        std::array<int*,2> d_num_indices_dblbuf{
            batch.d_num_indices.get(), 
            batch.d_num_indices_tmp.get()
        };

        GPUMultiMSA multiMSA;

        multiMSA.numMSAs = batchsize;
        multiMSA.columnPitchInElements = batch.msaColumnPitchInElements;
        multiMSA.counts = batch.d_counts.get();
        multiMSA.weights = batch.d_weights.get();
        multiMSA.coverages = batch.d_coverage.get();
        multiMSA.consensus = batch.d_consensus.get();
        multiMSA.support = batch.d_support.get();
        multiMSA.origWeights = batch.d_origWeights.get();
        multiMSA.origCoverages = batch.d_origCoverages.get();
        multiMSA.columnProperties = batch.d_msa_column_properties.get();

#if 0        

        static_assert(max_num_minimizations % 2 == 1, "");

        const std::size_t requiredTempStorageBytes = 
            SDIV(sizeof(bool) * batch.h_numCandidates[0], 128) * 128 // d_shouldBeKept + padding to align the next pointer
            + (sizeof(bool) * batchsize); // d_anchorIsFinished
        assert(batch.d_tempstorage.sizeInBytes() >= requiredTempStorageBytes);
        
        bool* d_shouldBeKept = (bool*)batch.d_tempstorage.get();
        bool* d_anchorIsFinished = d_shouldBeKept + (SDIV(sizeof(bool) * batch.h_numCandidates[0], 128) * 128);

        cudaMemsetAsync(d_anchorIsFinished, 0, sizeof(bool) * batchsize, streams[primary_stream_index]);

        for(int iteration = 0; iteration < max_num_minimizations; iteration++){

            callMsaCandidateRefinementKernel_singleiter_async(
                d_indices_dblbuf[(1 + iteration) % 2],
                d_indices_per_subject_dblbuf[(1 + iteration) % 2],
                d_num_indices_dblbuf[(1 + iteration) % 2],
                multiMSA,
                batch.d_alignment_best_alignment_flags.get(),
                batch.d_alignment_shifts.get(),
                batch.d_alignment_nOps.get(),
                batch.d_alignment_overlaps.get(),
                batch.d_subject_sequences_data.get(),
                batch.d_candidate_sequences_data.get(),
                batch.d_subject_sequences_lengths.get(),
                batch.d_candidate_sequences_lengths.get(),
                batch.d_subject_qualities.get(),
                batch.d_candidate_qualities.get(),
                d_shouldBeKept,
                batch.d_candidates_per_subject_prefixsum,
                batch.d_numAnchors.get(),
                batch.goodAlignmentProperties.maxErrorRate,
                batchsize,
                maxCandidates,
                batch.correctionOptions.useQualityScores,
                batch.encodedSequencePitchInInts,
                batch.qualityPitchInBytes,
                d_indices_dblbuf[(0 + iteration) % 2],
                d_indices_per_subject_dblbuf[(0 + iteration) % 2],
                batch.correctionOptions.estimatedCoverage,
                iteration,
                d_anchorIsFinished,
                streams[primary_stream_index],
                batch.kernelLaunchHandle
            );


        }

#else 

        const std::size_t requiredTempStorageBytes = sizeof(bool) * batch.h_numCandidates[0]; // d_shouldBeKept
            
        assert(batch.d_tempstorage.sizeInBytes() >= requiredTempStorageBytes);

        bool* d_shouldBeKept = (bool*)batch.d_tempstorage.get();

        callMsaCandidateRefinementKernel_multiiter_async(
            d_indices_dblbuf[1],
            d_indices_per_subject_dblbuf[1],
            d_num_indices_dblbuf[1],
            multiMSA,
            batch.d_alignment_best_alignment_flags.get(),
            batch.d_alignment_shifts.get(),
            batch.d_alignment_nOps.get(),
            batch.d_alignment_overlaps.get(),
            batch.d_subject_sequences_data.get(),
            batch.d_candidate_sequences_data.get(),
            batch.d_subject_sequences_lengths.get(),
            batch.d_candidate_sequences_lengths.get(),
            batch.d_subject_qualities.get(),
            batch.d_candidate_qualities.get(),
            d_shouldBeKept,
            batch.d_candidates_per_subject_prefixsum,
            batch.d_numAnchors.get(),
            batch.goodAlignmentProperties.maxErrorRate,
            batchsize,
            maxCandidates,
            batch.correctionOptions.useQualityScores,
            batch.encodedSequencePitchInInts,
            batch.qualityPitchInBytes,
            batch.d_indices.get(),
            batch.d_indices_per_subject.get(),
            batch.correctionOptions.estimatedCoverage,
            max_num_minimizations,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );
#endif 

        //At this point the msa is built, maybe minimized, and is ready to be used for correction

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
    }


	void correctSubjects(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

		std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;
        const auto batchsize = batch.correctionOptions.batchsize;

		const float avg_support_threshold = 1.0f-1.0f*batch.correctionOptions.estimatedErrorrate;
		const float min_support_threshold = 1.0f-3.0f*batch.correctionOptions.estimatedErrorrate;
		// coverage is always >= 1
		const float min_coverage_threshold = std::max(1.0f,
            batch.correctionOptions.m_coverage / 6.0f * batch.correctionOptions.estimatedCoverage);
        const float max_coverage_threshold = 0.5 * batch.correctionOptions.estimatedCoverage;

		// correct subjects
#if 0

        std::array<int*,2> d_indices_dblbuf_msa{
            batch.d_indices.get(), 
            batch.d_indices_tmp.get()
        };
        std::array<int*,2> d_indices_per_subject_dblbuf_msa{
            batch.d_indices_per_subject.get(), 
            batch.d_indices_per_subject_tmp.get()
        };

        const int* d_indices_per_subject_msa = d_indices_per_subject_dblbuf_msa[/*(max_num_minimizations % 2)*/1];
        const int* d_indices_msa = d_indices_dblbuf_msa[/*(max_num_minimizations % 2)*/1];


        cudaMemcpyAsync(batch.h_indices,
                        d_indices_msa,
                        batch.h_indices.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(batch.h_indices_per_subject,
                        d_indices_per_subject_msa,
                        batch.h_indices_per_subject.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(batch.h_msa_column_properties,
                        batch.d_msa_column_properties,
                        batch.d_msa_column_properties.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(batch.h_alignment_shifts,
                        batch.d_alignment_shifts,
                        batch.d_alignment_shifts.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(batch.h_alignment_best_alignment_flags,
                        batch.d_alignment_best_alignment_flags,
                        batch.d_alignment_best_alignment_flags.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(batch.h_subject_sequences_data,
                        batch.d_subject_sequences_data,
                        batch.d_subject_sequences_data.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(batch.h_subject_sequences_lengths,
            batch.d_subject_sequences_lengths,
            batch.d_subject_sequences_lengths.sizeInBytes(),
            D2H,
            streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(batch.h_candidate_sequences_data,
                        batch.d_candidate_sequences_data,
                        batch.d_candidate_sequences_data.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(batch.h_candidate_sequences_lengths,
            batch.d_candidate_sequences_lengths,
            batch.d_candidate_sequences_lengths.sizeInBytes(),
            D2H,
            streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(batch.h_consensus,
                        batch.d_consensus,
                        batch.d_consensus.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(batch.h_candidates_per_subject_prefixsum,
            batch.d_candidates_per_subject_prefixsum,
            batch.d_candidates_per_subject_prefixsum.sizeInBytes(),
            D2H,
            streams[primary_stream_index]); CUERR;
        cudaDeviceSynchronize(); CUERR;

        auto identity = [](auto i){return i;};


        for(int i = 0; i < batch.h_numAnchors[0]; i++){
            std::cout << "anchor id " <<  batch.h_subject_read_ids[i] << "\n";
            int numind = batch.h_indices_per_subject[i];
            if(numind > 0){
                std::vector<char> cands;
                cands.resize(batch.decodedSequencePitchInBytes * numind, 'F');
                std::vector<int> candlengths;
                candlengths.resize(numind);
                std::vector<int> candshifts;
                candshifts.resize(numind);

                std::string anchorString = get2BitString(
                    batch.h_subject_sequences_data.get() + i * batch.encodedSequencePitchInInts,
                    batch.h_subject_sequences_lengths[i]
                );

                for(int j = 0; j < numind; j++){
                    const size_t anchoroffset = batch.h_candidates_per_subject_prefixsum[i];
                    int index = batch.h_indices[anchoroffset + j];
                    char* dst = cands.data() + batch.decodedSequencePitchInBytes * j;
                    candlengths[j] = batch.h_candidate_sequences_lengths[anchoroffset + index];
                    candshifts[j] = batch.h_alignment_shifts[anchoroffset + index];

                    const unsigned int* candidateSequencePtr = batch.h_candidate_sequences_data.get() + (anchoroffset + index) * batch.encodedSequencePitchInInts;

                    assert(batch.h_alignment_best_alignment_flags[anchoroffset + index] != BestAlignment_t::None);

                    std::string candidatestring = get2BitString((unsigned int*)candidateSequencePtr, candlengths[j]);
                    if(batch.h_alignment_best_alignment_flags[anchoroffset + index] == BestAlignment_t::ReverseComplement){
                        candidatestring = reverseComplementString(candidatestring.c_str(), candidatestring.length());
                    }

                    std::copy(candidatestring.begin(), candidatestring.end(), dst);
                    //decode2BitSequence(dst, (const unsigned int*)candidateSequencePtr, 100, identity);
                    //std::cout << "Candidate: " << s << std::endl;
                }

                // std::cerr << "subjectColumnsBegin_incl = " << batch.h_msa_column_properties[i].subjectColumnsBegin_incl << "\n";
                // std::cerr << "subjectColumnsEnd_excl = " << batch.h_msa_column_properties[i].subjectColumnsEnd_excl << "\n";
                // std::cerr << "firstColumn_incl = " << batch.h_msa_column_properties[i].firstColumn_incl << "\n";
                // std::cerr << "lastColumn_excl = " << batch.h_msa_column_properties[i].lastColumn_excl << "\n";
                // std::cerr << "candshifts:\n";

                // for(int j = 0; j < numind; j++){
                //     std::cerr << candshifts[j] << " ";
                // }
                // std::cerr << "\n";

                const int msacolumns = batch.h_msa_column_properties[i].lastColumn_excl - batch.h_msa_column_properties[i].firstColumn_incl;

                printSequencesInMSA(std::cout,
                    anchorString.c_str(),
                    batch.h_subject_sequences_lengths[i],
                    cands.data(),
                    candlengths.data(),
                    numind,
                    candshifts.data(),
                    batch.h_msa_column_properties[i].subjectColumnsBegin_incl,
                    batch.h_msa_column_properties[i].subjectColumnsEnd_excl,
                    msacolumns,
                    batch.decodedSequencePitchInBytes);
                std::cout << "\n";
                std::string consensus = std::string{batch.h_consensus + i * batch.msaColumnPitchInElements, batch.h_consensus + i * batch.msaColumnPitchInElements + msacolumns};
                std::cout << "Consensus:\n   " << consensus << "\n\n";
                printSequencesInMSAConsEq(std::cout,
                    anchorString.c_str(),
                    batch.h_subject_sequences_lengths[i],
                    cands.data(),
                    candlengths.data(),
                    numind,
                    candshifts.data(),
                    consensus.c_str(),
                    batch.h_msa_column_properties[i].subjectColumnsBegin_incl,
                    batch.h_msa_column_properties[i].subjectColumnsEnd_excl,
                    msacolumns,
                    batch.decodedSequencePitchInBytes);
                std::cout << "\n";

                //std::exit(0);
            }
        }

        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
#endif

#if 0
        cudaDeviceSynchronize(); CUERR;

        cudaMemcpyAsync(batch.h_msa_column_properties,
            batch.d_msa_column_properties,
            batch.d_msa_column_properties.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(batch.h_counts,
            batch.d_counts,
            batch.d_counts.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(batch.h_weights,
            batch.d_weights,
            batch.d_weights.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(batch.h_coverage,
            batch.d_coverage,
            batch.d_coverage.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(batch.h_origWeights,
            batch.d_origWeights,
            batch.d_origWeights.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(batch.h_origCoverages,
            batch.d_origCoverages,
            batch.d_origCoverages.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(batch.h_support,
            batch.d_support,
            batch.d_support.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(batch.h_consensus,
            batch.d_consensus,
            batch.d_consensus.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;
        cudaDeviceSynchronize();

        const std::size_t msaColumnPitchInElements = batch.msaColumnPitchInElements;

        for(int i = 0; i < batch.h_numAnchors[0]; i++){
            if(batch.h_subject_read_ids[i] == 13){
                std::cerr << "subjectColumnsBegin_incl = " << batch.h_msa_column_properties[i].subjectColumnsBegin_incl << "\n";
                std::cerr << "subjectColumnsEnd_excl = " << batch.h_msa_column_properties[i].subjectColumnsEnd_excl << "\n";
                std::cerr << "lastColumn_excl = " << batch.h_msa_column_properties[i].lastColumn_excl << "\n";
                std::cerr << "counts: \n";
                int* counts = batch.h_counts + i * 4 * msaColumnPitchInElements;
                for(int k = 0; k < msaColumnPitchInElements; k++){
                    std::cerr << counts[0 * msaColumnPitchInElements + k] << ' ';
                }
                std::cerr << "\n";
                for(int k = 0; k < msaColumnPitchInElements; k++){
                    std::cerr << counts[1 * msaColumnPitchInElements + k] << ' ';
                }
                std::cerr << "\n";
                for(int k = 0; k < msaColumnPitchInElements; k++){
                    std::cerr << counts[2 * msaColumnPitchInElements + k] << ' ';
                }
                std::cerr << "\n";
                for(int k = 0; k < msaColumnPitchInElements; k++){
                    std::cerr << counts[3 * msaColumnPitchInElements + k] << ' ';
                }
                std::cerr << "\n";

                std::cerr << "weights: \n";
                float* weights = batch.h_weights + i * 4 * msaColumnPitchInElements;
                for(int k = 0; k < msaColumnPitchInElements; k++){
                    std::cerr << weights[0 * msaColumnPitchInElements + k] << ' ';
                }
                std::cerr << "\n";
                for(int k = 0; k < msaColumnPitchInElements; k++){
                    std::cerr << weights[1 * msaColumnPitchInElements + k] << ' ';
                }
                std::cerr << "\n";
                for(int k = 0; k < msaColumnPitchInElements; k++){
                    std::cerr << weights[2 * msaColumnPitchInElements + k] << ' ';
                }
                std::cerr << "\n";
                for(int k = 0; k < msaColumnPitchInElements; k++){
                    std::cerr << weights[3 * msaColumnPitchInElements + k] << ' ';
                }
                std::cerr << "\n";

                std::cerr << "coverage: \n";
                int* coverage = batch.h_coverage + i * msaColumnPitchInElements;
                for(int k = 0; k < msaColumnPitchInElements; k++){
                    std::cerr << coverage[k] << ' ';
                }
                std::cerr << "\n";

                std::cerr << "support: \n";
                float* support = batch.h_support + i * msaColumnPitchInElements;
                for(int k = 0; k < msaColumnPitchInElements; k++){
                    std::cerr << support[k] << ' ';
                }
                std::cerr << "\n";

                std::cerr << "consensus: \n";
                char* consensus = batch.h_consensus + i * msaColumnPitchInElements;
                for(int k = 0; k < msaColumnPitchInElements; k++){
                    std::cerr << consensus[k] << ' ';
                }
                std::cerr << "\n";
            }
        }

        
#endif        
        // cudaEventRecord(events[msa_build_finished_event_index], streams[primary_stream_index]); CUERR;
        // cudaStreamWaitEvent(streams[secondary_stream_index], events[msa_build_finished_event_index], 0); CUERR;

        std::array<int*,2> d_indices_per_subject_dblbuf{
            batch.d_indices_per_subject.get(), 
            batch.d_indices_per_subject_tmp.get()
        };
        std::array<int*,2> d_num_indices_dblbuf{
            batch.d_num_indices.get(), 
            batch.d_num_indices_tmp.get()
        };

        const int* d_indices_per_subject = d_indices_per_subject_dblbuf[/*max_num_minimizations % 2*/ 1];
        const int* d_num_indices = d_num_indices_dblbuf[/*max_num_minimizations % 2*/ 1];

        GPUMultiMSA multiMSA;

        multiMSA.numMSAs = batchsize;
        multiMSA.columnPitchInElements = batch.msaColumnPitchInElements;
        multiMSA.counts = batch.d_counts.get();
        multiMSA.weights = batch.d_weights.get();
        multiMSA.coverages = batch.d_coverage.get();
        multiMSA.consensus = batch.d_consensus.get();
        multiMSA.support = batch.d_support.get();
        multiMSA.origWeights = batch.d_origWeights.get();
        multiMSA.origCoverages = batch.d_origCoverages.get();
        multiMSA.columnProperties = batch.d_msa_column_properties.get();


        call_msaCorrectAnchorsKernel_async(
            batch.d_corrected_subjects.get(),
            batch.d_subject_is_corrected.get(),
            batch.d_is_high_quality_subject.get(),
            multiMSA,
            batch.d_subject_sequences_data.get(),
            batch.d_candidate_sequences_data.get(),
            batch.d_candidate_sequences_lengths.get(),
            d_indices_per_subject,
            batch.d_numAnchors.get(),
            batchsize,
            batch.encodedSequencePitchInInts,
            batch.decodedSequencePitchInBytes,
            batch.sequenceFileProperties.maxSequenceLength,
            batch.correctionOptions.estimatedErrorrate,
            batch.goodAlignmentProperties.maxErrorRate,
            avg_support_threshold,
            min_support_threshold,
            min_coverage_threshold,
            max_coverage_threshold,
            batch.correctionOptions.kmerlength,
            batch.sequenceFileProperties.maxSequenceLength,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );

        cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], events[correction_finished_event_index], 0); CUERR;

        //std::cerr << "cudaMemcpyAsync " << (void*)batch.h_indices_per_subject.get() << (void*) d_indices_per_subject << "\n";
        cudaMemcpyAsync(
            batch.h_indices_per_subject,
            d_indices_per_subject,
            sizeof(int) * batchsize,
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(batch.h_corrected_subjects,
                        batch.d_corrected_subjects,
                        batch.decodedSequencePitchInBytes * batchsize,
                        D2H,
                        streams[secondary_stream_index]); CUERR;
        cudaMemcpyAsync(batch.h_subject_is_corrected,
                        batch.d_subject_is_corrected,
                        sizeof(bool) * batchsize,
                        D2H,
                        streams[secondary_stream_index]); CUERR;
        cudaMemcpyAsync(batch.h_is_high_quality_subject,
                        batch.d_is_high_quality_subject,
                        sizeof(AnchorHighQualityFlag) * batchsize,
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        // cudaMemcpyAsync(batch.h_num_uncorrected_positions_per_subject,
        //                 batch.d_num_uncorrected_positions_per_subject,
        //                 sizeof(int) * batchsize,
        //                 D2H,
        //                 streams[secondary_stream_index]); CUERR;

        // cudaMemcpyAsync(batch.h_uncorrected_positions_per_subject,
        //                 batch.d_uncorrected_positions_per_subject,
        //                 sizeof(int) * batch.sequenceFileProperties.maxSequenceLength * batchsize,
        //                 D2H,
        //                 streams[secondary_stream_index]); CUERR;

        selectIndicesOfFlagsOnlyOneBlock<256><<<1,256,0, streams[primary_stream_index]>>>(
            batch.d_indices_of_corrected_subjects.get(),
            batch.d_num_indices_of_corrected_subjects.get(),
            batch.d_subject_is_corrected.get(),
            batch.d_numAnchors.get()
        ); CUERR;

        callConstructAnchorResultsKernelAsync(
            batch.d_editsPerCorrectedSubject.get(),
            batch.d_numEditsPerCorrectedSubject.get(),
            doNotUseEditsValue,
            batch.d_indices_of_corrected_subjects.get(),
            batch.d_num_indices_of_corrected_subjects.get(),
            batch.d_anchorContainsN.get(),
            batch.d_subject_sequences_data.get(),
            batch.d_subject_sequences_lengths.get(),
            batch.d_corrected_subjects.get(),
            batch.maxNumEditsPerSequence,
            batch.encodedSequencePitchInInts,
            batch.decodedSequencePitchInBytes,
            batch.editsPitchInBytes,
            batch.d_numAnchors.get(),
            batch.correctionOptions.batchsize,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );

        cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], events[correction_finished_event_index], 0); CUERR;

        cudaMemcpyAsync(
            batch.h_editsPerCorrectedSubject,
            batch.d_editsPerCorrectedSubject,
            batch.editsPitchInBytes * batchsize,
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(
            batch.h_numEditsPerCorrectedSubject,
            batch.d_numEditsPerCorrectedSubject,
            sizeof(int) * batchsize,
            D2H,
            streams[secondary_stream_index]
        ); CUERR;
        
	}



    void correctCandidates(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        const auto batchsize = batch.correctionOptions.batchsize;
        const auto maxCandidates = batch.numCandidatesLimit;

        const float min_support_threshold = 1.0f-3.0f*batch.correctionOptions.estimatedErrorrate;
        // coverage is always >= 1
        const float min_coverage_threshold = std::max(1.0f,
            batch.correctionOptions.m_coverage / 6.0f * batch.correctionOptions.estimatedCoverage);
        const int new_columns_to_correct = batch.correctionOptions.new_columns_to_correct;

        bool* const d_candidateCanBeCorrected = batch.d_alignment_isValid.get(); //repurpose

        int* const d_num_corrected_candidates_per_anchor = batch.d_num_corrected_candidates_per_anchor.get();
        const int* const d_numAnchors = batch.d_numAnchors.get();
        const int* const d_numCandidates = batch.d_numCandidates.get();

        auto isHqSubject = [] __device__ (const AnchorHighQualityFlag& flag){
            return flag.hq();
        };

        cub::TransformInputIterator<bool,decltype(isHqSubject), AnchorHighQualityFlag*>
            d_isHqSubject(batch.d_is_high_quality_subject,
                            isHqSubject);

        selectIndicesOfFlagsOnlyOneBlock<256><<<1,256,0, streams[primary_stream_index]>>>(
            batch.d_high_quality_subject_indices.get(),
            batch.d_num_high_quality_subject_indices.get(),
            d_isHqSubject,
            batch.d_numAnchors.get()
        ); CUERR;

        cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], events[correction_finished_event_index], 0); CUERR;

        cudaMemcpyAsync(batch.h_high_quality_subject_indices,
                        batch.d_high_quality_subject_indices,
                        sizeof(int) * batchsize,
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaMemcpyAsync(batch.h_num_high_quality_subject_indices,
                        batch.d_num_high_quality_subject_indices,
                        sizeof(int),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        generic_kernel<<<640, 128, 0, streams[primary_stream_index]>>>(
            [=] __device__ (){
                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                const int stride = blockDim.x * gridDim.x;

                for(int i = tid; i < batchsize; i += stride){
                    d_num_corrected_candidates_per_anchor[i] = 0;
                }

                for(int i = tid; i < maxCandidates; i += stride){
                    d_candidateCanBeCorrected[i] = 0;
                }
            }
        ); CUERR;


        std::array<int*,2> d_indices_dblbuf{
            batch.d_indices.get(), 
            batch.d_indices_tmp.get()
        };
        std::array<int*,2> d_indices_per_subject_dblbuf{
            batch.d_indices_per_subject.get(), 
            batch.d_indices_per_subject_tmp.get()
        };
        std::array<int*,2> d_num_indices_dblbuf{
            batch.d_num_indices.get(), 
            batch.d_num_indices_tmp.get()
        };

        const int* d_indices = d_indices_dblbuf[/*max_num_minimizations % 2*/1];
        const int* d_indices_per_subject = d_indices_per_subject_dblbuf[/*max_num_minimizations % 2*/1];
        const int* d_num_indices = d_num_indices_dblbuf[/*max_num_minimizations % 2*/1];

        GPUMultiMSA multiMSA;

        multiMSA.numMSAs = batchsize;
        multiMSA.columnPitchInElements = batch.msaColumnPitchInElements;
        multiMSA.counts = batch.d_counts.get();
        multiMSA.weights = batch.d_weights.get();
        multiMSA.coverages = batch.d_coverage.get();
        multiMSA.consensus = batch.d_consensus.get();
        multiMSA.support = batch.d_support.get();
        multiMSA.origWeights = batch.d_origWeights.get();
        multiMSA.origCoverages = batch.d_origCoverages.get();
        multiMSA.columnProperties = batch.d_msa_column_properties.get();

        callFlagCandidatesToBeCorrectedKernel_async(
            d_candidateCanBeCorrected,
            batch.d_num_corrected_candidates_per_anchor.get(),
            multiMSA,
            batch.d_alignment_shifts.get(),
            batch.d_candidate_sequences_lengths.get(),
            batch.d_anchorIndicesOfCandidates.get(),
            batch.d_is_high_quality_subject.get(),
            batch.d_candidates_per_subject_prefixsum,
            d_indices,
            d_indices_per_subject,
            d_numAnchors,
            d_numCandidates,
            min_support_threshold,
            min_coverage_threshold,
            new_columns_to_correct,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );

        size_t cubTempSize = batch.d_tempstorage.sizeInBytes();

        cub::DeviceSelect::Flagged(
            batch.d_tempstorage.get(),
            cubTempSize,
            cub::CountingInputIterator<int>(0),
            d_candidateCanBeCorrected,
            batch.d_indices_of_corrected_candidates.get(),
            batch.d_num_total_corrected_candidates.get(),
            maxCandidates,
            streams[primary_stream_index]
        ); CUERR;

        cudaEvent_t flaggingfinished = events[result_transfer_finished_event_index];

        cudaEventRecord(flaggingfinished, streams[primary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], flaggingfinished, 0); CUERR;

        //start result transfer of already calculated data in second stream

        cudaMemcpyAsync(
            batch.h_num_total_corrected_candidates.get(),
            batch.d_num_total_corrected_candidates.get(),
            sizeof(int),
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        //cudaEventRecord(events[numTotalCorrectedCandidates_event_index], streams[secondary_stream_index]); CUERR;

        cub::DeviceScan::ExclusiveSum(
            batch.d_tempstorage.get(), 
            cubTempSize, 
            batch.d_num_corrected_candidates_per_anchor.get(), 
            batch.d_num_corrected_candidates_per_anchor_prefixsum.get(), 
            batchsize, 
            streams[secondary_stream_index]
        );

        cudaMemcpyAsync(
            batch.h_num_corrected_candidates_per_anchor,
            batch.d_num_corrected_candidates_per_anchor,
            sizeof(int) * batchsize,
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(
            batch.h_num_corrected_candidates_per_anchor_prefixsum.get(),
            batch.d_num_corrected_candidates_per_anchor_prefixsum.get(),
            sizeof(int) * batchsize,
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        // cudaMemcpyAsync(
        //     batch.h_alignment_shifts,
        //     batch.d_alignment_shifts,
        //     sizeof(int) * maxCandidates, //actually only need sizeof(int) * num_total_corrected_candidates, but its not available on the host
        //     D2H,
        //     streams[secondary_stream_index]
        // ); CUERR;

        int* h_alignment_shifts = batch.h_alignment_shifts.get();
        const int* d_alignment_shifts = batch.d_alignment_shifts.get();
        int* h_indices_of_corrected_candidates = batch.h_indices_of_corrected_candidates.get();
        const int* d_indices_of_corrected_candidates = batch.d_indices_of_corrected_candidates.get();

        //copy alignment shifts and indices of corrected candidates from device to host

        generic_kernel<<<320, 256, 0, streams[secondary_stream_index]>>>(
            [=] __device__ (){
                using CopyType = int;

                const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
                const size_t stride = blockDim.x * gridDim.x;

                const int numElements = *d_numCandidates;

                for(int index = tid; index < numElements; index += stride){
                    h_alignment_shifts[index] = d_alignment_shifts[index];
                    h_indices_of_corrected_candidates[index] = d_indices_of_corrected_candidates[index];
                } 
            }
        ); CUERR;

        //compute candidate correction in first stream
#if 1
        callCorrectCandidatesKernel_async(
            batch.h_corrected_candidates.get(),
            batch.h_editsPerCorrectedCandidate.get(),
            batch.h_numEditsPerCorrectedCandidate.get(),
            multiMSA,
            batch.d_alignment_shifts.get(),
            batch.d_alignment_best_alignment_flags.get(),
            batch.d_candidate_sequences_data.get(),
            batch.d_candidate_sequences_lengths.get(),
            batch.d_candidateContainsN.get(),
            batch.d_indices_of_corrected_candidates.get(),
            batch.d_num_total_corrected_candidates.get(),
            batch.d_anchorIndicesOfCandidates.get(),
            d_numAnchors,
            d_numCandidates,
            doNotUseEditsValue,
            batch.maxNumEditsPerSequence,
            batch.encodedSequencePitchInInts,
            batch.decodedSequencePitchInBytes,
            batch.editsPitchInBytes,
            batch.sequenceFileProperties.maxSequenceLength,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );
#else
        callCorrectCandidatesKernel_async(
            batch.d_corrected_candidates.get(),
            batch.d_editsPerCorrectedCandidate.get(),
            batch.d_numEditsPerCorrectedCandidate.get(),
            multiMSA,
            batch.d_alignment_shifts.get(),
            batch.d_alignment_best_alignment_flags.get(),
            batch.d_candidate_sequences_data.get(),
            batch.d_candidate_sequences_lengths.get(),
            batch.d_candidateContainsN.get(),
            batch.d_indices_of_corrected_candidates.get(),
            batch.d_num_total_corrected_candidates.get(),
            batch.d_anchorIndicesOfCandidates.get(),
            d_numAnchors,
            d_numCandidates,
            doNotUseEditsValue,
            batch.maxNumEditsPerSequence,
            batch.encodedSequencePitchInInts,
            batch.decodedSequencePitchInBytes,
            batch.editsPitchInBytes,
            batch.sequenceFileProperties.maxSequenceLength,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        ); 
    
        //copy candidate correction results from device to host  
#if 0        
        generic_kernel<<<480, 256, 0, streams[primary_stream_index]>>>(
            [
                =,
                decodedSequencePitchInBytes = batch.decodedSequencePitchInBytes,
                h_corrected_candidates = batch.h_corrected_candidates.get(),
                h_numEditsPerCorrectedCandidate = batch.h_numEditsPerCorrectedCandidate.get(),
                d_corrected_candidates = batch.d_corrected_candidates.get(),
                d_numEditsPerCorrectedCandidate = batch.d_numEditsPerCorrectedCandidate.get(),
                d_num_total_corrected_candidates = batch.d_num_total_corrected_candidates.get()
            ] __device__ (){

                using CopyType = int4;

                const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
                const size_t stride = blockDim.x * gridDim.x;

                const int numElements = *d_num_total_corrected_candidates;

                auto copy = [&](auto* dest, auto* src, size_t numbytes){
                    const size_t iters = numbytes / sizeof(CopyType);

                    for(size_t index = tid; index < iters; index += stride){
                        ((CopyType*)dest)[index] = ((const CopyType*)src)[index];
                    }
                    //remainder
                    const size_t remainder = numbytes - sizeof(CopyType) * iters;
                    for(size_t index = tid; index < remainder; index += stride){
                        ((char*)dest)[sizeof(CopyType) * iters + index] 
                            = ((const char*)src)[sizeof(CopyType) * iters + index];
                    }
                };

                const size_t bytesCorrectedCandidates = numElements * decodedSequencePitchInBytes;
                copy(h_corrected_candidates, d_corrected_candidates, bytesCorrectedCandidates);

                const size_t bytesNumEdits = numElements * sizeof(int);
                copy(h_numEditsPerCorrectedCandidate, d_numEditsPerCorrectedCandidate, bytesNumEdits);
            }
        ); CUERR;    
        
        generic_kernel<<<480, 256, 0, streams[primary_stream_index]>>>(
            [
                =,
                editsPitchInBytes = batch.editsPitchInBytes,
                h_editsPerCorrectedCandidate = batch.h_editsPerCorrectedCandidate.get(),
                h_numEditsPerCorrectedCandidate = batch.h_numEditsPerCorrectedCandidate.get(),
                d_editsPerCorrectedCandidate = batch.d_editsPerCorrectedCandidate.get(),
                d_numEditsPerCorrectedCandidate = batch.d_numEditsPerCorrectedCandidate.get(),
                d_num_total_corrected_candidates = batch.d_num_total_corrected_candidates.get()
            ] __device__ (){

                const int numElements = *d_num_total_corrected_candidates;

                const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
                const size_t stride = blockDim.x * gridDim.x;

                auto tile = cg::tiled_partition<16>(cg::this_thread_block());

                const int tileId = tid / 16;
                const int numTiles = stride / 16;

                auto copyElements = [&](auto* dest, auto* src, size_t numElements){

                    for(size_t index = tile.thread_rank(); index < numElements; index += tile.size()){
                        dest[index] = src[index];
                    }
                };

                for(int k = tileId; k < numElements; k += numTiles){
                    const int numEdits = d_numEditsPerCorrectedCandidate[k];
                    if(numEdits > 0){
                        copyElements(
                            ((char*)h_editsPerCorrectedCandidate) + k * editsPitchInBytes, 
                            ((const char*)d_editsPerCorrectedCandidate) + k * editsPitchInBytes, 
                            numEdits
                        );
                    }
                }

            }
        ); CUERR;
#else 
        generic_kernel<<<480, 256, 0, streams[primary_stream_index]>>>(
            [
                =,
                decodedSequencePitchInBytes = batch.decodedSequencePitchInBytes,
                editsPitchInBytes = batch.editsPitchInBytes,
                h_corrected_candidates = batch.h_corrected_candidates.get(),
                h_editsPerCorrectedCandidate = batch.h_editsPerCorrectedCandidate.get(),
                h_numEditsPerCorrectedCandidate = batch.h_numEditsPerCorrectedCandidate.get(),
                d_corrected_candidates = batch.d_corrected_candidates.get(),
                d_editsPerCorrectedCandidate = batch.d_editsPerCorrectedCandidate.get(),
                d_numEditsPerCorrectedCandidate = batch.d_numEditsPerCorrectedCandidate.get(),
                d_num_total_corrected_candidates = batch.d_num_total_corrected_candidates.get()
            ] __device__ (){

                using CopyType = int4;

                const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
                const size_t stride = blockDim.x * gridDim.x;

                const int numElements = *d_num_total_corrected_candidates;

                auto copy = [&](auto* dest, auto* src, size_t numbytes){
                    const size_t iters = numbytes / sizeof(CopyType);

                    for(size_t index = tid; index < iters; index += stride){
                        ((CopyType*)dest)[index] = ((const CopyType*)src)[index];
                    }
                    //remainder
                    const size_t remainder = numbytes - sizeof(CopyType) * iters;
                    for(size_t index = tid; index < remainder; index += stride){
                        ((char*)dest)[sizeof(CopyType) * iters + index] 
                            = ((const char*)src)[sizeof(CopyType) * iters + index];
                    }
                };

                const size_t bytesCorrectedCandidates = numElements * decodedSequencePitchInBytes;
                copy(h_corrected_candidates, d_corrected_candidates, bytesCorrectedCandidates);

                const size_t bytesNumEdits = numElements * sizeof(int);
                copy(h_numEditsPerCorrectedCandidate, d_numEditsPerCorrectedCandidate, bytesNumEdits);

                const size_t bytesEdits = numElements * editsPitchInBytes;
                copy(h_editsPerCorrectedCandidate, d_editsPerCorrectedCandidate, bytesEdits);

            }
        ); CUERR;
#endif        
#endif        

        
    }


    void constructResults(Batch& batch){

        auto& outputData = batch.waitableOutputData.data;
        auto& rawResults = outputData.rawResults;

        auto& subjectIndicesToProcess = outputData.subjectIndicesToProcess;
        auto& candidateIndicesToProcess = outputData.candidateIndicesToProcess;

        subjectIndicesToProcess.clear();
        candidateIndicesToProcess.clear();

        subjectIndicesToProcess.reserve(rawResults.n_subjects);
        candidateIndicesToProcess.reserve(16 * rawResults.n_subjects);

        nvtx::push_range("preprocess anchor results",0);

        for(int subject_index = 0; subject_index < rawResults.n_subjects; subject_index++){
            const read_number readId = rawResults.h_subject_read_ids[subject_index];
            const bool isCorrected = rawResults.h_subject_is_corrected[subject_index];
            const bool isHQ = rawResults.h_is_high_quality_subject[subject_index].hq();

            if(isHQ){
                batch.correctionStatusFlagsPerRead[readId] |= readCorrectedAsHQAnchor;
            }

            if(isCorrected){
                subjectIndicesToProcess.emplace_back(subject_index);
            }else{
                batch.correctionStatusFlagsPerRead[readId] |= readCouldNotBeCorrectedAsAnchor;
            }
        }

        nvtx::pop_range();

        nvtx::push_range("preprocess candidate results",0);

        //int acc = 0;
        for(int subject_index = 0; subject_index < rawResults.n_subjects; subject_index++){

            const int globalOffset = rawResults.h_num_corrected_candidates_per_anchor_prefixsum[subject_index];
            //assert(globalOffset == acc);
            const int n_corrected_candidates = rawResults.h_num_corrected_candidates_per_anchor[subject_index];
            // if(n_corrected_candidates > 0){
            //     assert(
            //         rawResults.h_is_high_quality_subject[subject_index].hq()
            //         // std::find(
            //         //     rawResults.h_high_quality_subject_indices.get(),
            //         //     rawResults.h_high_quality_subject_indices.get() + rawResults.n_subjects,
            //         //     subject_index
            //         // ) != rawResults.h_high_quality_subject_indices.get() + rawResults.n_subjects
            //     );
                
            // }
            // assert(n_corrected_candidates <= rawResults.h_indices_per_subject[subject_index]);

            const int* const my_indices_of_corrected_candidates = rawResults.h_indices_of_corrected_candidates
                                                + globalOffset;

            for(int i = 0; i < n_corrected_candidates; ++i) {
                const int global_candidate_index = my_indices_of_corrected_candidates[i];

                const read_number candidate_read_id = rawResults.h_candidate_read_ids[global_candidate_index];

                bool savingIsOk = false;
                const std::uint8_t mask = batch.correctionStatusFlagsPerRead[candidate_read_id];
                if(!(mask & readCorrectedAsHQAnchor)) {
                    savingIsOk = true;
                }
                if (savingIsOk) {
                    //std::cerr << global_candidate_index << " will be corrected\n";
                    candidateIndicesToProcess.emplace_back(std::make_pair(subject_index, i));
                }else{
                    //std::cerr << global_candidate_index << " discarded\n";
                }
            }

            //acc += n_corrected_candidates;
        }

        nvtx::pop_range();

        const int numCorrectedAnchors = subjectIndicesToProcess.size();
        const int numCorrectedCandidates = candidateIndicesToProcess.size();

        //std::cerr << "numCorrectedCandidates " << numCorrectedCandidates << "\n";

        //  std::cerr << "\n" << "batch " << batch.id << " " 
        //      << numCorrectedAnchors << " " << numCorrectedCandidates << "\n";

        // nvtx::push_range("clear",1);
        // outputData.anchorCorrections.clear();
        // outputData.encodedAnchorCorrections.clear();
        // outputData.candidateCorrections.clear();
        // outputData.encodedCandidateCorrections.clear();
        // nvtx::pop_range();


        nvtx::push_range("resize",1);
        outputData.resizeAnchors(numCorrectedAnchors);
        outputData.resizeCandidates(numCorrectedCandidates);
        nvtx::pop_range();

        auto outputDataPtr = &outputData;

        auto unpackAnchors = [outputDataPtr](int begin, int end){
            nvtx::push_range("Anchor unpacking", 3);
            
            auto& outputData = *outputDataPtr;
            auto& rawResults = outputData.rawResults;
            const auto& subjectIndicesToProcess = outputData.subjectIndicesToProcess;
            
            for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                const int subject_index = subjectIndicesToProcess[positionInVector];

                auto& tmp = outputData.anchorCorrections[positionInVector];
                auto& tmpencoded = outputData.encodedAnchorCorrections[positionInVector];
                
                const read_number readId = rawResults.h_subject_read_ids[subject_index];

                tmp.hq = rawResults.h_is_high_quality_subject[subject_index].hq();                    
                tmp.type = TempCorrectedSequence::Type::Anchor;
                tmp.readId = readId;
                
                // const int numUncorrectedPositions = rawResults.h_num_uncorrected_positions_per_subject[subject_index];

                // if(numUncorrectedPositions > 0){
                //     tmp.uncorrectedPositionsNoConsensus.resize(numUncorrectedPositions);
                //     std::copy_n(rawResults.h_uncorrected_positions_per_subject + subject_index * batch.sequenceFileProperties.maxSequenceLength,
                //                 numUncorrectedPositions,
                //                 tmp.uncorrectedPositionsNoConsensus.begin());

                // }

                const int numEdits = rawResults.h_numEditsPerCorrectedSubject[positionInVector];
                if(numEdits != doNotUseEditsValue){
                    tmp.edits.resize(numEdits);
                    const TempCorrectedSequence::EncodedEdit* const gpuedits 
                        = (const TempCorrectedSequence::EncodedEdit*)(((const char*)rawResults.h_editsPerCorrectedSubject.get()) 
                            + positionInVector * rawResults.editsPitchInBytes);
                    std::copy_n(gpuedits, numEdits, tmp.edits.begin());
                    tmp.useEdits = true;
                }else{
                    tmp.edits.clear();
                    tmp.useEdits = false;

                    const char* const my_corrected_subject_data = rawResults.h_corrected_subjects + subject_index * rawResults.decodedSequencePitchInBytes;
                    const int subject_length = rawResults.h_subject_sequences_lengths[subject_index];
                    //tmp.sequence = std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length};       
                    tmp.sequence.assign(my_corrected_subject_data, subject_length);
                    
                    // auto isValidSequence = [](const std::string& s){
                    //     return std::all_of(s.begin(), s.end(), [](char c){
                    //         return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
                    //     });
                    // };
    
                    // if(!isValidSequence(tmp.sequence)){
                    //     std::cerr << "invalid sequence\n"; //std::cerr << tmp.sequence << "\n";
                    // }
                }

                //tmpencoded = tmp.encode();
                tmp.encodeInto(tmpencoded);

                // if(readId == 32141191 /* || readId == 10307280 || readId == 42537816*/){
                //     std::cerr << "readid = " << readId << ", anchor\n";
                //     std::cerr << "hq = " << tmp.hq;
                //     if(!tmp.useEdits){
                //         std::cerr << ", sequence = " << tmp.sequence << "\n";
                //     }else{
                //         std::cerr << "numEdits = " << tmp.edits.size();
                //         std::cerr << "\nedits: \n";
                //         for(int i = 0; i < int(tmp.edits.size()); i++){
                //             std::cerr << tmp.edits[i].base << ' ' << tmp.edits[i].pos << "\n";
                //         }
                //     }
                // }
            }

            nvtx::pop_range();
        };

        auto batchPtr = &batch;

        auto unpackcandidates = [outputDataPtr, batchPtr](int begin, int end){
            nvtx::push_range("candidate unpacking", 3);
            //std::cerr << "\n\n unpack candidates \n\n";
            
            auto& outputData = *outputDataPtr;
            auto& rawResults = outputData.rawResults;
            const auto& batch = *batchPtr;

            const auto& candidateIndicesToProcess = outputData.candidateIndicesToProcess;

            //std::cerr << "in unpackcandidates " << begin << " - " << end << "\n";

            for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                //TIMERSTARTCPU(setup);
                const int subject_index = candidateIndicesToProcess[positionInVector].first;
                const int candidateIndex = candidateIndicesToProcess[positionInVector].second;
                const read_number subjectReadId = rawResults.h_subject_read_ids[subject_index];

                auto& tmp = outputData.candidateCorrections[positionInVector];
                auto& tmpencoded = outputData.encodedCandidateCorrections[positionInVector];

                const size_t offsetForCorrectedCandidateData = rawResults.h_num_corrected_candidates_per_anchor_prefixsum[subject_index];

                const char* const my_corrected_candidates_data = rawResults.h_corrected_candidates
                                                + offsetForCorrectedCandidateData * rawResults.decodedSequencePitchInBytes;
                const int* const my_indices_of_corrected_candidates = rawResults.h_indices_of_corrected_candidates
                                                + offsetForCorrectedCandidateData;

                const TempCorrectedSequence::EncodedEdit* const my_editsPerCorrectedCandidate 
                    = (const TempCorrectedSequence::EncodedEdit*)(((const char*)rawResults.h_editsPerCorrectedCandidate.get()) 
                        + offsetForCorrectedCandidateData * rawResults.editsPitchInBytes);


                const int global_candidate_index = my_indices_of_corrected_candidates[candidateIndex];
                //std::cerr << global_candidate_index << "\n";
                const read_number candidate_read_id = rawResults.h_candidate_read_ids[global_candidate_index];

                const int candidate_shift = rawResults.h_alignment_shifts[global_candidate_index];
                
                //TIMERSTOPCPU(setup);
                if(batch.correctionOptions.new_columns_to_correct < candidate_shift){
                    std::cerr << "readid " << subjectReadId << " candidate readid " << candidate_read_id << " : "
                    << candidate_shift << " " << batch.correctionOptions.new_columns_to_correct <<"\n";
                }
                assert(batch.correctionOptions.new_columns_to_correct >= candidate_shift);
                
                //TIMERSTARTCPU(tmp);
                tmp.type = TempCorrectedSequence::Type::Candidate;
                tmp.shift = candidate_shift;
                tmp.readId = candidate_read_id;
                //TIMERSTOPCPU(tmp);
                //const bool originalReadContainsN = batch.readStorage->readContainsN(candidate_read_id);
                
                
                const int numEdits = rawResults.h_numEditsPerCorrectedCandidate[offsetForCorrectedCandidateData + candidateIndex];

                if(numEdits != doNotUseEditsValue){
                    tmp.edits.resize(numEdits);
                    const TempCorrectedSequence::EncodedEdit* gpuedits 
                        = (const TempCorrectedSequence::EncodedEdit*)(((const char*)my_editsPerCorrectedCandidate) 
                            + candidateIndex * rawResults.editsPitchInBytes);
                    std::copy_n(gpuedits, numEdits, tmp.edits.begin());
                    tmp.useEdits = true;
                }else{
                    const int candidate_length = rawResults.h_candidate_sequences_lengths[global_candidate_index];
                    const char* const candidate_data = my_corrected_candidates_data + candidateIndex * rawResults.decodedSequencePitchInBytes;
                    //tmp.sequence = std::string{candidate_data, candidate_data + candidate_length};
                    tmp.sequence.assign(candidate_data, candidate_length);
                    tmp.edits.clear();
                    tmp.useEdits = false;
                }

                //std::cerr << "subject_index " << subject_index << ", candidateIndex " << candidateIndex << ", result : " << tmp << "\n";

                //TIMERSTARTCPU(encode);
                //tmpencoded = tmp.encode();
                tmp.encodeInto(tmpencoded); 
                //TIMERSTOPCPU(encode);

            }

            nvtx::pop_range();
        };


        if(!batch.correctionOptions.correctCandidates){
            batch.threadPool->parallelFor(batch.pforHandle, 0, numCorrectedAnchors, [=](auto begin, auto end, auto /*threadId*/){
                unpackAnchors(begin, end);
            });
        }else{

#if 0            
            unpackAnchors(0, numCorrectedAnchors);
#else            
            nvtx::push_range("parallel anchor unpacking",1);

            batch.threadPool->parallelFor(batch.pforHandle, 0, numCorrectedAnchors, [=](auto begin, auto end, auto /*threadId*/){
                unpackAnchors(begin, end);
            });

            nvtx::pop_range();
#endif 


#if 0
            unpackcandidates(0, numCorrectedCandidates);
#else            
            nvtx::push_range("parallel candidate unpacking", 3);

            batch.threadPool->parallelFor(
                batch.pforHandle, 
                0, 
                numCorrectedCandidates, 
                [=](auto begin, auto end, auto /*threadId*/){
                    unpackcandidates(begin, end);
                },
                batch.threadPool->getConcurrency() * 4
            );

            nvtx::pop_range();
#endif            
        }

    }

 
    void saveResults(Batch& batch){
            
        auto function = [batchPtr = &batch,
            id = batch.id](){

            auto& batch = *batchPtr;
            auto& outputData = batch.waitableOutputData.data;

            const int numA = outputData.getNumAnchors();
            const int numC = outputData.getNumCandidates();

            nvtx::push_range("batch "+std::to_string(id)+" writeresultoutputhread"
                + std::to_string(numA) + " " + std::to_string(numC), 4);

#if 0                
            for(int i = 0; i < numA; i++){
                batch.saveCorrectedSequence(
                    std::move(outputData.anchorCorrections[i]), 
                    std::move(outputData.encodedAnchorCorrections[i])
                );
            }

            for(int i = 0; i < numC; i++){
                batch.saveCorrectedSequence(
                    std::move(outputData.candidateCorrections[i]), 
                    std::move(outputData.encodedCandidateCorrections[i])
                );
            }
#else 

            for(int i = 0; i < numA; i++){
                batch.saveCorrectedSequence(
                    &outputData.anchorCorrections[i], 
                    &outputData.encodedAnchorCorrections[i]
                );
            }

            for(int i = 0; i < numC; i++){
                batch.saveCorrectedSequence(
                    &outputData.candidateCorrections[i], 
                    &outputData.encodedCandidateCorrections[i]
                );
            }

#endif

            batch.waitableOutputData.signal();
            //std::cerr << "batch " << batch.id << " batch.waitableOutputData.signal() finished\n";

            nvtx::pop_range();
        };

		//function();

        nvtx::push_range("enqueue to outputthread", 2);
        batch.outputThread->enqueue(std::move(function));
        nvtx::pop_range();
	}




MemoryFileFixedSize<EncodedTempCorrectedSequence> 
correct_gpu(
        const GoodAlignmentProperties& goodAlignmentProperties,
        const CorrectionOptions& correctionOptions,
        const RuntimeOptions& runtimeOptions,
        const FileOptions& fileOptions,
        const MemoryOptions& memoryOptions,
        const SequenceFileProperties& sequenceFileProperties,
        Minhasher_t& minhasher,
        DistributedReadStorage& readStorage){

    assert(runtimeOptions.canUseGpu);
    //assert(runtimeOptions.max_candidates > 0);
    assert(runtimeOptions.deviceIds.size() > 0);

    const auto& deviceIds = runtimeOptions.deviceIds;

    const auto rsMemInfo = readStorage.getMemoryInfo();
    const auto mhMemInfo = minhasher.getMemoryInfo();

    std::size_t memoryAvailableBytesHost = memoryOptions.memoryTotalLimit;

    if(memoryAvailableBytesHost > rsMemInfo.host){
        memoryAvailableBytesHost -= rsMemInfo.host;
    }else{
        memoryAvailableBytesHost = 0;
    }

    if(memoryAvailableBytesHost > mhMemInfo.host){
        memoryAvailableBytesHost -= mhMemInfo.host;
    }else{
        memoryAvailableBytesHost = 0;
    }

    std::unique_ptr<std::atomic_uint8_t[]> correctionStatusFlagsPerRead = std::make_unique<std::atomic_uint8_t[]>(sequenceFileProperties.nReads);

    #pragma omp parallel for
    for(read_number i = 0; i < sequenceFileProperties.nReads; i++){
        correctionStatusFlagsPerRead[i] = 0;
    }

    std::cerr << "Status flags per reads require " << sizeof(std::atomic_uint8_t) * sequenceFileProperties.nReads / 1024. / 1024. << " MB\n";

    if(memoryAvailableBytesHost > sizeof(std::atomic_uint8_t) * sequenceFileProperties.nReads){
        memoryAvailableBytesHost -= sizeof(std::atomic_uint8_t) * sequenceFileProperties.nReads;
    }else{
        memoryAvailableBytesHost = 0;
    }

    const std::size_t availableMemoryInBytes = memoryAvailableBytesHost; //getAvailableMemoryInKB() * 1024;
    std::size_t memoryForPartialResultsInBytes = 0;

    if(availableMemoryInBytes > 2*(std::size_t(1) << 30)){
        memoryForPartialResultsInBytes = availableMemoryInBytes - 2*(std::size_t(1) << 30);
    }

    std::cerr << "Partial results may occupy " << (memoryForPartialResultsInBytes /1024. / 1024. / 1024.) 
        << " GB in memory. Remaining partial results will be stored in temp directory. \n";

    const std::string tmpfilename{fileOptions.tempdirectory + "/" + "MemoryFileFixedSizetmp"};
    MemoryFileFixedSize<EncodedTempCorrectedSequence> partialResults(memoryForPartialResultsInBytes, tmpfilename);

      //std::mutex outputstreamlock;

      BackgroundThread outputThread;

      const int threadPoolSize = std::max(1, runtimeOptions.threads - 3*int(deviceIds.size()));
      std::cerr << "threadpool size for correction = " << threadPoolSize << "\n";
      ThreadPool threadPool(threadPoolSize);

#ifndef DO_PROFILE
        cpu::RangeGenerator<read_number> readIdGenerator(sequenceFileProperties.nReads);
        //cpu::RangeGenerator<read_number> readIdGenerator(1000);
#else
        cpu::RangeGenerator<read_number> readIdGenerator(num_reads_to_profile);
#endif

        std::map<int,int> numCandidatesLimitPerGpu;

        for(int deviceId : deviceIds){
            cudaDeviceProp deviceProperties;
            cudaGetDeviceProperties(&deviceProperties, deviceId); CUERR;
            const int limitByThreads = deviceProperties.multiProcessorCount * deviceProperties.maxThreadsPerMultiProcessor;
            const int resultsPerMap = calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage);
            //maximum number of candidates of 1 read. limit must be at least this number to ensure progress
            const int minCandidates = resultsPerMap * minhasher.getNumberOfMaps();

            numCandidatesLimitPerGpu[deviceId] = std::max(minCandidates, limitByThreads);
            std::cerr << "Number of candidates per batch is limited to " << numCandidatesLimitPerGpu[deviceId] 
                << " for device id " << deviceId << "\n";
        }

      //std::mutex outputstreammutex;
    //   std::map<bool, int> useEditsCountMap;
    //   std::map<bool, int> useEditsSavedCountMap;
    //   std::map<int, int> numEditsHistogram;

     outputThread.start();

        std::vector<std::thread> batchExecutors;

      #ifdef DO_PROFILE
          cudaProfilerStart();
      #endif

        auto initBatchData = [&](auto& batchData, int deviceId){

            cudaSetDevice(deviceId); CUERR;

            std::array<cudaStream_t, nStreamsPerBatch> streams;
            for(int j = 0; j < nStreamsPerBatch; ++j) {
                cudaStreamCreate(&streams[j]); CUERR;
            }

            std::array<cudaEvent_t, nEventsPerBatch> events;
            for(int j = 0; j < nEventsPerBatch; ++j) {
                cudaEventCreateWithFlags(&events[j], cudaEventDisableTiming); CUERR;
            }

            batchData.id = -1;
            batchData.deviceId = deviceId;
            batchData.streams = std::move(streams);
            batchData.events = std::move(events);

            batchData.goodAlignmentProperties = goodAlignmentProperties;
            batchData.correctionOptions = correctionOptions;
            batchData.runtimeOptions = runtimeOptions;
            batchData.fileOptions = fileOptions;
            batchData.sequenceFileProperties = sequenceFileProperties;
            batchData.readIdGenerator = &readIdGenerator;
            batchData.minhasher = &minhasher;
            batchData.readStorage = &readStorage;
            batchData.correctionStatusFlagsPerRead = correctionStatusFlagsPerRead.get();
            batchData.saveCorrectedSequence 
                    = [&](const TempCorrectedSequence* tmp, const EncodedTempCorrectedSequence* encoded){
                //useEditsCountMap[tmp.useEdits]++;
                  //std::cerr << tmp << "\n";
                //std::unique_lock<std::mutex> l(outputstreammutex);
                if(!(tmp->hq && tmp->useEdits && tmp->edits.empty())){
                    //outputstream << tmp << '\n';
                    partialResults.storeElement(encoded);
                    //useEditsSavedCountMap[tmp.useEdits]++;
                    //numEditsHistogram[tmp.edits.size()]++;
      
                   // std::cerr << tmp.edits.size() << " " << encoded.data.capacity() << "\n";
                }
            };

            batchData.kernelLaunchHandle = make_kernel_launch_handle(deviceId);
            batchData.subjectSequenceGatherHandle = readStorage.makeGatherHandleSequences();
            batchData.candidateSequenceGatherHandle = readStorage.makeGatherHandleSequences();
            batchData.subjectQualitiesGatherHandle = readStorage.makeGatherHandleQualities();
            batchData.candidateQualitiesGatherHandle = readStorage.makeGatherHandleQualities();

            batchData.outputThread = &outputThread;
            batchData.backgroundWorker = nullptr;
            batchData.unpackWorker = nullptr;
            batchData.threadPool = &threadPool;
            batchData.threadsInThreadPool = threadPoolSize;
            batchData.encodedSequencePitchInInts = getEncodedNumInts2Bit(sequenceFileProperties.maxSequenceLength);
            batchData.decodedSequencePitchInBytes = SDIV(sequenceFileProperties.maxSequenceLength, 4) * 4;
            batchData.qualityPitchInBytes = SDIV(sequenceFileProperties.maxSequenceLength, 32) * 32;
            batchData.maxNumEditsPerSequence = std::max(1,sequenceFileProperties.maxSequenceLength / 7);

            //pad to multiple of 128 bytes
            batchData.editsPitchInBytes = SDIV(batchData.maxNumEditsPerSequence * sizeof(TempCorrectedSequence::EncodedEdit), 128) * 128;

            batchData.min_overlap = std::max(
                1, 
                std::max(
                    goodAlignmentProperties.min_overlap, 
                    int(sequenceFileProperties.maxSequenceLength * goodAlignmentProperties.min_overlap_ratio)
                )
            );
    
            batchData.msa_max_column_count = (3*sequenceFileProperties.maxSequenceLength - 2*batchData.min_overlap);
            //round up to 32 elements
            batchData.msaColumnPitchInElements = SDIV(batchData.msa_max_column_count, 32) * 32;
            
            batchData.numCandidatesLimit = numCandidatesLimitPerGpu[deviceId];

            const int resultsPerMap = calculateResultsPerMapThreshold(correctionOptions.estimatedCoverage);
            const int numMinhashMaps = minhasher.getNumberOfMaps();
            const int maxNumIds = resultsPerMap * numMinhashMaps * correctionOptions.batchsize;


            batchData.nextIterationData.init( 
                minhasher,
                &readStorage,
                &threadPool,
                &readIdGenerator,
                batchData.deviceId,
                correctionOptions.batchsize,
                batchData.numCandidatesLimit,
                batchData.encodedSequencePitchInInts,
                runtimeOptions.threads,
                numMinhashMaps,
                resultsPerMap
            );

            batchData.waitableOutputData.data.rawResults.init(
                correctionOptions.batchsize, 
                numCandidatesLimitPerGpu[deviceId], 
                maxNumIds,
                batchData.decodedSequencePitchInBytes, 
                batchData.maxNumEditsPerSequence,
                batchData.editsPitchInBytes
            );

            batchData.resize(
                correctionOptions.batchsize,
                correctionOptions,
                goodAlignmentProperties,
                sequenceFileProperties,
                minhasher.getNumberOfMaps()
            );

            #if 0
                MemoryUsage infobatch = batchData.getMemoryInfo();
                std::cerr << "Batch memory usage:\n";
                std::cerr << "host: " << infobatch.host << "\n";
                for(auto pair : infobatch.device){
                    std::cerr << "device id " << pair.first << ": " << pair.second << "\n";
                }

                MemoryUsage infonextiterdata = batchData.nextIterationData.getMemoryInfo();
                std::cerr << "nextiterationdata memory usage:\n";
                std::cerr << "host: " << infonextiterdata.host << "\n";
                for(auto pair : infonextiterdata.device){
                    std::cerr << "device id " << pair.first << ": " << pair.second << "\n";
                }

                MemoryUsage infooutputdata = batchData.waitableOutputData.data.rawResults.getMemoryInfo();
                std::cerr << "outputdata memory usage:\n";
                std::cerr << "host: " << infooutputdata.host << "\n";
                for(auto pair : infooutputdata.device){
                    std::cerr << "device id " << pair.first << ": " << pair.second << "\n";
                }
            #endif            
        };

        auto destroyBatchData = [&](auto& batchData){
            
            cudaSetDevice(batchData.deviceId); CUERR;
    
            batchData.nextIterationData.destroy();
            batchData.waitableOutputData.data.rawResults.destroy();
            batchData.destroy();

            // for(std::size_t i = 0; i < batchData.executionGraphs.size(); i++){
            //     if(batchData.executionGraphs[i].execgraph != nullptr){
            //         cudaGraphExecDestroy(batchData.executionGraphs[i].execgraph); CUERR;
            //         batchData.executionGraphs[i].execgraph = nullptr;
            //     }
            // }
    
            for(auto& stream : batchData.streams) {
                cudaStreamDestroy(stream); CUERR;
            }
    
            for(auto& event : batchData.events){
                cudaEventDestroy(event); CUERR;
            }            
        };

        auto showProgress = [&](std::int64_t totalCount, int seconds){
            if(runtimeOptions.showProgress){

                int hours = seconds / 3600;
                seconds = seconds % 3600;
                int minutes = seconds / 60;
                seconds = seconds % 60;
                
                printf("Processed %10lu of %10lu reads (Runtime: %03d:%02d:%02d)\r",
                totalCount, sequenceFileProperties.nReads,
                hours, minutes, seconds);
            }
        };

        auto updateShowProgressInterval = [](auto duration){
            return duration;
        };

        ProgressThread<std::int64_t> progressThread(sequenceFileProperties.nReads, showProgress, updateShowProgressInterval);


        auto processBatchUntilCandidateCorrectionStarted = [&](auto& batchData){
            //auto& streams = batchData.streams;
            //auto& events = batchData.events;

            auto pushrange = [&](const std::string& msg, int color){
                nvtx::push_range("batch "+std::to_string(batchData.id)+msg, color);
                //std::cerr << "batch "+std::to_string(batchData.id) << msg << "\n";
            };

            auto poprange = [&](){
                nvtx::pop_range();
                //cudaDeviceSynchronize(); CUERR;
            };
                
            pushrange("getNextBatchForCorrection", 0);
            
            getNextBatchForCorrection(batchData);

            poprange();

            if(batchData.h_numCandidates[0] == 0){
                return;
            }

            pushrange("getCandidateSequenceData", 1);

            getCandidateSequenceData(batchData, *batchData.readStorage);

            poprange();

            if(batchData.correctionOptions.useQualityScores) {
                pushrange("getQualities", 4);

                getQualities(batchData);

                poprange();
            }

#ifdef USE_CUDA_GRAPH
            buildGraphViaCapture(batchData);
            executeGraph(batchData);
#else            
            pushrange("getCandidateAlignments", 2);

            getCandidateAlignments(batchData);

            poprange();
            

            pushrange("buildMultipleSequenceAlignment", 5);

            buildMultipleSequenceAlignment(batchData);

            poprange();

        #ifdef USE_MSA_MINIMIZATION

            pushrange("removeCandidatesOfDifferentRegionFromMSA", 6);

            removeCandidatesOfDifferentRegionFromMSA(batchData);

            poprange();

        #endif


            pushrange("correctSubjects", 7);

            correctSubjects(batchData);

            poprange();

            if(batchData.correctionOptions.correctCandidates) {                        

                pushrange("correctCandidates", 8);

                correctCandidates(batchData);

                poprange();
                
            }
#endif            
        };

        auto processBatchResults = [&](auto& batchData){
            auto& streams = batchData.streams;
            
            #ifndef USE_CUDA_GRAPH
            auto& events = batchData.events;
            cudaEventRecord(events[0], streams[secondary_stream_index]); CUERR;
            cudaStreamWaitEvent(streams[primary_stream_index], events[0], 0); CUERR;   
            #endif

            cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
            // std::cerr << "batch " << batchData.id << " waitableOutputData.wait()\n";
            // batchData.waitableOutputData.wait();
            // std::cerr << "batch " << batchData.id << " waitableOutputData.wait() finished\n";

            batchData.waitableOutputData.wait();

            assert(!batchData.waitableOutputData.isBusy());

            //std::cerr << "batch " << batchData.id << " waitableOutputData.setBusy()\n";
            batchData.moveResultsToOutputData(batchData.waitableOutputData.data);

            batchData.waitableOutputData.setBusy();

            auto func = [batchDataPtr = &batchData](){
                //std::cerr << "batch " << batchDataPtr->id << " Backgroundworker func begin\n";
                auto& batchData = *batchDataPtr;
                auto pushrange = [&](const std::string& msg, int color){
                    nvtx::push_range("batch "+std::to_string(batchData.id)+msg, color);
                };
    
                auto poprange = [&](){
                    nvtx::pop_range();
                };

                pushrange("unpackClassicResults", 9);
    
                constructResults(batchData);
    
                poprange();
    
    
                pushrange("saveResults", 10);
    
                saveResults(batchData);
    
                poprange();

                //std::cerr << "batch " << batchDataPtr->id << " Backgroundworker func end\n";
    
                //batchData.hasUnprocessedResults = false;
            };

            func();
            //batchData.backgroundWorker->enqueue(func);
            //batchData.unpackWorker->enqueue(func);            
        };


        for(int deviceIdIndex = 0; deviceIdIndex < int(deviceIds.size()); ++deviceIdIndex) {
            constexpr int max_num_batches = 2;

            batchExecutors.emplace_back([&, deviceIdIndex](){
                const int deviceId = deviceIds[deviceIdIndex];

                std::array<BackgroundThread, max_num_batches> backgroundWorkerArray;
                std::array<BackgroundThread, max_num_batches> unpackWorkerArray;

                std::array<Batch, max_num_batches> batchDataArray;

                for(int i = 0; i < max_num_batches; i++){
                    initBatchData(batchDataArray[i], deviceId);
                    // if(i < int(deviceIds.size())){
                    //     initBatchData(batchDataArray[i], deviceIds[i]);
                    // }else{
                    //     initBatchData(batchDataArray[i], 0);
                    // }
                    batchDataArray[i].id = deviceIdIndex * max_num_batches + i;
                    batchDataArray[i].backgroundWorker = &backgroundWorkerArray[i];
                    batchDataArray[i].unpackWorker = &unpackWorkerArray[0]; //shared by all batches

                    backgroundWorkerArray[i].start();
                    unpackWorkerArray[i].start();
                }


// 1 batch
#if 0
                static_assert(1 <= max_num_batches, "");

                bool isFirstIteration = true;
                int batchIndex = 0;

                while(!(readIdGenerator.empty() 
                        && !batchDataArray[0].nextIterationData.syncFlag.isBusy()
                        && batchDataArray[0].nextIterationData.h_numAnchors[0] == 0
                        && !batchDataArray[0].waitableOutputData.isBusy())) {

                    auto& batchData = batchDataArray[batchIndex];

                    processBatchUntilCandidateCorrectionStarted(batchData);

                    if(batchData.h_numCandidates[0] == 0){
                        batchData.waitableOutputData.signal();
                        progressThread.addProgress(batchData.h_numAnchors[0]);
                        batchData.reset();
                        continue;
                    }

                    processBatchResults(batchData);

                    progressThread.addProgress(batchData.h_numAnchors[0]);
                    batchData.reset();                   
                }
#endif 

// 2 batches
#if 1
                static_assert(2 <= max_num_batches, "");

                bool isFirstIteration = true;
                int batchIndex = 0;

                while(!(readIdGenerator.empty() 
                        && !batchDataArray[0].nextIterationData.syncFlag.isBusy()
                        && batchDataArray[0].nextIterationData.h_numAnchors[0] == 0
                        && !batchDataArray[0].waitableOutputData.isBusy()
                        && !batchDataArray[1].nextIterationData.syncFlag.isBusy()
                        && batchDataArray[1].nextIterationData.h_numAnchors[0] == 0
                        && !batchDataArray[1].waitableOutputData.isBusy())) {

                    const int nextBatchIndex = 1 - batchIndex;
                    auto& currentBatchData = batchDataArray[batchIndex];
                    auto& nextBatchData = batchDataArray[nextBatchIndex];

                    if(isFirstIteration){
                        isFirstIteration = false;
                        processBatchUntilCandidateCorrectionStarted(currentBatchData);
                    }else{
                        processBatchUntilCandidateCorrectionStarted(nextBatchData);

                        if(currentBatchData.h_numCandidates[0] == 0){
                            currentBatchData.waitableOutputData.signal();
                            progressThread.addProgress(currentBatchData.h_numAnchors[0]);
                            currentBatchData.reset();
                            batchIndex = 1-batchIndex;
                            continue;
                        }

                        processBatchResults(currentBatchData);
    
                        progressThread.addProgress(currentBatchData.h_numAnchors[0]);
                        currentBatchData.reset();

                        batchIndex = 1-batchIndex;
                    }                
                }
#endif


// 3 batches
#if 0

                static_assert(3 <= max_num_batches, "");

                bool isFirstIteration = true;
                bool isSecondIteration = false;

                int batchIndex = 0;

                while(!(readIdGenerator.empty() 
                        && !batchDataArray[0].nextIterationData.syncFlag.isBusy()
                        && batchDataArray[0].nextIterationData.h_numAnchors[0] == 0
                        && !batchDataArray[0].waitableOutputData.isBusy()
                        && !batchDataArray[1].nextIterationData.syncFlag.isBusy()
                        && batchDataArray[1].nextIterationData.h_numAnchors[0] == 0
                        && !batchDataArray[1].waitableOutputData.isBusy()
                        && !batchDataArray[2].nextIterationData.syncFlag.isBusy()
                        && batchDataArray[2].nextIterationData.h_numAnchors[0] == 0
                        && !batchDataArray[2].waitableOutputData.isBusy())) {

                    const int nextBatchIndex = batchIndex == 2 ? 0 : 1 + batchIndex;
                    const int lastBatchIndex = nextBatchIndex == 2 ? 0 : 1 + nextBatchIndex;

                    auto& currentBatchData = batchDataArray[batchIndex];
                    auto& nextBatchData = batchDataArray[nextBatchIndex];
                    auto& lastBatchData = batchDataArray[lastBatchIndex];

                    if(isFirstIteration){
                        isFirstIteration = false;
                        processBatchUntilCandidateCorrectionStarted(currentBatchData);
                        processBatchUntilCandidateCorrectionStarted(nextBatchData);
                    }else{
                        processBatchUntilCandidateCorrectionStarted(lastBatchData);

                        if(currentBatchData.h_numCandidates[0] == 0){
                            currentBatchData.waitableOutputData.signal();
                            progressThread.addProgress(currentBatchData.h_numAnchors[0]);
                            currentBatchData.reset();
                            batchIndex = batchIndex == 2 ? 0 : 1 + batchIndex;
                            continue;
                        }

                        processBatchResults(currentBatchData);
    
                        progressThread.addProgress(currentBatchData.h_numAnchors[0]);
                        currentBatchData.reset();

                        batchIndex = batchIndex == 2 ? 0 : 1 + batchIndex;
                    }              
                }
#endif

                for(int i = 0; i < max_num_batches; i++){
                    backgroundWorkerArray[i].stopThread(BackgroundThread::StopType::FinishAndStop);
                    unpackWorkerArray[i].stopThread(BackgroundThread::StopType::FinishAndStop);
                }
                
                for(int i = 0; i < max_num_batches; i++){
                    //batchDataArray[i].backgroundWorker->stopThread(BackgroundThread::StopType::FinishAndStop);
                    //batchDataArray[i].unpackWorker->stopThread(BackgroundThread::StopType::FinishAndStop);
                    
#if 0                    
                    auto memInfo1 = readStorage.getMemoryInfoOfGatherHandleSequences(batchDataArray[i].subjectSequenceGatherHandle);
                    auto memInfo2 = readStorage.getMemoryInfoOfGatherHandleSequences(batchDataArray[i].candidateSequenceGatherHandle);
                    auto memInfo3 = readStorage.getMemoryInfoOfGatherHandleQualities(batchDataArray[i].subjectQualitiesGatherHandle);
                    auto memInfo4 = readStorage.getMemoryInfoOfGatherHandleQualities(batchDataArray[i].candidateQualitiesGatherHandle);

                    auto printMemoryInfo = [](const auto& meminfo){
                        std::cerr << "Host: " << meminfo.host << "\n";
                        for(const auto& pair : meminfo.device){
                            std::cerr << "Device " << pair.first << ": " << pair.second << "\n";
                        }
                    };

                    std::cerr << "Memory usage subjectSequenceGatherHandle\n";
                    printMemoryInfo(memInfo1);
                    std::cerr << "Memory usage candidateSequenceGatherHandle\n";
                    printMemoryInfo(memInfo2);
                    if(correctionOptions.useQualityScores){
                        std::cerr << "Memory usage subjectQualitiesGatherHandle\n";
                        printMemoryInfo(memInfo3);
                        std::cerr << "Memory usage candidateQualitiesGatherHandle\n";
                        printMemoryInfo(memInfo4);
                    }
#endif
                    destroyBatchData(batchDataArray[i]);
                }
            });
        }

        for(auto& executor : batchExecutors){
            executor.join();
        }

        progressThread.finished(); 
        
        std::cout << std::endl;

        threadPool.wait();
        outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

        assert(threadPool.empty());

      //outputstream.flush();
      partialResults.flush();

      #ifdef DO_PROFILE
          cudaProfilerStop();
      #endif



    //   for(const auto& batch : batches){
    //       std::cout << "size elements: " << batch.h_candidate_read_ids.size() << ", capacity elements " << batch.h_candidate_read_ids.capacity() << std::endl;
      
    //     }

    //     for(const auto& batch : batches){
    //         std::cerr << "Memory usage: \n";
    //         batch.printMemoryUsage();
    //         std::cerr << "Total: " << batch.getMemoryUsageInBytes() << " bytes\n";
    //         std::cerr << '\n';
    //     }


    correctionStatusFlagsPerRead.reset();

    //size_t occupiedMemory = minhasher.numBytes() + cpuReadStorage.size();

    // std::cerr << "useEditsCountMap\n";
    // for(const auto& pair : useEditsCountMap){
    //     std::cerr << int(pair.first) << " : " << pair.second << "\n";
    // }

    // std::cerr << "useEditsSavedCountMap\n";
    // for(const auto& pair : useEditsSavedCountMap){
    //     std::cerr << int(pair.first) << " : " << pair.second << "\n";
    // }

    // std::cerr << "numEditsHistogram\n";
    // for(const auto& pair : numEditsHistogram){
    //     std::cerr << int(pair.first) << " : " << pair.second << "\n";
    // }

    return partialResults;

}







}
}

