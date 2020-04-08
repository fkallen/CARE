

#include <gpu/correct_gpu.hpp>
#include <gpu/distributedreadstorage.hpp>
#include <gpu/nvtxtimelinemarkers.hpp>
#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <gpu/dataarrays.hpp>
#include <gpu/cubcachingallocator.cuh>
#include <gpu/minhashkernels.hpp>

#include <correctionresultprocessing.hpp>

#include <config.hpp>
#include <qualityscoreweights.hpp>
#include <sequence.hpp>
#include <featureextractor.hpp>
#include <forestclassifier.hpp>
//#include <nn_classifier.hpp>
#include <minhasher.hpp>
#include <options.hpp>
#include <candidatedistribution.hpp>
//#include <sequencefileio.hpp>
#include <rangegenerator.hpp>
#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>
#include <filehelpers.hpp>

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

#include <thrust/binary_search.h>

//#define CARE_GPU_DEBUG
//#define CARE_GPU_DEBUG_MEMCOPY
//#define CARE_GPU_DEBUG_PRINT_ARRAYS
//#define CARE_GPU_DEBUG_PRINT_MSA

#define MSA_IMPLICIT

//#define REARRANGE_INDICES
#define USE_MSA_MINIMIZATION

constexpr int max_num_minimizations = 5;

//#define DO_PROFILE

#ifdef DO_PROFILE
    constexpr size_t num_reads_to_profile = 100000;
#endif


#define USE_CUDA_GRAPH


#define LIMIT_MAX_CANDIDATES
constexpr int batchsizeCandidates = 200000;

namespace care{
namespace gpu{


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



    //constexpr std::uint8_t maxSavedCorrectedCandidatesPerRead = 5;

    //read status bitmask
    constexpr std::uint8_t readCorrectedAsHQAnchor = 1;
    constexpr std::uint8_t readCouldNotBeCorrectedAsAnchor = 2;

    constexpr int primary_stream_index = 0;
    constexpr int secondary_stream_index = 1;
    constexpr int nStreamsPerBatch = 2;

    constexpr int alignments_finished_event_index = 0;
    constexpr int quality_transfer_finished_event_index = 1;
    constexpr int indices_transfer_finished_event_index = 2;
    constexpr int correction_finished_event_index = 3;
    constexpr int result_transfer_finished_event_index = 4;
    constexpr int msadata_transfer_finished_event_index = 5;
    constexpr int alignment_data_transfer_h2d_finished_event_index = 6;
    constexpr int msa_build_finished_event_index = 7;
    constexpr int indices_calculated_event_index = 8;
    constexpr int num_indices_transfered_event_index = 9;
    constexpr int secondary_stream_finished_event_index = 10;
    constexpr int highqualityindices_event_index = 11;
    constexpr int numTotalCorrectedCandidates_event_index = 12;
    constexpr int nEventsPerBatch = 13;

    constexpr int doNotUseEditsValue = -1;

    struct TransitionFunctionData;

    struct SyncFlag{
        std::atomic<bool> busy{false};
        std::mutex m;
        std::condition_variable cv;

        void setBusy(){
            assert(busy == false);
            busy = true;
        }

        bool isBusy() const{
            return busy;
        }

        void wait(){
            if(isBusy()){
                std::unique_lock<std::mutex> l(m);
                while(isBusy()){
                    cv.wait(l);
                }
            }
        }

        void signal(){
            std::unique_lock<std::mutex> l(m);
            busy = false;
            cv.notify_all();
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
        PinnedBuffer<int> h_candidates_per_subject;

        DeviceBuffer<unsigned int> d_subject_sequences_data;
        DeviceBuffer<int> d_subject_sequences_lengths;
        DeviceBuffer<read_number> d_subject_read_ids;
        DeviceBuffer<read_number> d_candidate_read_ids;
        DeviceBuffer<int> d_candidates_per_subject;
        DeviceBuffer<int> d_candidates_per_subject_tmp;
        DeviceBuffer<int> d_candidates_per_subject_prefixsum;

        DeviceBuffer<read_number> d_candidate_read_ids_tmp;
        PinnedBuffer<std::uint64_t> h_minhashSignatures;
        DeviceBuffer<std::uint64_t> d_minhashSignatures;
        PinnedBuffer<int> h_numAnchors;
        PinnedBuffer<int> h_numCandidates;
        DeviceBuffer<int> d_numAnchors;
        DeviceBuffer<int> d_numCandidates;

        //private buffers
        DeviceBuffer<int> d_numLeftoverAnchors;
        DeviceBuffer<int> d_leftoverAnchorLengths;
        DeviceBuffer<read_number> d_leftoverAnchorReadIds;
        DeviceBuffer<int> d_numLeftoverCandidates;
        DeviceBuffer<read_number> d_leftoverCandidateReadIds;
        DeviceBuffer<int> d_leftoverCandidatesPerAnchors;
        DeviceBuffer<unsigned int> d_leftoverAnchorSequences;
        PinnedBuffer<read_number> h_leftoverAnchorReadIds;
        PinnedBuffer<int> h_numLeftoverAnchors;
        PinnedBuffer<int> h_numLeftoverCandidates;

        bool reallocOccurred = false;

        int n_subjects = -1;
        int n_new_subjects = -1;
        std::atomic<int> n_queries{-1};

        std::vector<Minhasher::Range_t> allRanges;
        std::vector<int> idsPerChunk;   
        std::vector<int> numAnchorsPerChunk;
        std::vector<int> idsPerChunkPrefixSum;
        std::vector<int> numAnchorsPerChunkPrefixSum;

        cudaStream_t stream;
        cudaEvent_t event;
        int deviceId;

        ThreadPool::ParallelForHandle pforHandle;

        MergeRangesGpuHandle<read_number> mergeRangesGpuHandle;

        SyncFlag syncFlag;
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

        PinnedBuffer<TempCorrectedSequence::Edit> h_editsPerCorrectedSubject;
        PinnedBuffer<int> h_numEditsPerCorrectedSubject;

        PinnedBuffer<TempCorrectedSequence::Edit> h_editsPerCorrectedCandidate;
        PinnedBuffer<int> h_numEditsPerCorrectedCandidate;

    };

    struct OutputData{
        std::vector<TempCorrectedSequence> anchorCorrections;
        std::vector<EncodedTempCorrectedSequence> encodedAnchorCorrections;
        std::vector<TempCorrectedSequence> candidateCorrections;
        std::vector<EncodedTempCorrectedSequence> encodedCandidateCorrections;

        std::vector<int> subjectIndicesToProcess;
        std::vector<std::pair<int,int>> candidateIndicesToProcess;

        UnprocessedCorrectionResults rawResults;
    };


    template<class T>
    struct WaitableData{
        T data;
        SyncFlag syncFlag;

        void setBusy(){
            syncFlag.setBusy();
        }

        bool isBusy() const{
            return syncFlag.isBusy();
        }

        void wait(){
            syncFlag.wait();
        }

        void signal(){
            syncFlag.signal();
        } 
    };

    struct CudaGraph{
        bool valid = false;
        int numAnchors = 0;
        int numCandidates = 0;
        cudaGraphExec_t execgraph = nullptr;
        int* d_anchorIndicesOfCandidates;
        int* d_numAnchors;
        int* d_candidates_per_subject;
        int* d_candidates_per_subject_prefixsum;
    };


    struct Batch {
        Batch() = default;
        Batch(const Batch&) = delete;
        Batch(Batch&&) = default;
        Batch& operator=(const Batch&) = delete;
        Batch& operator=(Batch&&) = default;

        NextIterationData nextIterationData;
        bool isFirstIteration = true;

        WaitableData<OutputData> waitableOutputData;

        bool combinedStreams = false;

        DataArrays dataArrays;
        bool hasUnprocessedResults = false;

		std::array<cudaStream_t, nStreamsPerBatch> streams;
		std::array<cudaEvent_t, nEventsPerBatch> events;

        TransitionFunctionData* transFuncData;
        BackgroundThread* outputThread;
        BackgroundThread* backgroundWorker;
        BackgroundThread* unpackWorker;

        ThreadPool* threadPool;
        int threadsInThreadPool = 1;

        ThreadPool::ParallelForHandle pforHandle;
        std::vector<Minhasher::Handle> minhashHandles;

        int id = -1;
        int deviceId = 0;

		KernelLaunchHandle kernelLaunchHandle;

        DistributedReadStorage::GatherHandleSequences subjectSequenceGatherHandle;
        DistributedReadStorage::GatherHandleSequences candidateSequenceGatherHandle;
        DistributedReadStorage::GatherHandleQualities subjectQualitiesGatherHandle;
        DistributedReadStorage::GatherHandleQualities candidateQualitiesGatherHandle;

        bool reallocResize = false;
        bool reallocInNextIterationData = false;

        int numCandidatesLimit = 0;

        int encodedSequencePitchInInts;
        int decodedSequencePitchInBytes;
        int qualityPitchInBytes;

        int maxNumEditsPerSequence;

        int msa_weights_pitch;
        int msa_pitch;

        int n_subjects;
        int n_queries;

        int graphindex = 0;
        std::array<CudaGraph,2> alignmentGraphs;

		void reset(){
            combinedStreams = false;
            n_subjects = 0;
            n_queries = 0;
            hasUnprocessedResults = false;
            reallocResize = false;
            reallocInNextIterationData = false;
        }

        void updateFromIterationData(NextIterationData& data){
            std::swap(dataArrays.h_subject_sequences_data, data.h_subject_sequences_data);
            std::swap(dataArrays.h_subject_sequences_lengths, data.h_subject_sequences_lengths);
            std::swap(dataArrays.h_subject_read_ids, data.h_subject_read_ids);
            std::swap(dataArrays.h_candidate_read_ids, data.h_candidate_read_ids);
            std::swap(dataArrays.h_candidates_per_subject, data.h_candidates_per_subject);           

            std::swap(dataArrays.d_subject_sequences_data, data.d_subject_sequences_data);
            std::swap(dataArrays.d_subject_sequences_lengths, data.d_subject_sequences_lengths);
            std::swap(dataArrays.d_subject_read_ids, data.d_subject_read_ids);
            std::swap(dataArrays.d_candidate_read_ids, data.d_candidate_read_ids);
            std::swap(dataArrays.d_candidates_per_subject, data.d_candidates_per_subject);
            std::swap(dataArrays.d_candidates_per_subject_prefixsum, data.d_candidates_per_subject_prefixsum);

            n_subjects = data.n_subjects;
            n_queries = data.n_queries;  

            data.n_subjects = 0;
            data.n_queries = 0;

            reallocInNextIterationData = data.reallocOccurred;
            data.reallocOccurred = false;

            graphindex = 1 - graphindex;
        }

        void moveResultsToOutputData(OutputData& outputData){
            auto& rawResults = outputData.rawResults;

            rawResults.n_subjects = n_subjects;
            rawResults.n_queries = n_queries;
            rawResults.encodedSequencePitchInInts = encodedSequencePitchInInts;
            rawResults.decodedSequencePitchInBytes = decodedSequencePitchInBytes;
            rawResults.maxNumEditsPerSequence = maxNumEditsPerSequence;

            std::swap(dataArrays.h_subject_read_ids, rawResults.h_subject_read_ids);
            std::swap(dataArrays.h_subject_is_corrected, rawResults.h_subject_is_corrected);
            std::swap(dataArrays.h_is_high_quality_subject, rawResults.h_is_high_quality_subject);
            std::swap(dataArrays.h_num_corrected_candidates_per_anchor, rawResults.h_num_corrected_candidates_per_anchor);
            std::swap(dataArrays.h_num_corrected_candidates_per_anchor_prefixsum, rawResults.h_num_corrected_candidates_per_anchor_prefixsum);
            std::swap(dataArrays.h_indices_of_corrected_candidates, rawResults.h_indices_of_corrected_candidates);
            std::swap(dataArrays.h_candidate_read_ids, rawResults.h_candidate_read_ids);
            std::swap(dataArrays.h_corrected_subjects, rawResults.h_corrected_subjects);
            std::swap(dataArrays.h_corrected_candidates, rawResults.h_corrected_candidates);
            std::swap(dataArrays.h_subject_sequences_lengths, rawResults.h_subject_sequences_lengths);
            std::swap(dataArrays.h_candidate_sequences_lengths, rawResults.h_candidate_sequences_lengths);
            std::swap(dataArrays.h_alignment_shifts, rawResults.h_alignment_shifts);

            std::swap(dataArrays.h_editsPerCorrectedSubject, rawResults.h_editsPerCorrectedSubject);
            std::swap(dataArrays.h_numEditsPerCorrectedSubject, rawResults.h_numEditsPerCorrectedSubject);

            std::swap(dataArrays.h_editsPerCorrectedCandidate, rawResults.h_editsPerCorrectedCandidate);
            std::swap(dataArrays.h_numEditsPerCorrectedCandidate, rawResults.h_numEditsPerCorrectedCandidate);

        }

	};

    struct SerializedFeature{
        char consensus;
        int position;
        read_number readId;
        std::string featureString;

        SerializedFeature(){}
        SerializedFeature(read_number r, int p, char c, const std::string& s)
            : readId(r), position(p), consensus(c), featureString(s){}
        SerializedFeature(read_number r, int p, char c, std::string&& s)
        : readId(r), position(p), consensus(c), featureString(std::move(s)){}
    };


    struct TransitionFunctionData {
		cpu::RangeGenerator<read_number>* readIdGenerator;
		const Minhasher* minhasher;
        const DistributedReadStorage* readStorage;
		CorrectionOptions correctionOptions;
        GoodAlignmentProperties goodAlignmentProperties;
        SequenceFileProperties sequenceFileProperties;
        RuntimeOptions runtimeOptions;
        MinhashOptions minhashOptions;
        AlignmentOptions alignmentOptions;
        FileOptions fileOptions;
		std::atomic_uint8_t* correctionStatusFlagsPerRead;
		std::ofstream* featurestream;
        std::function<void(const TempCorrectedSequence&, EncodedTempCorrectedSequence)> saveCorrectedSequence;
		std::function<void(read_number)> lock;
		std::function<void(read_number)> unlock;

        std::condition_variable isFinishedCV;
        std::mutex isFinishedMutex;

        std::function<void(const SerializedFeature&)> saveFeature;

        ForestClassifier fc;// = ForestClassifier{"./forests/testforest.so"};
        //NN_Correction_Classifier nnClassifier;

        std::vector<Minhasher::Handle> minhashHandles;
	};





    void initNextIterationData(NextIterationData& nextData, int deviceId){
        nextData.deviceId = deviceId;

        cudaSetDevice(deviceId); CUERR;
        cudaStreamCreate(&nextData.stream); CUERR;
        cudaEventCreate(&nextData.event); CUERR;

        nextData.mergeRangesGpuHandle = makeMergeRangesGpuHandle<read_number>();

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

        cudaStreamSynchronize(nextData.stream);
    }

    void destroyNextIterationData(NextIterationData& nextData){
        cudaSetDevice(nextData.deviceId); CUERR;
        cudaStreamDestroy(nextData.stream); CUERR;
        cudaEventDestroy(nextData.event); CUERR;

        nextData.h_subject_sequences_data.destroy();
        nextData.h_subject_sequences_lengths.destroy();
        nextData.h_subject_read_ids.destroy();
        nextData.h_candidate_read_ids.destroy();
        nextData.h_candidates_per_subject.destroy();

        nextData.d_subject_sequences_data.destroy();
        nextData.d_subject_sequences_lengths.destroy();
        nextData.d_subject_read_ids.destroy();
        nextData.d_candidate_read_ids.destroy();
        nextData.d_candidates_per_subject.destroy();
        nextData.d_candidates_per_subject_tmp.destroy();
        nextData.d_candidates_per_subject_prefixsum.destroy();

        nextData.d_candidate_read_ids_tmp.destroy();
        nextData.h_minhashSignatures.destroy();
        nextData.d_minhashSignatures.destroy();

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

        destroyMergeRangesGpuHandle(nextData.mergeRangesGpuHandle);
    }



    void getCandidateAlignments(Batch& batch);
    void buildMultipleSequenceAlignment(Batch& batch);
    void removeCandidatesOfDifferentRegionFromMSA(Batch& batch);
    void correctSubjects(Batch& batch);
    void correctCandidates(Batch& batch);
    


    void buildGraphViaCapture(Batch& batch){
        cudaSetDevice(batch.deviceId); CUERR;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        auto& graphwrap = batch.alignmentGraphs[batch.graphindex];

        if(!graphwrap.valid){
            std::cerr << "rebuild graph\n";

            if(graphwrap.execgraph != nullptr){
                cudaGraphExecDestroy(graphwrap.execgraph); CUERR;
            }
            
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
            if(batch.transFuncData->correctionOptions.correctCandidates){
                correctCandidates(batch);                
            }

            //join forked stream for valid capture
            cudaEventRecord(events[0], streams[secondary_stream_index]); CUERR;
            cudaStreamWaitEvent(streams[primary_stream_index], events[0], 0); CUERR;

            cudaGraph_t graph;
            cudaStreamEndCapture(streams[primary_stream_index], &graph); CUERR;
            
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

            graphwrap.execgraph = execGraph;

            graphwrap.valid = true;
        }
    }

    void executeGraph(Batch& batch){
        cudaSetDevice(batch.deviceId); CUERR;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        buildGraphViaCapture(batch);

        auto& graphwrap = batch.alignmentGraphs[batch.graphindex];

        assert(graphwrap.valid);
        cudaGraphLaunch(graphwrap.execgraph, streams[primary_stream_index]); CUERR;
    }




    void prepareNewDataForCorrection(Batch& batchData, int batchsize, const Minhasher& minhasher, const DistributedReadStorage& readStorage){
        NextIterationData& nextData = batchData.nextIterationData;
        const auto& transFuncData = *batchData.transFuncData;

        const int numCandidatesLimit = batchData.numCandidatesLimit;

        cudaSetDevice(nextData.deviceId); CUERR;

        const size_t encodedSequencePitchInInts = batchData.encodedSequencePitchInInts;

        nextData.reallocOccurred |= nextData.h_subject_sequences_data.resize(encodedSequencePitchInInts * batchsize);
        nextData.reallocOccurred |= nextData.d_subject_sequences_data.resize(encodedSequencePitchInInts * batchsize);
        nextData.reallocOccurred |= nextData.h_subject_sequences_lengths.resize(batchsize);
        nextData.reallocOccurred |= nextData.d_subject_sequences_lengths.resize(batchsize);
        nextData.reallocOccurred |= nextData.h_subject_read_ids.resize(batchsize);
        nextData.reallocOccurred |= nextData.d_subject_read_ids.resize(batchsize);
        nextData.reallocOccurred |= nextData.h_minhashSignatures.resize(maximum_number_of_maps * batchsize);
        nextData.reallocOccurred |= nextData.d_minhashSignatures.resize(maximum_number_of_maps * batchsize);

        const int maxNumThreads = batchData.transFuncData->runtimeOptions.threads;

        std::vector<Minhasher::Range_t>& allRanges = nextData.allRanges;
        std::vector<int>& idsPerChunk = nextData.idsPerChunk;
        std::vector<int>& numAnchorsPerChunk = nextData.numAnchorsPerChunk;
        std::vector<int>& idsPerChunkPrefixSum = nextData.idsPerChunkPrefixSum;
        std::vector<int>& numAnchorsPerChunkPrefixSum = nextData.numAnchorsPerChunkPrefixSum;

        allRanges.resize(maximum_number_of_maps * batchsize);
        idsPerChunk.resize(maxNumThreads, 0);   
        numAnchorsPerChunk.resize(maxNumThreads, 0);
        idsPerChunkPrefixSum.resize(maxNumThreads, 0);
        numAnchorsPerChunkPrefixSum.resize(maxNumThreads, 0);

        const int resultsPerMap = 2.5f * batchData.transFuncData->correctionOptions.estimatedCoverage;
            //minhasher.calculateResultsPerMapThreshold(batchData.transFuncData->correctionOptions.estimatedCoverage);
        const int maxNumIds = resultsPerMap * maximum_number_of_maps * batchsize;

        nextData.reallocOccurred |= nextData.h_candidate_read_ids.resize(maxNumIds + numCandidatesLimit);
        nextData.reallocOccurred |= nextData.d_candidate_read_ids.resize(maxNumIds + numCandidatesLimit);
        nextData.reallocOccurred |= nextData.d_candidate_read_ids_tmp.resize(maxNumIds + numCandidatesLimit);
        nextData.reallocOccurred |= nextData.h_candidates_per_subject.resize(batchsize);
        nextData.reallocOccurred |= nextData.d_candidates_per_subject.resize(2*batchsize);
        nextData.reallocOccurred |= nextData.d_candidates_per_subject_tmp.resize(2*batchsize);
        nextData.reallocOccurred |= nextData.d_candidates_per_subject_prefixsum.resize(batchsize+1);

        nextData.h_leftoverAnchorReadIds.resize(batchsize);
        nextData.d_leftoverAnchorReadIds.resize(batchsize);
        nextData.d_leftoverAnchorLengths.resize(batchsize);
        nextData.d_leftoverCandidateReadIds.resize(maxNumIds + numCandidatesLimit);
        nextData.d_leftoverCandidatesPerAnchors.resize(batchsize);
        nextData.d_leftoverAnchorSequences.resize(encodedSequencePitchInInts * batchsize);


        //data of new anchors is appended to leftover data

        const int numLeftoverAnchors = *nextData.h_numLeftoverAnchors.get();
        read_number* const readIdsBegin = nextData.h_leftoverAnchorReadIds.get();
        read_number* const readIdsEnd = transFuncData.readIdGenerator->next_n_into_buffer(
            batchsize - numLeftoverAnchors, 
            readIdsBegin + numLeftoverAnchors
        );
        nextData.n_new_subjects = std::distance(readIdsBegin + numLeftoverAnchors, readIdsEnd);
        
        nextData.n_subjects = nextData.n_new_subjects + numLeftoverAnchors;//debug

        nextData.h_numAnchors[0] = nextData.n_new_subjects + numLeftoverAnchors;

        if(nextData.n_subjects == 0){
            return;
        };

        cudaMemcpyAsync(
            nextData.d_numAnchors.get(),
            nextData.h_numAnchors.get(),
            sizeof(int),
            H2D,
            nextData.stream
        ); CUERR;

        cudaMemcpyAsync(
            nextData.d_leftoverAnchorReadIds.get() + numLeftoverAnchors,
            nextData.h_leftoverAnchorReadIds.get() + numLeftoverAnchors,
            sizeof(read_number) * nextData.n_new_subjects,
            H2D,
            nextData.stream
        ); CUERR;

        // std::cerr << "gather anchors\n";
        // get sequence data and length of new anchors.
        readStorage.gatherSequenceDataToGpuBufferAsync(
            batchData.threadPool,
            batchData.subjectSequenceGatherHandle,
            nextData.d_leftoverAnchorSequences.get() + numLeftoverAnchors * encodedSequencePitchInInts,
            encodedSequencePitchInInts,
            nextData.h_leftoverAnchorReadIds.get() + numLeftoverAnchors,
            nextData.d_leftoverAnchorReadIds.get() + numLeftoverAnchors,
            nextData.n_new_subjects,
            batchData.deviceId,
            nextData.stream,
            transFuncData.runtimeOptions.nCorrectorThreads
        );

        readStorage.gatherSequenceLengthsToGpuBufferAsync(
            nextData.d_leftoverAnchorLengths.get() + numLeftoverAnchors,
            batchData.deviceId,
            nextData.d_leftoverAnchorReadIds.get() + numLeftoverAnchors,
            nextData.n_new_subjects,            
            nextData.stream
        );

        //minhash the retrieved anchors to find candidate ids

        Batch* batchptr = &batchData;
        NextIterationData* nextDataPtr = &nextData;
        const Minhasher* minhasherPtr = &minhasher;

        const int kmerSize = batchData.transFuncData->minhashOptions.k;
        const int numHashFunctions = minhasherPtr->minparams.maps;

        callMinhashSignaturesKernel_async(
            nextData.d_minhashSignatures.get(),
            maximum_number_of_maps,
            nextData.d_leftoverAnchorSequences.get() + numLeftoverAnchors * encodedSequencePitchInInts,
            encodedSequencePitchInInts,
            nextData.n_new_subjects,
            nextData.d_leftoverAnchorLengths.get() + numLeftoverAnchors,
            kmerSize,
            numHashFunctions,
            nextData.stream
        );

        cudaMemcpyAsync(
            nextData.h_minhashSignatures.get(),
            nextData.d_minhashSignatures.get(),
            nextData.h_minhashSignatures.sizeInBytes(),
            H2D,
            nextData.stream
        ); CUERR;


        std::fill(idsPerChunk.begin(), idsPerChunk.end(), 0);
        std::fill(numAnchorsPerChunk.begin(), numAnchorsPerChunk.end(), 0);

        cudaStreamSynchronize(nextData.stream); CUERR; //wait for D2H transfers of signatures anchor data which is required for minhasher

        auto querySignatures2 = [&, batchptr, nextDataPtr, minhasherPtr](int begin, int end, int threadId){

            const int numSequences = end - begin;

            int totalNumResults = 0;

            nvtx::push_range("queryPrecalculatedSignatures", 6);
            minhasherPtr->queryPrecalculatedSignatures(
                nextData.h_minhashSignatures.get() + begin * maximum_number_of_maps,
                allRanges.data() + begin * maximum_number_of_maps,
                &totalNumResults, 
                numSequences
            );

            idsPerChunk[threadId] = totalNumResults;
            numAnchorsPerChunk[threadId] = numSequences;
            nvtx::pop_range();
        };

        int numChunksRequired = batchData.threadPool->parallelFor(
            nextData.pforHandle,
            0, 
            nextData.n_new_subjects, 
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
        //std::cerr << "totalNumIds = " << totalNumIds << "\n";
        const int numLeftoverCandidates = nextData.h_numLeftoverCandidates[0];

        auto copyCandidateIdsToContiguousMem = [&](int begin, int end, int threadId){
            nvtx::push_range("copyCandidateIdsToContiguousMem", 1);
            for(int chunkId = begin; chunkId < end; chunkId++){
                const auto hostdatabegin = nextData.h_candidate_read_ids.get() + idsPerChunkPrefixSum[chunkId];
                const auto devicedatabegin = nextData.d_candidate_read_ids_tmp.get() + numLeftoverCandidates 
                                                + idsPerChunkPrefixSum[chunkId];
                const size_t elementsInChunk = idsPerChunk[chunkId];

                const auto ranges = allRanges.data() + numAnchorsPerChunkPrefixSum[chunkId] * maximum_number_of_maps;

                auto dest = hostdatabegin;

                const int lmax = numAnchorsPerChunk[chunkId] * maximum_number_of_maps;

                for(int k = 0; k < lmax; k++){
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

                cudaMemcpyAsync(
                    devicedatabegin,
                    hostdatabegin,
                    sizeof(read_number) * elementsInChunk,
                    H2D,
                    nextData.stream
                ); CUERR;
            }
            nvtx::pop_range();
        };

        batchData.threadPool->parallelFor(
            nextData.pforHandle,
            0, 
            numChunksRequired, 
            [=](auto begin, auto end, auto threadId){
                copyCandidateIdsToContiguousMem(begin, end, threadId);
            }
        );

        // copyCandidateIdsToContiguousMem(0, 1, 0);

        nvtx::push_range("gpumakeUniqueQueryResults", 2);
        mergeRangesGpuAsync(
            nextDataPtr->mergeRangesGpuHandle, 
            nextData.d_leftoverCandidateReadIds.get() + numLeftoverCandidates,
            nextData.d_leftoverCandidatesPerAnchors.get() + numLeftoverAnchors,
            nextData.d_candidates_per_subject_prefixsum.get() + numLeftoverAnchors,
            nextData.d_candidate_read_ids_tmp.get() + numLeftoverCandidates,
            allRanges.data(), 
            maximum_number_of_maps * nextData.n_new_subjects, 
            nextData.d_leftoverAnchorReadIds.get() + numLeftoverAnchors,
            minhasherPtr->minparams.maps, 
            nextData.stream,
            MergeRangesKernelType::allcub
        );

        nvtx::pop_range();

        nvtx::push_range("leftover_calculation", 3);

        //fix the prefix sum to include the leftover data
        std::size_t cubTempBytes = sizeof(read_number) * (maxNumIds + numCandidatesLimit);
        void* cubTemp = nextData.d_candidate_read_ids_tmp.get();
        //d_candidates_per_subject_prefixsum[0] is 0
        cub::DeviceScan::InclusiveSum(
            cubTemp, 
            cubTempBytes,
            nextData.d_leftoverCandidatesPerAnchors.get(),
            nextData.d_candidates_per_subject_prefixsum.get() + 1,
            batchsize,
            nextData.stream
        ); CUERR;

        //find new numbers of leftover candidates and anchors
        {
            int* d_candidates_per_subject_prefixsum = nextData.d_candidates_per_subject_prefixsum.get();
            int* d_numAnchors = nextData.d_numAnchors.get();
            int* d_numCandidates = nextData.d_numCandidates.get();
            int* d_numLeftoverAnchors = nextData.d_numLeftoverAnchors.get();
            int* d_numLeftoverCandidates = nextData.d_numLeftoverCandidates.get();
            
            
            generic_kernel<<<1, 1, 0, nextData.stream>>>(
                [=]__device__(){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

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
                        if(tid == 0){
                            *d_numLeftoverAnchors = newNumLeftoverAnchors;
                            *d_numAnchors = numAnchors - newNumLeftoverAnchors;
                        }

                        if(index < numAnchors){

                            const int newNumLeftoverCandidates = totalNumCandidates - d_candidates_per_subject_prefixsum[index];
                            
                            if(tid == 0){
                                *d_numLeftoverCandidates = newNumLeftoverCandidates;
                                *d_numCandidates = totalNumCandidates - newNumLeftoverCandidates;
                            }
                        }else{
                            if(tid == 0){
                                *d_numLeftoverCandidates = 0;
                                *d_numCandidates = totalNumCandidates - 0;
                            }
                        }
                    }else{
                        if(tid == 0){
                            *d_numLeftoverAnchors = 0;
                            *d_numLeftoverCandidates = 0;
                            *d_numAnchors = numAnchors - 0;
                            *d_numCandidates = totalNumCandidates - 0;
                        }
                    }
                }
            ); CUERR;

        }

        //copy all data from leftover buffers to output buffers 
        //copy new leftover data from output buffers to the front of leftover buffers
        {
            int* d_numAnchors = nextData.d_numAnchors.get();
            int* d_numCandidates = nextData.d_numCandidates.get();
            int* d_numLeftoverAnchors = nextData.d_numLeftoverAnchors.get();
            int* d_numLeftoverCandidates = nextData.d_numLeftoverCandidates.get();
            int* d_candidates_per_subject = nextData.d_candidates_per_subject.get();
            int* d_leftoverCandidatesPerAnchors = nextData.d_leftoverCandidatesPerAnchors.get();

            unsigned int* d_leftoverAnchorSequences = nextData.d_leftoverAnchorSequences.get();
            int* d_leftoverAnchorLengths = nextData.d_leftoverAnchorLengths.get();
            read_number* d_leftoverAnchorReadIds = nextData.d_leftoverAnchorReadIds.get();
            read_number* d_leftoverCandidateReadIds = nextData.d_leftoverCandidateReadIds.get();
            
            read_number* d_subject_read_ids = nextData.d_subject_read_ids.get();
            int* d_subject_sequences_lengths = nextData.d_subject_sequences_lengths.get();
            read_number* d_candidate_read_ids = nextData.d_candidate_read_ids.get();
            unsigned int* d_subject_sequences_data = nextData.d_subject_sequences_data.get();
            
            //copy all data from leftover buffers to output buffers
            generic_kernel<<<320, 256, 0, nextData.stream>>>(
                [=]__device__(){
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
            generic_kernel<<<320, 256, 0, nextData.stream>>>(
                [=]__device__(){
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
        }

        nvtx::pop_range();

        cudaMemcpyAsync(
            nextData.h_numAnchors.get(),
            nextData.d_numAnchors.get(),
            sizeof(int),
            D2H,
            nextData.stream
        ); CUERR;

        cudaMemcpyAsync(
            nextData.h_numCandidates.get(),
            nextData.d_numCandidates.get(),
            sizeof(int),
            D2H,
            nextData.stream
        ); CUERR;

        cudaMemcpyAsync(
            nextData.h_numLeftoverAnchors.get(),
            nextData.d_numLeftoverAnchors.get(),
            sizeof(int),
            D2H,
            nextData.stream
        ); CUERR;

        cudaMemcpyAsync(
            nextData.h_numLeftoverCandidates.get(),
            nextData.d_numLeftoverCandidates.get(),
            sizeof(int),
            D2H,
            nextData.stream
        ); CUERR;

        cudaStreamSynchronize(nextData.stream); CUERR;

        // std::cerr << "final n_subjects " << nextData.h_numAnchors[0] << " ";
        // std::cerr << "final n_queries " << nextData.h_numCandidates[0] << " ";
        // std::cerr << "n_new_subjects " << nextData.n_new_subjects << " ";
        // std::cerr << "numLeftoverAnchors " << nextData.h_numLeftoverAnchors[0] << " ";
        // std::cerr << "numLeftoverCandidates " << nextData.h_numLeftoverCandidates[0] << "\n";

        nextData.n_subjects = nextData.h_numAnchors[0];
        nextData.n_queries = nextData.h_numCandidates[0];

        cudaMemcpyAsync(
            nextData.h_candidate_read_ids.get(),
            nextData.d_candidate_read_ids.get(),
            sizeof(read_number) * nextData.n_queries,
            D2H,
            nextData.stream
        ); CUERR;

        cudaMemcpyAsync(
            nextData.h_leftoverAnchorReadIds.get(),
            nextData.d_leftoverAnchorReadIds.get(),
            sizeof(read_number) * nextData.h_numLeftoverAnchors[0],
            D2H,
            nextData.stream
        ); CUERR;

        cudaMemcpyAsync(
            nextData.h_subject_read_ids.get(),
            nextData.d_subject_read_ids.get(),
            sizeof(read_number) * nextData.h_numAnchors[0],
            D2H,
            nextData.stream
        ); CUERR;

        // cudaMemcpyAsync(
        //     nextData.h_candidates_per_subject.get(),
        //     nextData.d_candidates_per_subject.get(),
        //     sizeof(int) * (nextData.n_subjects),
        //     D2H,
        //     nextData.stream
        // ); CUERR; 

        cudaStreamSynchronize(nextData.stream); CUERR;


        

    }



    void getNextBatchForCorrection(Batch& batchData){
        Batch* batchptr = &batchData;

        auto getDataForNextIteration = [batchptr](){
            nvtx::push_range("prepareNewDataForCorrection",1);
            prepareNewDataForCorrection(
                *batchptr, 
                batchptr->transFuncData->correctionOptions.batchsize,
                *batchptr->transFuncData->minhasher,
                *batchptr->transFuncData->readStorage                
            );
            cudaStreamSynchronize(batchptr->nextIterationData.stream); CUERR;
            if(batchptr->nextIterationData.n_subjects > 0){                
                batchptr->nextIterationData.syncFlag.signal();
            }else{
                batchptr->nextIterationData.n_queries = 0;
                batchptr->nextIterationData.syncFlag.signal();
            }
            nvtx::pop_range();
        };

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

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batchData.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batchData.events;

        cudaEventRecord(events[0], batchData.nextIterationData.stream); CUERR;
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

    }


    void resizeArraysFixedCandidateBatchsize(Batch& batchData){

        //allocate memory required for batch processing

        auto& dataArrays = batchData.dataArrays;
        const auto& transFuncData = *(batchData.transFuncData);
        auto& streams = batchData.streams;

        const int min_overlap = std::max(1, std::max(transFuncData.goodAlignmentProperties.min_overlap, 
            int(transFuncData.sequenceFileProperties.maxSequenceLength 
                * transFuncData.goodAlignmentProperties.min_overlap_ratio)));

        const int sequence_pitch = batchData.decodedSequencePitchInBytes;

        int msa_max_column_count = (3*transFuncData.sequenceFileProperties.maxSequenceLength - 2*min_overlap);
        batchData.msa_pitch = SDIV(sizeof(char)*msa_max_column_count, 4) * 4;
        batchData.msa_weights_pitch = SDIV(sizeof(float)*msa_max_column_count, 4) * 4;
        size_t msa_weights_pitch_floats = batchData.msa_weights_pitch / sizeof(float);

        const auto maxCandidates = batchData.numCandidatesLimit;
        const auto batchsize = batchData.transFuncData->correctionOptions.batchsize;
        const auto encodedSeqPitchInts = batchData.encodedSequencePitchInInts;
        const auto qualPitchBytes = batchData.qualityPitchInBytes;

        const auto maxNumEditsPerSequence = batchData.maxNumEditsPerSequence;
        
        const auto n_subjects = batchData.n_subjects;
        const auto n_queries = batchData.n_queries;
        
        //sequence input data
        //following arrays are initialized by next iteration data:
        //h_subject_sequences_data, h_subject_sequences_length
        //h_subject_read_ids, h_candidate_read_ids
        //d_subject_sequences_data, d_subject_sequences_length, 
        //d_candidates_per_subject, d_candidates_per_subject_prefixsum
        //d_subject_read_ids, d_candidate_read_ids


        batchData.reallocResize |= dataArrays.h_numAnchors.resize(1);
        batchData.reallocResize |= dataArrays.h_numCandidates.resize(1);
        batchData.reallocResize |= dataArrays.d_numAnchors.resize(1);
        batchData.reallocResize |= dataArrays.d_numCandidates.resize(1);


        batchData.reallocResize |= dataArrays.h_candidate_sequences_data.resize(maxCandidates * encodedSeqPitchInts);
        batchData.reallocResize |= dataArrays.h_transposedCandidateSequencesData.resize(maxCandidates * encodedSeqPitchInts);
        batchData.reallocResize |= dataArrays.h_subject_sequences_lengths.resize(batchsize);
        batchData.reallocResize |= dataArrays.h_candidate_sequences_lengths.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.h_anchorIndicesOfCandidates.resize(maxCandidates);

        batchData.reallocResize |= dataArrays.d_subject_sequences_data.resize(batchsize * encodedSeqPitchInts);
        batchData.reallocResize |= dataArrays.d_candidate_sequences_data.resize(maxCandidates * encodedSeqPitchInts);
        batchData.reallocResize |= dataArrays.d_transposedCandidateSequencesData.resize(maxCandidates * encodedSeqPitchInts);
        batchData.reallocResize |= dataArrays.d_subject_sequences_lengths.resize(batchsize);
        batchData.reallocResize |= dataArrays.d_candidate_sequences_lengths.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.d_anchorIndicesOfCandidates.resize(maxCandidates);

        

        //alignment output

        // batchData.reallocResize |= dataArrays.h_alignment_scores.resize(2*maxCandidates);
        // batchData.reallocResize |= dataArrays.h_alignment_overlaps.resize(2*maxCandidates);
        batchData.reallocResize |= dataArrays.h_alignment_shifts.resize(2*maxCandidates);
        // batchData.reallocResize |= dataArrays.h_alignment_nOps.resize(2*maxCandidates);
        // batchData.reallocResize |= dataArrays.h_alignment_isValid.resize(2*maxCandidates);
        // batchData.reallocResize |= dataArrays.h_alignment_best_alignment_flags.resize(maxCandidates);

        //batchData.reallocResize |= dataArrays.d_alignment_scores.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.d_alignment_overlaps.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.d_alignment_shifts.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.d_alignment_nOps.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.d_alignment_isValid.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.d_alignment_best_alignment_flags.resize(maxCandidates);

        // candidate indices

        batchData.reallocResize |= dataArrays.h_indices.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.h_indices_per_subject.resize(batchsize);
        batchData.reallocResize |= dataArrays.h_num_indices.resize(1);

        batchData.reallocResize |= dataArrays.d_indices.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.d_indices_per_subject.resize(batchsize);
        batchData.reallocResize |= dataArrays.d_num_indices.resize(1);
        batchData.reallocResize |= dataArrays.d_indices_tmp.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.d_indices_per_subject_tmp.resize(batchsize);
        batchData.reallocResize |= dataArrays.d_num_indices_tmp.resize(1);

        batchData.reallocResize |= dataArrays.h_indices_of_corrected_subjects.resize(batchsize);
        batchData.reallocResize |= dataArrays.h_num_indices_of_corrected_subjects.resize(1);
        batchData.reallocResize |= dataArrays.d_indices_of_corrected_subjects.resize(batchsize);
        batchData.reallocResize |= dataArrays.d_num_indices_of_corrected_subjects.resize(1);

        batchData.reallocResize |= dataArrays.h_editsPerCorrectedSubject.resize(batchsize * maxNumEditsPerSequence);
        batchData.reallocResize |= dataArrays.h_numEditsPerCorrectedSubject.resize(batchsize);
        batchData.reallocResize |= dataArrays.h_anchorContainsN.resize(batchsize);

        batchData.reallocResize |= dataArrays.d_editsPerCorrectedSubject.resize(batchsize * maxNumEditsPerSequence);
        batchData.reallocResize |= dataArrays.d_numEditsPerCorrectedSubject.resize(batchsize);
        batchData.reallocResize |= dataArrays.d_anchorContainsN.resize(batchsize);

        batchData.reallocResize |= dataArrays.h_editsPerCorrectedCandidate.resize(maxCandidates * maxNumEditsPerSequence);
        batchData.reallocResize |= dataArrays.h_numEditsPerCorrectedCandidate.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.h_candidateContainsN.resize(maxCandidates);

        batchData.reallocResize |= dataArrays.d_editsPerCorrectedCandidate.resize(maxCandidates * maxNumEditsPerSequence);
        batchData.reallocResize |= dataArrays.d_numEditsPerCorrectedCandidate.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.d_candidateContainsN.resize(maxCandidates);

        //qualitiy scores
        if(transFuncData.correctionOptions.useQualityScores) {
            batchData.reallocResize |= dataArrays.h_subject_qualities.resize(batchsize * qualPitchBytes);
            batchData.reallocResize |= dataArrays.h_candidate_qualities.resize(maxCandidates * qualPitchBytes);

            batchData.reallocResize |= dataArrays.d_subject_qualities.resize(batchsize * qualPitchBytes);
            batchData.reallocResize |= dataArrays.d_candidate_qualities.resize(maxCandidates * qualPitchBytes);
            batchData.reallocResize |= dataArrays.d_candidate_qualities_transposed.resize(maxCandidates * qualPitchBytes);            
        }


        //correction results

        batchData.reallocResize |= dataArrays.h_corrected_subjects.resize(batchsize * sequence_pitch);
        batchData.reallocResize |= dataArrays.h_corrected_candidates.resize(maxCandidates * sequence_pitch);
        batchData.reallocResize |= dataArrays.h_num_corrected_candidates_per_anchor.resize(batchsize);
        batchData.reallocResize |= dataArrays.h_num_corrected_candidates_per_anchor_prefixsum.resize(batchsize);
        batchData.reallocResize |= dataArrays.h_num_total_corrected_candidates.resize(1);
        batchData.reallocResize |= dataArrays.h_subject_is_corrected.resize(batchsize);
        batchData.reallocResize |= dataArrays.h_indices_of_corrected_candidates.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.h_num_uncorrected_positions_per_subject.resize(batchsize);
        batchData.reallocResize |= dataArrays.h_uncorrected_positions_per_subject.resize(batchsize * transFuncData.sequenceFileProperties.maxSequenceLength);
        
        batchData.reallocResize |= dataArrays.d_corrected_subjects.resize(batchsize * sequence_pitch);
        batchData.reallocResize |= dataArrays.d_corrected_candidates.resize(maxCandidates * sequence_pitch);
        batchData.reallocResize |= dataArrays.d_num_corrected_candidates_per_anchor.resize(batchsize);
        batchData.reallocResize |= dataArrays.d_num_corrected_candidates_per_anchor_prefixsum.resize(batchsize);
        batchData.reallocResize |= dataArrays.d_num_total_corrected_candidates.resize(1);
        batchData.reallocResize |= dataArrays.d_subject_is_corrected.resize(batchsize);
        batchData.reallocResize |= dataArrays.d_indices_of_corrected_candidates.resize(maxCandidates);
        batchData.reallocResize |= dataArrays.d_num_uncorrected_positions_per_subject.resize(batchsize);
        batchData.reallocResize |= dataArrays.d_uncorrected_positions_per_subject.resize(batchsize * transFuncData.sequenceFileProperties.maxSequenceLength);

        batchData.reallocResize |= dataArrays.h_is_high_quality_subject.resize(batchsize);
        batchData.reallocResize |= dataArrays.h_high_quality_subject_indices.resize(batchsize);
        batchData.reallocResize |= dataArrays.h_num_high_quality_subject_indices.resize(1);

        batchData.reallocResize |= dataArrays.d_is_high_quality_subject.resize(batchsize);
        batchData.reallocResize |= dataArrays.d_high_quality_subject_indices.resize(batchsize);
        batchData.reallocResize |= dataArrays.d_num_high_quality_subject_indices.resize(1);

        //multiple sequence alignment

        batchData.reallocResize |= dataArrays.h_consensus.resize(batchsize * batchData.msa_pitch);
        batchData.reallocResize |= dataArrays.h_support.resize(batchsize * msa_weights_pitch_floats);
        batchData.reallocResize |= dataArrays.h_coverage.resize(batchsize * msa_weights_pitch_floats);
        batchData.reallocResize |= dataArrays.h_origWeights.resize(batchsize * msa_weights_pitch_floats);
        batchData.reallocResize |= dataArrays.h_origCoverages.resize(batchsize * msa_weights_pitch_floats);
        batchData.reallocResize |= dataArrays.h_msa_column_properties.resize(batchsize);
        batchData.reallocResize |= dataArrays.h_counts.resize(batchsize * 4 * msa_weights_pitch_floats);
        batchData.reallocResize |= dataArrays.h_weights.resize(batchsize * 4 * msa_weights_pitch_floats);

        batchData.reallocResize |= dataArrays.d_consensus.resize(batchsize * batchData.msa_pitch);
        batchData.reallocResize |= dataArrays.d_support.resize(batchsize * msa_weights_pitch_floats);
        batchData.reallocResize |= dataArrays.d_coverage.resize(batchsize * msa_weights_pitch_floats);
        batchData.reallocResize |= dataArrays.d_origWeights.resize(batchsize * msa_weights_pitch_floats);
        batchData.reallocResize |= dataArrays.d_origCoverages.resize(batchsize * msa_weights_pitch_floats);
        batchData.reallocResize |= dataArrays.d_msa_column_properties.resize(batchsize);
        batchData.reallocResize |= dataArrays.d_counts.resize(batchsize * 4 * msa_weights_pitch_floats);
        batchData.reallocResize |= dataArrays.d_weights.resize(batchsize * 4 * msa_weights_pitch_floats);


        batchData.reallocResize |= dataArrays.d_canExecute.resize(1);

           
        
        
        std::size_t flagTemp = sizeof(bool) * maxCandidates;
        std::size_t popcountShdTempBytes = 0; 
        
        call_popcount_shifted_hamming_distance_kernel_async(
            nullptr,
            popcountShdTempBytes,
            dataArrays.d_alignment_overlaps.get(),
            dataArrays.d_alignment_shifts.get(),
            dataArrays.d_alignment_nOps.get(),
            dataArrays.d_alignment_isValid.get(),
            dataArrays.d_alignment_best_alignment_flags.get(),
            dataArrays.d_subject_sequences_data.get(),
            dataArrays.d_candidate_sequences_data.get(),
            dataArrays.d_subject_sequences_lengths.get(),
            dataArrays.d_candidate_sequences_lengths.get(),
            dataArrays.d_candidates_per_subject_prefixsum.get(),
            dataArrays.h_candidates_per_subject.get(),
            dataArrays.d_candidates_per_subject.get(),
            dataArrays.d_anchorIndicesOfCandidates.get(),
            dataArrays.d_numAnchors.get(),
            dataArrays.d_numCandidates.get(),
            batchsize,
            maxCandidates,
            transFuncData.sequenceFileProperties.maxSequenceLength,
            batchData.encodedSequencePitchInInts,
            transFuncData.goodAlignmentProperties.min_overlap,
            transFuncData.goodAlignmentProperties.maxErrorRate,
            transFuncData.goodAlignmentProperties.min_overlap_ratio,
            transFuncData.correctionOptions.estimatedErrorrate,
            //batchData.maxSubjectLength,
            streams[primary_stream_index],
            batchData.kernelLaunchHandle
        );
        
        // this buffer will also serve as temp storage for cub. The required memory for cub 
        // is less than popcountShdTempBytes.
        popcountShdTempBytes = std::max(flagTemp, popcountShdTempBytes);
        batchData.reallocResize |= dataArrays.d_tempstorage.resize(popcountShdTempBytes);

        if(batchData.reallocResize){
            //invalidate all graphs
            for(int i = 0; i < 2; i++){
                batchData.alignmentGraphs[i].valid = false;
            }
        }else{
            if(batchData.reallocInNextIterationData){
                //invalidate current graph
                batchData.alignmentGraphs[batchData.graphindex].valid = false;
            }else{
                ; //all good
            }
        }

        // if(batchData.reallocResize){
        //     auto& graph = batchData.alignmentGraphs[batchData.graphindex];

        //     graph.d_anchorIndicesOfCandidates = dataArrays.d_anchorIndicesOfCandidates.get();
        //     graph.d_numAnchors = dataArrays.d_numAnchors.get();
        //     graph.d_candidates_per_subject = dataArrays.d_candidates_per_subject.get();
        //     graph.d_candidates_per_subject_prefixsum = dataArrays.d_candidates_per_subject_prefixsum.get();
        // }
        
        {
            int numAnchors = batchData.n_subjects;
            int numCandidates = batchData.n_queries;
            int* d_numAnchorsPtr = dataArrays.d_numAnchors.get();
            int* d_numCandidatesPtr = dataArrays.d_numCandidates.get();
            bool* d_canExecutePtr = dataArrays.d_canExecute.get();
            int* d_numTotalCorrectedCandidatePtr = dataArrays.d_num_total_corrected_candidates.get();

            generic_kernel<<<1,1,0,streams[primary_stream_index]>>>([=] __device__ (){
                *d_numAnchorsPtr = numAnchors;
                *d_numCandidatesPtr = numCandidates;
                *d_canExecutePtr = true;
                *d_numTotalCorrectedCandidatePtr = 0;
            }); CUERR;
        }
        // call_fill_kernel_async(
        //     dataArrays.d_canExecute.get(),
        //     1,
        //     true,
        //     streams[primary_stream_index]
        // );

        // call_fill_kernel_async(
        //     dataArrays.d_num_total_corrected_candidates.get(),
        //     1,
        //     0,
        //     streams[primary_stream_index]
        // );
    }





    void getCandidateSequenceData(Batch& batch, const DistributedReadStorage& readStorage){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        DataArrays& dataArrays = batch.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        const auto batchsize = batch.transFuncData->correctionOptions.batchsize;
        const auto maxCandidates = batch.numCandidatesLimit;

        // readStorage.gatherSequenceLengthsToGpuBufferAsync(
        //                                 dataArrays.d_subject_sequences_lengths.get(),
        //                                 batch.deviceId,
        //                                 dataArrays.d_subject_read_ids.get(),
        //                                 batch.n_subjects,   
        //                                 streams[primary_stream_index]);

        cudaMemcpyAsync(
            dataArrays.h_subject_sequences_lengths,
            dataArrays.d_subject_sequences_lengths, //filled by nextiteration data
            sizeof(int) * batchsize,
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        readStorage.readsContainN_async(
            batch.deviceId,
            dataArrays.d_anchorContainsN.get(), 
            dataArrays.d_subject_read_ids.get(), 
            dataArrays.d_numAnchors,
            batchsize, 
            streams[primary_stream_index]
        );

        readStorage.readsContainN_async(
            batch.deviceId,
            dataArrays.d_candidateContainsN.get(), 
            dataArrays.d_candidate_read_ids.get(), 
            dataArrays.d_numCandidates,
            maxCandidates, 
            streams[primary_stream_index]
        );  

        readStorage.gatherSequenceLengthsToGpuBufferAsync(
                                        dataArrays.d_candidate_sequences_lengths.get(),
                                        batch.deviceId,
                                        dataArrays.d_candidate_read_ids.get(),
                                        batch.n_queries,            
                                        streams[primary_stream_index]);
        // std::cerr << "gather candidates\n";
        readStorage.gatherSequenceDataToGpuBufferAsync(
            batch.threadPool,
            batch.candidateSequenceGatherHandle,
            dataArrays.d_candidate_sequences_data.get(),
            batch.encodedSequencePitchInInts,
            dataArrays.h_candidate_read_ids,
            dataArrays.d_candidate_read_ids,
            batch.n_queries,
            batch.deviceId,
            streams[primary_stream_index],
            transFuncData.runtimeOptions.nCorrectorThreads);

        call_transpose_kernel(
            dataArrays.d_transposedCandidateSequencesData.get(), 
            dataArrays.d_candidate_sequences_data.get(), 
            batch.n_queries, 
            batch.encodedSequencePitchInInts, 
            batch.encodedSequencePitchInInts, 
            streams[primary_stream_index]
        );


        cudaEventRecord(events[alignment_data_transfer_h2d_finished_event_index], streams[primary_stream_index]); CUERR;

        cudaStreamWaitEvent(streams[secondary_stream_index], events[alignment_data_transfer_h2d_finished_event_index], 0) ;


        cudaMemcpyAsync(dataArrays.h_candidate_sequences_lengths,
                        dataArrays.d_candidate_sequences_lengths,
                        sizeof(int) * maxCandidates,
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

    }

    void getQualities(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        DataArrays& dataArrays = batch.dataArrays;

		std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;

        const auto* gpuReadStorage = transFuncData.readStorage;

		if(transFuncData.correctionOptions.useQualityScores) {
            // std::cerr << "gather anchor qual\n";
            gpuReadStorage->gatherQualitiesToGpuBufferAsync(
                batch.threadPool,
                batch.subjectQualitiesGatherHandle,
                dataArrays.d_subject_qualities,
                batch.qualityPitchInBytes,
                dataArrays.h_subject_read_ids,
                dataArrays.d_subject_read_ids,
                batch.n_subjects,
                batch.deviceId,
                streams[primary_stream_index],
                transFuncData.runtimeOptions.nCorrectorThreads);
            // std::cerr << "gather candidate qual\n";
            gpuReadStorage->gatherQualitiesToGpuBufferAsync(
                batch.threadPool,
                batch.candidateQualitiesGatherHandle,
                dataArrays.d_candidate_qualities,
                batch.qualityPitchInBytes,
                dataArrays.h_candidate_read_ids.get(),
                dataArrays.d_candidate_read_ids.get(),
                batch.n_queries,
                batch.deviceId,
                streams[primary_stream_index],
                transFuncData.runtimeOptions.nCorrectorThreads);
        }

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
	}


	void getCandidateAlignments(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

		DataArrays& dataArrays = batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        const auto batchsize = batch.transFuncData->correctionOptions.batchsize;
        const auto maxCandidates = batch.numCandidatesLimit;
        
        auto& graphwrap = batch.alignmentGraphs[batch.graphindex];

        auto run0 = [&](){
            {
                const int* numAnchorsPtr = dataArrays.d_numAnchors.get();
                int* d_anchorIndicesOfCandidates = dataArrays.d_anchorIndicesOfCandidates.get();
                int* d_candidates_per_subject = dataArrays.d_candidates_per_subject.get();
                int* d_candidates_per_subject_prefixsum = dataArrays.d_candidates_per_subject_prefixsum.get();

                setAnchorIndicesOfCandidateskernel<1024, 128>
                        <<<1024, 128, 0, streams[primary_stream_index]>>>(
                            dataArrays.d_anchorIndicesOfCandidates.get(),
                    dataArrays.d_numAnchors.get(),
                    dataArrays.d_candidates_per_subject.get(),
                    dataArrays.d_candidates_per_subject_prefixsum.get()
                );
            }
        };

        auto run1 = [&](){
            std::size_t tempBytes = dataArrays.d_tempstorage.sizeInBytes();

            call_popcount_shifted_hamming_distance_kernel_async(
                dataArrays.d_tempstorage.get(),
                tempBytes,
                dataArrays.d_alignment_overlaps.get(),
                dataArrays.d_alignment_shifts.get(),
                dataArrays.d_alignment_nOps.get(),
                dataArrays.d_alignment_isValid.get(),
                dataArrays.d_alignment_best_alignment_flags.get(),
                dataArrays.d_subject_sequences_data.get(),
                dataArrays.d_candidate_sequences_data.get(),
                dataArrays.d_subject_sequences_lengths.get(),
                dataArrays.d_candidate_sequences_lengths.get(),
                dataArrays.d_candidates_per_subject_prefixsum.get(),
                dataArrays.h_candidates_per_subject.get(),
                dataArrays.d_candidates_per_subject.get(),
                dataArrays.d_anchorIndicesOfCandidates.get(),
                dataArrays.d_numAnchors.get(),
                dataArrays.d_numCandidates.get(),
                batchsize,
                maxCandidates,
                transFuncData.sequenceFileProperties.maxSequenceLength,
                batch.encodedSequencePitchInInts,
                transFuncData.goodAlignmentProperties.min_overlap,
                transFuncData.goodAlignmentProperties.maxErrorRate,
                transFuncData.goodAlignmentProperties.min_overlap_ratio,
                transFuncData.correctionOptions.estimatedErrorrate,
                //batch.maxSubjectLength,
                streams[primary_stream_index],
                batch.kernelLaunchHandle
            );
        };

        auto run2 = [&](){
            call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                dataArrays.getDeviceAlignmentResultPointers(),
                dataArrays.d_candidates_per_subject_prefixsum.get(),
                dataArrays.d_numAnchors.get(),
                dataArrays.d_numCandidates.get(),
                batchsize,
                maxCandidates,
                transFuncData.correctionOptions.estimatedErrorrate,
                transFuncData.correctionOptions.estimatedCoverage * transFuncData.correctionOptions.m_coverage,
                streams[primary_stream_index],
                batch.kernelLaunchHandle
            );
        };

        auto run3 = [&](){
            callSelectIndicesOfGoodCandidatesKernelAsync(
                dataArrays.d_indices.get(),
                dataArrays.d_indices_per_subject.get(),
                dataArrays.d_num_indices.get(),
                dataArrays.d_alignment_best_alignment_flags.get(),
                dataArrays.d_candidates_per_subject.get(),
                dataArrays.d_candidates_per_subject_prefixsum.get(),
                dataArrays.d_anchorIndicesOfCandidates.get(),
                dataArrays.d_numAnchors.get(),
                dataArrays.d_numCandidates.get(),
                batchsize,
                maxCandidates,
                streams[primary_stream_index],
                batch.kernelLaunchHandle
            );
        };

        auto run = [&](){
            run0();
            run1();
            run2();
            run3();
        };

        // if(!graphwrap.valid){
        //     std::cerr << "rebuild alignmentgraph\n";

        //     graphwrap.d_numAnchors = dataArrays.d_numAnchors.get();
        //     graphwrap.d_anchorIndicesOfCandidates = dataArrays.d_anchorIndicesOfCandidates.get();
        //     graphwrap.d_candidates_per_subject = dataArrays.d_candidates_per_subject.get();
        //     graphwrap.d_candidates_per_subject_prefixsum = dataArrays.d_candidates_per_subject_prefixsum.get();
    

        //     if(graphwrap.execgraph != nullptr){
        //         cudaGraphExecDestroy(graphwrap.execgraph); CUERR;
        //     }
            
        //     cudaStreamBeginCapture(streams[primary_stream_index], cudaStreamCaptureModeRelaxed); CUERR;

        //     run0();
        //     run1();
        //     run2();
        //     run3();

        //     cudaGraph_t graph;
        //     cudaStreamEndCapture(streams[primary_stream_index], &graph); CUERR;
            
        //     cudaGraphExec_t execGraph;
        //     cudaGraphNode_t errorNode;
        //     auto logBuffer = std::make_unique<char[]>(1025);
        //     std::fill_n(logBuffer.get(), 1025, 0);
        //     cudaError_t status = cudaGraphInstantiate(&execGraph, graph, &errorNode, logBuffer.get(), 1025);
        //     if(status != cudaSuccess){
        //         if(logBuffer[1024] != '\0'){
        //             std::cerr << "cudaGraphInstantiate: truncated error message: ";
        //             std::copy_n(logBuffer.get(), 1025, std::ostream_iterator<char>(std::cerr, ""));
        //             std::cerr << "\n";
        //         }else{
        //             std::cerr << "cudaGraphInstantiate: error message: ";
        //             std::cerr << logBuffer.get();
        //             std::cerr << "\n";
        //         }
        //         CUERR;
        //     }            

        //     cudaGraphDestroy(graph); CUERR;

        //     graphwrap.execgraph = execGraph;

        //     graphwrap.valid = true;
        // }

        // assert(graphwrap.valid);

        // assert(graphwrap.d_numAnchors == dataArrays.d_numAnchors.get());
        // assert(graphwrap.d_anchorIndicesOfCandidates == dataArrays.d_anchorIndicesOfCandidates.get());
        // assert(graphwrap.d_candidates_per_subject == dataArrays.d_candidates_per_subject.get());
        // assert(graphwrap.d_candidates_per_subject_prefixsum == dataArrays.d_candidates_per_subject_prefixsum.get());


        run();
        //cudaGraphLaunch(graphwrap.execgraph, streams[primary_stream_index]); CUERR;



        // cudaMemcpyAsync(dataArrays.h_num_indices,
        //                 dataArrays.d_num_indices,
        //                 sizeof(int),
        //                 D2H,
        //                 streams[primary_stream_index]); CUERR;       

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

        //std::cerr << "After alignment: " << *dataArrays.h_num_indices << " / " << dataArrays.n_queries << "\n";
	}

    void buildMultipleSequenceAlignment(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        DataArrays& dataArrays = batch.dataArrays;

        const auto batchsize = batch.transFuncData->correctionOptions.batchsize;
        const auto maxCandidates = batch.numCandidatesLimit;

		const float desiredAlignmentMaxErrorRate = transFuncData.goodAlignmentProperties.maxErrorRate;
        //const float desiredAlignmentMaxErrorRate = transFuncData.correctionOptions.estimatedErrorrate * 4.0f;

        //std::cout << "msa_init" << std::endl;


        callBuildMSAKernel_async(
            dataArrays.d_msa_column_properties.get(),
            dataArrays.d_counts.get(),
            dataArrays.d_weights.get(),
            dataArrays.d_coverage.get(),
            dataArrays.d_origWeights.get(),
            dataArrays.d_origCoverages.get(),
            dataArrays.d_support.get(),
            dataArrays.d_consensus.get(),
            dataArrays.d_alignment_overlaps.get(),
            dataArrays.d_alignment_shifts.get(),
            dataArrays.d_alignment_nOps.get(),
            dataArrays.d_alignment_best_alignment_flags.get(),
            dataArrays.d_subject_sequences_data.get(),
            dataArrays.d_subject_sequences_lengths.get(),
            dataArrays.d_transposedCandidateSequencesData.get(),
            dataArrays.d_candidate_sequences_lengths.get(),
            dataArrays.d_subject_qualities.get(),
            dataArrays.d_candidate_qualities.get(),
            transFuncData.correctionOptions.useQualityScores,
            batch.encodedSequencePitchInInts,
            batch.qualityPitchInBytes,
            batch.msa_pitch,
            batch.msa_weights_pitch / sizeof(float),
            dataArrays.d_indices,
            dataArrays.d_indices_per_subject,
            dataArrays.d_candidates_per_subject_prefixsum,
            dataArrays.d_numAnchors.get(),
            dataArrays.d_numCandidates.get(),
            batchsize,
            maxCandidates,
            dataArrays.d_canExecute,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );



        //batch.dataArrays.copyEverythingToHostForDebugging();

        //At this point the msa is built

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
	}





    void removeCandidatesOfDifferentRegionFromMSA(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        DataArrays& dataArrays = batch.dataArrays;
        const auto batchsize = batch.transFuncData->correctionOptions.batchsize;
        const auto maxCandidates = batch.numCandidatesLimit;
        

        const float desiredAlignmentMaxErrorRate = transFuncData.goodAlignmentProperties.maxErrorRate;
        //const float desiredAlignmentMaxErrorRate = transFuncData.correctionOptions.estimatedErrorrate * 4.0f;

        /*bool* d_shouldBeKept = nullptr; //flag per candidate which shows whether the candidate should remain in the msa, or not.

        cubCachingAllocator.DeviceAllocate(
            (void**)&d_shouldBeKept, 
            sizeof(bool) * batch.n_queries, 
            streams[primary_stream_index]
        ); CUERR;*/
        
        assert(batch.dataArrays.d_tempstorage.sizeInBytes() >= sizeof(bool) * batch.n_queries);
        
        bool* d_shouldBeKept = (bool*)batch.dataArrays.d_tempstorage.get();



        // int* fooindices;
        // int* fooindicespersubject;
        // int* foonumindices;

        // cudaMallocManaged(&fooindices, sizeof(int) *batch.n_queries); CUERR;
        // cudaMallocManaged(&fooindicespersubject, sizeof(int) *batch.n_subjects); CUERR;
        // cudaMallocManaged(&foonumindices, sizeof(int)); CUERR;

        std::array<int*,2> d_indices_dblbuf{
            dataArrays.d_indices.get(), 
            dataArrays.d_indices_tmp.get()
        };
        std::array<int*,2> d_indices_per_subject_dblbuf{
            dataArrays.d_indices_per_subject.get(), 
            dataArrays.d_indices_per_subject_tmp.get()
        };
        std::array<int*,2> d_num_indices_dblbuf{
            dataArrays.d_num_indices.get(), 
            dataArrays.d_num_indices_tmp.get()
        };

        for(int iteration = 0; iteration < max_num_minimizations; iteration++){
#if 0
            {
                //Initialize d_shouldBeKept array

                const int N = batch.n_queries;
                bool* d_canExecute = dataArrays.d_canExecute.get();
                generic_kernel<<<SDIV(batch.n_queries, 128), 128, 0, streams[primary_stream_index]>>>(
                    [=] __device__ (){
                        if(*d_canExecute){
                            const int index = threadIdx.x + blockIdx.x * 128;
                            if(index < N){
                                d_shouldBeKept[index] = false;
                            }
                        }
                    }
                ); CUERR;
            }


            //select candidates which are to be removed
            call_msa_findCandidatesOfDifferentRegion_kernel_async(
                dataArrays.d_indices_tmp.get(),
                dataArrays.d_indices_per_subject_tmp.get(),
                dataArrays.d_num_indices_tmp.get(),
                dataArrays.getDeviceMSAPointers(),
                dataArrays.getDeviceAlignmentResultPointers(),
                dataArrays.getDeviceSequencePointers(),
                d_shouldBeKept,
                dataArrays.d_candidates_per_subject_prefixsum,
                batch.n_subjects,
                batch.n_queries,
                batch.encodedSequencePitchInInts,
                batch.msa_pitch,
                batch.msa_weights_pitch,
                dataArrays.d_indices,
                dataArrays.d_indices_per_subject,
                transFuncData.correctionOptions.estimatedCoverage,
                dataArrays.d_canExecute.get(),
                streams[primary_stream_index],
                batch.kernelLaunchHandle,
                dataArrays.d_subject_read_ids,
                false
            );  CUERR;


            // cudaDeviceSynchronize(); CUERR;

            // // auto shouldbekept = std::make_unique<bool[]>(batch.n_queries);
            // // cudaMemcpy(shouldbekept.get(), d_shouldBeKept, sizeof(bool) * batch.n_queries, D2H); CUERR;

            // // cudaDeviceSynchronize(); CUERR;

            // std::vector<int> updatedindices(batch.n_queries);
            // std::vector<int> updatedindicespersubject(batch.n_subjects);
            // std::vector<int> updatednumindices(1);
            
            // cudaMemcpy(updatedindices.data(), d_newIndices, sizeof(int) * batch.n_queries, D2H); CUERR;            
            // cudaMemcpy(updatedindicespersubject.data(), dataArrays.d_indices_per_subject.get(), sizeof(int) * batch.n_subjects, D2H); CUERR;            
            // cudaMemcpy(updatednumindices.data(), dataArrays.d_num_indices_tmp.get(), sizeof(int), D2H); CUERR;

            // cudaDeviceSynchronize(); CUERR;

            

            // std::cerr << "old indices per subject: ";
            // for(int i = 0; i < 10; i++){
            //     std::cerr << dataArrays.h_indices_per_subject[i] << " ";
            // }
            // std::cerr << "\n";

            // std::cerr << "old num indices: ";
            // std::cerr << *dataArrays.h_num_indices;
            // std::cerr << "\n";

            // std::cerr << "upd indices per subject: ";
            // for(int i = 0; i < 10; i++){
            //     std::cerr << updatedindicespersubject[i] << " ";
            // }
            // std::cerr << "\n";

            // std::cerr << "upd num indices: ";
            // std::cerr << updatednumindices[0];
            // std::cerr << "\n";

            // std::exit(0);

            {

                /*
                    copy new indicesPerSubject (d_indices_per_subject_tmp) to old indicesPerSubject (d_indices_per_subject)
                    if new value is equal to old value, set new value to 0
                */

                int* d_indices_per_subject = dataArrays.d_indices_per_subject.get();
                int* d_indices_per_subject_tmp = dataArrays.d_indices_per_subject_tmp.get();
                bool* d_canExecute = dataArrays.d_canExecute.get();
                const int n_subjects = batch.n_subjects;
                cudaStream_t stream = streams[primary_stream_index];

                dim3 block(128,1,1);
                dim3 grid(SDIV(batch.n_subjects, block.x),1,1);
                generic_kernel<<<grid, block, 0, stream>>>(
                    [=] __device__ (){
                        if(*d_canExecute){
                            const int tid = threadIdx.x + blockDim.x * blockIdx.x;
                            if(tid < n_subjects){
                                const int oldValue = d_indices_per_subject[tid];
                                const int newValue = d_indices_per_subject_tmp[tid];

                                d_indices_per_subject[tid] = newValue;
                                
                                if(oldValue == newValue){
                                    d_indices_per_subject_tmp[tid] = 0;
                                }
                            }
                        }
                    }
                ); CUERR;
            }

            {
                //set d_canExecute flag. reconstructing the msa and performing another minimization step 
                // is only neccessary if the indices changed, and if there are any indices left.

                const int* d_num_indices_tmp = dataArrays.d_num_indices_tmp.get();
                const int* d_num_indices = dataArrays.d_num_indices.get();
                bool* d_canExecute = dataArrays.d_canExecute.get();

                generic_kernel<<<1,1, 0, streams[primary_stream_index]>>>(
                    [=] __device__ (){
                        if(*d_num_indices_tmp > *d_num_indices){
                            printf("%d %d\n", *d_num_indices_tmp, *d_num_indices);
                            assert(*d_num_indices_tmp <= *d_num_indices);
                        }

                        if(*d_num_indices_tmp > 0 && *d_num_indices_tmp < *d_num_indices){
                            *d_canExecute = true;
                        }else{
                            *d_canExecute = false;
                        }
                    }
                ); CUERR;

            }

            
            std::swap(dataArrays.d_indices, dataArrays.d_indices_tmp);
            std::swap(dataArrays.d_num_indices_tmp, dataArrays.d_num_indices);

            callBuildMSAKernel_async(
                dataArrays.d_msa_column_properties.get(),
                dataArrays.d_counts.get(),
                dataArrays.d_weights.get(),
                dataArrays.d_coverage.get(),
                dataArrays.d_origWeights.get(),
                dataArrays.d_origCoverages.get(),
                dataArrays.d_support.get(),
                dataArrays.d_consensus.get(),
                dataArrays.d_alignment_overlaps.get(),
                dataArrays.d_alignment_shifts.get(),
                dataArrays.d_alignment_nOps.get(),
                dataArrays.d_alignment_best_alignment_flags.get(),
                dataArrays.d_subject_sequences_data.get(),
                dataArrays.d_subject_sequences_lengths.get(),
                dataArrays.d_transposedCandidateSequencesData.get(),
                dataArrays.d_candidate_sequences_lengths.get(),
                dataArrays.d_subject_qualities.get(),
                dataArrays.d_candidate_qualities.get(),
                transFuncData.correctionOptions.useQualityScores,
                batch.encodedSequencePitchInInts,
                batch.qualityPitchInBytes,
                batch.msa_pitch,
                batch.msa_weights_pitch / sizeof(float),
                dataArrays.d_indices,
                dataArrays.d_indices_per_subject_tmp,
                dataArrays.d_candidates_per_subject_prefixsum,
                dataArrays.d_numAnchors.get(),
                dataArrays.d_numCandidates.get(),
                batch.n_subjects,
                batch.n_queries,
                dataArrays.d_canExecute,
                streams[primary_stream_index],
                batch.kernelLaunchHandle
            );


#else 
            callMsaFindCandidatesOfDifferentRegionAndRemoveThemKernel_async(
                d_indices_dblbuf[(1 + iteration) % 2],
                d_indices_per_subject_dblbuf[(1 + iteration) % 2],
                d_num_indices_dblbuf[(1 + iteration) % 2],
                dataArrays.d_msa_column_properties.get(),
                dataArrays.d_consensus.get(),
                dataArrays.d_coverage.get(),
                dataArrays.d_counts.get(),
                dataArrays.d_weights.get(),
                dataArrays.d_support.get(),
                dataArrays.d_origCoverages.get(),
                dataArrays.d_origWeights.get(),
                dataArrays.d_alignment_best_alignment_flags.get(),
                dataArrays.d_alignment_shifts.get(),
                dataArrays.d_alignment_nOps.get(),
                dataArrays.d_alignment_overlaps.get(),
                dataArrays.d_subject_sequences_data.get(),
                dataArrays.d_candidate_sequences_data.get(),
                dataArrays.d_transposedCandidateSequencesData.get(),
                dataArrays.d_subject_sequences_lengths.get(),
                dataArrays.d_candidate_sequences_lengths.get(),
                dataArrays.d_subject_qualities.get(),
                dataArrays.d_candidate_qualities.get(),
                d_shouldBeKept,
                dataArrays.d_candidates_per_subject_prefixsum,
                dataArrays.d_numAnchors.get(),
                dataArrays.d_numCandidates.get(),
                batchsize,
                maxCandidates,
                transFuncData.correctionOptions.useQualityScores,
                batch.encodedSequencePitchInInts,
                batch.qualityPitchInBytes,
                batch.msa_pitch,
                batch.msa_weights_pitch / sizeof(float),
                d_indices_dblbuf[(0 + iteration) % 2],
                d_indices_per_subject_dblbuf[(0 + iteration) % 2],
                transFuncData.correctionOptions.estimatedCoverage,
                dataArrays.d_canExecute,
                iteration,
                dataArrays.d_subject_read_ids.get(),
                streams[primary_stream_index],
                batch.kernelLaunchHandle
            );

            // callMsaFindCandidatesOfDifferentRegionAndRemoveThemKernel_async(
            //     dataArrays.d_indices_tmp.get(),
            //     dataArrays.d_indices_per_subject_tmp.get(),
            //     dataArrays.d_num_indices_tmp.get(),
            //     dataArrays.d_msa_column_properties.get(),
            //     dataArrays.d_consensus.get(),
            //     dataArrays.d_coverage.get(),
            //     dataArrays.d_counts.get(),
            //     dataArrays.d_weights.get(),
            //     dataArrays.d_support.get(),
            //     dataArrays.d_origCoverages.get(),
            //     dataArrays.d_origWeights.get(),
            //     dataArrays.d_alignment_best_alignment_flags.get(),
            //     dataArrays.d_alignment_shifts.get(),
            //     dataArrays.d_alignment_nOps.get(),
            //     dataArrays.d_alignment_overlaps.get(),
            //     dataArrays.d_subject_sequences_data.get(),
            //     dataArrays.d_candidate_sequences_data.get(),
            //     dataArrays.d_transposedCandidateSequencesData.get(),
            //     dataArrays.d_subject_sequences_lengths.get(),
            //     dataArrays.d_candidate_sequences_lengths.get(),
            //     dataArrays.d_subject_qualities.get(),
            //     dataArrays.d_candidate_qualities.get(),
            //     d_shouldBeKept,
            //     dataArrays.d_candidates_per_subject_prefixsum,
            //     dataArrays.d_numAnchors.get(),
            //     dataArrays.d_numCandidates.get(),
            //     batch.n_subjects,
            //     batch.n_queries,
            //     transFuncData.correctionOptions.useQualityScores,
            //     batch.encodedSequencePitchInInts,
            //     batch.qualityPitchInBytes,
            //     batch.msa_pitch,
            //     batch.msa_weights_pitch / sizeof(float),
            //     dataArrays.d_indices,
            //     dataArrays.d_indices_per_subject,
            //     transFuncData.correctionOptions.estimatedCoverage,
            //     dataArrays.d_canExecute,
            //     iteration,
            //     dataArrays.d_subject_read_ids.get(),
            //     streams[primary_stream_index],
            //     batch.kernelLaunchHandle
            // );

            // std::swap(dataArrays.d_indices, dataArrays.d_indices_tmp);
            // std::swap(dataArrays.d_indices_per_subject, dataArrays.d_indices_per_subject_tmp);
            // std::swap(dataArrays.d_num_indices_tmp, dataArrays.d_num_indices);

        }


#endif
        //cubCachingAllocator.DeviceFree(d_shouldBeKept); CUERR;
        
        //At this point the msa is built, maybe minimized, and is ready to be used for correction

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
    }


	void correctSubjects(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;

        cudaSetDevice(batch.deviceId); CUERR;

		DataArrays& dataArrays = batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;
        const auto batchsize = batch.transFuncData->correctionOptions.batchsize;
        const auto maxCandidates = batch.numCandidatesLimit;

		const float avg_support_threshold = 1.0f-1.0f*transFuncData.correctionOptions.estimatedErrorrate;
		const float min_support_threshold = 1.0f-3.0f*transFuncData.correctionOptions.estimatedErrorrate;
		// coverage is always >= 1
		const float min_coverage_threshold = std::max(1.0f,
					transFuncData.correctionOptions.m_coverage / 6.0f * transFuncData.correctionOptions.estimatedCoverage);
        const float max_coverage_threshold = 0.5 * transFuncData.correctionOptions.estimatedCoverage;

		// correct subjects
#if 0
        cudaMemcpyAsync(dataArrays.h_num_indices,
                        dataArrays.d_num_indices,
                        dataArrays.d_num_indices.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_indices,
                        dataArrays.d_indices,
                        dataArrays.d_indices.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_indices_per_subject,
                        dataArrays.d_indices_per_subject,
                        dataArrays.d_indices_per_subject.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_msa_column_properties,
                        dataArrays.d_msa_column_properties,
                        dataArrays.d_msa_column_properties.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_alignment_shifts,
                        dataArrays.d_alignment_shifts,
                        dataArrays.d_alignment_shifts.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_alignment_best_alignment_flags,
                        dataArrays.d_alignment_best_alignment_flags,
                        dataArrays.d_alignment_best_alignment_flags.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_subject_sequences_data,
                        dataArrays.d_subject_sequences_data,
                        dataArrays.d_subject_sequences_data.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_candidate_sequences_data,
                        dataArrays.d_candidate_sequences_data,
                        dataArrays.d_candidate_sequences_data.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_consensus,
                        dataArrays.d_consensus,
                        dataArrays.d_consensus.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaDeviceSynchronize(); CUERR;

        auto identity = [](auto i){return i;};


        for(int i = 0; i < dataArrays.n_subjects; i++){
            std::cout << "Subject  : " << batch.tasks[i].subject_string << ", subject id " <<  batch.tasks[i].readId << std::endl;
            int numind = dataArrays.h_indices_per_subject[i];
            if(numind > 0){
                std::vector<char> cands;
                cands.resize(128 * numind, 'F');
                std::vector<int> candlengths;
                candlengths.resize(numind);
                std::vector<int> candshifts;
                candshifts.resize(numind);
                for(int j = 0; j < numind; j++){
                    int index = dataArrays.h_indices[dataArrays.h_indices_per_subject_prefixsum[i] + j];
                    char* dst = cands.data() + 128 * j;
                    candlengths[j] = dataArrays.h_candidate_sequences_lengths[index];
                    candshifts[j] = dataArrays.h_alignment_shifts[index];

                    const unsigned int* candidateSequencePtr = dataArrays.h_candidate_sequences_data.get() + index * batch.encodedSequencePitchInInts;

                    assert(dataArrays.h_alignment_best_alignment_flags[index] != BestAlignment_t::None);

                    std::string candidatestring = get2BitString((unsigned int*)candidateSequencePtr, dataArrays.h_candidate_sequences_lengths[index]);
                    if(dataArrays.h_alignment_best_alignment_flags[index] == BestAlignment_t::ReverseComplement){
                        candidatestring = reverseComplementString(candidatestring.c_str(), candidatestring.length());
                    }

                    std::copy(candidatestring.begin(), candidatestring.end(), dst);
                    //decode2BitSequence(dst, (const unsigned int*)candidateSequencePtr, 100, identity);
                    //std::cout << "Candidate: " << s << std::endl;
                }

                printSequencesInMSA(std::cout,
                                         batch.tasks[i].subject_string.c_str(),
                                         dataArrays.h_subject_sequences_lengths[i],
                                         cands.data(),
                                         candlengths.data(),
                                         numind,
                                         candshifts.data(),
                                         dataArrays.h_msa_column_properties[i].subjectColumnsBegin_incl,
                                         dataArrays.h_msa_column_properties[i].subjectColumnsEnd_excl,
                                         dataArrays.h_msa_column_properties[i].lastColumn_excl - dataArrays.h_msa_column_properties[i].firstColumn_incl,
                                         128);
                 std::cout << "\n";
                 std::string consensus = std::string{dataArrays.h_consensus + i * batch.msa_pitch, dataArrays.h_consensus + (i+1) * batch.msa_pitch};
                 std::cout << "Consensus:\n   " << consensus << "\n\n";
                 printSequencesInMSAConsEq(std::cout,
                                          batch.tasks[i].subject_string.c_str(),
                                          dataArrays.h_subject_sequences_lengths[i],
                                          cands.data(),
                                          candlengths.data(),
                                          numind,
                                          candshifts.data(),
                                          consensus.c_str(),
                                          dataArrays.h_msa_column_properties[i].subjectColumnsBegin_incl,
                                          dataArrays.h_msa_column_properties[i].subjectColumnsEnd_excl,
                                          dataArrays.h_msa_column_properties[i].lastColumn_excl - dataArrays.h_msa_column_properties[i].firstColumn_incl,
                                          128);
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

        cudaMemcpyAsync(dataArrays.h_msa_column_properties,
            dataArrays.d_msa_column_properties,
            dataArrays.d_msa_column_properties.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(dataArrays.h_counts,
            dataArrays.d_counts,
            dataArrays.d_counts.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(dataArrays.h_weights,
            dataArrays.d_weights,
            dataArrays.d_weights.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(dataArrays.h_coverage,
            dataArrays.d_coverage,
            dataArrays.d_coverage.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(dataArrays.h_origWeights,
            dataArrays.d_origWeights,
            dataArrays.d_origWeights.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(dataArrays.h_origCoverages,
            dataArrays.d_origCoverages,
            dataArrays.d_origCoverages.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(dataArrays.h_support,
            dataArrays.d_support,
            dataArrays.d_support.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(dataArrays.h_consensus,
            dataArrays.d_consensus,
            dataArrays.d_consensus.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;
        cudaDeviceSynchronize();

        std::size_t msa_weights_row_pitch_floats = batch.msa_weights_pitch / sizeof(float);
        std::size_t msa_row_pitch = batch.msa_pitch;

        for(int i = 0; i < batch.n_subjects; i++){
            if(dataArrays.h_subject_read_ids[i] == 13){
                std::cerr << "subjectColumnsBegin_incl = " << dataArrays.h_msa_column_properties[i].subjectColumnsBegin_incl << "\n";
                std::cerr << "subjectColumnsEnd_excl = " << dataArrays.h_msa_column_properties[i].subjectColumnsEnd_excl << "\n";
                std::cerr << "lastColumn_excl = " << dataArrays.h_msa_column_properties[i].lastColumn_excl << "\n";
                std::cerr << "counts: \n";
                int* counts = dataArrays.h_counts + i * 4 * msa_weights_row_pitch_floats;
                for(int k = 0; k < msa_weights_row_pitch_floats; k++){
                    std::cerr << counts[0 * msa_weights_row_pitch_floats + k];
                }
                std::cerr << "\n";
                for(int k = 0; k < msa_weights_row_pitch_floats; k++){
                    std::cerr << counts[1 * msa_weights_row_pitch_floats + k];
                }
                std::cerr << "\n";
                for(int k = 0; k < msa_weights_row_pitch_floats; k++){
                    std::cerr << counts[2 * msa_weights_row_pitch_floats + k];
                }
                std::cerr << "\n";
                for(int k = 0; k < msa_weights_row_pitch_floats; k++){
                    std::cerr << counts[3 * msa_weights_row_pitch_floats + k];
                }
                std::cerr << "\n";

                std::cerr << "weights: \n";
                float* weights = dataArrays.h_weights + i * 4 * msa_weights_row_pitch_floats;
                for(int k = 0; k < msa_weights_row_pitch_floats; k++){
                    std::cerr << weights[0 * msa_weights_row_pitch_floats + k];
                }
                std::cerr << "\n";
                for(int k = 0; k < msa_weights_row_pitch_floats; k++){
                    std::cerr << weights[1 * msa_weights_row_pitch_floats + k];
                }
                std::cerr << "\n";
                for(int k = 0; k < msa_weights_row_pitch_floats; k++){
                    std::cerr << weights[2 * msa_weights_row_pitch_floats + k];
                }
                std::cerr << "\n";
                for(int k = 0; k < msa_weights_row_pitch_floats; k++){
                    std::cerr << weights[3 * msa_weights_row_pitch_floats + k];
                }
                std::cerr << "\n";

                std::cerr << "coverage: \n";
                int* coverage = dataArrays.h_coverage + i * msa_weights_row_pitch_floats;
                for(int k = 0; k < msa_weights_row_pitch_floats; k++){
                    std::cerr << coverage[k];
                }
                std::cerr << "\n";

                std::cerr << "support: \n";
                float* support = dataArrays.h_support + i * msa_weights_row_pitch_floats;
                for(int k = 0; k < msa_weights_row_pitch_floats; k++){
                    std::cerr << support[k];
                }
                std::cerr << "\n";

                std::cerr << "consensus: \n";
                char* consensus = dataArrays.h_consensus + i * msa_row_pitch;
                for(int k = 0; k < msa_row_pitch; k++){
                    std::cerr << consensus[k];
                }
                std::cerr << "\n";
            }
        }

        
#endif        
        // cudaEventRecord(events[msa_build_finished_event_index], streams[primary_stream_index]); CUERR;
        // cudaStreamWaitEvent(streams[secondary_stream_index], events[msa_build_finished_event_index], 0); CUERR;

        std::array<int*,2> d_indices_dblbuf{
            dataArrays.d_indices.get(), 
            dataArrays.d_indices_tmp.get()
        };
        std::array<int*,2> d_indices_per_subject_dblbuf{
            dataArrays.d_indices_per_subject.get(), 
            dataArrays.d_indices_per_subject_tmp.get()
        };
        std::array<int*,2> d_num_indices_dblbuf{
            dataArrays.d_num_indices.get(), 
            dataArrays.d_num_indices_tmp.get()
        };

        const int* d_indices = d_indices_dblbuf[max_num_minimizations % 2];
        const int* d_indices_per_subject = d_indices_per_subject_dblbuf[max_num_minimizations % 2];
        const int* d_num_indices = d_num_indices_dblbuf[max_num_minimizations % 2];

        call_msa_correct_subject_implicit_kernel_async(
            dataArrays.getDeviceMSAPointers(),
            dataArrays.getDeviceAlignmentResultPointers(),
            dataArrays.getDeviceSequencePointers(),
            dataArrays.getDeviceCorrectionResultPointers(),
            d_indices,
            d_indices_per_subject,
            dataArrays.d_numAnchors.get(),
            batchsize,
            batch.encodedSequencePitchInInts,
            batch.decodedSequencePitchInBytes,
            batch.msa_pitch,
            batch.msa_weights_pitch,
            transFuncData.sequenceFileProperties.maxSequenceLength,
            transFuncData.correctionOptions.estimatedErrorrate,
            transFuncData.goodAlignmentProperties.maxErrorRate,
            avg_support_threshold,
            min_support_threshold,
            min_coverage_threshold,
            max_coverage_threshold,
            transFuncData.correctionOptions.kmerlength,
            transFuncData.sequenceFileProperties.maxSequenceLength,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );

        cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], events[correction_finished_event_index], 0); CUERR;

        //std::cerr << "cudaMemcpyAsync " << (void*)dataArrays.h_indices_per_subject.get() << (void*) d_indices_per_subject << "\n";
        cudaMemcpyAsync(
            dataArrays.h_indices_per_subject,
            d_indices_per_subject,
            sizeof(int) * batchsize,
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(dataArrays.h_corrected_subjects,
                        dataArrays.d_corrected_subjects,
                        batch.decodedSequencePitchInBytes * batchsize,
                        D2H,
                        streams[secondary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_subject_is_corrected,
                        dataArrays.d_subject_is_corrected,
                        sizeof(bool) * batchsize,
                        D2H,
                        streams[secondary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_is_high_quality_subject,
                        dataArrays.d_is_high_quality_subject,
                        sizeof(AnchorHighQualityFlag) * batchsize,
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        // cudaMemcpyAsync(dataArrays.h_num_uncorrected_positions_per_subject,
        //                 dataArrays.d_num_uncorrected_positions_per_subject,
        //                 sizeof(int) * batchsize,
        //                 D2H,
        //                 streams[secondary_stream_index]); CUERR;

        // cudaMemcpyAsync(dataArrays.h_uncorrected_positions_per_subject,
        //                 dataArrays.d_uncorrected_positions_per_subject,
        //                 sizeof(int) * transFuncData.sequenceFileProperties.maxSequenceLength * batchsize,
        //                 D2H,
        //                 streams[secondary_stream_index]); CUERR;

        selectIndicesOfFlagsOnlyOneBlock<256><<<1,256,0, streams[primary_stream_index]>>>(
            dataArrays.d_indices_of_corrected_subjects.get(),
            dataArrays.d_num_indices_of_corrected_subjects.get(),
            dataArrays.d_subject_is_corrected.get(),
            dataArrays.d_numAnchors.get()
        );

        callConstructAnchorResultsKernelAsync(
            dataArrays.d_editsPerCorrectedSubject.get(),
            dataArrays.d_numEditsPerCorrectedSubject.get(),
            doNotUseEditsValue,
            dataArrays.d_indices_of_corrected_subjects.get(),
            dataArrays.d_num_indices_of_corrected_subjects.get(),
            dataArrays.d_anchorContainsN.get(),
            dataArrays.d_subject_sequences_data.get(),
            dataArrays.d_subject_sequences_lengths.get(),
            dataArrays.d_corrected_subjects.get(),
            batch.maxNumEditsPerSequence,
            batch.encodedSequencePitchInInts,
            batch.decodedSequencePitchInBytes,
            dataArrays.d_numAnchors.get(),
            batch.transFuncData->correctionOptions.batchsize,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );

        cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], events[correction_finished_event_index], 0); CUERR;

        cudaMemcpyAsync(
            dataArrays.h_editsPerCorrectedSubject,
            dataArrays.d_editsPerCorrectedSubject,
            sizeof(TempCorrectedSequence::Edit) * batch.maxNumEditsPerSequence * batchsize,
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(
            dataArrays.h_numEditsPerCorrectedSubject,
            dataArrays.d_numEditsPerCorrectedSubject,
            sizeof(int) * batchsize,
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        // cudaMemcpyAsync(
        //     dataArrays.h_indices_of_corrected_subjects,
        //     dataArrays.d_indices_of_corrected_subjects,
        //     sizeof(int) * batchsize,
        //     D2H,
        //     streams[secondary_stream_index]
        // ); CUERR;

        // cudaMemcpyAsync(
        //     dataArrays.h_num_indices_of_corrected_subjects,
        //     dataArrays.d_num_indices_of_corrected_subjects,
        //     sizeof(int),
        //     D2H,
        //     streams[secondary_stream_index]
        // ); CUERR;

		//cudaEventRecord(events[result_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;

		//if(transFuncData.correctionOptions.correctCandidates) {
            // find subject ids of subjects with high quality multiple sequence alignment

            // auto isHqSubject = [] __device__ (const AnchorHighQualityFlag& flag){
            //     return flag.hq();
            // };

            // cub::TransformInputIterator<bool,decltype(isHqSubject), AnchorHighQualityFlag*>
            //     d_isHqSubject(dataArrays.d_is_high_quality_subject,
            //                     isHqSubject);

            // selectIndicesOfFlagsOnlyOneBlock<256><<<1,256,0, streams[primary_stream_index]>>>(
            //     dataArrays.d_high_quality_subject_indices.get(),
            //     dataArrays.d_num_high_quality_subject_indices.get(),
            //     d_isHqSubject,
            //     dataArrays.d_numAnchors.get()
            // );

            // cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
            // cudaStreamWaitEvent(streams[secondary_stream_index], events[correction_finished_event_index], 0); CUERR;

            // cudaMemcpyAsync(dataArrays.h_high_quality_subject_indices,
            //                 dataArrays.d_high_quality_subject_indices,
            //                 sizeof(int) * batchsize,
            //                 D2H,
            //                 streams[secondary_stream_index]); CUERR;

            // cudaMemcpyAsync(dataArrays.h_num_high_quality_subject_indices,
            //                 dataArrays.d_num_high_quality_subject_indices,
            //                 sizeof(int),
            //                 D2H,
            //                 streams[secondary_stream_index]); CUERR;
		//}

        
	}



    void correctCandidates(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;

        cudaSetDevice(batch.deviceId); CUERR;

        DataArrays& dataArrays = batch.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        const auto batchsize = batch.transFuncData->correctionOptions.batchsize;
        const auto maxCandidates = batch.numCandidatesLimit;

        const float min_support_threshold = 1.0f-3.0f*transFuncData.correctionOptions.estimatedErrorrate;
        // coverage is always >= 1
        const float min_coverage_threshold = std::max(1.0f,
                    transFuncData.correctionOptions.m_coverage / 6.0f * transFuncData.correctionOptions.estimatedCoverage);
        const int new_columns_to_correct = transFuncData.correctionOptions.new_columns_to_correct;


        bool* const d_candidateCanBeCorrected = dataArrays.d_alignment_isValid.get(); //repurpose

        int* const d_num_corrected_candidates_per_anchor = dataArrays.d_num_corrected_candidates_per_anchor.get();
        const int* const d_numAnchors = dataArrays.d_numAnchors.get();
        const int* const d_numCandidates = dataArrays.d_numCandidates.get();

        auto isHqSubject = [] __device__ (const AnchorHighQualityFlag& flag){
            return flag.hq();
        };

        cub::TransformInputIterator<bool,decltype(isHqSubject), AnchorHighQualityFlag*>
            d_isHqSubject(dataArrays.d_is_high_quality_subject,
                            isHqSubject);

        selectIndicesOfFlagsOnlyOneBlock<256><<<1,256,0, streams[primary_stream_index]>>>(
            dataArrays.d_high_quality_subject_indices.get(),
            dataArrays.d_num_high_quality_subject_indices.get(),
            d_isHqSubject,
            dataArrays.d_numAnchors.get()
        );

        cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], events[correction_finished_event_index], 0); CUERR;

        cudaMemcpyAsync(dataArrays.h_high_quality_subject_indices,
                        dataArrays.d_high_quality_subject_indices,
                        sizeof(int) * batchsize,
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_num_high_quality_subject_indices,
                        dataArrays.d_num_high_quality_subject_indices,
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
            dataArrays.d_indices.get(), 
            dataArrays.d_indices_tmp.get()
        };
        std::array<int*,2> d_indices_per_subject_dblbuf{
            dataArrays.d_indices_per_subject.get(), 
            dataArrays.d_indices_per_subject_tmp.get()
        };
        std::array<int*,2> d_num_indices_dblbuf{
            dataArrays.d_num_indices.get(), 
            dataArrays.d_num_indices_tmp.get()
        };

        const int* d_indices = d_indices_dblbuf[max_num_minimizations % 2];
        const int* d_indices_per_subject = d_indices_per_subject_dblbuf[max_num_minimizations % 2];
        const int* d_num_indices = d_num_indices_dblbuf[max_num_minimizations % 2];

        callFlagCandidatesToBeCorrectedKernel_async(
            d_candidateCanBeCorrected,
            dataArrays.d_num_corrected_candidates_per_anchor.get(),
            dataArrays.d_support.get(),
            dataArrays.d_coverage.get(),
            dataArrays.d_msa_column_properties.get(),
            dataArrays.d_alignment_shifts.get(),
            dataArrays.d_candidate_sequences_lengths.get(),
            dataArrays.d_anchorIndicesOfCandidates.get(),
            dataArrays.d_is_high_quality_subject.get(),
            dataArrays.d_candidates_per_subject_prefixsum,
            d_indices,
            d_indices_per_subject,
            d_numAnchors,
            d_numCandidates,
            batch.msa_weights_pitch / sizeof(float),
            min_support_threshold,
            min_coverage_threshold,
            new_columns_to_correct,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );

        size_t cubTempSize = dataArrays.d_tempstorage.sizeInBytes();

        cub::DeviceSelect::Flagged(
            dataArrays.d_tempstorage.get(),
            cubTempSize,
            cub::CountingInputIterator<int>(0),
            d_candidateCanBeCorrected,
            dataArrays.d_indices_of_corrected_candidates.get(),
            dataArrays.d_num_total_corrected_candidates.get(),
            maxCandidates,
            streams[primary_stream_index]
        ); CUERR;

        cudaEvent_t flaggingfinished = events[result_transfer_finished_event_index];

        cudaEventRecord(flaggingfinished, streams[primary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], flaggingfinished, 0); CUERR;

        //start result transfer of already calculated data in second stream

        cudaMemcpyAsync(
            dataArrays.h_num_total_corrected_candidates.get(),
            dataArrays.d_num_total_corrected_candidates.get(),
            sizeof(int),
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        //cudaEventRecord(events[numTotalCorrectedCandidates_event_index], streams[secondary_stream_index]); CUERR;

        cub::DeviceScan::ExclusiveSum(
            dataArrays.d_tempstorage.get(), 
            cubTempSize, 
            dataArrays.d_num_corrected_candidates_per_anchor.get(), 
            dataArrays.d_num_corrected_candidates_per_anchor_prefixsum.get(), 
            batchsize, 
            streams[secondary_stream_index]
        );

        cudaMemcpyAsync(
            dataArrays.h_num_corrected_candidates_per_anchor,
            dataArrays.d_num_corrected_candidates_per_anchor,
            sizeof(int) * batchsize,
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(
            dataArrays.h_num_corrected_candidates_per_anchor_prefixsum.get(),
            dataArrays.d_num_corrected_candidates_per_anchor_prefixsum.get(),
            sizeof(int) * batchsize,
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        // cudaMemcpyAsync(
        //     dataArrays.h_alignment_shifts,
        //     dataArrays.d_alignment_shifts,
        //     sizeof(int) * maxCandidates, //actually only need sizeof(int) * num_total_corrected_candidates, but its not available on the host
        //     D2H,
        //     streams[secondary_stream_index]
        // ); CUERR;

        int* h_alignment_shifts = dataArrays.h_alignment_shifts.get();
        const int* d_alignment_shifts = dataArrays.d_alignment_shifts.get();
        int* h_indices_of_corrected_candidates = dataArrays.h_indices_of_corrected_candidates.get();
        const int* d_indices_of_corrected_candidates = dataArrays.d_indices_of_corrected_candidates.get();

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

        callCorrectCandidatesWithGroupKernel2_async(
            dataArrays.h_corrected_candidates.get(),
            dataArrays.h_editsPerCorrectedCandidate.get(),
            dataArrays.h_numEditsPerCorrectedCandidate.get(),
            dataArrays.d_msa_column_properties.get(),
            dataArrays.d_consensus.get(),
            dataArrays.d_support.get(),
            dataArrays.d_alignment_shifts.get(),
            dataArrays.d_alignment_best_alignment_flags.get(),
            dataArrays.d_candidate_sequences_data.get(),
            dataArrays.d_candidate_sequences_lengths.get(),
            dataArrays.d_candidateContainsN.get(),
            dataArrays.d_indices_of_corrected_candidates.get(),
            dataArrays.d_num_total_corrected_candidates.get(),
            dataArrays.d_anchorIndicesOfCandidates.get(),
            d_numAnchors,
            d_numCandidates,
            doNotUseEditsValue,
            batch.maxNumEditsPerSequence,
            batch.encodedSequencePitchInInts,
            batch.decodedSequencePitchInBytes,
            batch.msa_pitch,
            batch.msa_weights_pitch,
            transFuncData.sequenceFileProperties.maxSequenceLength,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );       
        
        //cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
        
        // cudaMemcpyAsync(
        //     dataArrays.h_numEditsPerCorrectedCandidate,
        //     dataArrays.d_numEditsPerCorrectedCandidate,
        //     sizeof(int) * maxCandidates, //actually only need sizeof(int) * num_total_corrected_candidates, but its not available on the host
        //     D2H,
        //     streams[primary_stream_index]
        // ); CUERR;
        
        // cudaMemcpyAsync(
        //     dataArrays.h_indices_of_corrected_candidates,
        //     dataArrays.d_indices_of_corrected_candidates,
        //     sizeof(int) * maxCandidates, //actually only need sizeof(int) * num_total_corrected_candidates, but its not available on the host
        //     D2H,
        //     streams[primary_stream_index]
        // ); CUERR;

        // const int resultsToCopy = batch.n_queries * 0.2f;

        // cudaMemcpyAsync(
        //     dataArrays.h_corrected_candidates,
        //     dataArrays.d_corrected_candidates,
        //     batch.decodedSequencePitchInBytes * resultsToCopy,
        //     D2H,
        //     streams[primary_stream_index]
        // ); CUERR;

        // cudaMemcpyAsync(
        //     dataArrays.h_editsPerCorrectedCandidate,
        //     dataArrays.d_editsPerCorrectedCandidate,
        //     sizeof(TempCorrectedSequence::Edit) * batch.maxNumEditsPerSequence * resultsToCopy,
        //     D2H,
        //     streams[primary_stream_index]
        // ); CUERR;

        // int* d_remainingResultsToCopy = dataArrays.d_num_high_quality_subject_indices.get(); //reuse
        // const int* tmpptr = dataArrays.d_num_total_corrected_candidates.get();

        // generic_kernel<<<1,1,0, streams[primary_stream_index]>>>([=] __device__ (){
        //     *d_remainingResultsToCopy = max(0, *tmpptr - resultsToCopy);
        // }); CUERR;
        
        // callMemcpy2DKernel(
        //     dataArrays.h_corrected_candidates.get() + batch.decodedSequencePitchInBytes * resultsToCopy,
        //     dataArrays.d_corrected_candidates.get() + batch.decodedSequencePitchInBytes * resultsToCopy,
        //     d_remainingResultsToCopy,
        //     batch.decodedSequencePitchInBytes,
        //     batch.n_queries - resultsToCopy,
        //     streams[primary_stream_index]
        // ); CUERR;
        
        // callMemcpy2DKernel(
        //     dataArrays.h_editsPerCorrectedCandidate.get() + batch.maxNumEditsPerSequence * resultsToCopy,
        //     dataArrays.d_editsPerCorrectedCandidate.get() + batch.maxNumEditsPerSequence * resultsToCopy,
        //     d_remainingResultsToCopy,
        //     batch.maxNumEditsPerSequence,
        //     batch.n_queries - resultsToCopy,
        //     streams[primary_stream_index]
        // ); CUERR;

        // {
        //     char* const h_corrected_candidates = dataArrays.h_corrected_candidates.get();
        //     char* const d_corrected_candidates = dataArrays.d_corrected_candidates.get();
        //     const int* const d_num_total_corrected_candidates = dataArrays.d_num_total_corrected_candidates.get();
            
        //     auto* h_editsPerCorrectedCandidate = dataArrays.h_editsPerCorrectedCandidate.get();
        //     auto* d_editsPerCorrectedCandidate = dataArrays.d_editsPerCorrectedCandidate.get();

        //     std::size_t decodedSequencePitchInBytes = batch.decodedSequencePitchInBytes;
        //     std::size_t maxNumEditsPerSequence = batch.maxNumEditsPerSequence;
            
        //     generic_kernel<<<640, 256, 0, streams[primary_stream_index]>>>(
        //         [=] __device__ (){
        //             using CopyType = int;

        //             const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        //             const size_t stride = blockDim.x * gridDim.x;

        //             const int numElements = *d_num_total_corrected_candidates;

        //             const size_t bytesToCopy1 = numElements * decodedSequencePitchInBytes;

        //             const int fullIntsToCopy1 = bytesToCopy1 / sizeof(CopyType);

        //             for(int index = tid; index < fullIntsToCopy1; index += stride){
        //                 ((CopyType*)h_corrected_candidates)[index] = 
        //                     ((const CopyType*)d_corrected_candidates)[index];
        //             }

        //             const int remainingBytes1 = bytesToCopy1 - fullIntsToCopy1 * sizeof(CopyType);

        //             if(tid < remainingBytes1){
        //                 h_corrected_candidates[fullIntsToCopy1 * sizeof(CopyType) + tid] 
        //                     = d_corrected_candidates[fullIntsToCopy1 * sizeof(CopyType) + tid];
        //             }

        //             const size_t bytesToCopy2 = numElements * maxNumEditsPerSequence;

        //             const int fullIntsToCopy2 = bytesToCopy2 / sizeof(CopyType);

        //             for(int index = tid; index < fullIntsToCopy2; index += stride){
        //                 ((CopyType*)h_editsPerCorrectedCandidate)[index] = 
        //                     ((const CopyType*)d_editsPerCorrectedCandidate)[index];
        //             }

        //             const int remainingBytes2 = bytesToCopy2 - fullIntsToCopy2 * sizeof(CopyType);

        //             if(tid < remainingBytes2){
        //                 h_editsPerCorrectedCandidate[fullIntsToCopy2 * sizeof(CopyType) + tid] 
        //                     = d_editsPerCorrectedCandidate[fullIntsToCopy2 * sizeof(CopyType) + tid];
        //             }


                    
        //         }
        //     ); CUERR;

        // }

        // callMemcpy2DKernel(
        //     dataArrays.h_corrected_candidates.get(),
        //     dataArrays.d_corrected_candidates.get(),
        //     dataArrays.d_num_total_corrected_candidates.get(),
        //     batch.decodedSequencePitchInBytes,
        //     batch.n_queries,
        //     streams[primary_stream_index]
        // ); CUERR;
        
        // callMemcpy2DKernel(
        //     dataArrays.h_editsPerCorrectedCandidate.get(),
        //     dataArrays.d_editsPerCorrectedCandidate.get(),
        //     dataArrays.d_num_total_corrected_candidates.get(),
        //     batch.maxNumEditsPerSequence,
        //     batch.n_queries,
        //     streams[primary_stream_index]
        // ); CUERR;
        
    }

    void copyCorrectedCandidatesToHost(Batch& batch){

        // cudaSetDevice(batch.deviceId); CUERR;

        // DataArrays& dataArrays = batch.dataArrays;
        // std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        // std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        // cudaEventSynchronize(events[numTotalCorrectedCandidates_event_index]); CUERR;

        // const int numTotalCorrectedCandidates = *dataArrays.h_num_total_corrected_candidates.get();
        // //std::cerr << numTotalCorrectedCandidates << " / " << batch.n_queries << "\n";

        // cudaEventSynchronize(events[correction_finished_event_index]); CUERR;        



        // cudaMemcpyAsync(
        //     dataArrays.h_corrected_candidates,
        //     dataArrays.d_corrected_candidates,
        //     batch.decodedSequencePitchInBytes * numTotalCorrectedCandidates,
        //     D2H,
        //     streams[primary_stream_index]
        // ); CUERR;

        // cudaMemcpyAsync(
        //     dataArrays.h_editsPerCorrectedCandidate,
        //     dataArrays.d_editsPerCorrectedCandidate,
        //     sizeof(TempCorrectedSequence::Edit) * batch.maxNumEditsPerSequence * numTotalCorrectedCandidates,
        //     D2H,
        //     streams[primary_stream_index]
        // ); CUERR;

        // // cubCachingAllocator.DeviceFree(dataArrays.d_compactCorrectedCandidates); CUERR;
        // // cubCachingAllocator.DeviceFree(dataArrays.d_compactEditsPerCorrectedCandidate); CUERR;

        //  cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
    }


    void constructResults(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;

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
                transFuncData.correctionStatusFlagsPerRead[readId] |= readCorrectedAsHQAnchor;
            }

            if(isCorrected){
                subjectIndicesToProcess.emplace_back(subject_index);
            }else{
                transFuncData.correctionStatusFlagsPerRead[readId] |= readCouldNotBeCorrectedAsAnchor;
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
                const std::uint8_t mask = transFuncData.correctionStatusFlagsPerRead[candidate_read_id];
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
        outputData.anchorCorrections.resize(numCorrectedAnchors);
        outputData.encodedAnchorCorrections.resize(numCorrectedAnchors);
        outputData.candidateCorrections.resize(numCorrectedCandidates);
        outputData.encodedCandidateCorrections.resize(numCorrectedCandidates);
        nvtx::pop_range();

        auto outputDataPtr = &outputData;
        auto transFuncDataPtr = batch.transFuncData;

        auto unpackAnchors = [outputDataPtr, transFuncDataPtr](int begin, int end){
            nvtx::push_range("Anchor unpacking", 3);
            
            auto& outputData = *outputDataPtr;
            auto& rawResults = outputData.rawResults;
            const auto& transFuncData = *transFuncDataPtr;
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
                //     std::copy_n(rawResults.h_uncorrected_positions_per_subject + subject_index * transFuncData.sequenceFileProperties.maxSequenceLength,
                //                 numUncorrectedPositions,
                //                 tmp.uncorrectedPositionsNoConsensus.begin());

                // }

                const int numEdits = rawResults.h_numEditsPerCorrectedSubject[positionInVector];
                if(numEdits != doNotUseEditsValue){
                    tmp.edits.resize(numEdits);
                    const auto* gpuedits = rawResults.h_editsPerCorrectedSubject + positionInVector * rawResults.maxNumEditsPerSequence;
                    std::copy_n(gpuedits, numEdits, tmp.edits.begin());
                    tmp.useEdits = true;
                }else{
                    tmp.edits.clear();
                    tmp.useEdits = false;

                    const char* const my_corrected_subject_data = rawResults.h_corrected_subjects + subject_index * rawResults.decodedSequencePitchInBytes;
                    const int subject_length = rawResults.h_subject_sequences_lengths[subject_index];
                    tmp.sequence = std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length};       
                    
                    auto isValidSequence = [](const std::string& s){
                        return std::all_of(s.begin(), s.end(), [](char c){
                            return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
                        });
                    };
    
                    if(!isValidSequence(tmp.sequence)){
                        std::cerr << "invalid sequence\n"; //std::cerr << tmp.sequence << "\n";
                    }
                }

                tmpencoded = tmp.encode();

                // if(readId == 13){
                //     std::cerr << "readid = 13, anchor\n";
                //     std::cerr << "hq = " << tmp.hq << ", sequence = " << tmp.sequence << "\n";
                //     std::cerr << "\nedits: ";
                //     for(int i = 0; i < int(tmp.edits.size()); i++){
                //         std::cerr << tmp.edits[i].base << ' ' << tmp.edits[i].pos << "\n";
                //     }
                // }
            }

            nvtx::pop_range();
        };

        auto unpackcandidates = [outputDataPtr, transFuncDataPtr](int begin, int end){
            nvtx::push_range("candidate unpacking", 3);
            //std::cerr << "\n\n unpack candidates \n\n";
            
            auto& outputData = *outputDataPtr;
            auto& rawResults = outputData.rawResults;
            const auto& transFuncData = *transFuncDataPtr;

            const auto& subjectIndicesToProcess = outputData.subjectIndicesToProcess;
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
                const TempCorrectedSequence::Edit* const my_editsPerCorrectedCandidate = rawResults.h_editsPerCorrectedCandidate
                                                        + offsetForCorrectedCandidateData * rawResults.maxNumEditsPerSequence;


                const int global_candidate_index = my_indices_of_corrected_candidates[candidateIndex];
                //std::cerr << global_candidate_index << "\n";
                const read_number candidate_read_id = rawResults.h_candidate_read_ids[global_candidate_index];

                const int candidate_shift = rawResults.h_alignment_shifts[global_candidate_index];
                
                //TIMERSTOPCPU(setup);
                if(transFuncData.correctionOptions.new_columns_to_correct < candidate_shift){
                    std::cerr << "readid " << subjectReadId << " candidate readid " << candidate_read_id << " : "
                    << candidate_shift << " " << transFuncData.correctionOptions.new_columns_to_correct <<"\n";
                }
                assert(transFuncData.correctionOptions.new_columns_to_correct >= candidate_shift);
                
                //TIMERSTARTCPU(tmp);
                tmp.type = TempCorrectedSequence::Type::Candidate;
                tmp.shift = candidate_shift;
                tmp.readId = candidate_read_id;
                //TIMERSTOPCPU(tmp);
                //const bool originalReadContainsN = transFuncData.readStorage->readContainsN(candidate_read_id);
                
                
                const int numEdits = rawResults.h_numEditsPerCorrectedCandidate[global_candidate_index];
                if(numEdits != doNotUseEditsValue){
                    tmp.edits.resize(numEdits);
                    const auto* gpuedits = my_editsPerCorrectedCandidate + candidateIndex * rawResults.maxNumEditsPerSequence;
                    std::copy_n(gpuedits, numEdits, tmp.edits.begin());
                    tmp.useEdits = true;
                }else{
                    const int candidate_length = rawResults.h_candidate_sequences_lengths[global_candidate_index];
                    const char* const candidate_data = my_corrected_candidates_data + candidateIndex * rawResults.decodedSequencePitchInBytes;
                    tmp.sequence = std::string{candidate_data, candidate_data + candidate_length};
                    tmp.edits.clear();
                    tmp.useEdits = false;
                }

                //TIMERSTARTCPU(encode);
                tmpencoded = tmp.encode();
                //TIMERSTOPCPU(encode);

                // if(candidate_read_id == 13){
                //     std::cerr << "readid = 13, as candidate of anchor with id " << subjectReadId << "\n";
                //     std::cerr << "hq = " << tmp.hq << ", sequence = " << tmp.sequence;
                //     std::cerr << "\nedits: ";
                //     for(int i = 0; i < int(tmp.edits.size()); i++){
                //         std::cerr << tmp.edits[i].base << ' ' << tmp.edits[i].pos << "\n";
                //     }
                // }
            }

            nvtx::pop_range();
        };


        if(!transFuncData.correctionOptions.correctCandidates){
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

        const auto& transFuncData = *batch.transFuncData;
            
        auto function = [batchPtr = &batch,
            transFuncData = &transFuncData,
            id = batch.id](){

            auto& batch = *batchPtr;
            auto& outputData = batch.waitableOutputData.data;

            const int numA = outputData.anchorCorrections.size();
            const int numC = outputData.candidateCorrections.size();

            nvtx::push_range("batch "+std::to_string(id)+" writeresultoutputhread"
                + std::to_string(numA) + " " + std::to_string(numC), 4);

            for(int i = 0; i < numA; i++){
                transFuncData->saveCorrectedSequence(
                    std::move(outputData.anchorCorrections[i]), 
                    std::move(outputData.encodedAnchorCorrections[i])
                );
            }

            for(int i = 0; i < numC; i++){
                transFuncData->saveCorrectedSequence(
                    std::move(outputData.candidateCorrections[i]), 
                    std::move(outputData.encodedCandidateCorrections[i])
                );
            }

            batch.waitableOutputData.signal();
            //std::cerr << "batch " << batch.id << " batch.waitableOutputData.signal() finished\n";

            nvtx::pop_range();
        };

		//function();

        nvtx::push_range("enqueue to outputthread", 2);
        batch.outputThread->enqueue(std::move(function));
        nvtx::pop_range();
	}




void correct_gpu(
        const MinhashOptions& minhashOptions,
        const AlignmentOptions& alignmentOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        const CorrectionOptions& correctionOptions,
        const RuntimeOptions& runtimeOptions,
        const FileOptions& fileOptions,
        const MemoryOptions& memoryOptions,
        const SequenceFileProperties& sequenceFileProperties,
        Minhasher& minhasher,
        DistributedReadStorage& readStorage,
        std::uint64_t maxCandidatesPerRead){

    assert(runtimeOptions.canUseGpu);
    //assert(runtimeOptions.max_candidates > 0);
    assert(runtimeOptions.deviceIds.size() > 0);

    const auto& deviceIds = runtimeOptions.deviceIds;

    std::vector<std::string> tmpfiles{fileOptions.tempdirectory + "/" + fileOptions.outputfilename + "_tmp"};
    std::vector<std::string> featureTmpFiles{fileOptions.tempdirectory + "/" + fileOptions.outputfilename + "_features"};

    //std::vector<std::atomic_uint8_t> correctionStatusFlagsPerRead;
    //std::size_t nLocksForProcessedFlags = runtimeOptions.nCorrectorThreads * 1000;
    //std::unique_ptr<std::mutex[]> locksForProcessedFlags(new std::mutex[nLocksForProcessedFlags]);

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

    std::cerr << "correctionStatusFlagsPerRead bytes: " << sizeof(std::atomic_uint8_t) * sequenceFileProperties.nReads / 1024. / 1024. << " MB\n";

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

    MemoryFileFixedSize<EncodedTempCorrectedSequence> partialResults(memoryForPartialResultsInBytes, tmpfiles[0]);

    //   std::ofstream outputstream;
    //   std::unique_ptr<SequenceFileWriter> writer;

      //if candidate correction is not enabled, it is possible to write directly into the result file
      // if(!correctionOptions.correctCandidates){
      //     //writer = std::move(makeSequenceWriter(fileOptions.outputfile, FileFormat::FASTQGZ));
      //     outputstream = std::move(std::ofstream(fileOptions.outputfile));
      //     if(!outputstream){
      //         throw std::runtime_error("Could not open output file " + tmpfiles[0]);
      //     }
      // }else{
        //   outputstream = std::move(std::ofstream(tmpfiles[0]));
        //   if(!outputstream){
        //       throw std::runtime_error("Could not open output file " + tmpfiles[0]);
        //   }
     // }


      std::ofstream featurestream;
      //if(correctionOptions.extractFeatures){
          featurestream = std::move(std::ofstream(featureTmpFiles[0]));
          if(!featurestream && correctionOptions.extractFeatures){
              throw std::runtime_error("Could not open output feature file");
          }
      //}

      //std::mutex outputstreamlock;

      TransitionFunctionData transFuncData;

      const int nParallelBatches = runtimeOptions.gpuParallelBatches;
      const int batchsize = correctionOptions.batchsize;

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
            const int limit = deviceProperties.multiProcessorCount * deviceProperties.maxThreadsPerMultiProcessor;
            numCandidatesLimitPerGpu[deviceId] = limit;
            std::cerr << "Number of candidates per batch is limited to " << limit 
                << " for device id " << deviceId << "\n";
        }

        

        
   


      //transFuncData.mybatchgen = &mybatchgen;
      transFuncData.goodAlignmentProperties = goodAlignmentProperties;
      transFuncData.correctionOptions = correctionOptions;
      transFuncData.runtimeOptions = runtimeOptions;
      transFuncData.minhashOptions = minhashOptions;
      transFuncData.alignmentOptions = alignmentOptions;
      transFuncData.fileOptions = fileOptions;
      transFuncData.sequenceFileProperties = sequenceFileProperties;

      transFuncData.readIdGenerator = &readIdGenerator;
      transFuncData.minhasher = &minhasher;
      transFuncData.readStorage = &readStorage;
      transFuncData.correctionStatusFlagsPerRead = correctionStatusFlagsPerRead.get();
      transFuncData.featurestream = &featurestream;

      //std::mutex outputstreammutex;
      std::map<bool, int> useEditsCountMap;
      std::map<bool, int> useEditsSavedCountMap;
      std::map<int, int> numEditsHistogram;

      transFuncData.saveCorrectedSequence = [&](TempCorrectedSequence tmp, EncodedTempCorrectedSequence encoded){
          //useEditsCountMap[tmp.useEdits]++;
            //std::cerr << tmp << "\n";
          //std::unique_lock<std::mutex> l(outputstreammutex);
          if(!(tmp.hq && tmp.useEdits && tmp.edits.empty())){
              //outputstream << tmp << '\n';
              partialResults.storeElement(std::move(encoded));
              //useEditsSavedCountMap[tmp.useEdits]++;
              //numEditsHistogram[tmp.edits.size()]++;

             // std::cerr << tmp.edits.size() << " " << encoded.data.capacity() << "\n";
          }
      };

      transFuncData.lock = [&](read_number readId){
                       // read_number index = readId % transFuncData.nLocksForProcessedFlags;
                       // transFuncData.locksForProcessedFlags[index].lock();
                   };
      transFuncData.unlock = [&](read_number readId){
                         // read_number index = readId % transFuncData.nLocksForProcessedFlags;
                         // transFuncData.locksForProcessedFlags[index].unlock();
                     };

      if(transFuncData.correctionOptions.correctionType == CorrectionType::Forest){
         transFuncData.fc = ForestClassifier{fileOptions.forestfilename};
      }

      transFuncData.minhashHandles.resize(threadPoolSize);


#if 0
      NN_Correction_Classifier_Base nnClassifierBase;
      NN_Correction_Classifier nnClassifier;
      if(correctionOptions.correctionType == CorrectionType::Convnet){
          nnClassifierBase = std::move(NN_Correction_Classifier_Base{"./nn_sources", fileOptions.nnmodelfilename});
          nnClassifier = std::move(NN_Correction_Classifier{&nnClassifierBase});
      }
#endif 
      // BEGIN CORRECTION


     outputThread.start();

        std::vector<std::thread> batchExecutors;

      #ifdef DO_PROFILE
          cudaProfilerStart();
      #endif

        auto initBatchData = [&](auto& batchData, int deviceId){

            cudaSetDevice(deviceId); CUERR;

            DataArrays dataArrays;

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
            batchData.dataArrays = std::move(dataArrays);
            batchData.streams = std::move(streams);
            batchData.events = std::move(events);
            batchData.kernelLaunchHandle = make_kernel_launch_handle(deviceId);
            batchData.subjectSequenceGatherHandle = readStorage.makeGatherHandleSequences();
            batchData.candidateSequenceGatherHandle = readStorage.makeGatherHandleSequences();
            batchData.subjectQualitiesGatherHandle = readStorage.makeGatherHandleQualities();
            batchData.candidateQualitiesGatherHandle = readStorage.makeGatherHandleQualities();
            batchData.transFuncData = &transFuncData;
            batchData.outputThread = &outputThread;
            batchData.backgroundWorker = nullptr;
            batchData.unpackWorker = nullptr;
            batchData.threadPool = &threadPool;
            batchData.threadsInThreadPool = threadPoolSize;
            batchData.minhashHandles.resize(threadPoolSize);
            batchData.encodedSequencePitchInInts = getEncodedNumInts2Bit(sequenceFileProperties.maxSequenceLength);
            batchData.decodedSequencePitchInBytes = SDIV(sequenceFileProperties.maxSequenceLength, 4) * 4;
            batchData.qualityPitchInBytes = SDIV(sequenceFileProperties.maxSequenceLength, 32) * 32;
            batchData.maxNumEditsPerSequence = std::max(1,sequenceFileProperties.maxSequenceLength / 7);

            initNextIterationData(batchData.nextIterationData, batchData.deviceId); 


            
            batchData.numCandidatesLimit = numCandidatesLimitPerGpu[deviceId];
        };

        auto destroyBatchData = [&](auto& batchData){
            
            cudaSetDevice(batchData.deviceId); CUERR;
    
            batchData.dataArrays.reset();
            destroyNextIterationData(batchData.nextIterationData);

            for(int i = 0; i < 2; i++){
                if(batchData.alignmentGraphs[i].execgraph != nullptr){
                    cudaGraphExecDestroy(batchData.alignmentGraphs[i].execgraph); CUERR;
                }
            }
    
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
                
                if(totalCount == std::int64_t(sequenceFileProperties.nReads)){
                    std::cerr << '\n';
                }

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

            if(batchData.n_queries == 0){
                return;
            }

            pushrange("resizeArrays", 3);
            resizeArraysFixedCandidateBatchsize(batchData);

            poprange();

            pushrange("getCandidateSequenceData", 1);

            getCandidateSequenceData(batchData, *transFuncData.readStorage);

            poprange();

            if(transFuncData.correctionOptions.useQualityScores) {
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

            if(transFuncData.correctionOptions.correctCandidates) {                        

                pushrange("correctCandidates", 8);

                correctCandidates(batchData);

                poprange();
                
            }
#endif            
        };

        auto copyCandidateCorrectionsToHostAndJoinStreams = [&](auto& batchData){
            // auto& streams = batchData.streams;
            // auto& events = batchData.events;

            auto pushrange = [&](const std::string& msg, int color){
                nvtx::push_range("batch "+std::to_string(batchData.id)+msg, color);
            };

            auto poprange = [&](){
                nvtx::pop_range();
            };

            if(transFuncData.correctionOptions.correctCandidates) {

                pushrange("copyCorrectedCandidatesToHost", 8);

                copyCorrectedCandidatesToHost(batchData);

                poprange();                
            }

            // cudaEventRecord(events[secondary_stream_finished_event_index], streams[secondary_stream_index]); CUERR;
            // cudaStreamWaitEvent(streams[primary_stream_index], events[secondary_stream_finished_event_index], 0); CUERR;   

            //cudaDeviceSynchronize(); CUERR;

            batchData.hasUnprocessedResults = true;
        };

        auto processBatchResults = [&](auto& batchData){
            auto& streams = batchData.streams;
            //auto& events = batchData.events;

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
                    batchDataArray[i].unpackWorker = &unpackWorkerArray[i];

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
                        && batchDataArray[0].nextIterationData.n_subjects == 0
                        && !batchDataArray[0].waitableOutputData.isBusy())) {

                    auto& batchData = batchDataArray[batchIndex];

                    processBatchUntilCandidateCorrectionStarted(batchData);

                    if(batchData.n_queries == 0){
                        batchData.waitableOutputData.signal();
                        progressThread.addProgress(batchData.n_subjects);
                        batchData.reset();
                        continue;
                    }

                    copyCandidateCorrectionsToHostAndJoinStreams(batchData);

                    processBatchResults(batchData);

                    progressThread.addProgress(batchData.n_subjects);
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
                        && batchDataArray[0].nextIterationData.n_subjects == 0
                        && !batchDataArray[0].waitableOutputData.isBusy()
                        && !batchDataArray[1].nextIterationData.syncFlag.isBusy()
                        && batchDataArray[1].nextIterationData.n_subjects == 0
                        && !batchDataArray[1].waitableOutputData.isBusy())) {

                    const int nextBatchIndex = 1 - batchIndex;
                    auto& currentBatchData = batchDataArray[batchIndex];
                    auto& nextBatchData = batchDataArray[nextBatchIndex];

                    if(isFirstIteration){
                        isFirstIteration = false;
                        processBatchUntilCandidateCorrectionStarted(currentBatchData);
                    }else{
                        processBatchUntilCandidateCorrectionStarted(nextBatchData);

                        if(currentBatchData.n_queries == 0){
                            currentBatchData.waitableOutputData.signal();
                            progressThread.addProgress(currentBatchData.n_subjects);
                            currentBatchData.reset();
                            batchIndex = 1-batchIndex;
                            continue;
                        }

                        copyCandidateCorrectionsToHostAndJoinStreams(currentBatchData);
                        processBatchResults(currentBatchData);
    
                        progressThread.addProgress(currentBatchData.n_subjects);
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
                        && batchDataArray[0].nextIterationData.n_subjects == 0
                        && !batchDataArray[0].waitableOutputData.isBusy()
                        && !batchDataArray[1].nextIterationData.syncFlag.isBusy()
                        && batchDataArray[1].nextIterationData.n_subjects == 0
                        && !batchDataArray[1].waitableOutputData.isBusy()
                        && !batchDataArray[2].nextIterationData.syncFlag.isBusy()
                        && batchDataArray[2].nextIterationData.n_subjects == 0
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

                        if(currentBatchData.n_queries == 0){
                            currentBatchData.waitableOutputData.signal();
                            progressThread.addProgress(currentBatchData.n_subjects);
                            currentBatchData.reset();
                            batchIndex = batchIndex == 2 ? 0 : 1 + batchIndex;
                            continue;
                        }

                        copyCandidateCorrectionsToHostAndJoinStreams(currentBatchData);
                        processBatchResults(currentBatchData);
    
                        progressThread.addProgress(currentBatchData.n_subjects);
                        currentBatchData.reset();

                        batchIndex = batchIndex == 2 ? 0 : 1 + batchIndex;
                    }              
                }
#endif
                
                for(int i = 0; i < max_num_batches; i++){
                    batchDataArray[i].backgroundWorker->stopThread(BackgroundThread::StopType::FinishAndStop);
                    batchDataArray[i].unpackWorker->stopThread(BackgroundThread::StopType::FinishAndStop);
                    destroyBatchData(batchDataArray[i]);
                }
            });
        }

        for(auto& executor : batchExecutors){
            executor.join();
        }

        progressThread.finished();        
        threadPool.wait();
        outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

      //flushCachedResults();
      //outputstream.flush();
      featurestream.flush();
      partialResults.flush();

      #ifdef DO_PROFILE
          cudaProfilerStop();
      #endif



    //   for(const auto& batch : batches){
    //       std::cout << "size elements: " << batch.dataArrays.h_candidate_read_ids.size() << ", capacity elements " << batch.dataArrays.h_candidate_read_ids.capacity() << std::endl;
      
    //     }

    //     for(const auto& batch : batches){
    //         std::cerr << "Memory usage: \n";
    //         batch.dataArrays.printMemoryUsage();
    //         std::cerr << "Total: " << batch.dataArrays.getMemoryUsageInBytes() << " bytes\n";
    //         std::cerr << '\n';
    //     }


      correctionStatusFlagsPerRead.reset();

      //size_t occupiedMemory = minhasher.numBytes() + cpuReadStorage.size();

      minhasher.destroy();
      readStorage.destroy();

      std::cerr << "useEditsCountMap\n";
      for(const auto& pair : useEditsCountMap){
          std::cerr << int(pair.first) << " : " << pair.second << "\n";
      }

      std::cerr << "useEditsSavedCountMap\n";
      for(const auto& pair : useEditsSavedCountMap){
          std::cerr << int(pair.first) << " : " << pair.second << "\n";
      }

      std::cerr << "numEditsHistogram\n";
      for(const auto& pair : numEditsHistogram){
          std::cerr << int(pair.first) << " : " << pair.second << "\n";
      }

      

      #ifndef DO_PROFILE

      //if candidate correction is enabled, only the read id and corrected sequence of corrected reads is written to outputfile
      //outputfile needs to be sorted by read id
      //then, the corrected reads from the output file have to be merged with the original input file to get headers, uncorrected reads, and quality scores
      if(true || correctionOptions.correctCandidates){

        const std::size_t availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
        std::size_t memoryForSorting = 0;

        if(availableMemoryInBytes > 1*(std::size_t(1) << 30)){
            memoryForSorting = availableMemoryInBytes - 1*(std::size_t(1) << 30);
        }

          std::cout << "begin merge" << std::endl;

          if(!correctionOptions.extractFeatures){

              std::cout << "begin merging reads" << std::endl;

              TIMERSTARTCPU(merge);

              constructOutputFileFromResults(
                    fileOptions.tempdirectory,
                    sequenceFileProperties.nReads, 
                    fileOptions.inputfile, 
                    fileOptions.format, 
                    partialResults, 
                    memoryForSorting,
                    fileOptions.outputfile, 
                    false
                );

              TIMERSTOPCPU(merge);

              std::cout << "end merging reads" << std::endl;

          }

          filehelpers::deleteFiles(tmpfiles);
      }

      //concatenate feature files of each thread into one file

      if(correctionOptions.extractFeatures){
          std::cout << "begin merging features" << std::endl;

          std::stringstream commandbuilder;

          commandbuilder << "cat";

          for(const auto& featureFile : featureTmpFiles){
              commandbuilder << " \"" << featureFile << "\"";
          }

          commandbuilder << " > \"" << fileOptions.outputfile << "_features\"";

          const std::string command = commandbuilder.str();
          TIMERSTARTCPU(concat_feature_files);
          int r1 = std::system(command.c_str());
          TIMERSTOPCPU(concat_feature_files);

          if(r1 != 0){
              std::cerr << "Warning. Feature files could not be concatenated!\n";
              std::cerr << "This command returned a non-zero error value: \n";
              std::cerr << command +  '\n';
              std::cerr << "Please concatenate the following files manually\n";
              for(const auto& s : featureTmpFiles)
                  std::cerr << s << '\n';
          }else{
            filehelpers::deleteFiles(featureTmpFiles);
          }

          std::cout << "end merging features" << std::endl;
      }else{
        filehelpers::deleteFiles(featureTmpFiles);
      }

      std::cout << "end merge" << std::endl;

      #endif



}







}
}

