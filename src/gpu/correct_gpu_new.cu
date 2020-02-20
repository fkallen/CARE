#if 1

#include <gpu/correct_gpu.hpp>
#include <gpu/distributedreadstorage.hpp>
#include <gpu/nvtxtimelinemarkers.hpp>
#include <gpu/kernels.hpp>
#include <gpu/dataarrays.hpp>
#include <gpu/cubcachingallocator.cuh>
#include <gpu/minhashkernels.hpp>

#include <config.hpp>
#include <qualityscoreweights.hpp>
#include <sequence.hpp>
#include <featureextractor.hpp>
#include <forestclassifier.hpp>
//#include <nn_classifier.hpp>
#include <minhasher.hpp>
#include <options.hpp>
#include <candidatedistribution.hpp>
#include <sequencefileio.hpp>
#include <rangegenerator.hpp>
#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>

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

#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>

//#define CARE_GPU_DEBUG
//#define CARE_GPU_DEBUG_MEMCOPY
//#define CARE_GPU_DEBUG_PRINT_ARRAYS
//#define CARE_GPU_DEBUG_PRINT_MSA

#define MSA_IMPLICIT

//#define REARRANGE_INDICES
#define USE_MSA_MINIMIZATION

//#define DO_PROFILE

#ifdef DO_PROFILE
    constexpr size_t num_reads_to_profile = 100000;
#endif



namespace care{
namespace gpu{
namespace test{

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
    constexpr int nEventsPerBatch = 11;

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
        SimpleAllocationPinnedHost<unsigned int> h_subject_sequences_data;
        SimpleAllocationPinnedHost<int> h_subject_sequences_lengths;
        SimpleAllocationPinnedHost<read_number> h_subject_read_ids;
        SimpleAllocationPinnedHost<read_number> h_candidate_read_ids;
        SimpleAllocationPinnedHost<int> h_candidates_per_subject;
        SimpleAllocationPinnedHost<int> h_candidates_per_subject_prefixsum;
        SimpleAllocationPinnedHost<bool> h_anchorContainsN;
        SimpleAllocationPinnedHost<bool> h_candidateContainsN;

        SimpleAllocationDevice<unsigned int> d_subject_sequences_data;
        SimpleAllocationDevice<int> d_subject_sequences_lengths;
        SimpleAllocationDevice<read_number> d_subject_read_ids;
        SimpleAllocationDevice<read_number> d_candidate_read_ids;
        SimpleAllocationDevice<int> d_candidates_per_subject;
        SimpleAllocationDevice<int> d_candidates_per_subject_prefixsum;
        SimpleAllocationDevice<bool> d_anchorContainsN;
        SimpleAllocationDevice<bool> d_candidateContainsN;

        SimpleAllocationDevice<read_number> d_candidate_read_ids_tmp;

        int n_subjects = -1;
        std::atomic<int> n_queries{-1};

        std::vector<std::string> decodedSubjectStrings;

        cudaStream_t stream;
        cudaEvent_t event;
        int deviceId;

        ThreadPool::ParallelForHandle pforHandle;

        MergeRangesGpuHandle<read_number> mergeRangesGpuHandle;

        SyncFlag syncFlag;
    };

    struct UnprocessedCorrectionResults{
        int n_subjects;
        int n_queries;
        int decodedSequencePitchInBytes;
        int encodedSequencePitchInInts;
        int maxNumEditsPerSequence;

        std::vector<std::string> decodedSubjectStrings;
        SimpleAllocationPinnedHost<read_number> h_subject_read_ids;
        SimpleAllocationPinnedHost<bool> h_subject_is_corrected;
        SimpleAllocationPinnedHost<AnchorHighQualityFlag> h_is_high_quality_subject;
        SimpleAllocationPinnedHost<int> h_num_corrected_candidates;
        SimpleAllocationPinnedHost<int> h_indices_of_corrected_candidates;
        SimpleAllocationPinnedHost<int> h_candidates_per_subject_prefixsum;
        SimpleAllocationPinnedHost<read_number> h_candidate_read_ids;
        SimpleAllocationPinnedHost<char> h_corrected_subjects;
        SimpleAllocationPinnedHost<char> h_corrected_candidates;
        SimpleAllocationPinnedHost<int> h_subject_sequences_lengths;
        SimpleAllocationPinnedHost<int> h_num_uncorrected_positions_per_subject;
        SimpleAllocationPinnedHost<int> h_uncorrected_positions_per_subject;
        SimpleAllocationPinnedHost<int> h_candidate_sequences_lengths;
        SimpleAllocationPinnedHost<int> h_alignment_shifts;
        SimpleAllocationPinnedHost<unsigned int> h_candidate_sequences_data;

        SimpleAllocationPinnedHost<TempCorrectedSequence::Edit> h_editsPerCorrectedSubject;
        SimpleAllocationPinnedHost<int> h_numEditsPerCorrectedSubject;
        SimpleAllocationPinnedHost<int> h_indices_of_corrected_subjects;
        SimpleAllocationPinnedHost<int> h_num_indices_of_corrected_subjects;

        SimpleAllocationPinnedHost<TempCorrectedSequence::Edit> h_editsPerCorrectedCandidate;
        SimpleAllocationPinnedHost<int> h_numEditsPerCorrectedCandidate;

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
        std::vector<std::string> decodedSubjectStrings;

		std::array<cudaStream_t, nStreamsPerBatch> streams;
		std::array<cudaEvent_t, nEventsPerBatch> events;

        TransitionFunctionData* transFuncData;
        BackgroundThread* outputThread;
        BackgroundThread* backgroundWorker;

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

        int encodedSequencePitchInInts;
        int decodedSequencePitchInBytes;
        int qualityPitchInBytes;

        int maxNumEditsPerSequence;

        int msa_weights_pitch;
        int msa_pitch;

        int n_subjects;
        int n_queries;

        int max_n_queries = 0;

		void reset(){
            combinedStreams = false;
            n_subjects = 0;
            n_queries = 0;
            hasUnprocessedResults = false;
        }

        void updateFromIterationData(NextIterationData& data){
            std::swap(dataArrays.h_subject_sequences_data, data.h_subject_sequences_data);
            std::swap(dataArrays.h_subject_sequences_lengths, data.h_subject_sequences_lengths);
            std::swap(dataArrays.h_subject_read_ids, data.h_subject_read_ids);
            std::swap(dataArrays.h_candidate_read_ids, data.h_candidate_read_ids);
            std::swap(dataArrays.h_candidates_per_subject, data.h_candidates_per_subject);
            std::swap(dataArrays.h_candidates_per_subject_prefixsum, data.h_candidates_per_subject_prefixsum);
            std::swap(dataArrays.h_anchorContainsN, data.h_anchorContainsN);
            

            std::swap(dataArrays.d_subject_sequences_data, data.d_subject_sequences_data);
            std::swap(dataArrays.d_subject_sequences_lengths, data.d_subject_sequences_lengths);
            std::swap(dataArrays.d_subject_read_ids, data.d_subject_read_ids);
            std::swap(dataArrays.d_candidate_read_ids, data.d_candidate_read_ids);
            std::swap(dataArrays.d_candidates_per_subject, data.d_candidates_per_subject);
            std::swap(dataArrays.d_candidates_per_subject_prefixsum, data.d_candidates_per_subject_prefixsum);
            std::swap(dataArrays.d_anchorContainsN, data.d_anchorContainsN);
            std::swap(dataArrays.d_candidateContainsN, data.d_candidateContainsN);


            std::swap(decodedSubjectStrings, data.decodedSubjectStrings);

            n_subjects = data.n_subjects;
            n_queries = data.n_queries;  

            data.n_subjects = 0;
            data.n_queries = 0;
            data.decodedSubjectStrings.clear();
        }

        void moveResultsToOutputData(OutputData& outputData){
            auto& rawResults = outputData.rawResults;

            rawResults.n_subjects = n_subjects;
            rawResults.n_queries = n_queries;
            rawResults.encodedSequencePitchInInts = encodedSequencePitchInInts;
            rawResults.decodedSequencePitchInBytes = decodedSequencePitchInBytes;
            rawResults.maxNumEditsPerSequence = maxNumEditsPerSequence;
            
            std::swap(decodedSubjectStrings, rawResults.decodedSubjectStrings);

            std::swap(dataArrays.h_subject_read_ids, rawResults.h_subject_read_ids);
            std::swap(dataArrays.h_subject_is_corrected, rawResults.h_subject_is_corrected);
            std::swap(dataArrays.h_is_high_quality_subject, rawResults.h_is_high_quality_subject);
            std::swap(dataArrays.h_num_corrected_candidates, rawResults.h_num_corrected_candidates);
            std::swap(dataArrays.h_indices_of_corrected_candidates, rawResults.h_indices_of_corrected_candidates);
            std::swap(dataArrays.h_candidates_per_subject_prefixsum, rawResults.h_candidates_per_subject_prefixsum);
            std::swap(dataArrays.h_candidate_read_ids, rawResults.h_candidate_read_ids);
            std::swap(dataArrays.h_corrected_subjects, rawResults.h_corrected_subjects);
            std::swap(dataArrays.h_corrected_candidates, rawResults.h_corrected_candidates);
            std::swap(dataArrays.h_subject_sequences_lengths, rawResults.h_subject_sequences_lengths);
            std::swap(dataArrays.h_num_uncorrected_positions_per_subject, rawResults.h_num_uncorrected_positions_per_subject);
            std::swap(dataArrays.h_uncorrected_positions_per_subject, rawResults.h_uncorrected_positions_per_subject);
            std::swap(dataArrays.h_candidate_sequences_lengths, rawResults.h_candidate_sequences_lengths);
            std::swap(dataArrays.h_alignment_shifts, rawResults.h_alignment_shifts);
            std::swap(dataArrays.h_candidate_sequences_data, rawResults.h_candidate_sequences_data);

            std::swap(dataArrays.h_editsPerCorrectedSubject, rawResults.h_editsPerCorrectedSubject);
            std::swap(dataArrays.h_numEditsPerCorrectedSubject, rawResults.h_numEditsPerCorrectedSubject);

            std::swap(dataArrays.h_editsPerCorrectedCandidate, rawResults.h_editsPerCorrectedCandidate);
            std::swap(dataArrays.h_numEditsPerCorrectedCandidate, rawResults.h_numEditsPerCorrectedCandidate);

            std::swap(dataArrays.h_indices_of_corrected_subjects, rawResults.h_indices_of_corrected_subjects);
            std::swap(dataArrays.h_num_indices_of_corrected_subjects, rawResults.h_num_indices_of_corrected_subjects);
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



    void build_msa_async(MSAPointers d_msapointers,
                    AlignmentResultPointers d_alignmentresultpointers,
                    ReadSequencesPointers d_sequencePointers,
                    ReadQualitiesPointers d_qualityPointers,
                    const int* d_candidates_per_subject_prefixsum,
                    const int* d_indices,
                    const int* d_indices_per_subject,
                    int n_subjects,
                    int n_queries,
                    const int* d_num_indices,
                    float expectedAffectedIndicesFraction,
                    bool useQualityScores,
                    float desiredAlignmentMaxErrorRate,
                    int maximum_sequence_length,
                    int encodedSequencePitchInInts,
                    int qualityPitchInBytes,
                    size_t msa_pitch,
                    size_t msa_weights_pitch,
                    const bool* d_canExecute,
                    cudaStream_t stream,
                    gpu::KernelLaunchHandle& kernelLaunchHandle){

        call_msa_init_kernel_async_exp(
                d_msapointers,
                d_alignmentresultpointers,
                d_sequencePointers,
                d_indices,
                d_indices_per_subject,
                d_candidates_per_subject_prefixsum,
                n_subjects,
                n_queries,
                d_canExecute,
                stream,
                kernelLaunchHandle);

        call_msa_add_sequences_kernel_implicit_async(
                    d_msapointers,
                    d_alignmentresultpointers,
                    d_sequencePointers,
                    d_qualityPointers,
                    d_candidates_per_subject_prefixsum,
                    d_indices,
                    d_indices_per_subject,
                    n_subjects,
                    n_queries,
                    d_num_indices,
                    expectedAffectedIndicesFraction,
                    useQualityScores,
                    desiredAlignmentMaxErrorRate,
                    maximum_sequence_length,
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    msa_pitch,
                    msa_weights_pitch,
                    d_canExecute,
                    stream,
                    kernelLaunchHandle,
                    false);

        call_msa_find_consensus_implicit_kernel_async(
                    d_msapointers,
                    d_sequencePointers,
                    d_indices_per_subject,
                    n_subjects,
                    encodedSequencePitchInInts,
                    msa_pitch,
                    msa_weights_pitch,
                    d_canExecute,
                    stream,
                    kernelLaunchHandle);
    };


    void initNextIterationData(NextIterationData& nextData, int deviceId){
        nextData.deviceId = deviceId;

        cudaSetDevice(deviceId); CUERR;
        cudaStreamCreate(&nextData.stream); CUERR;
        cudaEventCreate(&nextData.event); CUERR;

        nextData.mergeRangesGpuHandle = makeMergeRangesGpuHandle<read_number>();
    }

    void destroyNextIterationData(NextIterationData& nextData){
        cudaSetDevice(nextData.deviceId); CUERR;
        cudaStreamDestroy(nextData.stream); CUERR;
        cudaEventDestroy(nextData.event); CUERR;

        nextData.h_subject_sequences_data = std::move(SimpleAllocationPinnedHost<unsigned int>{});
        nextData.h_subject_sequences_lengths = std::move(SimpleAllocationPinnedHost<int>{});
        nextData.h_subject_read_ids = std::move(SimpleAllocationPinnedHost<read_number>{});
        nextData.h_candidate_read_ids = std::move(SimpleAllocationPinnedHost<read_number>{});
        nextData.h_candidates_per_subject = std::move(SimpleAllocationPinnedHost<int>{});
        nextData.h_candidates_per_subject_prefixsum = std::move(SimpleAllocationPinnedHost<int>{});

        nextData.d_subject_sequences_data = std::move(SimpleAllocationDevice<unsigned int>{});
        nextData.d_subject_sequences_lengths = std::move(SimpleAllocationDevice<int>{});
        nextData.d_subject_read_ids = std::move(SimpleAllocationDevice<read_number>{});
        nextData.d_candidate_read_ids = std::move(SimpleAllocationDevice<read_number>{});
        nextData.d_candidates_per_subject = std::move(SimpleAllocationDevice<int>{});
        nextData.d_candidates_per_subject_prefixsum = std::move(SimpleAllocationDevice<int>{});

        nextData.d_candidate_read_ids_tmp.destroy();

        destroyMergeRangesGpuHandle(nextData.mergeRangesGpuHandle);
    }

    void getSubjectDataOfNextIteration(Batch& batchData, int batchsize, const DistributedReadStorage& readStorage){
        NextIterationData& nextData = batchData.nextIterationData;
        const auto& transFuncData = *batchData.transFuncData;

        nextData.h_subject_sequences_data.resize(batchData.encodedSequencePitchInInts * batchsize);
        nextData.d_subject_sequences_data.resize(batchData.encodedSequencePitchInInts * batchsize);
        nextData.h_subject_sequences_lengths.resize(batchsize);
        nextData.d_subject_sequences_lengths.resize(batchsize);
        nextData.h_subject_read_ids.resize(batchsize);
        nextData.d_subject_read_ids.resize(batchsize);
        //nextData.h_anchorContainsN.resize(batchsize);
        nextData.d_anchorContainsN.resize(batchsize);        

        read_number* const readIdsBegin = nextData.h_subject_read_ids.get();
        read_number* const readIdsEnd = transFuncData.readIdGenerator->next_n_into_buffer(batchsize, readIdsBegin);
        nextData.n_subjects = std::distance(readIdsBegin, readIdsEnd);

        if(nextData.n_subjects == 0){
            return;
        };

        //copy read ids to device. gather sequences + lengths for those ids and copy them back to host
        cudaMemcpyAsync(
            nextData.d_subject_read_ids,
            nextData.h_subject_read_ids,
            nextData.h_subject_read_ids.sizeInBytes(),
            H2D,
            nextData.stream
        ); CUERR;

        readStorage.gatherSequenceDataToGpuBufferAsync(
            batchData.threadPool,
            batchData.subjectSequenceGatherHandle,
            nextData.d_subject_sequences_data.get(),
            batchData.encodedSequencePitchInInts,
            nextData.h_subject_read_ids,
            nextData.d_subject_read_ids,
            nextData.n_subjects,
            batchData.deviceId,
            nextData.stream,
            transFuncData.runtimeOptions.nCorrectorThreads
        );

        readStorage.gatherSequenceLengthsToGpuBufferAsync(
            nextData.d_subject_sequences_lengths.get(),
            batchData.deviceId,
            nextData.d_subject_read_ids.get(),
            nextData.n_subjects,            
            nextData.stream
        );

        cudaMemcpyAsync(
            nextData.h_subject_sequences_data,
            nextData.d_subject_sequences_data,
            nextData.d_subject_sequences_data.sizeInBytes(),
            D2H,
            nextData.stream
        ); CUERR;

        cudaMemcpyAsync(
            nextData.h_subject_sequences_lengths,
            nextData.d_subject_sequences_lengths,
            nextData.d_subject_sequences_lengths.sizeInBytes(),
            D2H,
            nextData.stream
        ); CUERR;

        readStorage.readsContainN_async(
            nextData.d_anchorContainsN.get(), 
            nextData.d_subject_read_ids.get(),
            nextData.n_subjects, 
            nextData.stream
        );

        // for(int i = 0; i < nextData.n_subjects; i++){
        //     const read_number anchorReadId = nextData.h_subject_read_ids[i];
        //     const bool containsN = readStorage.readContainsN(anchorReadId);
        //     nextData.h_anchorContainsN[i] = containsN;
        // }

        // cudaMemcpyAsync(
        //     nextData.d_anchorContainsN,
        //     nextData.h_anchorContainsN,
        //     nextData.h_anchorContainsN.sizeInBytes(),
        //     H2D,
        //     nextData.stream
        // ); CUERR;
        
    }

    void determineCandidateReadIdsOfNextIteration(
            Batch& batchData, 
            const Minhasher& minhasher, 
            const DistributedReadStorage& readStorage){

        NextIterationData& nextData = batchData.nextIterationData;

        //minhash the retrieved anchors to find candidate ids

        Batch* batchptr = &batchData;
        NextIterationData* nextDataPtr = &nextData;
        const Minhasher* minhasherPtr = &minhasher;

        nextData.n_queries = 0;

        std::vector<std::vector<std::string>> decodedSubjectStringsPerThread(batchData.threadsInThreadPool);

        auto calculateMinhash = [&, batchptr, nextDataPtr, minhasherPtr](int begin, int end, int threadId){

            auto& transFuncData = *(batchptr->transFuncData);
            const int hits_per_candidate = transFuncData.correctionOptions.hits_per_candidate;

            auto& minhashHandle = batchptr->minhashHandles[threadId];                      

            std::vector<std::string> decodedSubjectStrings(end-begin);

            nvtx::push_range("decode sequences", 1);
            for(int i = begin; i < end; i++){
                const unsigned int* sequenceptr = nextDataPtr->h_subject_sequences_data.get() + i * batchptr->encodedSequencePitchInInts;
                const int sequencelength = nextDataPtr->h_subject_sequences_lengths[i];

                decodedSubjectStrings[i - begin] = get2BitString(sequenceptr, sequencelength);
            }
            nvtx::pop_range();


            nvtx::push_range("calculateMinhashSignatures", 5);
            minhasherPtr->calculateMinhashSignatures(
                minhashHandle,
                decodedSubjectStrings
            );
            nvtx::pop_range();
            nvtx::push_range("queryPrecalculatedSignatures", 6);
            minhasherPtr->queryPrecalculatedSignatures(
                minhashHandle, 
                end - begin
            );
            nvtx::pop_range();
            nvtx::push_range("makeUniqueQueryResults", 7);
            minhasherPtr->makeUniqueQueryResults(
                minhashHandle, 
                end - begin
            );
            nvtx::pop_range();

            // nvtx::push_range("gpumakeUniqueQueryResults", 2);
            // OperationResult gpumergeresults = mergeRangesGpu(
            //     nextDataPtr->mergeRangesGpuHandle, 
            //     minhashHandle.multiranges.data(), 
            //     minhashHandle.multiranges.size(), 
            //     minhasherPtr->minparams.maps, 
            //     nextData.stream,
            //     MergeRangesKernelType::allcub
            // );
            // nvtx::pop_range();            

            // {
            //     //assert that gpu results are identical to cpu results
            //     const int total = minhashHandle.numResultsPerSequencePrefixSum.back();

            //     for(int i = 0; i < total; i++){
            //         const auto a = minhashHandle.multiallUniqueResults[i];
            //         const auto b = gpumergeresults.candidateIds[i];
            //         if(a != b){
            //             std::cerr << "error pos i = " << i << " cpu = " << a << ", gpu = " << b << "\n";
            //         }
            //         assert(a == b);
            //     }

            //     for(int i = begin; i < end; i++){
            //         const auto a = minhashHandle.numResultsPerSequence[i-begin];
            //         const auto b = gpumergeresults.candidateIdsPerSequence[i-begin];

            //         if(a != b){
            //             std::cerr << "error2 pos i = " << i << " cpu = " << a << ", gpu = " << b << "\n";
            //         }
            //         assert(a == b);
            //     }
                
            // }

            nvtx::push_range("remove self", 3);

            int initialNumberOfCandidates = 0;  

            auto multiresultbegin = minhashHandle.multiresults().begin();
            
            for(int i = begin; i < end; i++){
                const read_number readId = nextDataPtr->h_subject_read_ids[i];

                auto multiresultend = multiresultbegin + minhashHandle.numResultsPerSequence[i-begin];   
                auto readIdPos = std::lower_bound(multiresultbegin, multiresultend, readId);
                
                if(readIdPos != multiresultend && *readIdPos == readId) {
                    std::copy(readIdPos+1, multiresultend, readIdPos); //remove readId from range

                    minhashHandle.numResultsPerSequence[i-begin] -= 1;
                }

                multiresultbegin = multiresultend;

                initialNumberOfCandidates += minhashHandle.numResultsPerSequence[i-begin];
            }
            nvtx::pop_range();

            nextDataPtr->n_queries += initialNumberOfCandidates;

            decodedSubjectStringsPerThread[threadId] = std::move(decodedSubjectStrings);
        };

        cudaStreamSynchronize(nextData.stream); CUERR; //wait for D2H transfers of anchor data which is required for minhasher

        int numChunksRequired = batchData.threadsInThreadPool;

#if 1       
        numChunksRequired = batchData.threadPool->parallelFor(
            nextData.pforHandle,
            0, 
            nextData.n_subjects, 
            [=](auto begin, auto end, auto threadId){
                calculateMinhash(begin, end, threadId);
            }
        );
#else 

        calculateMinhash(0, nextData.n_subjects, 0);

#endif


        nextData.decodedSubjectStrings.clear();
        for(int i = 0; i < numChunksRequired; i++){
            nextData.decodedSubjectStrings.insert(
                nextData.decodedSubjectStrings.end(),
                decodedSubjectStringsPerThread[i].begin(),
                decodedSubjectStringsPerThread[i].end()
            );
        }

        nextData.h_candidate_read_ids.resize(nextData.n_queries);
        nextData.d_candidate_read_ids.resize(nextData.n_queries);
        nextData.h_candidates_per_subject.resize(nextData.n_subjects);
        nextData.d_candidates_per_subject.resize(nextData.n_subjects);
        nextData.h_candidates_per_subject_prefixsum.resize(nextData.n_subjects+1);
        nextData.d_candidates_per_subject_prefixsum.resize(nextData.n_subjects+1);  
        nextData.h_candidateContainsN.resize(nextData.n_queries);
        nextData.d_candidateContainsN.resize(nextData.n_queries);

        //copy candidate ids to pinned buffer, then to gpu
        auto destbegin = nextData.h_candidate_read_ids.get();
        for(int i = 0; i < numChunksRequired; i++){
            auto& minhashHandle = batchData.minhashHandles[i];

            for(int k = 0; k < int(minhashHandle.numResultsPerSequence.size()); k++){
                const int numCandidateIds = minhashHandle.numResultsPerSequence[k];               

                auto srcbegin = minhashHandle.multiresults().begin() + minhashHandle.numResultsPerSequencePrefixSum[k];
                auto srcend = srcbegin + numCandidateIds;
                auto destend = destbegin + numCandidateIds;

                destbegin = std::copy(srcbegin, srcend, destbegin);
            }            
        }

        // std::cerr << "candidate ids \n\n";
        // for(int i = 0; i < nextData.n_queries; i++){
        //     std::cerr << nextData.h_candidate_read_ids[i] << "\n";
        // }

        cudaMemcpyAsync(
            nextData.d_candidate_read_ids.get(),
            nextData.h_candidate_read_ids.get(),
            nextData.h_candidate_read_ids.sizeInBytes(),
            H2D,
            nextData.stream
        ); CUERR;

        nextData.h_candidates_per_subject_prefixsum[0] = 0;

        //make candidates per subject prefix sum
        int subject_index = 0;
        for(int i = 0; i < numChunksRequired; i++){
            auto& minhashHandle = batchData.minhashHandles[i];

            for(int k = 0; k < int(minhashHandle.numResultsPerSequence.size()); k++){
                const int numCandidateIds = minhashHandle.numResultsPerSequence[k];               

                nextData.h_candidates_per_subject[subject_index] = numCandidateIds;
                nextData.h_candidates_per_subject_prefixsum[subject_index+1] 
                    = nextData.h_candidates_per_subject_prefixsum[subject_index] + numCandidateIds;

                subject_index++;
            }            
        }

        // std::cerr << "num candidate ids \n\n";
        // for(int i = 0; i < nextData.n_subjects; i++){
        //     std::cerr << nextData.h_candidates_per_subject[i] << "\n";
        // }

        cudaMemcpyAsync(
            nextData.d_candidates_per_subject.get(),
            nextData.h_candidates_per_subject.get(),
            nextData.h_candidates_per_subject.sizeInBytes(),
            H2D,
            nextData.stream
        ); CUERR;

        cudaMemcpyAsync(
            nextData.d_candidates_per_subject_prefixsum.get(),
            nextData.h_candidates_per_subject_prefixsum.get(),
            nextData.h_candidates_per_subject_prefixsum.sizeInBytes(),
            H2D,
            nextData.stream
        ); CUERR;

        for(int i = 0; i < nextData.n_queries; i++){
            const read_number candidateReadId = nextData.h_candidate_read_ids[i];
            const bool containsN = readStorage.readContainsN(candidateReadId);
            nextData.h_candidateContainsN[i] = containsN;
        }

        // std::cerr << "contains \n\n";
        // for(int i = 0; i < nextData.n_queries; i++){
        //     std::cerr << nextData.h_candidateContainsN[i] << "\n";
        // }

        // std::exit(0);

        cudaMemcpyAsync(
            nextData.d_candidateContainsN,
            nextData.h_candidateContainsN,
            nextData.h_candidateContainsN.sizeInBytes(),
            H2D,
            nextData.stream
        ); CUERR;    
    }

    void determineCandidateReadIdsOfNextIterationGpu(
            Batch& batchData, 
            const Minhasher& minhasher, 
            const DistributedReadStorage& readStorage){

        NextIterationData& nextData = batchData.nextIterationData;

        //minhash the retrieved anchors to find candidate ids

        Batch* batchptr = &batchData;
        NextIterationData* nextDataPtr = &nextData;
        const Minhasher* minhasherPtr = &minhasher;

        nextData.n_queries = 0;

        //std::vector<std::vector<std::string>> decodedSubjectStringsPerThread(batchData.threadsInThreadPool);

        auto makeSignatures = [&, batchptr, nextDataPtr, minhasherPtr](int begin, int end, int threadId){

            auto& minhashHandle = batchptr->minhashHandles[threadId];                      

            std::vector<std::string> decodedSubjectStrings(end-begin);

            nvtx::push_range("decode sequences", 1);
            for(int i = begin; i < end; i++){
                const unsigned int* sequenceptr = nextDataPtr->h_subject_sequences_data.get() + i * batchptr->encodedSequencePitchInInts;
                const int sequencelength = nextDataPtr->h_subject_sequences_lengths[i];

                decodedSubjectStrings[i - begin] = get2BitString(sequenceptr, sequencelength);
            }
            nvtx::pop_range();


            nvtx::push_range("calculateMinhashSignatures", 5);
            minhasherPtr->calculateMinhashSignatures(
                minhashHandle,
                decodedSubjectStrings
            );
            nvtx::pop_range();

            //decodedSubjectStringsPerThread[threadId] = std::move(decodedSubjectStrings);
        };

        auto querySignatures = [&, batchptr, nextDataPtr, minhasherPtr](int begin, int end, int threadId){

            auto& minhashHandle = batchptr->minhashHandles[threadId];                      

            nvtx::push_range("queryPrecalculatedSignatures", 6);
            minhasherPtr->queryPrecalculatedSignatures(
                minhashHandle, 
                end - begin
            );
            nvtx::pop_range();
        };


        cudaStreamSynchronize(nextData.stream); CUERR; //wait for D2H transfers of anchor data which is required for minhasher

        int numChunksRequired = batchData.threadPool->parallelFor(
            nextData.pforHandle,
            0, 
            nextData.n_subjects, 
            [=](auto begin, auto end, auto threadId){
                makeSignatures(begin, end, threadId);
            }
        );
        batchData.threadPool->parallelFor(
            nextData.pforHandle,
            0, 
            nextData.n_subjects, 
            [=](auto begin, auto end, auto threadId){
                querySignatures(begin, end, threadId);
            }
        );

        std::vector<std::pair<const read_number*, const read_number*>> allRanges;
        std::vector<int> idsPerChunkPrefixSum(numChunksRequired+1, 0);

        int totalNumIds = 0;
        for(int i = 0; i < numChunksRequired; i++){
            allRanges.insert(
                allRanges.end(), 
                batchptr->minhashHandles[i].multiranges.begin(),
                batchptr->minhashHandles[i].multiranges.end()
            );

            for(const auto& range : batchptr->minhashHandles[i].multiranges){
                totalNumIds += std::distance(range.first, range.second);
            }

            idsPerChunkPrefixSum[i+1] = totalNumIds;
        }

        nextData.h_candidate_read_ids.resize(totalNumIds);
        nextData.d_candidate_read_ids.resize(totalNumIds);
        nextData.d_candidate_read_ids_tmp.resize(totalNumIds);
        nextData.h_candidates_per_subject.resize(nextData.n_subjects);
        nextData.d_candidates_per_subject.resize(nextData.n_subjects);
        nextData.h_candidates_per_subject_prefixsum.resize(nextData.n_subjects+1);
        nextData.d_candidates_per_subject_prefixsum.resize(nextData.n_subjects+1);
        //nextData.h_candidateContainsN.resize(totalNumIds);
        nextData.d_candidateContainsN.resize(totalNumIds);

        auto copyCandidateIdsToContiguousMem = [&](int begin, int end, int threadId){
            nvtx::push_range("copyCandidateIdsToContiguousMem", 1);
            for(int chunkId = begin; chunkId < end; chunkId++){
                const auto hostdatabegin = nextData.h_candidate_read_ids.get() + idsPerChunkPrefixSum[chunkId];
                const auto devicedatabegin = nextData.d_candidate_read_ids_tmp.get() + idsPerChunkPrefixSum[chunkId];
                const size_t elementsInChunk = idsPerChunkPrefixSum[chunkId+1] - idsPerChunkPrefixSum[chunkId];

                const auto& ranges = batchptr->minhashHandles[chunkId].multiranges;

                auto dest = hostdatabegin;

                const int lmax = ranges.size();
                for(int k = 0; k < lmax; k++){
                    constexpr int nextprefetch = 2;
                    if(k+nextprefetch < lmax){
                        __builtin_prefetch(ranges[k+nextprefetch].first, 0, 0);
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


        nvtx::push_range("gpumakeUniqueQueryResults", 2);
        mergeRangesGpuAsync(
            nextDataPtr->mergeRangesGpuHandle, 
            nextData.d_candidate_read_ids.get(),
            nextData.d_candidates_per_subject.get(),
            nextData.d_candidates_per_subject_prefixsum.get(),
            nextData.d_candidate_read_ids_tmp.get(),
            allRanges.data(), 
            allRanges.size(), 
            nextData.d_subject_read_ids.get(),
            minhasherPtr->minparams.maps, 
            nextData.stream,
            MergeRangesKernelType::allcub
        );

        cudaMemcpyAsync(
            nextData.h_candidates_per_subject_prefixsum.get(),
            nextData.d_candidates_per_subject_prefixsum.get(),
            sizeof(int) * (nextData.n_subjects + 1),
            D2H,
            nextData.stream
        ); CUERR;

        cudaStreamSynchronize(nextData.stream); CUERR;

        nextData.n_queries = nextData.h_candidates_per_subject_prefixsum[nextData.n_subjects];

        nextData.h_candidate_read_ids.resize(nextData.n_queries);
        nextData.d_candidate_read_ids.resize(nextData.n_queries);

        nextData.d_candidateContainsN.resize(nextData.n_queries);

        cudaMemcpyAsync(
            nextData.h_candidate_read_ids.get(),
            nextData.d_candidate_read_ids.get(),
            sizeof(read_number) * nextData.n_queries,
            D2H,
            nextData.stream
        ); CUERR;

        cudaMemcpyAsync(
            nextData.h_candidates_per_subject.get(),
            nextData.d_candidates_per_subject.get(),
            sizeof(int) * (nextData.n_subjects),
            D2H,
            nextData.stream
        ); CUERR;        

        readStorage.readsContainN_async(
            nextData.d_candidateContainsN.get(), 
            nextData.d_candidate_read_ids.get(), 
            //nextData.d_candidates_per_subject_prefixsum.get() + nextData.n_subjects,
            nextData.n_queries, 
            nextData.stream
        );

        cudaStreamSynchronize(nextData.stream); CUERR;
        nvtx::pop_range();

        

        //nextData.h_candidateContainsN.resize(nextData.n_queries);

        // cudaMemcpyAsync(
        //     nextData.h_candidateContainsN,
        //     nextData.d_candidateContainsN,
        //     nextData.n_queries * sizeof(bool),
        //     H2D,
        //     nextData.stream
        // ); CUERR;  

        // cudaDeviceSynchronize(); CUERR;

        // for(int i = 0; i < nextData.n_queries; i++){
        //     const read_number candidateReadId = nextData.h_candidate_read_ids[i];
        //     const bool containsN = readStorage.readContainsN(candidateReadId);
        //     assert(containsN == nextData.h_candidateContainsN[i]);
        // }

        // std::cerr << "contains \n\n";
        // for(int i = 0; i < nextData.n_queries; i++){
        //     std::cerr << nextData.h_candidateContainsN[i] << "\n";
        // }

        // std::exit(0);

        // cudaMemcpyAsync(
        //     nextData.d_candidateContainsN,
        //     nextData.h_candidateContainsN,
        //     nextData.h_candidateContainsN.sizeInBytes(),
        //     H2D,
        //     nextData.stream
        // ); CUERR;    
    }



    void getNextBatchOfSubjectsAndDetermineCandidateReadIds(Batch& batchData){
        Batch* batchptr = &batchData;

        auto getDataForNextIteration = [batchptr](){
            nvtx::push_range("getSubjectDataOfNextIteration",1);
            getSubjectDataOfNextIteration(
                *batchptr, 
                batchptr->transFuncData->correctionOptions.batchsize,
                *batchptr->transFuncData->readStorage
            );
            nvtx::pop_range();

            nvtx::push_range("determineCandidateReadIdsOfNextIteration",2);

            if(batchptr->nextIterationData.n_subjects > 0){
                determineCandidateReadIdsOfNextIterationGpu(
                    *batchptr, 
                    *batchptr->transFuncData->minhasher,
                    *batchptr->transFuncData->readStorage
                );
                cudaStreamSynchronize(batchptr->nextIterationData.stream); CUERR;
                batchptr->nextIterationData.syncFlag.signal();
            }else{
                batchptr->nextIterationData.n_queries = 0;
                batchptr->nextIterationData.syncFlag.signal();
            }
            nvtx::pop_range();
        };

        if(batchData.isFirstIteration){
            batchData.nextIterationData.syncFlag.setBusy();

            getDataForNextIteration();        
         
            batchData.isFirstIteration = false;
        }else{
            batchData.nextIterationData.syncFlag.wait(); //wait until data is available
        }

        batchData.updateFromIterationData(batchData.nextIterationData);        
            
        batchData.nextIterationData.syncFlag.setBusy();
#if 1   
        //asynchronously prepare data for next iteration
        batchData.backgroundWorker->enqueue(
            getDataForNextIteration
        );
#else  
        getDataForNextIteration();
#endif

    }

    void resizeArrays(Batch& batchData){

        //allocate memory required for batch processing

        auto& dataArrays = batchData.dataArrays;
        const auto& transFuncData = *(batchData.transFuncData);
        auto& streams = batchData.streams;

        nvtx::push_range("set_problem_dimensions", 4);

        const int min_overlap = std::max(1, std::max(transFuncData.goodAlignmentProperties.min_overlap, 
            int(transFuncData.sequenceFileProperties.maxSequenceLength 
                * transFuncData.goodAlignmentProperties.min_overlap_ratio)));

        const int sequence_pitch = batchData.decodedSequencePitchInBytes;

        int msa_max_column_count = (3*transFuncData.sequenceFileProperties.maxSequenceLength - 2*min_overlap);
        batchData.msa_pitch = SDIV(sizeof(char)*msa_max_column_count, 4) * 4;
        batchData.msa_weights_pitch = SDIV(sizeof(float)*msa_max_column_count, 4) * 4;
        size_t msa_weights_pitch_floats = batchData.msa_weights_pitch / sizeof(float);

        if(batchData.max_n_queries < batchData.n_queries){
            batchData.max_n_queries = batchData.n_queries;
            std::cerr << "resize necessary\n";
        }
        
        //sequence input data
        //following arrays are initialized by next iteration data:
        //h_subject_sequences_data, h_candidates_per_subject, h_candidates_per_subject_prefixsum
        //h_subject_read_ids, h_candidate_read_ids
        //d_subject_sequences_data, d_candidates_per_subject, d_candidates_per_subject_prefixsum
        //d_subject_read_ids, d_candidate_read_ids

        dataArrays.h_candidate_sequences_data.resize(batchData.n_queries * batchData.encodedSequencePitchInInts);
        dataArrays.h_transposedCandidateSequencesData.resize(batchData.n_queries * batchData.encodedSequencePitchInInts);
        dataArrays.h_subject_sequences_lengths.resize(batchData.n_subjects);
        dataArrays.h_candidate_sequences_lengths.resize(batchData.n_queries);
        dataArrays.h_anchorIndicesOfCandidates.resize(batchData.n_queries);

        dataArrays.d_subject_sequences_data.resize(batchData.n_subjects * batchData.encodedSequencePitchInInts);
        dataArrays.d_candidate_sequences_data.resize(batchData.n_queries * batchData.encodedSequencePitchInInts);
        dataArrays.d_transposedCandidateSequencesData.resize(batchData.n_queries * batchData.encodedSequencePitchInInts);
        dataArrays.d_subject_sequences_lengths.resize(batchData.n_subjects);
        dataArrays.d_candidate_sequences_lengths.resize(batchData.n_queries);
        dataArrays.d_anchorIndicesOfCandidates.resize(batchData.n_queries);

        

        //alignment output

        dataArrays.h_alignment_scores.resize(2*batchData.n_queries);
        dataArrays.h_alignment_overlaps.resize(2*batchData.n_queries);
        dataArrays.h_alignment_shifts.resize(2*batchData.n_queries);
        dataArrays.h_alignment_nOps.resize(2*batchData.n_queries);
        dataArrays.h_alignment_isValid.resize(2*batchData.n_queries);
        dataArrays.h_alignment_best_alignment_flags.resize(batchData.n_queries);

        dataArrays.d_alignment_scores.resize(2*batchData.n_queries);
        dataArrays.d_alignment_overlaps.resize(2*batchData.n_queries);
        dataArrays.d_alignment_shifts.resize(2*batchData.n_queries);
        dataArrays.d_alignment_nOps.resize(2*batchData.n_queries);
        dataArrays.d_alignment_isValid.resize(2*batchData.n_queries);
        dataArrays.d_alignment_best_alignment_flags.resize(batchData.n_queries);

        // candidate indices

        dataArrays.h_indices.resize(batchData.n_queries);
        dataArrays.h_indices_per_subject.resize(batchData.n_subjects);
        dataArrays.h_num_indices.resize(1);

        dataArrays.d_indices.resize(batchData.n_queries);
        dataArrays.d_indices_per_subject.resize(batchData.n_subjects);
        dataArrays.d_num_indices.resize(1);
        dataArrays.d_indices_tmp.resize(batchData.n_queries);
        dataArrays.d_indices_per_subject_tmp.resize(batchData.n_subjects);
        dataArrays.d_num_indices_tmp.resize(1);

        dataArrays.h_indices_of_corrected_subjects.resize(batchData.n_subjects);
        dataArrays.h_num_indices_of_corrected_subjects.resize(1);
        dataArrays.d_indices_of_corrected_subjects.resize(batchData.n_subjects);
        dataArrays.d_num_indices_of_corrected_subjects.resize(1);

        dataArrays.h_editsPerCorrectedSubject.resize(batchData.n_subjects * batchData.maxNumEditsPerSequence);
        dataArrays.h_numEditsPerCorrectedSubject.resize(batchData.n_subjects);
        dataArrays.h_anchorContainsN.resize(batchData.n_subjects);

        dataArrays.d_editsPerCorrectedSubject.resize(batchData.n_subjects * batchData.maxNumEditsPerSequence);
        dataArrays.d_numEditsPerCorrectedSubject.resize(batchData.n_subjects);
        dataArrays.d_anchorContainsN.resize(batchData.n_subjects);

        dataArrays.h_editsPerCorrectedCandidate.resize(batchData.n_queries * batchData.maxNumEditsPerSequence);
        dataArrays.h_numEditsPerCorrectedCandidate.resize(batchData.n_queries);
        dataArrays.h_candidateContainsN.resize(batchData.n_queries);

        dataArrays.d_editsPerCorrectedCandidate.resize(batchData.n_queries * batchData.maxNumEditsPerSequence);
        dataArrays.d_numEditsPerCorrectedCandidate.resize(batchData.n_queries);
        dataArrays.d_candidateContainsN.resize(batchData.n_queries);



        //qualitiy scores
        if(transFuncData.correctionOptions.useQualityScores) {
            dataArrays.h_subject_qualities.resize(batchData.n_subjects * batchData.qualityPitchInBytes);
            dataArrays.h_candidate_qualities.resize(batchData.n_queries * batchData.qualityPitchInBytes);

            dataArrays.d_subject_qualities.resize(batchData.n_subjects * batchData.qualityPitchInBytes);
            dataArrays.d_candidate_qualities.resize(batchData.n_queries * batchData.qualityPitchInBytes);
            dataArrays.d_candidate_qualities_transposed.resize(batchData.n_queries * batchData.qualityPitchInBytes);
            dataArrays.d_candidate_qualities_tmp.resize(batchData.n_queries * batchData.qualityPitchInBytes);
        }


        //correction results

        dataArrays.h_corrected_subjects.resize(batchData.n_subjects * sequence_pitch);
        dataArrays.h_corrected_candidates.resize(batchData.n_queries * sequence_pitch);
        dataArrays.h_num_corrected_candidates.resize(batchData.n_subjects);
        dataArrays.h_subject_is_corrected.resize(batchData.n_subjects);
        dataArrays.h_indices_of_corrected_candidates.resize(batchData.n_queries);
        dataArrays.h_num_uncorrected_positions_per_subject.resize(batchData.n_subjects);
        dataArrays.h_uncorrected_positions_per_subject.resize(batchData.n_subjects * transFuncData.sequenceFileProperties.maxSequenceLength);

        dataArrays.d_corrected_subjects.resize(batchData.n_subjects * sequence_pitch);
        dataArrays.d_corrected_candidates.resize(batchData.n_queries * sequence_pitch);
        dataArrays.d_num_corrected_candidates.resize(batchData.n_subjects);
        dataArrays.d_subject_is_corrected.resize(batchData.n_subjects);
        dataArrays.d_indices_of_corrected_candidates.resize(batchData.n_queries);
        dataArrays.d_num_uncorrected_positions_per_subject.resize(batchData.n_subjects);
        dataArrays.d_uncorrected_positions_per_subject.resize(batchData.n_subjects * transFuncData.sequenceFileProperties.maxSequenceLength);

        dataArrays.h_is_high_quality_subject.resize(batchData.n_subjects);
        dataArrays.h_high_quality_subject_indices.resize(batchData.n_subjects);
        dataArrays.h_num_high_quality_subject_indices.resize(1);

        dataArrays.d_is_high_quality_subject.resize(batchData.n_subjects);
        dataArrays.d_high_quality_subject_indices.resize(batchData.n_subjects);
        dataArrays.d_num_high_quality_subject_indices.resize(1);

        //multiple sequence alignment

        dataArrays.h_consensus.resize(batchData.n_subjects * batchData.msa_pitch);
        dataArrays.h_support.resize(batchData.n_subjects * msa_weights_pitch_floats);
        dataArrays.h_coverage.resize(batchData.n_subjects * msa_weights_pitch_floats);
        dataArrays.h_origWeights.resize(batchData.n_subjects * msa_weights_pitch_floats);
        dataArrays.h_origCoverages.resize(batchData.n_subjects * msa_weights_pitch_floats);
        dataArrays.h_msa_column_properties.resize(batchData.n_subjects);
        dataArrays.h_counts.resize(batchData.n_subjects * 4 * msa_weights_pitch_floats);
        dataArrays.h_weights.resize(batchData.n_subjects * 4 * msa_weights_pitch_floats);

        dataArrays.d_consensus.resize(batchData.n_subjects * batchData.msa_pitch);
        dataArrays.d_support.resize(batchData.n_subjects * msa_weights_pitch_floats);
        dataArrays.d_coverage.resize(batchData.n_subjects * msa_weights_pitch_floats);
        dataArrays.d_origWeights.resize(batchData.n_subjects * msa_weights_pitch_floats);
        dataArrays.d_origCoverages.resize(batchData.n_subjects * msa_weights_pitch_floats);
        dataArrays.d_msa_column_properties.resize(batchData.n_subjects);
        dataArrays.d_counts.resize(batchData.n_subjects * 4 * msa_weights_pitch_floats);
        dataArrays.d_weights.resize(batchData.n_subjects * 4 * msa_weights_pitch_floats);


        dataArrays.d_canExecute.resize(1);

        nvtx::pop_range();

        std::size_t temp_storage_bytes = 0;
        std::size_t max_temp_storage_bytes = 0;
        cub::DeviceHistogram::HistogramRange((void*)nullptr, temp_storage_bytes,
                    (int*)nullptr, (int*)nullptr,
                    batchData.n_subjects+1,
                    (int*)nullptr,
                    batchData.n_queries,
                    streams[primary_stream_index]); CUERR;

        max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

        cub::DeviceSelect::Flagged((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                    (bool*)nullptr, (int*)nullptr, (int*)nullptr,
                    batchData.n_queries,
                    streams[primary_stream_index]); CUERR;

        max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

        cub::DeviceScan::ExclusiveSum((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                    (int*)nullptr,
                    batchData.n_subjects,
                    streams[primary_stream_index]); CUERR;

        cub::DeviceScan::InclusiveSum((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                    (int*)nullptr,
                    batchData.n_subjects,
                    streams[primary_stream_index]); CUERR;

        cub::DeviceSegmentedRadixSort::SortPairs((void*)nullptr,
                                                temp_storage_bytes,
                                                (const char*) nullptr,
                                                (char*)nullptr,
                                                (const int*)nullptr,
                                                (int*)nullptr,
                                                batchData.n_queries,
                                                batchData.n_subjects,
                                                (const int*)nullptr,
                                                (const int*)nullptr,
                                                0,
                                                3,
                                                streams[primary_stream_index]);

        max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);
        temp_storage_bytes = max_temp_storage_bytes;
        dataArrays.set_cub_temp_storage_size(max_temp_storage_bytes);

        call_fill_kernel_async(
            dataArrays.d_canExecute.get(),
            1,
            true,
            streams[primary_stream_index]
        );
    }





    void getCandidateSequenceData(Batch& batchData, const DistributedReadStorage& readStorage){

        cudaSetDevice(batchData.deviceId); CUERR;

        const auto& transFuncData = *batchData.transFuncData;

        DataArrays& dataArrays = batchData.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = batchData.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batchData.events;

        readStorage.gatherSequenceLengthsToGpuBufferAsync(
                                        dataArrays.d_subject_sequences_lengths.get(),
                                        batchData.deviceId,
                                        dataArrays.d_subject_read_ids.get(),
                                        batchData.n_subjects,   
                                        streams[primary_stream_index]);

        readStorage.gatherSequenceLengthsToGpuBufferAsync(
                                        dataArrays.d_candidate_sequences_lengths.get(),
                                        batchData.deviceId,
                                        dataArrays.d_candidate_read_ids.get(),
                                        batchData.n_queries,            
                                        streams[primary_stream_index]);

        readStorage.gatherSequenceDataToGpuBufferAsync(
            batchData.threadPool,
            batchData.candidateSequenceGatherHandle,
            dataArrays.d_candidate_sequences_data.get(),
            batchData.encodedSequencePitchInInts,
            dataArrays.h_candidate_read_ids,
            dataArrays.d_candidate_read_ids,
            batchData.n_queries,
            batchData.deviceId,
            streams[primary_stream_index],
            transFuncData.runtimeOptions.nCorrectorThreads);

        call_transpose_kernel(
            dataArrays.d_transposedCandidateSequencesData.get(), 
            dataArrays.d_candidate_sequences_data.get(), 
            batchData.n_queries, 
            batchData.encodedSequencePitchInInts, 
            batchData.encodedSequencePitchInInts, 
            streams[primary_stream_index]
        );


        cudaEventRecord(events[alignment_data_transfer_h2d_finished_event_index], streams[primary_stream_index]); CUERR;

        cudaStreamWaitEvent(streams[secondary_stream_index], events[alignment_data_transfer_h2d_finished_event_index], 0) ;

        // cudaMemcpyAsync(dataArrays.h_subject_sequences_data,
        //                 dataArrays.d_subject_sequences_data,
        //                 dataArrays.d_subject_sequences_data.sizeInBytes(),
        //                 D2H,
        //                 streams[secondary_stream_index]); CUERR;

        // cudaMemcpyAsync(dataArrays.h_candidate_sequences_data,
        //                 dataArrays.d_candidate_sequences_data,
        //                 dataArrays.d_candidate_sequences_data.sizeInBytes(),
        //                 D2H,
        //                 streams[secondary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_subject_sequences_lengths,
                        dataArrays.d_subject_sequences_lengths,
                        dataArrays.d_subject_sequences_lengths.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_candidate_sequences_lengths,
                        dataArrays.d_candidate_sequences_lengths,
                        dataArrays.d_candidate_sequences_lengths.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

    }


	void getCandidateAlignments(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

		DataArrays& dataArrays = batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        {
            int n_queries = batch.n_queries;
            int n_anchors = batch.n_subjects;
            int* d_anchorIndicesOfCandidates = dataArrays.d_anchorIndicesOfCandidates.get();
            int* d_candidates_per_subject = dataArrays.d_candidates_per_subject.get();
            int* d_candidates_per_subject_prefixsum = dataArrays.d_candidates_per_subject_prefixsum.get();

            generic_kernel
                <<<256, 128, 0, streams[primary_stream_index]>>>
            ([=] __device__ {
                for(int anchorIndex = blockIdx.x; anchorIndex < n_anchors; anchorIndex += 256){
                    const int offset = d_candidates_per_subject_prefixsum[anchorIndex];
                    const int numCandidatesOfAnchor = d_candidates_per_subject[anchorIndex];
                    int* const beginptr = &d_anchorIndicesOfCandidates[offset];

                    for(int localindex = threadIdx.x; localindex < numCandidatesOfAnchor; localindex += 128){
                        beginptr[localindex] = anchorIndex;
                    }
                }
                
            });

        }

        //cudaStreamWaitEvent(streams[primary_stream_index], events[alignment_data_transfer_h2d_finished_event_index], 0); CUERR;
        
        call_popcount_shifted_hamming_distance_kernel_async(
                    dataArrays.getDeviceAlignmentResultPointers(),
                    dataArrays.getDeviceSequencePointers(),
                    dataArrays.d_candidates_per_subject_prefixsum,
                    dataArrays.h_candidates_per_subject,
                    dataArrays.d_candidates_per_subject,
                    dataArrays.d_anchorIndicesOfCandidates.get(),
                    batch.n_subjects,
                    batch.n_queries,
                    transFuncData.sequenceFileProperties.maxSequenceLength,
                    batch.encodedSequencePitchInInts,
                    transFuncData.goodAlignmentProperties.min_overlap,
                    transFuncData.goodAlignmentProperties.maxErrorRate,
                    transFuncData.goodAlignmentProperties.min_overlap_ratio,
                    transFuncData.correctionOptions.estimatedErrorrate,
                    //batch.maxSubjectLength,
                    streams[primary_stream_index],
                    batch.kernelLaunchHandle);

        cudaEventRecord(events[alignments_finished_event_index], streams[primary_stream_index]); CUERR;

		//Compare each forward alignment with the correspoding reverse complement alignment and keep the best, if any.
		//    If reverse complement is the best, it is copied into the first half, replacing the forward alignment

        // call_cuda_find_best_alignment_kernel_async_exp(
        //             dataArrays.getDeviceAlignmentResultPointers(),
        //             dataArrays.getDeviceSequencePointers(),
        //             dataArrays.d_candidates_per_subject_prefixsum.get(),
        //             batch.n_subjects,
		// 			batch.n_queries,
        //             transFuncData.goodAlignmentProperties.min_overlap_ratio,
        //             transFuncData.goodAlignmentProperties.min_overlap,
        //             transFuncData.correctionOptions.estimatedErrorrate,
        //             streams[primary_stream_index],
        //             batch.kernelLaunchHandle,
        //             dataArrays.h_subject_read_ids[0]);

		//choose the most appropriate subset of alignments from the good alignments.
		//This sets d_alignment_best_alignment_flags[i] = BestAlignment_t::None for all non-appropriate alignments

		call_cuda_filter_alignments_by_mismatchratio_kernel_async(
					dataArrays.getDeviceAlignmentResultPointers(),
					dataArrays.d_candidates_per_subject_prefixsum.get(),
					batch.n_subjects,
					batch.n_queries,
					transFuncData.correctionOptions.estimatedErrorrate,
					transFuncData.correctionOptions.estimatedCoverage * transFuncData.correctionOptions.m_coverage,
					streams[primary_stream_index],
					batch.kernelLaunchHandle);


        callSelectIndicesOfGoodCandidatesKernelAsync(
            dataArrays.d_indices.get(),
            dataArrays.d_indices_per_subject.get(),
            dataArrays.d_num_indices.get(),
            dataArrays.d_alignment_best_alignment_flags.get(),
            dataArrays.d_candidates_per_subject.get(),
            dataArrays.d_candidates_per_subject_prefixsum.get(),
            dataArrays.d_anchorIndicesOfCandidates.get(),
            batch.n_subjects,
			batch.n_queries,
            streams[primary_stream_index],
			batch.kernelLaunchHandle
        );

        // cudaEventRecord(events[indices_transfer_finished_event_index], streams[primary_stream_index]); CUERR;
        // cudaStreamWaitEvent(streams[secondary_stream_index], events[indices_transfer_finished_event_index], 0); CUERR;

        // cudaMemcpyAsync(dataArrays.h_num_indices,
        //                 dataArrays.d_num_indices,
        //                 sizeof(int),
        //                 D2H,
        //                 streams[secondary_stream_index]); CUERR;

        // cudaMemcpyAsync(dataArrays.h_indices,
        //                 dataArrays.d_indices,
        //                 dataArrays.d_indices.sizeInBytes(),
        //                 D2H,
        //                 streams[secondary_stream_index]); CUERR;

        // cudaMemcpyAsync(dataArrays.h_indices_per_subject,
        //                 dataArrays.d_indices_per_subject,
        //                 dataArrays.d_indices_per_subject.sizeInBytes(),
        //                 D2H,
        //                 streams[secondary_stream_index]); CUERR;

        // cudaEventRecord(events[indices_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;
       

        

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

        //std::cerr << "After alignment: " << *dataArrays.h_num_indices << " / " << dataArrays.n_queries << "\n";
	}


    void getQualities(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        DataArrays& dataArrays = batch.dataArrays;

		std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;

        const auto* gpuReadStorage = transFuncData.readStorage;

		if(transFuncData.correctionOptions.useQualityScores) {

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

            cudaEventRecord(events[quality_transfer_finished_event_index], streams[primary_stream_index]); CUERR;

            // cudaStreamWaitEvent(streams[secondary_stream_index], events[quality_transfer_finished_event_index], 0); CUERR;

            // cudaMemcpyAsync(dataArrays.h_subject_qualities,
            //                 dataArrays.d_subject_qualities,
            //                 dataArrays.d_subject_qualities.sizeInBytes(),
            //                 D2H,
            //                 streams[secondary_stream_index]);

            // cudaMemcpyAsync(dataArrays.h_candidate_qualities,
            //                 dataArrays.d_candidate_qualities,
            //                 dataArrays.d_candidate_qualities.sizeInBytes(),
            //                 D2H,
            //                 streams[secondary_stream_index]);
        }

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
	}


    void buildMultipleSequenceAlignment(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        DataArrays& dataArrays = batch.dataArrays;

        // if(transFuncData.correctionOptions.useQualityScores){
		//      cudaStreamWaitEvent(streams[primary_stream_index], events[quality_transfer_finished_event_index], 0); CUERR;
        // }

		const float desiredAlignmentMaxErrorRate = transFuncData.goodAlignmentProperties.maxErrorRate;
        //const float desiredAlignmentMaxErrorRate = transFuncData.correctionOptions.estimatedErrorrate * 4.0f;

        //std::cout << "msa_init" << std::endl;

        

        build_msa_async(dataArrays.getDeviceMSAPointers(),
                        dataArrays.getDeviceAlignmentResultPointers(),
                        dataArrays.getDeviceSequencePointers(),
                        dataArrays.getDeviceQualityPointers(),
                        dataArrays.d_candidates_per_subject_prefixsum,
                        dataArrays.d_indices,
                        dataArrays.d_indices_per_subject,
                        batch.n_subjects,
                        batch.n_queries,
                        dataArrays.d_num_indices,
                        1.0f,
                        transFuncData.correctionOptions.useQualityScores,
                        desiredAlignmentMaxErrorRate,
                        transFuncData.sequenceFileProperties.maxSequenceLength,
                        batch.encodedSequencePitchInInts,
                        batch.qualityPitchInBytes,
                        batch.msa_pitch,
                        batch.msa_weights_pitch,
                        dataArrays.d_canExecute,
                        streams[primary_stream_index],
                        batch.kernelLaunchHandle);

        //batch.dataArrays.copyEverythingToHostForDebugging();

        //At this point the msa is built
        cudaEventRecord(events[msa_build_finished_event_index], streams[primary_stream_index]); CUERR;

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
	}





    void removeCandidatesOfDifferentRegionFromMSA(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        DataArrays& dataArrays = batch.dataArrays;

        constexpr int max_num_minimizations = 5;

        const float desiredAlignmentMaxErrorRate = transFuncData.goodAlignmentProperties.maxErrorRate;
        //const float desiredAlignmentMaxErrorRate = transFuncData.correctionOptions.estimatedErrorrate * 4.0f;

        bool* d_shouldBeKept = nullptr; //flag per candidate which shows whether the candidate should remain in the msa, or not.

        cubCachingAllocator.DeviceAllocate(
            (void**)&d_shouldBeKept, 
            sizeof(bool) * batch.n_queries, 
            streams[primary_stream_index]
        ); CUERR;



        // int* fooindices;
        // int* fooindicespersubject;
        // int* foonumindices;

        // cudaMallocManaged(&fooindices, sizeof(int) *batch.n_queries); CUERR;
        // cudaMallocManaged(&fooindicespersubject, sizeof(int) *batch.n_subjects); CUERR;
        // cudaMallocManaged(&foonumindices, sizeof(int)); CUERR;


        for(int iteration = 0; iteration < max_num_minimizations; iteration++){

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
                desiredAlignmentMaxErrorRate,
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

            build_msa_async(dataArrays.getDeviceMSAPointers(),
                            dataArrays.getDeviceAlignmentResultPointers(),
                            dataArrays.getDeviceSequencePointers(),
                            dataArrays.getDeviceQualityPointers(),
                            dataArrays.d_candidates_per_subject_prefixsum,
                            dataArrays.d_indices,
                            dataArrays.d_indices_per_subject_tmp,
                            batch.n_subjects,
                            batch.n_queries,
                            dataArrays.d_num_indices,
                            0.05f, //
                            transFuncData.correctionOptions.useQualityScores,
                            desiredAlignmentMaxErrorRate,
                            transFuncData.sequenceFileProperties.maxSequenceLength,
                            batch.encodedSequencePitchInInts,
                            batch.qualityPitchInBytes,
                            batch.msa_pitch,
                            batch.msa_weights_pitch,
                            dataArrays.d_canExecute,
                            streams[primary_stream_index],
                            batch.kernelLaunchHandle);
        }

        cubCachingAllocator.DeviceFree(d_shouldBeKept); CUERR;
        
        // {
        //     //std::cerr << "minimization finished\n";

        //     cudaEventRecord(events[msa_build_finished_event_index], streams[primary_stream_index]); CUERR;
        //     cudaStreamWaitEvent(streams[secondary_stream_index], events[msa_build_finished_event_index], 0); CUERR;

        //     cudaMemcpyAsync(dataArrays.h_num_indices,
        //                     dataArrays.d_num_indices,
        //                     dataArrays.d_num_indices.sizeInBytes(),
        //                     D2H,
        //                     streams[secondary_stream_index]); CUERR;

        //     cudaMemcpyAsync(dataArrays.h_indices,
        //                     dataArrays.d_indices,
        //                     dataArrays.d_indices.sizeInBytes(),
        //                     D2H,
        //                     streams[secondary_stream_index]); CUERR;

        //     cudaMemcpyAsync(dataArrays.h_indices_per_subject,
        //                     dataArrays.d_indices_per_subject,
        //                     dataArrays.d_indices_per_subject.sizeInBytes(),
        //                     D2H,
        //                     streams[secondary_stream_index]); CUERR;

        //                 //update host qscores accordingly
        //                 /*cudaMemcpyAsync(dataArrays.h_candidate_qualities,
        //                                 dataArrays.d_candidate_qualities,
        //                                 dataArrays.d_candidate_qualities.sizeInBytes(),
        //                                 D2H,
        //                                 streams[secondary_stream_index]);*/

        //     cudaEventRecord(events[indices_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;
        // }


        //At this point the msa is built, maybe minimized, and is ready to be used for correction

        cudaEventRecord(events[msa_build_finished_event_index], streams[primary_stream_index]); CUERR;

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
    }


	void correctSubjects(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;

        cudaSetDevice(batch.deviceId); CUERR;

		DataArrays& dataArrays = batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

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

        call_msa_correct_subject_implicit_kernel_async(
                    dataArrays.getDeviceMSAPointers(),
                    dataArrays.getDeviceAlignmentResultPointers(),
                    dataArrays.getDeviceSequencePointers(),
                    dataArrays.getDeviceCorrectionResultPointers(),
                    dataArrays.d_indices,
                    dataArrays.d_indices_per_subject,
                    batch.n_subjects,
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
                    batch.kernelLaunchHandle);

        cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], events[correction_finished_event_index], 0); CUERR;

        cudaMemcpyAsync(dataArrays.h_corrected_subjects,
                        dataArrays.d_corrected_subjects,
                        dataArrays.d_corrected_subjects.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_subject_is_corrected,
                        dataArrays.d_subject_is_corrected,
                        dataArrays.d_subject_is_corrected.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_is_high_quality_subject,
                        dataArrays.d_is_high_quality_subject,
                        dataArrays.d_is_high_quality_subject.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_num_uncorrected_positions_per_subject,
                        dataArrays.d_num_uncorrected_positions_per_subject,
                        dataArrays.d_num_uncorrected_positions_per_subject.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_uncorrected_positions_per_subject,
                        dataArrays.d_uncorrected_positions_per_subject,
                        dataArrays.d_uncorrected_positions_per_subject.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        size_t cubTempSize = dataArrays.d_cub_temp_storage.sizeInBytes();

        cub::DeviceSelect::Flagged(
            dataArrays.d_cub_temp_storage.get(),
            cubTempSize,
            cub::CountingInputIterator<int>(0),
            dataArrays.d_subject_is_corrected.get(),
            dataArrays.d_indices_of_corrected_subjects.get(),
            dataArrays.d_num_indices_of_corrected_subjects.get(),
            batch.n_subjects,
            streams[primary_stream_index]
        ); CUERR;

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
            batch.n_subjects,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );

        cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], events[correction_finished_event_index], 0); CUERR;

        cudaMemcpyAsync(
            dataArrays.h_editsPerCorrectedSubject,
            dataArrays.d_editsPerCorrectedSubject,
            dataArrays.d_editsPerCorrectedSubject.sizeInBytes(),
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(
            dataArrays.h_numEditsPerCorrectedSubject,
            dataArrays.d_numEditsPerCorrectedSubject,
            dataArrays.d_numEditsPerCorrectedSubject.sizeInBytes(),
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(
            dataArrays.h_indices_of_corrected_subjects,
            dataArrays.d_indices_of_corrected_subjects,
            dataArrays.d_indices_of_corrected_subjects.sizeInBytes(),
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(
            dataArrays.h_num_indices_of_corrected_subjects,
            dataArrays.d_num_indices_of_corrected_subjects,
            dataArrays.d_num_indices_of_corrected_subjects.sizeInBytes(),
            D2H,
            streams[secondary_stream_index]
        ); CUERR;

		cudaEventRecord(events[result_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;

		if(transFuncData.correctionOptions.correctCandidates) {
            // find subject ids of subjects with high quality multiple sequence alignment

            size_t cubTempSize = dataArrays.d_cub_temp_storage.sizeInBytes();

            auto isHqSubject = [] __device__ (const AnchorHighQualityFlag& flag){
                return flag.hq();
            };

            cub::TransformInputIterator<bool,decltype(isHqSubject), AnchorHighQualityFlag*>
                d_isHqSubject(dataArrays.d_is_high_quality_subject,
                                isHqSubject);

            cub::DeviceSelect::Flagged(dataArrays.d_cub_temp_storage.get(),
                        cubTempSize,
                        cub::CountingInputIterator<int>(0),
                        d_isHqSubject,
                        dataArrays.d_high_quality_subject_indices.get(),
                        dataArrays.d_num_high_quality_subject_indices.get(),
                        batch.n_subjects,
                        streams[primary_stream_index]); CUERR;

            cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
            cudaStreamWaitEvent(streams[secondary_stream_index], events[correction_finished_event_index], 0); CUERR;

            cudaMemcpyAsync(dataArrays.h_high_quality_subject_indices,
                            dataArrays.d_high_quality_subject_indices,
                            dataArrays.d_high_quality_subject_indices.sizeInBytes(),
                            D2H,
                            streams[secondary_stream_index]); CUERR;

            cudaMemcpyAsync(dataArrays.h_num_high_quality_subject_indices,
                            dataArrays.d_num_high_quality_subject_indices,
                            dataArrays.d_num_high_quality_subject_indices.sizeInBytes(),
                            D2H,
                            streams[secondary_stream_index]); CUERR;

            cudaEventRecord(events[result_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;

            //cudaStreamWaitEvent(streams[primary_stream_index], events[indices_transfer_finished_event_index], 0); CUERR;

            //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
		}

        
	}



    void correctCandidates(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;

        cudaSetDevice(batch.deviceId); CUERR;

        DataArrays& dataArrays = batch.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        const float min_support_threshold = 1.0f-3.0f*transFuncData.correctionOptions.estimatedErrorrate;
        // coverage is always >= 1
        const float min_coverage_threshold = std::max(1.0f,
                    transFuncData.correctionOptions.m_coverage / 6.0f * transFuncData.correctionOptions.estimatedCoverage);
        const int new_columns_to_correct = transFuncData.correctionOptions.new_columns_to_correct;


        //wait for transfer of h_indices_per_subject to host
        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
        //cudaEventSynchronize(events[indices_transfer_finished_event_index]); CUERR;
#if 0
        call_msa_correct_candidates_kernel_async(
                dataArrays.getDeviceMSAPointers(),
                dataArrays.getDeviceAlignmentResultPointers(),
                dataArrays.getDeviceSequencePointers(),
                dataArrays.getDeviceCorrectionResultPointers(),
                dataArrays.d_indices,
                dataArrays.d_indices_per_subject,
                dataArrays.d_candidates_per_subject_prefixsum,
                batch.n_subjects,
                batch.n_queries,
                dataArrays.d_num_indices,
                batch.encodedSequencePitchInInts,
                batch.decodedSequencePitchInBytes,
                batch.msa_pitch,
                batch.msa_weights_pitch,
                min_support_threshold,
                min_coverage_threshold,
                new_columns_to_correct,
                transFuncData.sequenceFileProperties.maxSequenceLength,
                streams[primary_stream_index],
                batch.kernelLaunchHandle);
#else 
        callCorrectCandidatesWithGroupKernel_async(
            dataArrays.getDeviceMSAPointers(),
            dataArrays.getDeviceAlignmentResultPointers(),
            dataArrays.getDeviceSequencePointers(),
            dataArrays.getDeviceCorrectionResultPointers(),
            dataArrays.d_editsPerCorrectedCandidate.get(),
            dataArrays.d_numEditsPerCorrectedCandidate.get(),
            dataArrays.d_candidateContainsN.get(),
            doNotUseEditsValue,
            batch.maxNumEditsPerSequence,
            dataArrays.d_indices,
            dataArrays.d_indices_per_subject,
            dataArrays.d_candidates_per_subject_prefixsum,
            batch.n_subjects,
            batch.n_queries,
            dataArrays.d_num_indices,
            batch.encodedSequencePitchInInts,
            batch.decodedSequencePitchInBytes,
            batch.msa_pitch,
            batch.msa_weights_pitch,
            min_support_threshold,
            min_coverage_threshold,
            new_columns_to_correct,
            transFuncData.sequenceFileProperties.maxSequenceLength,
            streams[primary_stream_index],
            batch.kernelLaunchHandle
        );

#endif

                
        cudaMemcpyAsync(
            dataArrays.h_editsPerCorrectedCandidate,
            dataArrays.d_editsPerCorrectedCandidate,
            dataArrays.d_editsPerCorrectedCandidate.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(
            dataArrays.h_numEditsPerCorrectedCandidate,
            dataArrays.d_numEditsPerCorrectedCandidate,
            dataArrays.d_numEditsPerCorrectedCandidate.sizeInBytes(),
            D2H,
            streams[primary_stream_index]
        ); CUERR;

        cudaMemcpyAsync(dataArrays.h_corrected_candidates,
                        dataArrays.d_corrected_candidates,
                        dataArrays.d_corrected_candidates.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_num_corrected_candidates,
                        dataArrays.d_num_corrected_candidates,
                        dataArrays.d_num_corrected_candidates.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_indices_of_corrected_candidates,
                        dataArrays.d_indices_of_corrected_candidates,
                        dataArrays.d_indices_of_corrected_candidates.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_alignment_shifts,
                        dataArrays.d_alignment_shifts,
                        dataArrays.d_alignment_shifts.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;
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

            // if(readId == 13158000){
            //     std::cerr << "readid = 13158000, stats\n";
            //     std::cerr << "isCorrected " << isCorrected << ", isHQ " << isHQ << "\n";
            //     auto& dataArrays = batch.dataArrays;
            //     std::cerr << "num candidates " << dataArrays.h_candidates_per_subject[subject_index] 
            //         << "num good candidates " << dataArrays.h_indices_per_subject[subject_index] << "\n";
            //     std::cerr << "good candidate ids:\n";

            //     const int globalOffset = rawResults.h_candidates_per_subject_prefixsum[subject_index];

            //     for(int i = 0; i < dataArrays.h_indices_per_subject[subject_index]; i++){
            //         const int index = dataArrays.h_indices[globalOffset + i];
            //         const read_number candidateId = rawResults.h_candidate_read_ids[globalOffset + index];
            //         std::cerr << candidateId << "\n";
            //     }
            // }
        }

        for(int subject_index = 0; subject_index < rawResults.n_subjects; subject_index++){

            const int globalOffset = rawResults.h_candidates_per_subject_prefixsum[subject_index];

            const int n_corrected_candidates = rawResults.h_num_corrected_candidates[subject_index];
            const int* const my_indices_of_corrected_candidates = rawResults.h_indices_of_corrected_candidates
                                                + globalOffset;

            for(int i = 0; i < n_corrected_candidates; ++i) {
                const int localCandidateIndex = my_indices_of_corrected_candidates[i];
                const int global_candidate_index = globalOffset + localCandidateIndex;

                const read_number candidate_read_id = rawResults.h_candidate_read_ids[global_candidate_index];

                bool savingIsOk = false;
                const std::uint8_t mask = transFuncData.correctionStatusFlagsPerRead[candidate_read_id];
                if(!(mask & readCorrectedAsHQAnchor)) {
                    savingIsOk = true;
                }
                if (savingIsOk) {
                    candidateIndicesToProcess.emplace_back(std::make_pair(subject_index, i));
                }
            }
        }

        const int numCorrectedAnchors = subjectIndicesToProcess.size();
        const int numCorrectedCandidates = candidateIndicesToProcess.size();

        // std::cerr << "\n" << "batch " << batch.id << " " 
        //     << numCorrectedAnchors << " " << numCorrectedCandidates << "\n";

        outputData.anchorCorrections.clear();
        outputData.encodedAnchorCorrections.clear();
        outputData.candidateCorrections.clear();
        outputData.encodedCandidateCorrections.clear();

        outputData.anchorCorrections.resize(numCorrectedAnchors);
        outputData.encodedAnchorCorrections.resize(numCorrectedAnchors);
        outputData.candidateCorrections.resize(numCorrectedCandidates);
        outputData.encodedCandidateCorrections.resize(numCorrectedCandidates);

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

                const char* const my_corrected_subject_data = rawResults.h_corrected_subjects + subject_index * rawResults.decodedSequencePitchInBytes;
                const read_number readId = rawResults.h_subject_read_ids[subject_index];

                const int subject_length = rawResults.h_subject_sequences_lengths[subject_index];

                tmp.hq = rawResults.h_is_high_quality_subject[subject_index].hq();                    
                tmp.type = TempCorrectedSequence::Type::Anchor;
                tmp.readId = readId;
                tmp.sequence = std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length};

                // const int numUncorrectedPositions = rawResults.h_num_uncorrected_positions_per_subject[subject_index];

                // if(numUncorrectedPositions > 0){
                //     tmp.uncorrectedPositionsNoConsensus.resize(numUncorrectedPositions);
                //     std::copy_n(rawResults.h_uncorrected_positions_per_subject + subject_index * transFuncData.sequenceFileProperties.maxSequenceLength,
                //                 numUncorrectedPositions,
                //                 tmp.uncorrectedPositionsNoConsensus.begin());

                // }

                auto isValidSequence = [](const std::string& s){
                    return std::all_of(s.begin(), s.end(), [](char c){
                        return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
                    });
                };

                if(!isValidSequence(tmp.sequence)){
                    std::cerr << tmp.sequence << "\n";
                }

                
                tmp.edits.clear();
       

                const int numEdits = rawResults.h_numEditsPerCorrectedSubject[positionInVector];
                if(numEdits != doNotUseEditsValue){
                    tmp.edits.resize(numEdits);
                    const auto* gpuedits = rawResults.h_editsPerCorrectedSubject + positionInVector * rawResults.maxNumEditsPerSequence;
                    std::copy_n(gpuedits, numEdits, tmp.edits.begin());
                    tmp.useEdits = true;
                }else{
                    tmp.useEdits = false;
                }

                tmpencoded = tmp.encode();

                // if(readId == 13158000){
                //     std::cerr << "readid = 13158000, anchor\n";
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

                const size_t offset = rawResults.h_candidates_per_subject_prefixsum[subject_index];

                const char* const my_corrected_candidates_data = rawResults.h_corrected_candidates
                                                + offset * rawResults.decodedSequencePitchInBytes;
                const int* const my_indices_of_corrected_candidates = rawResults.h_indices_of_corrected_candidates
                                                + offset;           

                const int localCandidateIndex = my_indices_of_corrected_candidates[candidateIndex];
                const int global_candidate_index = offset + localCandidateIndex;

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
                
                tmp.edits.clear();
                const int numEdits = rawResults.h_numEditsPerCorrectedCandidate[global_candidate_index];
                if(numEdits != doNotUseEditsValue){
                    tmp.edits.resize(numEdits);
                    const auto* gpuedits = rawResults.h_editsPerCorrectedCandidate + global_candidate_index * rawResults.maxNumEditsPerSequence;
                    std::copy_n(gpuedits, numEdits, tmp.edits.begin());
                    tmp.useEdits = true;
                }else{
                    const int candidate_length = rawResults.h_candidate_sequences_lengths[global_candidate_index];
                    const char* const candidate_data = my_corrected_candidates_data + candidateIndex * rawResults.decodedSequencePitchInBytes;
                    tmp.sequence = std::string{candidate_data, candidate_data + candidate_length};
                    tmp.useEdits = false;
                }

                //TIMERSTARTCPU(encode);
                tmpencoded = tmp.encode();
                //TIMERSTOPCPU(encode);

                // if(candidate_read_id == 13158000){
                //     std::cerr << "readid = 13158000, as candidate of anchor with id " << subjectReadId << "\n";
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
            batch.threadPool->parallelFor(batch.pforHandle, 0, numCorrectedAnchors, [=](auto begin, auto end, auto /*threadId*/){
                unpackAnchors(begin, end);
            });
#endif 

#if 0
            unpackcandidates(0, numCorrectedCandidates);
#else            
            batch.threadPool->parallelFor(
                batch.pforHandle, 
                0, 
                numCorrectedCandidates, 
                [=](auto begin, auto end, auto /*threadId*/){
                    unpackcandidates(begin, end);
                },
                batch.threadPool->getConcurrency() * 4
            );
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




void correct_gpu(const MinhashOptions& minhashOptions,
                  const AlignmentOptions& alignmentOptions,
                  const GoodAlignmentProperties& goodAlignmentProperties,
                  const CorrectionOptions& correctionOptions,
                  const RuntimeOptions& runtimeOptions,
                  const FileOptions& fileOptions,
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

      std::unique_ptr<SequenceFileReader> reader = makeSequenceReader(fileOptions.inputfile, fileOptions.format);

      //std::vector<std::atomic_uint8_t> correctionStatusFlagsPerRead;
      //std::size_t nLocksForProcessedFlags = runtimeOptions.nCorrectorThreads * 1000;
      //std::unique_ptr<std::mutex[]> locksForProcessedFlags(new std::mutex[nLocksForProcessedFlags]);

      std::unique_ptr<std::atomic_uint8_t[]> correctionStatusFlagsPerRead = std::make_unique<std::atomic_uint8_t[]>(sequenceFileProperties.nReads);

      #pragma omp parallel for
      for(read_number i = 0; i < sequenceFileProperties.nReads; i++){
          correctionStatusFlagsPerRead[i] = 0;
      }

      std::cerr << "correctionStatusFlagsPerRead bytes: " << sizeof(std::atomic_uint8_t) * sequenceFileProperties.nReads / 1024. / 1024. << " MB\n";

    const std::size_t availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
    std::size_t memoryForPartialResultsInBytes = 0;

    if(availableMemoryInBytes > (std::size_t(1) << 30)){
        memoryForPartialResultsInBytes = availableMemoryInBytes - (std::size_t(1) << 30);
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
        //cpu::RangeGenerator<read_number> readIdGenerator(10000);
#else
        cpu::RangeGenerator<read_number> readIdGenerator(num_reads_to_profile);
#endif


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
            batchData.backgroundWorker = nullptr;//&backgroundWorkers[i];
            batchData.threadPool = &threadPool;
            batchData.threadsInThreadPool = threadPoolSize;
            batchData.minhashHandles.resize(threadPoolSize);
            batchData.encodedSequencePitchInInts = getEncodedNumInts2Bit(sequenceFileProperties.maxSequenceLength);
            batchData.decodedSequencePitchInBytes = SDIV(sequenceFileProperties.maxSequenceLength, 4) * 4;
            batchData.qualityPitchInBytes = SDIV(sequenceFileProperties.maxSequenceLength, 32) * 32;
            batchData.maxNumEditsPerSequence = std::max(1,sequenceFileProperties.maxSequenceLength / 7);

            batchData.max_n_queries = batchsize * 2.5 * correctionOptions.estimatedCoverage;
            
            initNextIterationData(batchData.nextIterationData, batchData.deviceId); 
        };

        auto destroyBatchData = [&](auto& batchData){
            
            cudaSetDevice(batchData.deviceId); CUERR;
    
            batchData.dataArrays.reset();
            destroyNextIterationData(batchData.nextIterationData);
    
            for(auto& stream : batchData.streams) {
                cudaStreamDestroy(stream); CUERR;
            }
    
            for(auto& event : batchData.events){
                cudaEventDestroy(event); CUERR;
            }            
        };

        auto showProgress = [&](std::int64_t totalCount, int seconds){
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
        };

        auto updateShowProgressInterval = [](auto duration){
            return duration;
        };

        ProgressThread<std::int64_t> progressThread(sequenceFileProperties.nReads, showProgress, updateShowProgressInterval);


        auto processBatchUntilResultTransferIsInitiated = [&](auto& batchData){
            auto& streams = batchData.streams;
            auto& events = batchData.events;

            auto pushrange = [&](const std::string& msg, int color){
                nvtx::push_range("batch "+std::to_string(batchData.id)+msg, color);
            };

            auto poprange = [&](){
                nvtx::pop_range();
            };
                
            pushrange("getNextBatchOfSubjectsAndDetermineCandidateReadIds", 0);
            
            getNextBatchOfSubjectsAndDetermineCandidateReadIds(batchData);

            poprange();

            if(batchData.n_queries == 0){
                return;
            }

            pushrange("resizeArrays", 3);

            resizeArrays(batchData);

            poprange();

            pushrange("getCandidateSequenceData", 1);

            getCandidateSequenceData(batchData, *transFuncData.readStorage);

            poprange();

            if(transFuncData.correctionOptions.useQualityScores) {
                pushrange("getQualities", 4);

                getQualities(batchData);

                poprange();
            }


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

            cudaEventRecord(events[secondary_stream_finished_event_index], streams[secondary_stream_index]); CUERR;
            cudaStreamWaitEvent(streams[primary_stream_index], events[secondary_stream_finished_event_index], 0); CUERR;   

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
            
        };


        for(int deviceIdIndex = 0; deviceIdIndex < int(deviceIds.size()); ++deviceIdIndex) {
            batchExecutors.emplace_back([&, deviceIdIndex](){
                const int deviceId = deviceIds[deviceIdIndex];

                std::array<BackgroundThread, 2> backgroundWorkerArray;

                std::array<Batch, 2> batchDataArray;

                for(int i = 0; i < 2; i++){
                    initBatchData(batchDataArray[i], deviceId);
                    batchDataArray[i].id = deviceIdIndex * 2 + i;
                    batchDataArray[i].backgroundWorker = &backgroundWorkerArray[i];
                }

                backgroundWorkerArray[0].start();
                backgroundWorkerArray[1].start();

                bool isFirstIteration = true;

                int batchIndex = 0;
#if 0
                while(!(readIdGenerator.empty() 
                        && !batchDataArray[0].nextIterationData.syncFlag.isBusy()
                        && batchDataArray[0].nextIterationData.n_subjects == 0
                        && !batchDataArray[0].waitableOutputData.isBusy())) {

                    auto& batchData = batchDataArray[batchIndex];

                    processBatchUntilResultTransferIsInitiated(batchData);

                    if(batchData.n_queries == 0){
                        batchData.waitableOutputData.signal();
                        progressThread.addProgress(batchData.n_subjects);
                        batchData.reset();
                        continue;
                    }

                    processBatchResults(batchData);

                    progressThread.addProgress(batchData.n_subjects);
                    batchData.reset();                   
                }

                std::cerr << "exit while loop\n";
#else 
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
//std::cerr << "\nprocessBatchUntilResultTransferIsInitiated batch " << currentBatchData.id << "\n";
                        processBatchUntilResultTransferIsInitiated(currentBatchData);
                    }else{

                        //std::cerr << "\nprocessBatchUntilResultTransferIsInitiated batch " << nextBatchData.id << "\n";
                        processBatchUntilResultTransferIsInitiated(nextBatchData);

                        if(currentBatchData.n_queries == 0){
                            currentBatchData.waitableOutputData.signal();
                            progressThread.addProgress(currentBatchData.n_subjects);
                            currentBatchData.reset();
                            batchIndex = 1-batchIndex;
                            continue;
                        }
                        //std::cerr << "\processBatchResults batch " << currentBatchData.id << "\n";
                        processBatchResults(currentBatchData);
    
                        progressThread.addProgress(currentBatchData.n_subjects);
                        currentBatchData.reset();

                        batchIndex = 1-batchIndex;
                    }                
                }

#endif
                std::cerr << "batchDataArray[0].max_n_queries: " << batchDataArray[0].max_n_queries << "\n";
                std::cerr << "batchDataArray[1].max_n_queries: " << batchDataArray[1].max_n_queries << "\n";
                batchDataArray[0].backgroundWorker->stopThread(BackgroundThread::StopType::FinishAndStop);
                batchDataArray[1].backgroundWorker->stopThread(BackgroundThread::StopType::FinishAndStop);
                destroyBatchData(batchDataArray[0]);
                destroyBatchData(batchDataArray[1]);
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

          std::cout << "begin merge" << std::endl;

          if(!correctionOptions.extractFeatures){

              std::cout << "begin merging reads" << std::endl;

              TIMERSTARTCPU(merge);

              mergeResultFiles(
                                fileOptions.tempdirectory,
                                sequenceFileProperties.nReads, 
                                fileOptions.inputfile, 
                                fileOptions.format, 
                                partialResults, 
                                fileOptions.outputfile, 
                                false);

              TIMERSTOPCPU(merge);

              std::cout << "end merging reads" << std::endl;

          }

          deleteFiles(tmpfiles);
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
              deleteFiles(featureTmpFiles);
          }

          std::cout << "end merging features" << std::endl;
      }else{
          deleteFiles(featureTmpFiles);
      }

      std::cout << "end merge" << std::endl;

      #endif



}






















}
}

}



#endif
