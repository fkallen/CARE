#if 1

#include <gpu/correct_gpu.hpp>
#include <gpu/distributedreadstorage.hpp>
#include <gpu/nvtxtimelinemarkers.hpp>
#include <gpu/kernels.hpp>
#include <gpu/dataarrays.hpp>
#include <gpu/cubcachingallocator.cuh>

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


    struct CorrectionTask {
        CorrectionTask(){
        }

        CorrectionTask(read_number readId)
            :   active(true),
            corrected(false),
            highQualityAlignment(false),
            readId(readId)
        {
        }

        CorrectionTask(const CorrectionTask& other)
            : active(other.active),
            corrected(other.corrected),
            highQualityAlignment(other.highQualityAlignment),
            readId(other.readId),
            subject_string(other.subject_string),
            candidate_read_ids(other.candidate_read_ids),
            correctionEqualsOriginal(other.correctionEqualsOriginal),
            corrected_subject(other.corrected_subject),
            corrected_candidates(other.corrected_candidates),
            corrected_candidates_read_ids(other.corrected_candidates_read_ids),
            corrected_candidates_shifts(other.corrected_candidates_shifts),
            corrected_candidate_equals_uncorrected(other.corrected_candidate_equals_uncorrected),
            uncorrectedPositionsNoConsensus(other.uncorrectedPositionsNoConsensus),
            anchoroutput(other.anchoroutput),
            candidatesoutput(other.candidatesoutput){
        }

        CorrectionTask(CorrectionTask&& other){
            operator=(other);
        }

        CorrectionTask& operator=(const CorrectionTask& other){
            CorrectionTask tmp(other);
            swap(*this, tmp);
            return *this;
        }

        CorrectionTask& operator=(CorrectionTask&& other){
            swap(*this, other);
            return *this;
        }

        friend void swap(CorrectionTask& l, CorrectionTask& r) noexcept{
            using std::swap;

            swap(l.active, r.active);
            swap(l.corrected, r.corrected);
            swap(l.highQualityAlignment, r.highQualityAlignment);
            swap(l.readId, r.readId);
            swap(l.subject_string, r.subject_string);
            swap(l.candidate_read_ids, r.candidate_read_ids);
            swap(l.correctionEqualsOriginal, r.correctionEqualsOriginal);
            swap(l.corrected_subject, r.corrected_subject);
            swap(l.corrected_candidates_read_ids, r.corrected_candidates_read_ids);
            swap(l.corrected_candidates_shifts, r.corrected_candidates_shifts);
            swap(l.corrected_candidate_equals_uncorrected, r.corrected_candidate_equals_uncorrected);
            swap(l.uncorrectedPositionsNoConsensus, r.uncorrectedPositionsNoConsensus);
            swap(l.anchoroutput, r.anchoroutput);
            swap(l.candidatesoutput, r.candidatesoutput);
        }

        bool active;
        bool corrected;
        bool highQualityAlignment;
        read_number readId;

        std::vector<read_number> candidate_read_ids;

        std::string subject_string;

        bool correctionEqualsOriginal;
        std::string corrected_subject;
        std::vector<std::string> corrected_candidates;
        std::vector<read_number> corrected_candidates_read_ids;
        std::vector<int> corrected_candidates_shifts;
        std::vector<bool> corrected_candidate_equals_uncorrected;
        std::vector<int> uncorrectedPositionsNoConsensus;

        EncodedTempCorrectedSequence encodedAnchoroutput;
        std::vector<EncodedTempCorrectedSequence> encodedCandidatesoutput;
        TempCorrectedSequence anchoroutput;
        std::vector<TempCorrectedSequence> candidatesoutput;
    };

    enum class BatchState : int{
		Unprepared,
        FindCandidateIds,
		CopyReads,
		StartAlignment,
        RearrangeIndices,
		CopyQualities,
		BuildMSA,
        ImproveMSA,
		StartClassicCorrection,
		StartForestCorrection,
        StartConvnetCorrection,
        StartClassicCandidateCorrection,
        CombineStreams,
		UnpackClassicResults,
		WriteResults,
		WriteFeatures,
		Finished,
	};

    static constexpr int nBatchStates = static_cast<int>(BatchState::Finished)+1;

    std::string nameOf(const BatchState&);

    struct TransitionFunctionData;

    struct FindCandidateIdsDataFrame{
        std::atomic_int initialNumberOfCandidates{0};
        int finishedChunks = 0;
        int chunksToWaitFor = 0;
        std::mutex m;
        std::condition_variable cv;

        void reset(){
            initialNumberOfCandidates = 0;
            finishedChunks = 0;
            chunksToWaitFor = 0;
        }
    };

    struct UnpackClassicResultsDataFrame{
        int finishedChunks = 0;
        int chunksToWaitFor = 0;
        std::mutex m;
        std::condition_variable cv;

        void reset(){
            finishedChunks = 0;
            chunksToWaitFor = 0;
        }
    };

    struct NextIterationData{
        SimpleAllocationPinnedHost<unsigned int> h_subject_sequences_data;
        SimpleAllocationPinnedHost<int> h_subject_sequences_lengths;
        SimpleAllocationPinnedHost<read_number> h_subject_read_ids;

        SimpleAllocationDevice<unsigned int> d_subject_sequences_data;
        SimpleAllocationDevice<int> d_subject_sequences_lengths;
        SimpleAllocationDevice<read_number> d_subject_read_ids;

        std::vector<std::string> decodedSubjectStrings;

        std::vector<CorrectionTask> tasks;
        int initialNumberOfAnchorIds = -1;
        std::atomic<int> initialNumberOfCandidates{-1};

        cudaStream_t stream;
        cudaEvent_t event;
        int deviceId;

        ThreadPool::ParallelForHandle pforHandle;

        bool done = true;
        std::mutex mDone;
        std::condition_variable cvDone;

        void wait(){
            if(!isDone()){
                std::unique_lock<std::mutex> l(mDone);
                while(!isDone()){
                    cvDone.wait(l);
                }
            }
        }

        void signal(){
            std::unique_lock<std::mutex> l(mDone);
            done = true;
            cvDone.notify_all();
        }

        bool isDone() const{
            return done;
        }
    };
#if 0
    struct BatchResultData{
        std::vector<CorrectionTask> tasks;

        SimpleAllocationPinnedHost<char> h_corrected_subjects;
        SimpleAllocationPinnedHost<bool> h_subject_is_corrected;
        SimpleAllocationPinnedHost<AnchorHighQualityFlag> h_is_high_quality_subject;
        SimpleAllocationPinnedHost<int> h_subject_sequences_lengths;
        SimpleAllocationPinnedHost<int> h_num_uncorrected_positions_per_subject;
        SimpleAllocationPinnedHost<int> h_uncorrected_positions_per_subject;

        SimpleAllocationPinnedHost<int> h_num_corrected_candidates;
        SimpleAllocationPinnedHost<char> h_corrected_candidates;
        SimpleAllocationPinnedHost<unsigned int> h_candidate_sequences_data;
        SimpleAllocationPinnedHost<int> h_indices_of_corrected_candidates;
        SimpleAllocationPinnedHost<int> h_indices_per_subject_prefixsum;

        SimpleAllocationPinnedHost<int> h_candidate_sequences_lengths;
        SimpleAllocationPinnedHost<int> h_alignment_shifts;
        SimpleAllocationPinnedHost<read_number> h_candidate_read_ids;

        int maximum_sequence_length;
        int sequence_pitch;
        int encoded_sequence_pitch;

        ThreadPool::ParallelForHandle pforHandle;

        bool done = true;
        std::mutex mDone;
        std::condition_variable cvDone;

        void wait(){
            if(!isDone()){
                std::unique_lock<std::mutex> l(mDone);
                while(!isDone()){
                    cvDone.wait(l);
                }
            }
        }

        void signal(){
            std::unique_lock<std::mutex> l(mDone);
            done = true;
            cvDone.notify_all();
        }

        bool isDone() const{
            return done;
        }
    };
#endif 


    template<class T>
    struct WaitableData{
        T data;

        std::atomic<bool> done{true};
        std::mutex mDone;
        std::condition_variable cvDone;

        void wait(){
            if(!isDone()){
                std::unique_lock<std::mutex> l(mDone);
                while(!isDone()){
                    cvDone.wait(l);
                }
            }
        }

        void signal(){
            std::unique_lock<std::mutex> l(mDone);
            done = true;
            cvDone.notify_all();
        }

        bool isDone() const{
            return done;
        }
    };


    struct Batch {

 
        Batch() = default;
        Batch(const Batch&) = delete;
        Batch(Batch&&) = default;
        Batch& operator=(const Batch&) = delete;
        Batch& operator=(Batch&&) = default;

        struct OutputData{
            std::vector<TempCorrectedSequence> anchorCorrections;
            std::vector<EncodedTempCorrectedSequence> encodedAnchorCorrections;
            std::vector<TempCorrectedSequence> candidateCorrections;
            std::vector<EncodedTempCorrectedSequence> encodedCandidateCorrections;

            std::vector<int> subjectIndicesToProcess;
            std::vector<std::pair<int,int>> candidateIndicesToProcess;
        };

        NextIterationData nextIterationData;
        bool isFirstIteration = true;

        WaitableData<OutputData> waitableOutputData;

        WaitableData<int> reuseFlag;

        //BatchResultData resultData;

		std::vector<CorrectionTask> tasks;
		int initialNumberOfCandidates = 0;
		BatchState state = BatchState::Unprepared;

        bool handledReadIds = false;

        bool combinedStreams = false;

        int initialNumberOfAnchorIds = 0;

        DataArrays dataArrays;
        bool hasUnprocessedResults = false;

        bool doImproveMSA = false;
        int numMinimizations = 0;
        int previousNumIndices = 0;

        std::vector<std::string> decodedSubjectStrings;

		std::array<cudaStream_t, nStreamsPerBatch> streams;
		std::array<cudaEvent_t, nEventsPerBatch> events;

        std::array<std::atomic_int, nBatchStates> waitCounts{};
        int activeWaitIndex = 0;
        //std::vector<std::unique_ptr<WaitCallbackData>> callbackDataList;

        TransitionFunctionData* transFuncData;
        BackgroundThread* outputThread;
        BackgroundThread* backgroundWorker;

        ThreadPool* threadPool;
        int threadsInThreadPool = 1;

        ThreadPool::ParallelForHandle pforHandle;
        std::vector<Minhasher::Handle> minhashHandles;

        bool isTerminated = false;

        int id = -1;
        int deviceId = 0;

		KernelLaunchHandle kernelLaunchHandle;

        DistributedReadStorage::GatherHandleSequences subjectSequenceGatherHandle2;
        DistributedReadStorage::GatherHandleSequences candidateSequenceGatherHandle2;
        DistributedReadStorage::GatherHandleLengths subjectLengthGatherHandle2;
        DistributedReadStorage::GatherHandleLengths candidateLengthGatherHandle2;
        DistributedReadStorage::GatherHandleQualities subjectQualitiesGatherHandle2;
        DistributedReadStorage::GatherHandleQualities candidateQualitiesGatherHandle2;

        int encodedSequencePitchInInts;
        int decodedSequencePitchInBytes;
        int qualityPitchInBytes;

        int msa_weights_pitch;
        int msa_pitch;

        int n_subjects;
        int n_queries;

        FindCandidateIdsDataFrame findcandidateidsDataFrame;
        UnpackClassicResultsDataFrame unpackclassicresultsDataFrame;

        std::atomic_int statesInProgress{0};

        void setState(BatchState s){
            std::cerr << "batch " << nameOf(state) << " -> " << nameOf(s) << "\n";

            state = s;
        }

        void setState(BatchState s, BatchState expected){
            if(state != expected){
                std::cerr << "batch " << id << nameOf(state) << " ( " << nameOf(expected) << " )" << " -> " << nameOf(s) << "\n";
                assert(false);
            }
            state = s;
        }

		void reset(){
            tasks.clear();

    		findcandidateidsDataFrame.reset();
            unpackclassicresultsDataFrame.reset();

            statesInProgress = 0;

    		state = BatchState::Unprepared;

            handledReadIds = false;

            combinedStreams = false;

            initialNumberOfAnchorIds = 0;
            initialNumberOfCandidates = 0;
            hasUnprocessedResults = false;

            //assert(std::all_of(waitCounts.begin(), waitCounts.end(), [](const auto& i){return i == 0;}));

            activeWaitIndex = 0;

            doImproveMSA = false;
            numMinimizations = 0;
            previousNumIndices = 0;
        }

        void updateFromIterationData(NextIterationData& nextIterationData){
            //std::cerr << "update from iteration data\n";
            std::swap(dataArrays.h_subject_sequences_data, nextIterationData.h_subject_sequences_data);
            std::swap(dataArrays.h_subject_sequences_lengths, nextIterationData.h_subject_sequences_lengths);
            std::swap(dataArrays.h_subject_read_ids, nextIterationData.h_subject_read_ids);
            std::swap(dataArrays.d_subject_sequences_data, nextIterationData.d_subject_sequences_data);
            std::swap(dataArrays.d_subject_sequences_lengths, nextIterationData.d_subject_sequences_lengths);
            std::swap(dataArrays.d_subject_read_ids, nextIterationData.d_subject_read_ids);
            std::swap(tasks, nextIterationData.tasks);
            std::swap(decodedSubjectStrings, nextIterationData.decodedSubjectStrings);

            initialNumberOfAnchorIds = nextIterationData.initialNumberOfAnchorIds;
            initialNumberOfCandidates = nextIterationData.initialNumberOfCandidates;  

            nextIterationData.tasks.clear();
            nextIterationData.initialNumberOfAnchorIds = 0;
            nextIterationData.initialNumberOfCandidates = 0;

            //std::cerr << "update from iteration data finished\n";
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

    std::string nameOf(const BatchState& state){
        switch(state) {
        case BatchState::Unprepared: return "Unprepared";
        case BatchState::FindCandidateIds: return "FindCandidateIds";
        case BatchState::CopyReads: return "CopyReads";
        case BatchState::StartAlignment: return "StartAlignment";
        case BatchState::RearrangeIndices: return "RearrangeIndices";
        case BatchState::CopyQualities: return "CopyQualities";
        case BatchState::BuildMSA: return "BuildMSA";
        case BatchState::ImproveMSA: return "ImproveMSA";
        case BatchState::StartClassicCorrection: return "StartClassicCorrection";
        case BatchState::StartForestCorrection: return "StartForestCorrection";
        case BatchState::StartConvnetCorrection: return "StartConvnetCorrection";
        case BatchState::StartClassicCandidateCorrection: return "StartClassicCandidateCorrection";
        case BatchState::UnpackClassicResults: return "UnpackClassicResults";
        case BatchState::WriteResults: return "WriteResults";
        case BatchState::WriteFeatures: return "WriteFeatures";
        case BatchState::Finished: return "Finished";
        default: assert(false); return "None";
        }
    }




    /*for each active read pair in tasks, if at least one read has more than threshold candidates:
        1. find out candidates whose paired candidate is a candidate of the paired read
        2. for each read with more than threshold candidates:
             remove all candidates which are not found in step 1.

        tasks without pair remain unchanged.
        modified tasks with no candidates left are set inactive.
        read pairs are identified by read id.
        reads A and B are considered a pair if readIdA / 2 == readIdB / 2
    */
    void
    removeNonPairedCandidatesFromHighCoverageTasks(std::vector<CorrectionTask>& tasks,
                                        size_t threshold){
        //paired reads have consecutive read ids, where the first read id of the read pair is even, the second read id is odd.
        auto pairedReadId = [](read_number readId){
            return readId / 2;
        };
        auto compByPairedReadId = [&](read_number l, read_number r){
            return pairedReadId(l) < pairedReadId(r);
        };

        //for each read, only keep candidates whose pair is a candidate of the paired read
        std::vector<read_number> candidatepairs;
        std::vector<read_number> newcandidateids;
        size_t index = 0;
        while(index < tasks.size()){
            if(index == tasks.size() -1 ){
                //std::cerr << "no pair found\n";
                tasks[index].active = false;

                index += 1;
                continue;
            }
            auto& taskl = tasks[index];
            auto& taskr = tasks[index+1];

            const read_number readIdl = taskl.readId;
            const read_number readIdr = taskr.readId;

            //check if tasks[index] and tasks[index+1] are a paired read.
            if(taskl.active && taskr.active && pairedReadId(readIdl) == pairedReadId(readIdr)){
                const size_t oldNumCandidatesl = taskl.candidate_read_ids.size();
                const size_t oldNumCandidatesr = taskr.candidate_read_ids.size();

                //check threshold
                const bool updatel = oldNumCandidatesl > threshold;
                const bool updater = oldNumCandidatesr > threshold;

                if(updatel || updater){

                    candidatepairs.clear();
                    candidatepairs.resize(std::min(taskl.candidate_read_ids.size(), taskr.candidate_read_ids.size()));

                    const auto pairsend = std::set_intersection(taskl.candidate_read_ids.begin(), taskl.candidate_read_ids.end(),
                                                          taskr.candidate_read_ids.begin(), taskr.candidate_read_ids.end(),
                                                          candidatepairs.begin(),
                                                          compByPairedReadId);

                    const size_t numcandidatepairs = std::distance(candidatepairs.begin(), pairsend);

                    if(updatel){
                        newcandidateids.clear();
                        newcandidateids.resize(std::min(numcandidatepairs, taskl.candidate_read_ids.size()));

                        const auto newcandidateidsend = std::set_intersection(taskl.candidate_read_ids.begin(), taskl.candidate_read_ids.end(),
                                                                        candidatepairs.begin(), pairsend,
                                                                        newcandidateids.begin(),
                                                                        compByPairedReadId);
                        newcandidateids.erase(newcandidateidsend, newcandidateids.end());
                        std::swap(newcandidateids, taskl.candidate_read_ids);

                        if(taskl.candidate_read_ids.size() == 0){
                            taskl.active = false;
                        }
                    }

                    if(updater){
                        newcandidateids.clear();
                        newcandidateids.resize(std::min(numcandidatepairs, taskr.candidate_read_ids.size()));

                        const auto newcandidateidsend = std::set_intersection(taskr.candidate_read_ids.begin(), taskr.candidate_read_ids.end(),
                                                                        candidatepairs.begin(), pairsend,
                                                                        newcandidateids.begin(),
                                                                        compByPairedReadId);
                        newcandidateids.erase(newcandidateidsend, newcandidateids.end());
                        std::swap(newcandidateids, taskr.candidate_read_ids);

                        if(taskr.candidate_read_ids.size() == 0){
                            taskr.active = false;
                        }
                    }

                    /*assert(std::all_of(taskl.candidate_read_ids.begin(),
                                        taskl.candidate_read_ids.end(),
                                        [&](auto readId){
                                            return readId < transFuncData.gpuReadStorage->cpuReadStorage->getNumberOfSequences();
                                        }));

                    assert(std::all_of(taskr.candidate_read_ids.begin(),
                                        taskr.candidate_read_ids.end(),
                                        [&](auto readId){
                                            return readId < transFuncData.gpuReadStorage->cpuReadStorage->getNumberOfSequences();
                                        }));*/

                    //const size_t deltal = oldNumCandidatesl - taskl.candidate_read_ids.size();
                    //const size_t deltar = oldNumCandidatesr - taskr.candidate_read_ids.size();

                    //std::cerr << "removed " << deltal << "( " << oldNumCandidatesl << " ), " << deltar << "( " << oldNumCandidatesr << " )\n";

                }

                //assert(taskl.candidate_read_ids.size() <= std::size_t(transFuncData.runtimeOptions.max_candidates));
                //assert(taskr.candidate_read_ids.size() <= std::size_t(transFuncData.runtimeOptions.max_candidates));

                index += 2;
            }else{
                //std::cerr << "no pair found\n";

                index += 1;
            }
        }
    }


    void build_msa_async(MSAPointers d_msapointers,
                    AlignmentResultPointers d_alignmentresultpointers,
                    ReadSequencesPointers d_sequencePointers,
                    ReadQualitiesPointers d_qualityPointers,
                    const int* d_candidates_per_subject_prefixsum,
                    const int* d_indices,
                    const int* d_indices_per_subject,
                    const int* d_indices_per_subject_prefixsum,
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
                d_indices_per_subject_prefixsum,
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
                    d_indices_per_subject_prefixsum,
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
    }

    void destroyNextIterationData(NextIterationData& nextData){
        cudaSetDevice(nextData.deviceId); CUERR;
        cudaStreamDestroy(nextData.stream); CUERR;
        cudaEventDestroy(nextData.event); CUERR;

        std::cerr << "destroyNextIterationData\n"; CUERR;

        nextData.h_subject_sequences_data = std::move(SimpleAllocationPinnedHost<unsigned int>{});
        nextData.h_subject_sequences_lengths = std::move(SimpleAllocationPinnedHost<int>{});
        nextData.h_subject_read_ids = std::move(SimpleAllocationPinnedHost<read_number>{});

        nextData.d_subject_sequences_data = std::move(SimpleAllocationDevice<unsigned int>{});
        nextData.d_subject_sequences_lengths = std::move(SimpleAllocationDevice<int>{});
        nextData.d_subject_read_ids = std::move(SimpleAllocationDevice<read_number>{});

        nextData.tasks.clear();
        nextData.tasks.shrink_to_fit();

        std::cerr << "destroyNextIterationData finished\n"; CUERR;
    }

    void getSubjectDataOfNextIteration(Batch& batchData, int batchsize, const DistributedReadStorage& readStorage){
        NextIterationData& nextData = batchData.nextIterationData;
        const auto& transFuncData = *batchData.transFuncData;

       // std::cerr << "getSubjectDataOfNextIteration\n"; CUERR;

        nextData.h_subject_sequences_data.resize(batchData.encodedSequencePitchInInts * batchsize);
        nextData.d_subject_sequences_data.resize(batchData.encodedSequencePitchInInts * batchsize);
        nextData.h_subject_sequences_lengths.resize(batchsize);
        nextData.d_subject_sequences_lengths.resize(batchsize);
        nextData.h_subject_read_ids.resize(batchsize);
        nextData.d_subject_read_ids.resize(batchsize);
        //nextData.tasks.reserve(batchsize);

        read_number* const readIdsBegin = nextData.h_subject_read_ids.get();
        read_number* const readIdsEnd = transFuncData.readIdGenerator->next_n_into_buffer(batchsize, readIdsBegin);
        nextData.initialNumberOfAnchorIds = std::distance(readIdsBegin, readIdsEnd);

        if(nextData.initialNumberOfAnchorIds == 0){
            nextData.signal();
            return;
        };

        //copy read ids to device. gather sequences + lengths for those ids and copy them back to host
        cudaMemcpyAsync(
            nextData.d_subject_read_ids,
            nextData.h_subject_read_ids,
            nextData.h_subject_read_ids.sizeInBytes(),
            H2D,
            nextData.stream); CUERR;

        //cudaStreamSynchronize(nextData.stream); CUERR;

        readStorage.gatherSequenceDataToGpuBufferAsync(
            batchData.threadPool,
            batchData.subjectSequenceGatherHandle2,
            nextData.d_subject_sequences_data.get(),
            batchData.encodedSequencePitchInInts,
            nextData.h_subject_read_ids,
            nextData.d_subject_read_ids,
            nextData.initialNumberOfAnchorIds,
            batchData.deviceId,
            nextData.stream,
            transFuncData.runtimeOptions.nCorrectorThreads);

        //    cudaStreamSynchronize(nextData.stream); CUERR;

        readStorage.gatherSequenceLengthsToGpuBufferAsync(
            nextData.d_subject_sequences_lengths.get(),
            batchData.deviceId,
            nextData.d_subject_read_ids.get(),
            nextData.initialNumberOfAnchorIds,            
            nextData.stream);

        //    cudaStreamSynchronize(nextData.stream); CUERR;

        cudaMemcpyAsync(
            nextData.h_subject_sequences_data,
            nextData.d_subject_sequences_data,
            nextData.d_subject_sequences_data.sizeInBytes(),
            D2H,
            nextData.stream); CUERR;

        //    cudaStreamSynchronize(nextData.stream); CUERR;

        cudaMemcpyAsync(
            nextData.h_subject_sequences_lengths,
            nextData.d_subject_sequences_lengths,
            nextData.d_subject_sequences_lengths.sizeInBytes(),
            D2H,
            nextData.stream); CUERR;

        //    cudaStreamSynchronize(nextData.stream); CUERR;
    }

    void determineCandidateReadIdsOfNextIteration(Batch& batchData, const Minhasher& minhasher){
        NextIterationData& nextData = batchData.nextIterationData;
        nextData.tasks.resize(nextData.initialNumberOfAnchorIds);

        //minhash the retrieved anchors to find candidate ids

        Batch* batchptr = &batchData;
        NextIterationData* nextDataPtr = &nextData;
        const Minhasher* minhasherPtr = &minhasher;

        nextData.initialNumberOfCandidates = 0;

        std::vector<std::vector<std::string>> decodedSubjectStringsPerThread(batchData.threadsInThreadPool);

        auto maketasks = [&, batchptr, nextDataPtr, minhasherPtr](int begin, int end, int threadId){

            auto& transFuncData = *(batchptr->transFuncData);
            const int hits_per_candidate = transFuncData.correctionOptions.hits_per_candidate;

            auto& minhashHandle = batchptr->minhashHandles[threadId];

            int initialNumberOfCandidates = 0;

            std::vector<std::string> decodedSubjectStrings(end-begin);

            nvtx::push_range("init tasks", 0);
            for(int i = begin; i < end; i++){
                auto& task = nextDataPtr->tasks[i];

                const read_number readId = nextDataPtr->h_subject_read_ids[i];

                task = CorrectionTask(readId);
            }
            nvtx::pop_range();

            nvtx::push_range("decode sequences", 1);
            for(int i = begin; i < end; i++){
                auto& task = nextDataPtr->tasks[i];
                const unsigned int* sequenceptr = nextDataPtr->h_subject_sequences_data.get() + i * batchptr->encodedSequencePitchInInts;
                const int sequencelength = nextDataPtr->h_subject_sequences_lengths[i];

                decodedSubjectStrings[i - begin] = get2BitString(sequenceptr, sequencelength);
                //task.subject_string = get2BitString(sequenceptr, sequencelength);
            }
            nvtx::pop_range();

#if 0
            nvtx::push_range("minhashing", 2);
            for(int i = begin; i < end; i++){
                auto& task = nextDataPtr->tasks[i];
                minhasherPtr->getCandidates_any_map(
                    minhashHandle,
                    decodedSubjectStrings[i - begin],
                    transFuncData.runtimeOptions.max_candidates
                );
                std::swap(task.candidate_read_ids, minhashHandle.result());
            }
            nvtx::pop_range();

            nvtx::push_range("remove self", 3);
            
            for(int i = begin; i < end; i++){
                auto& task = nextDataPtr->tasks[i];
                task.candidate_read_ids.clear();
                std::copy(minhashHandle.multiresults() + numResultsOfSequence(), task.candidate_read_ids.begin());
                //task.subject_string = std::move(decodedSubjectStrings[i - begin]);
                task.subject_string = decodedSubjectStrings[i - begin];

                auto readIdPos = std::lower_bound(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.readId);
                //TIMERSTOPCPU(lower_bound);

                if(readIdPos != task.candidate_read_ids.end() && *readIdPos == task.readId) {
                    //TIMERSTARTCPU(erase);
                    task.candidate_read_ids.erase(readIdPos);
                    //TIMERSTOPCPU(erase);
                }

                std::size_t myNumCandidates = task.candidate_read_ids.size();

                //assert(myNumCandidates <= std::size_t(transFuncData.runtimeOptions.max_candidates));

                if(myNumCandidates == 0) {
                    task.active = false;
                }
                initialNumberOfCandidates += myNumCandidates;
            }
            nvtx::pop_range();
#else 
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

            nvtx::push_range("remove self", 3);

            auto multiresultbegin = minhashHandle.multiresults().begin();
            
            for(int i = begin; i < end; i++){
                auto& task = nextDataPtr->tasks[i];
                task.candidate_read_ids.clear();
                task.candidate_read_ids.resize(minhashHandle.numResultsOfSequence(i-begin));
                auto multiresultend = multiresultbegin + minhashHandle.numResultsOfSequence(i-begin);               

                std::copy(
                    multiresultbegin,
                    multiresultend, 
                    task.candidate_read_ids.begin()
                );
                multiresultbegin = multiresultend;
                //task.subject_string = std::move(decodedSubjectStrings[i - begin]);
                task.subject_string = decodedSubjectStrings[i - begin];

                auto readIdPos = std::lower_bound(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.readId);
                //TIMERSTOPCPU(lower_bound);

                if(readIdPos != task.candidate_read_ids.end() && *readIdPos == task.readId) {
                    //TIMERSTARTCPU(erase);
                    task.candidate_read_ids.erase(readIdPos);
                    //TIMERSTOPCPU(erase);
                }

                std::size_t myNumCandidates = task.candidate_read_ids.size();

                //assert(myNumCandidates <= std::size_t(transFuncData.runtimeOptions.max_candidates));

                if(myNumCandidates == 0) {
                    task.active = false;
                }
                initialNumberOfCandidates += myNumCandidates;
            }
            nvtx::pop_range();
#endif
            
            

            // for(int i = begin; i < end; i++){
            //     auto& task = nextDataPtr->tasks[i];

            //     const read_number readId = nextDataPtr->h_subject_read_ids[i];

            //     task = CorrectionTask(readId);
            //     const bool ok = true;

            //     if(ok){
            //         const unsigned int* sequenceptr = nextDataPtr->h_subject_sequences_data.get() + i * batchptr->encodedSequencePitchInInts;
            //         const int sequencelength = nextDataPtr->h_subject_sequences_lengths[i];

            //         //TIMERSTARTCPU(get2BitString);
            //         task.subject_string = get2BitString(sequenceptr, sequencelength);
            //         //TIMERSTOPCPU(get2BitString);

            //         //TIMERSTARTCPU(getCandidates);
            //         minhasherPtr->getCandidates_any_map(
            //             minhashHandle,
            //             task.subject_string,
            //             transFuncData.runtimeOptions.max_candidates
            //         );

            //         std::swap(task.candidate_read_ids, minhashHandle.result());
            //         //TIMERSTOPCPU(getCandidates);

            //         //TIMERSTARTCPU(lower_bound);
            //         auto readIdPos = std::lower_bound(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.readId);
            //         //TIMERSTOPCPU(lower_bound);

            //         if(readIdPos != task.candidate_read_ids.end() && *readIdPos == task.readId) {
            //             //TIMERSTARTCPU(erase);
            //             task.candidate_read_ids.erase(readIdPos);
            //             //TIMERSTOPCPU(erase);
            //         }

            //         std::size_t myNumCandidates = task.candidate_read_ids.size();

            //         //assert(myNumCandidates <= std::size_t(transFuncData.runtimeOptions.max_candidates));

            //         if(myNumCandidates == 0) {
            //             task.active = false;
            //         }
            //     }else{
            //         task.active = false;
            //     }

            //    const int myNumCandidates = int(task.candidate_read_ids.size());
            //    initialNumberOfCandidates += myNumCandidates;
            //}

            nextDataPtr->initialNumberOfCandidates += initialNumberOfCandidates;
            decodedSubjectStringsPerThread[threadId] = std::move(decodedSubjectStrings);
        };

        cudaStreamSynchronize(nextData.stream); CUERR; //wait for D2H transfers of anchor data which is required for minhasher

        batchData.threadPool->parallelFor(
            nextData.pforHandle,
            0, 
            nextData.initialNumberOfAnchorIds, 
            [=](auto begin, auto end, auto threadId){
                maketasks(begin, end, threadId);
            }
        );

        nextData.decodedSubjectStrings.clear();
        for(int i = 0; i < batchData.threadsInThreadPool; i++){
            nextData.decodedSubjectStrings.insert(
                nextData.decodedSubjectStrings.end(),
                decodedSubjectStringsPerThread[i].begin(),
                decodedSubjectStringsPerThread[i].end()
            );
        }

        // auto it = std::remove_if(nextData.tasks.begin(), nextData.tasks.end(), [](const auto& t){return !t.active;});
        // nextData.tasks.erase(it, nextData.tasks.end());
        int cur = 0;
        for(int i = 0; i < int(nextData.tasks.size()); i++){
            if(nextData.tasks[i].active && i != cur){
                nextData.tasks[cur] = std::move(nextData.tasks[i]);
                nextData.decodedSubjectStrings[cur] = std::move(nextData.decodedSubjectStrings[i]);
                // std::copy_n(
                //     nextData.h_subject_sequences_data.get() + i,
                //     batchData.encodedSequencePitchInInts,
                //     nextData.h_subject_sequences_data.get() + cur
                // );
                cur++;
            }
        }

        nextData.tasks.erase(nextData.tasks.begin() + cur, nextData.tasks.end());
        nextData.decodedSubjectStrings.erase(nextData.decodedSubjectStrings.begin() + cur, nextData.decodedSubjectStrings.end());

        // cudaMemcpyAsync(
        //     nextData.d_subject_sequences_data,
        //     nextData.h_subject_sequences_data,
        //     nextData.h_subject_sequences_data.sizeInBytes(),
        //     H2D,
        //     nextData.stream); CUERR;

        //     cudaStreamSynchronize(nextData.stream); CUERR;

        nextData.signal();
    }

#if 0
    void makeBatchResultData(Batch& batch, BatchResultData& resultData){
        // resultData.tasks = std::move(batch.tasks);

        // auto& da = batch.dataArrays;
        // std::swap(resultData.h_corrected_subjects, da.h_corrected_subjects);
        // std::swap(resultData.h_subject_is_corrected, da.h_subject_is_corrected);
        // std::swap(resultData.h_is_high_quality_subject, da.h_is_high_quality_subject);
        // std::swap(resultData.h_subject_sequences_lengths, da.h_subject_sequences_lengths);
        // std::swap(resultData.h_num_uncorrected_positions_per_subject, da.h_num_uncorrected_positions_per_subject);
        // std::swap(resultData.h_uncorrected_positions_per_subject, da.h_uncorrected_positions_per_subject);

        // const auto& transFuncData = *batch.transFuncData;

        // if(transFuncData.correctionOptions.correctCandidates){
        //     std::swap(resultData.h_num_corrected_candidates, da.h_num_corrected_candidates);
        //     std::swap(resultData.h_corrected_candidates, da.h_corrected_candidates);
        //     std::swap(resultData.h_candidate_sequences_data, da.h_candidate_sequences_data);            
        //     std::swap(resultData.h_indices_of_corrected_candidates, da.h_indices_of_corrected_candidates);
        //     std::swap(resultData.h_indices_per_subject_prefixsum, da.h_indices_per_subject_prefixsum);
        //     std::swap(resultData.h_candidate_sequences_lengths, da.h_candidate_sequences_lengths);
        //     std::swap(resultData.h_alignment_shifts, da.h_alignment_shifts);
        //     std::swap(resultData.h_candidate_read_ids, da.h_candidate_read_ids);
        // }

        // resultData.maximum_sequence_length = da.maximum_sequence_length;
        // resultData.sequence_pitch = da.sequence_pitch;
        // resultData.encoded_sequence_pitch = da.encoded_sequence_pitch;
    }
#endif

    void getNextBatchOfSubjectsAndDetermineCandidateReadIds(Batch& batchData){

        if(batchData.isFirstIteration){
            batchData.nextIterationData.done = false;

            getSubjectDataOfNextIteration(
                batchData, 
                batchData.transFuncData->correctionOptions.batchsize,
                *batchData.transFuncData->readStorage
            );

            if(batchData.nextIterationData.initialNumberOfAnchorIds > 0){
                determineCandidateReadIdsOfNextIteration(
                    batchData, 
                    *batchData.transFuncData->minhasher
                );
            }else{
                batchData.nextIterationData.initialNumberOfCandidates = 0;
            }         
         
            batchData.isFirstIteration = false;
        }else{
            batchData.nextIterationData.wait(); //wait until data is available
        }


        if(batchData.nextIterationData.initialNumberOfCandidates == 0){
            return; 
        }else{
            batchData.updateFromIterationData(batchData.nextIterationData);

            //asynchronously prepare data for next iteration
            Batch* batchptr = &batchData;
            batchData.nextIterationData.done = false;
            batchData.backgroundWorker->enqueue(
                [batchptr](){
                    nvtx::push_range("getSubjectDataOfNextIteration",1);
                    getSubjectDataOfNextIteration(
                        *batchptr, 
                        batchptr->transFuncData->correctionOptions.batchsize,
                        *batchptr->transFuncData->readStorage
                    );
                    nvtx::pop_range();

                    nvtx::push_range("determineCandidateReadIdsOfNextIteration",2);
        
                    if(batchptr->nextIterationData.initialNumberOfAnchorIds > 0){
                        determineCandidateReadIdsOfNextIteration(
                            *batchptr, 
                            *batchptr->transFuncData->minhasher
                        );
                    }else{
                        batchptr->nextIterationData.initialNumberOfCandidates = 0;
                        batchptr->nextIterationData.signal();
                    }
                    nvtx::pop_range();
                }
            );

            //allocate memory required for batch processing

            auto& dataArrays = batchData.dataArrays;
            const auto& transFuncData = *(batchData.transFuncData);
            auto& streams = batchData.streams;

            nvtx::push_range("set_problem_dimensions", 4);

            batchData.n_subjects = int(batchData.tasks.size());
            batchData.n_queries = batchData.initialNumberOfCandidates;

            const int min_overlap = std::max(1, std::max(transFuncData.goodAlignmentProperties.min_overlap, 
                int(transFuncData.sequenceFileProperties.maxSequenceLength 
                    * transFuncData.goodAlignmentProperties.min_overlap_ratio)));
    
            const int sequence_pitch = batchData.decodedSequencePitchInBytes;

            int msa_max_column_count = (3*transFuncData.sequenceFileProperties.maxSequenceLength - 2*min_overlap);
            batchData.msa_pitch = SDIV(sizeof(char)*msa_max_column_count, 4) * 4;
            batchData.msa_weights_pitch = SDIV(sizeof(float)*msa_max_column_count, 4) * 4;
            size_t msa_weights_pitch_floats = batchData.msa_weights_pitch / sizeof(float);
    
            //sequence input data
    
            dataArrays.h_subject_sequences_data.resize(batchData.n_subjects * batchData.encodedSequencePitchInInts);
            dataArrays.h_candidate_sequences_data.resize(batchData.n_queries * batchData.encodedSequencePitchInInts);
            dataArrays.h_transposedCandidateSequencesData.resize(batchData.n_queries * batchData.encodedSequencePitchInInts);
            dataArrays.h_subject_sequences_lengths.resize(batchData.n_subjects);
            dataArrays.h_candidate_sequences_lengths.resize(batchData.n_queries);
            dataArrays.h_candidates_per_subject.resize(batchData.n_subjects);
            dataArrays.h_candidates_per_subject_prefixsum.resize((batchData.n_subjects + 1));
            dataArrays.h_subject_read_ids.resize(batchData.n_subjects);
            dataArrays.h_candidate_read_ids.resize(batchData.n_queries);
    
            dataArrays.d_subject_sequences_data.resize(batchData.n_subjects * batchData.encodedSequencePitchInInts);
            dataArrays.d_candidate_sequences_data.resize(batchData.n_queries * batchData.encodedSequencePitchInInts);
            dataArrays.d_transposedCandidateSequencesData.resize(batchData.n_queries * batchData.encodedSequencePitchInInts);
            dataArrays.d_subject_sequences_lengths.resize(batchData.n_subjects);
            dataArrays.d_candidate_sequences_lengths.resize(batchData.n_queries);
            dataArrays.d_candidates_per_subject.resize(batchData.n_subjects);
            dataArrays.d_candidates_per_subject_prefixsum.resize((batchData.n_subjects + 1));
            dataArrays.d_subject_read_ids.resize(batchData.n_subjects);
            dataArrays.d_candidate_read_ids.resize(batchData.n_queries);
    
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
            dataArrays.h_indices_per_subject_prefixsum.resize((batchData.n_subjects + 1));
            dataArrays.h_num_indices.resize(1);
    
            dataArrays.d_indices.resize(batchData.n_queries);
            dataArrays.d_indices_per_subject.resize(batchData.n_subjects);
            dataArrays.d_indices_per_subject_prefixsum.resize((batchData.n_subjects + 1));
            dataArrays.d_num_indices.resize(1);
            dataArrays.d_num_indices_tmp.resize(1);
    
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
                        batchData.initialNumberOfCandidates,
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
                                                    batchData.initialNumberOfCandidates,
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
            //dataArrays.zero_gpu(streams[primary_stream_index]);
        }

    }


    void getAnchorReadIds(Batch& batchData, int batchsize){

    }

    void getAnchorData(Batch& batchData, const DistributedReadStorage& readStorage){

    }

    void calculateAnchorMinhashSignature(Batch& batchData){

    }

    void determineCandidateReadIds(Batch& batchData){

    }


    void getCandidateSequenceData(Batch& batchData, const DistributedReadStorage& readStorage){

        cudaSetDevice(batchData.deviceId); CUERR;

        const auto& transFuncData = *batchData.transFuncData;

        DataArrays& dataArrays = batchData.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = batchData.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batchData.events;

        dataArrays.h_candidates_per_subject_prefixsum[0] = 0;
        for(size_t i = 0; i < batchData.tasks.size(); i++){
            const size_t num = batchData.tasks[i].candidate_read_ids.size();
            dataArrays.h_candidates_per_subject[i] = num;
            dataArrays.h_candidates_per_subject_prefixsum[i+1] = dataArrays.h_candidates_per_subject_prefixsum[i] + num;
        }

        for(size_t i = 0; i < batchData.tasks.size(); i++){
            const auto& task = batchData.tasks[i];
            dataArrays.h_subject_read_ids[i] = task.readId;

            const int offset = dataArrays.h_candidates_per_subject_prefixsum[i];
            std::copy(task.candidate_read_ids.begin(),
                        task.candidate_read_ids.end(),
                        dataArrays.h_candidate_read_ids + offset);
        }

        cudaMemcpyAsync(dataArrays.d_subject_read_ids,
                        dataArrays.h_subject_read_ids,
                        dataArrays.h_subject_read_ids.sizeInBytes(),
                        H2D,
                        streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.d_candidate_read_ids,
                        dataArrays.h_candidate_read_ids,
                        dataArrays.h_candidate_read_ids.sizeInBytes(),
                        H2D,
                        streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.d_candidates_per_subject,
                        dataArrays.h_candidates_per_subject,
                        dataArrays.h_candidates_per_subject.sizeInBytes(),
                        H2D,
                        streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.d_candidates_per_subject_prefixsum,
                        dataArrays.h_candidates_per_subject_prefixsum,
                        dataArrays.h_candidates_per_subject_prefixsum.sizeInBytes(),
                        H2D,
                        streams[primary_stream_index]); CUERR;

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
            batchData.subjectSequenceGatherHandle2,
            dataArrays.d_subject_sequences_data.get(),
            batchData.encodedSequencePitchInInts,
            dataArrays.h_subject_read_ids,
            dataArrays.d_subject_read_ids,
            batchData.n_subjects,
            batchData.deviceId,
            streams[primary_stream_index],
            transFuncData.runtimeOptions.nCorrectorThreads);

        readStorage.gatherSequenceDataToGpuBufferAsync(
            batchData.threadPool,
            batchData.candidateSequenceGatherHandle2,
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

        cudaMemcpyAsync(dataArrays.h_subject_sequences_data,
                        dataArrays.d_subject_sequences_data,
                        dataArrays.d_subject_sequences_data.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_candidate_sequences_data,
                        dataArrays.d_candidate_sequences_data,
                        dataArrays.d_candidate_sequences_data.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

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

        //cudaStreamWaitEvent(streams[primary_stream_index], events[alignment_data_transfer_h2d_finished_event_index], 0); CUERR;
        
        call_popcount_shifted_hamming_distance_kernel_async(
                    dataArrays.getDeviceAlignmentResultPointers(),
                    dataArrays.getDeviceSequencePointers(),
                    dataArrays.d_candidates_per_subject_prefixsum,
                    dataArrays.h_candidates_per_subject,
                    dataArrays.d_candidates_per_subject,
                    batch.n_subjects,
                    batch.n_queries,
                    transFuncData.sequenceFileProperties.maxSequenceLength,
                    batch.encodedSequencePitchInInts,
                    transFuncData.goodAlignmentProperties.min_overlap,
                    transFuncData.goodAlignmentProperties.maxErrorRate,
                    transFuncData.goodAlignmentProperties.min_overlap_ratio,
                    //batch.maxSubjectLength,
                    streams[primary_stream_index],
                    batch.kernelLaunchHandle);

        cudaEventRecord(events[alignments_finished_event_index], streams[primary_stream_index]); CUERR;
#if 0
        auto identity = [](auto i){return i;};

        cudaMemcpyAsync(dataArrays.h_alignment_best_alignment_flags,
                        dataArrays.d_alignment_best_alignment_flags,
                        dataArrays.d_alignment_best_alignment_flags.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_alignment_scores,
                        dataArrays.d_alignment_scores,
                        dataArrays.d_alignment_scores.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_alignment_overlaps,
                        dataArrays.d_alignment_overlaps,
                        dataArrays.d_alignment_overlaps.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_alignment_shifts,
                    dataArrays.d_alignment_shifts,
                    dataArrays.d_alignment_shifts.sizeInBytes(),
                    D2H,
                    streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_alignment_nOps,
                    dataArrays.d_alignment_nOps,
                    dataArrays.d_alignment_nOps.sizeInBytes(),
                    D2H,
                    streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_alignment_isValid,
                    dataArrays.d_alignment_isValid,
                    dataArrays.d_alignment_isValid.sizeInBytes(),
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

                    cudaMemcpyAsync(dataArrays.h_subject_sequences_lengths,
                                    dataArrays.d_subject_sequences_lengths,
                                    dataArrays.d_subject_sequences_lengths.sizeInBytes(),
                                    D2H,
                                    streams[primary_stream_index]); CUERR;

                    cudaMemcpyAsync(dataArrays.h_candidate_sequences_lengths,
                                    dataArrays.d_candidate_sequences_lengths,
                                    dataArrays.d_candidate_sequences_lengths.sizeInBytes(),
                                    D2H,
                                    streams[primary_stream_index]); CUERR;

        cudaDeviceSynchronize(); CUERR;

        for(int i = 0; i < dataArrays.n_subjects; i++){
            // std::string s; s.resize(128);
            // decode2BitSequence(&s[0], (const unsigned int*)dataArrays.h_subject_sequences_data.get() + i * batch.encodedSequencePitchInInts * sizeof(unsigned int), 100, identity);
            // std::cout << "Subject  : " << s << " " << batch.tasks[i].readId << std::endl;

            if(dataArrays.n_queries > 0){
                for(int j = 0; j < dataArrays.n_queries; j++){
                    // std::string s; s.resize(128);
                    // decode2BitSequence(&s[0], (const unsigned int*)dataArrays.h_candidate_sequences_data.get() + j * batch.encodedSequencePitchInInts * sizeof(unsigned int), 100, identity);
                    //const char* hostptr = transFuncData.readStorage->fetchSequenceData_ptr(batch.tasks[i].candidate_read_ids[j]);
                    //std::string hostsequence = get2BitString((const unsigned int*)hostptr, 100, identity);
                    //std::string s = get2BitString((const unsigned int*)(dataArrays.h_candidate_sequences_data.get() + j * batch.encodedSequencePitchInInts * sizeof(unsigned int)), 100, identity);
                    // if(hostsequence != s){
                    //     std::cout << "host " << hostsequence << std::endl;
                    //     std::cout << "device " << s << std::endl;
                    // }
                    //std::cout << "Candidate  : " << s << " " << batch.tasks[i].candidate_read_ids[j] << std::endl;
                    std::cout << "Candidate  : " << batch.tasks[i].candidate_read_ids[j] << std::endl;
                    std::cout << "Fwd alignment: " << dataArrays.h_alignment_scores[j] << " "
                                << dataArrays.h_alignment_overlaps[j] << " "
                                << dataArrays.h_alignment_shifts[j] << " "
                                << dataArrays.h_alignment_nOps[j] << " "
                                << dataArrays.h_alignment_isValid[j] << std::endl;
                    std::cout << "Rev alignment: " << dataArrays.h_alignment_scores[j + dataArrays.n_queries] << " "
                                << dataArrays.h_alignment_overlaps[j + dataArrays.n_queries] << " "
                                << dataArrays.h_alignment_shifts[j + dataArrays.n_queries] << " "
                                << dataArrays.h_alignment_nOps[j + dataArrays.n_queries] << " "
                                << dataArrays.h_alignment_isValid[j + dataArrays.n_queries] << std::endl;
                }
            }
        }
        std::exit(0);
#endif
		//Compare each forward alignment with the correspoding reverse complement alignment and keep the best, if any.
		//    If reverse complement is the best, it is copied into the first half, replacing the forward alignment

        call_cuda_find_best_alignment_kernel_async_exp(
                    dataArrays.getDeviceAlignmentResultPointers(),
                    dataArrays.getDeviceSequencePointers(),
                    dataArrays.d_candidates_per_subject_prefixsum.get(),
                    batch.n_subjects,
					batch.n_queries,
                    transFuncData.goodAlignmentProperties.min_overlap_ratio,
                    transFuncData.goodAlignmentProperties.min_overlap,
                    transFuncData.correctionOptions.estimatedErrorrate,
                    streams[primary_stream_index],
                    batch.kernelLaunchHandle,
                    dataArrays.h_subject_read_ids[0]);

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


        //initialize indices with -1. this allows to calculate the histrogram later on
        //without knowing the number of valid indices
        call_fill_kernel_async(dataArrays.d_indices.get(), batch.n_queries, -1, streams[primary_stream_index]);

        auto select_op = [] __device__ (const BestAlignment_t& flag){
            return flag != BestAlignment_t::None;
        };

        cub::TransformInputIterator<bool,decltype(select_op), BestAlignment_t*>
            d_isGoodAlignment(dataArrays.d_alignment_best_alignment_flags,
                            select_op);

        //Writes indices of candidates with alignmentflag != None to d_indices
        size_t cubTempSize = dataArrays.d_cub_temp_storage.sizeInBytes();

        cub::DeviceSelect::Flagged(dataArrays.d_cub_temp_storage.get(),
                                    cubTempSize,
                                    cub::CountingInputIterator<int>(0),
                                    d_isGoodAlignment,
                                    dataArrays.d_indices.get(),
                                    dataArrays.d_num_indices.get(),
                                    batch.n_queries,
                                    streams[primary_stream_index]); CUERR;

        //calculate indices_per_subject
        cub::DeviceHistogram::HistogramRange(dataArrays.d_cub_temp_storage.get(),
                    cubTempSize,
                    dataArrays.d_indices.get(),
                    dataArrays.d_indices_per_subject.get(),
                    batch.n_subjects+1,
                    dataArrays.d_candidates_per_subject_prefixsum.get(),
                    batch.n_queries,
                    streams[primary_stream_index]); CUERR;

        //calculate indices_per_subject_prefixsum
        call_set_kernel_async(dataArrays.d_indices_per_subject_prefixsum.get(),
                                0,
                                0,
                                streams[primary_stream_index]);

        cub::DeviceScan::InclusiveSum(dataArrays.d_cub_temp_storage.get(),
                    cubTempSize,
                    dataArrays.d_indices_per_subject.get(),
                    dataArrays.d_indices_per_subject_prefixsum.get()+1,
                    batch.n_subjects,
                    streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_num_indices,
                        dataArrays.d_num_indices,
                        sizeof(int),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaEventRecord(events[num_indices_transfered_event_index], streams[primary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], events[num_indices_transfered_event_index], 0); CUERR;

        cudaMemcpyAsync(dataArrays.h_indices,
                        dataArrays.d_indices,
                        dataArrays.d_indices.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_indices_per_subject,
                        dataArrays.d_indices_per_subject,
                        dataArrays.d_indices_per_subject.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_indices_per_subject_prefixsum,
                        dataArrays.d_indices_per_subject_prefixsum,
                        dataArrays.d_indices_per_subject_prefixsum.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaEventRecord(events[indices_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;
        //cudaStreamWaitEvent(streams[primary_stream_index], events[indices_transfer_finished_event_index], 0); CUERR;

        // {
        //     cudaDeviceSynchronize(); CUERR;
        //     unsigned int* d_candidateDataTmp = nullptr;
        //     unsigned int* d_candidateData = (unsigned int*)dataArrays.d_candidate_sequences_data.get();

        //     const int nIndices = dataArrays.h_num_indices[0];
        //     const int* d_indices = dataArrays.d_indices.get();
        //     const int numCols = batch.encodedSequencePitchInInts;
            



        //     cubCachingAllocator.DeviceAllocate((void**)&d_candidateDataTmp,
        //                                         sizeof(unsigned int) * nIndices,
        //                                         streams[primary_stream_index]); CUERR;

        //     dim3 block(256,1,1);
        //     dim3 grid(std::min(65535, SDIV(nIndices * numCols, 256)),1,1);

        //     generic_kernel<<<grid, block, 0, streams[primary_stream_index]>>>([=] __device__ (){
        //         for(size_t i = threadIdx.x + size_t(blockIdx.x) * 256; i < nIndices * numCols; i += size_t(256) * gridDim.x){
        //             const int outputrow = i / numCols;
        //             const int inputrow = d_indices[outputrow];
        //             const int col = i % numCols;
        //             d_candidateDataTmp[size_t(outputrow) * numCols + col] 
        //                     = d_candidateData[size_t(inputrow) * numCols + col];
        //         }
        //     }); CUERR;

        //     cubCachingAllocator.DeviceFree(d_candidateDataTmp); CUERR;

        // }

        

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

        //std::cerr << "After alignment: " << *dataArrays.h_num_indices << " / " << dataArrays.n_queries << "\n";
	}

    void rearrangeIndices(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        DataArrays& dataArrays = batch.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

#if 0



        int* d_indices_segmented_partitioned;
        BestAlignment_t* d_alignment_best_alignment_flags_compact;
        BestAlignment_t* d_alignment_best_alignment_flags_discardedoutput;

        cubCachingAllocator.DeviceAllocate((void**)&d_indices_segmented_partitioned,
                                            sizeof(int) * batch.n_queries,
                                            streams[primary_stream_index]); CUERR;

        cubCachingAllocator.DeviceAllocate((void**)&d_alignment_best_alignment_flags_compact,
                                            sizeof(BestAlignment_t) * batch.n_queries,
                                            streams[primary_stream_index]); CUERR;

        cubCachingAllocator.DeviceAllocate((void**)&d_alignment_best_alignment_flags_discardedoutput,
                                            sizeof(BestAlignment_t) * batch.n_queries,
                                            streams[primary_stream_index]); CUERR;

        //compact alignment flags according to indices
        call_compact_kernel_async(d_alignment_best_alignment_flags_compact,
                                dataArrays.d_alignment_best_alignment_flags,
                                dataArrays.d_indices,
                                dataArrays.h_num_indices[0],
                                streams[primary_stream_index]);

        //partition d_indices according to d_alignment_best_alignment_flags
        //with this partitioning branch divergence in kernels is reduced

        //segmented partitioning of indices is achieved by using segmented radix sort of pairs,
        //where each pair is composed of (key: d_alignment_best_alignment_flags[d_indices[i]], value: d_indices[i])
        static_assert(sizeof(char) == sizeof(BestAlignment_t), "");

        size_t cubTempSize = dataArrays.d_cub_temp_storage.sizeInBytes();

        cub::DeviceSegmentedRadixSort::SortPairs(dataArrays.d_cub_temp_storage.get(),
                                                cubTempSize,
                                                (const char*) d_alignment_best_alignment_flags_compact,
                                                (char*)d_alignment_best_alignment_flags_discardedoutput,
                                                dataArrays.d_indices,
                                                d_indices_segmented_partitioned,
                                                dataArrays.h_num_indices[0],
                                                batch.n_subjects,
                                                dataArrays.d_indices_per_subject_prefixsum,
                                                dataArrays.d_indices_per_subject_prefixsum+1,
                                                0,
                                                3,
                                                streams[primary_stream_index]);

        cudaMemcpyAsync(dataArrays.d_indices, d_indices_segmented_partitioned, sizeof(int) * (dataArrays.h_num_indices[0]), D2D, streams[primary_stream_index]); CUERR;

        cubCachingAllocator.DeviceFree(d_alignment_best_alignment_flags_compact);
        cubCachingAllocator.DeviceFree(d_alignment_best_alignment_flags_discardedoutput);
        cubCachingAllocator.DeviceFree(d_indices_segmented_partitioned);
#endif

        cudaEventRecord(events[indices_calculated_event_index], streams[primary_stream_index]); CUERR;

        //copy indices of usable candidates. these are required on the host for coping quality scores, and for creating results.
        cudaStreamWaitEvent(streams[secondary_stream_index], events[indices_calculated_event_index], 0); CUERR;

        cudaMemcpyAsync(dataArrays.h_num_indices,
                        dataArrays.d_num_indices,
                        dataArrays.d_num_indices.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_indices,
                        dataArrays.d_indices,
                        dataArrays.d_indices.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_indices_per_subject,
                        dataArrays.d_indices_per_subject,
                        dataArrays.d_indices_per_subject.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_indices_per_subject_prefixsum,
                        dataArrays.d_indices_per_subject_prefixsum,
                        dataArrays.d_indices_per_subject_prefixsum.sizeInBytes(),
                        D2H,
                        streams[secondary_stream_index]); CUERR;

        cudaEventRecord(events[indices_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[primary_stream_index], events[indices_transfer_finished_event_index], 0); CUERR;

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

            gpuReadStorage->gatherQualitiesToGpuBufferAsync(
                batch.threadPool,
                batch.subjectQualitiesGatherHandle2,
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
                batch.candidateQualitiesGatherHandle2,
                dataArrays.d_candidate_qualities,
                batch.qualityPitchInBytes,
                dataArrays.h_candidate_read_ids.get(),
                dataArrays.d_candidate_read_ids.get(),
                batch.n_queries,
                batch.deviceId,
                streams[primary_stream_index],
                transFuncData.runtimeOptions.nCorrectorThreads);

            cudaEventRecord(events[quality_transfer_finished_event_index], streams[primary_stream_index]); CUERR;

            cudaStreamWaitEvent(streams[secondary_stream_index], events[quality_transfer_finished_event_index], 0); CUERR;

            cudaMemcpyAsync(dataArrays.h_subject_qualities,
                            dataArrays.d_subject_qualities,
                            dataArrays.d_subject_qualities.sizeInBytes(),
                            D2H,
                            streams[secondary_stream_index]);

            cudaMemcpyAsync(dataArrays.h_candidate_qualities,
                            dataArrays.d_candidate_qualities,
                            dataArrays.d_candidate_qualities.sizeInBytes(),
                            D2H,
                            streams[secondary_stream_index]);
        }

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
	}


    void buildMultipleSequenceAlignment(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        DataArrays& dataArrays = batch.dataArrays;

        if(transFuncData.correctionOptions.useQualityScores){
		     cudaStreamWaitEvent(streams[primary_stream_index], events[quality_transfer_finished_event_index], 0); CUERR;
        }

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
                        dataArrays.d_indices_per_subject_prefixsum,
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
        int* d_shouldBeKept_positions = nullptr;
        int* d_newIndices = nullptr;
        int* d_indices_per_subject_tmp = nullptr;

        cubCachingAllocator.DeviceAllocate(
            (void**)&d_shouldBeKept, 
            sizeof(bool) * batch.n_queries, 
            streams[primary_stream_index]
        ); CUERR;

        cubCachingAllocator.DeviceAllocate(
            (void**)&d_shouldBeKept_positions, 
            sizeof(int) * batch.n_queries, 
            streams[primary_stream_index]
        ); CUERR;

        cubCachingAllocator.DeviceAllocate(
            (void**)&d_newIndices, 
            sizeof(int) * batch.n_queries, 
            streams[primary_stream_index]
        ); CUERR;

        cubCachingAllocator.DeviceAllocate(
            (void**)&d_indices_per_subject_tmp, 
            sizeof(int) * batch.n_subjects, 
            streams[primary_stream_index]
        ); CUERR;

        for(int iteration = 0; iteration < max_num_minimizations; iteration++){

            {
                //Initialize d_shouldBeKept array

                const int N = batch.n_queries;
                const int* d_num_indices = dataArrays.d_num_indices.get();
            
                generic_kernel<<<SDIV(batch.n_queries, 128), 128, 0, streams[primary_stream_index]>>>(
                    [=] __device__ (){
                        const int index = threadIdx.x + blockIdx.x * 128;
                        const int maxValidIndex = *d_num_indices;
                        if(index < N){
                            d_shouldBeKept[index] = (index < maxValidIndex);
                        }
                    }
                ); CUERR;
            }

            //select candidates which are to be removed
            call_msa_findCandidatesOfDifferentRegion_kernel_async(
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
                        dataArrays.d_indices_per_subject_prefixsum,
                        desiredAlignmentMaxErrorRate,
                        transFuncData.correctionOptions.estimatedCoverage,
                        dataArrays.d_canExecute.get(),
                        streams[primary_stream_index],
                        batch.kernelLaunchHandle,
                        dataArrays.d_subject_read_ids,
                        false);  CUERR;

            //save current indices_per_subject
            cudaMemcpyAsync(d_indices_per_subject_tmp,
                dataArrays.d_indices_per_subject,
                sizeof(int) * batch.n_subjects,
                D2D,
                streams[primary_stream_index]); CUERR;

            call_fill_kernel_async(d_newIndices, batch.n_queries, -1, streams[primary_stream_index]);

            size_t cubTempSize = dataArrays.d_cub_temp_storage.sizeInBytes();

            cub::DeviceSelect::Flagged(dataArrays.d_cub_temp_storage.get(),
                        cubTempSize,
                        cub::CountingInputIterator<int>{0},
                        d_shouldBeKept,
                        d_shouldBeKept_positions,
                        dataArrays.d_num_indices_tmp.get(),
                        batch.n_queries,
                        streams[primary_stream_index]); CUERR;

            call_compact_kernel_async(d_newIndices,
                dataArrays.d_indices.get(),
                d_shouldBeKept_positions,
                dataArrays.d_num_indices_tmp.get(),
                batch.n_queries,
                streams[primary_stream_index]);

            // d_newIndices now contains *d_num_indices_tmp valid indices, followed by (n_queries - *d_num_indices_tmp) times -1

            cudaMemcpyAsync(dataArrays.d_indices,
                d_newIndices,
                sizeof(int) * batch.n_queries,
                D2D,
                streams[primary_stream_index]); CUERR;

            //calculate indices per subject. since candidates_per_subject_prefixsum only contains numbers >= 0, the -1 entries in d_newIndices will be ignored.
            cub::DeviceHistogram::HistogramRange(dataArrays.d_cub_temp_storage.get(),
                        cubTempSize,
                        d_newIndices,
                        dataArrays.d_indices_per_subject.get(),
                        batch.n_subjects+1,
                        dataArrays.d_candidates_per_subject_prefixsum.get(),
                        batch.n_queries,
                        streams[primary_stream_index]); CUERR;

            //make indices per subject prefixsum
            call_set_kernel_async(dataArrays.d_indices_per_subject_prefixsum.get(),
                                    0,
                                    0,
                                    streams[primary_stream_index]);

            cub::DeviceScan::InclusiveSum(dataArrays.d_cub_temp_storage.get(),
                        cubTempSize,
                        dataArrays.d_indices_per_subject.get(),
                        dataArrays.d_indices_per_subject_prefixsum.get()+1,
                        batch.n_subjects,
                        streams[primary_stream_index]); CUERR;

            {
                /*
                    compare old indices_per_subject , which are stored in indices_per_subject_tmp
                    to the new indices_per_subject.
                    set value in indices_per_subject_tmp to 0 if old value and new value are equal, else the value is set to the new value.
                    this prevents rebuilding the MSAs of subjects whose indices where not changed by minimization (all indices are kept)
                */

                const int* d_indices_per_subject = dataArrays.d_indices_per_subject;
                const int n_subjects = batch.n_subjects;
                cudaStream_t stream = streams[primary_stream_index];

                dim3 block(128,1,1);
                dim3 grid(SDIV(batch.n_subjects, block.x),1,1);
                generic_kernel<<<grid, block, 0, stream>>>(
                    [=] __device__ (){
                        for(int i = threadIdx.x + blockDim.x * blockIdx.x; i < n_subjects; i += blockDim.x * gridDim.x){
                            if(d_indices_per_subject[i] == d_indices_per_subject_tmp[i]){
                                d_indices_per_subject_tmp[i] = 0;
                            }else{
                                d_indices_per_subject_tmp[i] = d_indices_per_subject[i];
                            }
                        }
                    }
                ); CUERR;

                /*const int* d_num_indices = dataArrays.d_num_indices;
                generic_kernel<<<1, 1, 0, stream>>>([=] __device__ (){
                    int sum = 0;
                    for(int i = 0; i < n_subjects; i++){
                        sum += d_indices_per_subject_tmp[i];
                    }
                    printf("sum = %d, totalindices %d\n", sum, *d_num_indices);
                }); CUERR;*/
            }

            {
                //set d_canExecute flag. reconstructing the msa and performing another minimization step 
                // is only neccessary if the indices changed, and if there are any indices left.

                const int* d_num_indices_tmp = dataArrays.d_num_indices_tmp.get();
                const int* d_num_indices = dataArrays.d_num_indices.get();
                bool* d_canExecute = dataArrays.d_canExecute.get();

                generic_kernel<<<1,1, 0, streams[primary_stream_index]>>>(
                    [=] __device__ (){
                        assert(*d_num_indices_tmp <= *d_num_indices);

                        if(*d_num_indices_tmp > 0 && *d_num_indices_tmp < *d_num_indices){
                            *d_canExecute = true;
                        }else{
                            *d_canExecute = false;
                        }
                    }
                ); CUERR;

            }

            std::swap(dataArrays.d_num_indices_tmp, dataArrays.d_num_indices);

            //cudaMemcpyAsync(dataArrays.h_num_indices, dataArrays.d_num_indices, sizeof(int), D2H, streams[primary_stream_index]);  CUERR;
            //cudaEventRecord(events[num_indices_transfered_event_index], streams[primary_stream_index]); CUERR;

            build_msa_async(dataArrays.getDeviceMSAPointers(),
                            dataArrays.getDeviceAlignmentResultPointers(),
                            dataArrays.getDeviceSequencePointers(),
                            dataArrays.getDeviceQualityPointers(),
                            dataArrays.d_candidates_per_subject_prefixsum,
                            dataArrays.d_indices,
                            d_indices_per_subject_tmp,
                            dataArrays.d_indices_per_subject_prefixsum,
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
        cubCachingAllocator.DeviceFree(d_newIndices); CUERR;
        cubCachingAllocator.DeviceFree(d_indices_per_subject_tmp); CUERR;
        cubCachingAllocator.DeviceFree(d_shouldBeKept_positions); CUERR;
        
        {
            //std::cerr << "minimization finished\n";

            cudaEventRecord(events[msa_build_finished_event_index], streams[primary_stream_index]); CUERR;
            cudaStreamWaitEvent(streams[secondary_stream_index], events[msa_build_finished_event_index], 0); CUERR;

            cudaMemcpyAsync(dataArrays.h_num_indices,
                            dataArrays.d_num_indices,
                            dataArrays.d_num_indices.sizeInBytes(),
                            D2H,
                            streams[secondary_stream_index]); CUERR;

            cudaMemcpyAsync(dataArrays.h_indices,
                            dataArrays.d_indices,
                            dataArrays.d_indices.sizeInBytes(),
                            D2H,
                            streams[secondary_stream_index]); CUERR;

            cudaMemcpyAsync(dataArrays.h_indices_per_subject,
                            dataArrays.d_indices_per_subject,
                            dataArrays.d_indices_per_subject.sizeInBytes(),
                            D2H,
                            streams[secondary_stream_index]); CUERR;

            cudaMemcpyAsync(dataArrays.h_indices_per_subject_prefixsum,
                            dataArrays.d_indices_per_subject_prefixsum,
                            dataArrays.d_indices_per_subject_prefixsum.sizeInBytes(),
                            D2H,
                            streams[secondary_stream_index]); CUERR;

                        //update host qscores accordingly
                        /*cudaMemcpyAsync(dataArrays.h_candidate_qualities,
                                        dataArrays.d_candidate_qualities,
                                        dataArrays.d_candidate_qualities.sizeInBytes(),
                                        D2H,
                                        streams[secondary_stream_index]);*/

            cudaEventRecord(events[indices_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;
        }


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

        cudaMemcpyAsync(dataArrays.h_indices_per_subject_prefixsum,
                        dataArrays.d_indices_per_subject_prefixsum,
                        dataArrays.d_indices_per_subject_prefixsum.sizeInBytes(),
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
                    dataArrays.d_indices_per_subject_prefixsum,
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

        call_msa_correct_candidates_kernel_async(
                dataArrays.getDeviceMSAPointers(),
                dataArrays.getDeviceAlignmentResultPointers(),
                dataArrays.getDeviceSequencePointers(),
                dataArrays.getDeviceCorrectionResultPointers(),
                dataArrays.d_indices,
                dataArrays.d_indices_per_subject,
                dataArrays.d_indices_per_subject_prefixsum,
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

#if 0 
    void unpackClassicResults2(Batch& batch){

        const auto& transFuncDataPtr = batch.transFuncData;
        assert(transFuncDataPtr->correctionOptions.correctionType == CorrectionType::Classic);

        constexpr BatchState expectedState = BatchState::UnpackClassicResults;

        assert(batch.state == expectedState);

        cudaSetDevice(batch.deviceId); CUERR;

        auto& events = batch.events;
        auto& streams = batch.streams;



        cudaError_t errort = cudaEventQuery(events[correction_finished_event_index]);
        if(errort != cudaSuccess){
            std::cout << "error cudaEventQuery\n";
            std::exit(0);
        }
        assert(cudaEventQuery(events[correction_finished_event_index]) == cudaSuccess); CUERR;
        assert(cudaEventQuery(events[result_transfer_finished_event_index]) == cudaSuccess); CUERR;

        batch.resultData.wait();
        batch.resultData.done = false;

        makeBatchResultData(batch, batch.resultData);

        BatchResultData* resultDataPtr = &batch.resultData;

        //batch.dataArrays.copyEverythingToHostForDebugging();

        auto unpackAnchors = [resultDataPtr,transFuncDataPtr](int begin, int end){

            const auto& transFuncData = *transFuncDataPtr;
            auto& resultData = *resultDataPtr;

            //std::cerr << "in unpackAnchors " << begin << " - " << end << "\n";

            for(int subject_index = begin; subject_index < end; ++subject_index) {
                auto& task = resultData.tasks[subject_index];
                const char* const my_corrected_subject_data = resultData.h_corrected_subjects + subject_index * resultData.sequence_pitch;
                task.corrected = resultData.h_subject_is_corrected[subject_index];
                task.highQualityAlignment = resultData.h_is_high_quality_subject[subject_index].hq();

                // if(task.readId == 5383){
                //     dataArrays.printActiveDataOfSubject(subject_index, std::cerr);
                // }

                if(task.corrected) {
                    const int subject_length = resultData.h_subject_sequences_lengths[subject_index];
                    task.corrected_subject = std::move(std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length});

                    //task.correctionEqualsOriginal = task.corrected_subject == task.subject_string;

                    const int numUncorrectedPositions = resultData.h_num_uncorrected_positions_per_subject[subject_index];
                    if(numUncorrectedPositions > 0){
                        task.uncorrectedPositionsNoConsensus.resize(numUncorrectedPositions);
                        std::copy_n(resultData.h_uncorrected_positions_per_subject + subject_index * resultData.maximum_sequence_length,
                                    numUncorrectedPositions,
                                    task.uncorrectedPositionsNoConsensus.begin());

                    }

                    auto isValidSequence = [](const std::string& s){
                        return std::all_of(s.begin(), s.end(), [](char c){
                            return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
                        });
                    };

                    if(!isValidSequence(task.corrected_subject)){
                        std::cout << task.corrected_subject << std::endl;
                    }

                    if(task.highQualityAlignment){
                        transFuncData.correctionStatusFlagsPerRead[task.readId] |= readCorrectedAsHQAnchor;
                    }
                    //transFuncData->unlock(task.readId);

                    const bool originalReadContainsN = transFuncData.readStorage->readContainsN(task.readId);

                    if(!originalReadContainsN){
                        const int maxEdits = subject_length / 7;
                        int edits = 0;
                        for(int i = 0; i < subject_length && edits <= maxEdits; i++){
                            if(task.corrected_subject[i] != task.subject_string[i]){
                                task.anchoroutput.edits.emplace_back(i, task.corrected_subject[i]);
                                edits++;
                            }
                        }
                        task.anchoroutput.useEdits = edits <= maxEdits;
                    }else{
                        task.anchoroutput.useEdits = false;
                    }

                    task.anchoroutput.hq = task.highQualityAlignment;                    
                    task.anchoroutput.type = TempCorrectedSequence::Type::Anchor;
                    task.anchoroutput.readId = task.readId;
                    task.anchoroutput.sequence = std::move(task.corrected_subject);
                    task.anchoroutput.uncorrectedPositionsNoConsensus = std::move(task.uncorrectedPositionsNoConsensus);

                }else{

                    transFuncData.correctionStatusFlagsPerRead[task.readId] |= readCouldNotBeCorrectedAsAnchor;

                }
            }
        };

        auto unpackcandidates = [resultDataPtr,transFuncDataPtr](int begin, int end){
            const auto& transFuncData = *transFuncDataPtr;
            auto& resultData = *resultDataPtr;

            //std::cerr << "in unpackcandidates " << begin << " - " << end << "\n";

            for(int subject_index = begin; subject_index < end; ++subject_index) {
                auto& task = resultData.tasks[subject_index];

                const int n_corrected_candidates = resultData.h_num_corrected_candidates[subject_index];
                const char* const my_corrected_candidates_data = resultData.h_corrected_candidates
                                                + resultData.h_indices_per_subject_prefixsum[subject_index] * resultData.sequence_pitch;
                const int* const my_indices_of_corrected_candidates = resultData.h_indices_of_corrected_candidates
                                                + resultData.h_indices_per_subject_prefixsum[subject_index];


                task.corrected_candidates_shifts.resize(n_corrected_candidates);
                task.corrected_candidates_read_ids.resize(n_corrected_candidates);
                task.corrected_candidates.resize(n_corrected_candidates);
                task.corrected_candidate_equals_uncorrected.resize(n_corrected_candidates);
                task.candidatesoutput.reserve(n_corrected_candidates);

                // if(task.readId == 10){
                //     for(int i = 0; i < n_corrected_candidates; ++i) {
                //         std::cerr << my_indices_of_corrected_candidates[i] << " ";
                //     }
                //     std::cerr << std::endl;
                // }

                for(int i = 0; i < n_corrected_candidates; ++i) {
                    const int global_candidate_index = my_indices_of_corrected_candidates[i];

                    const read_number candidate_read_id = resultData.h_candidate_read_ids[global_candidate_index];

                    bool savingIsOk = false;
                    const std::uint8_t mask = transFuncData.correctionStatusFlagsPerRead[candidate_read_id];
                    if(!(mask & readCorrectedAsHQAnchor)) {
                        savingIsOk = true;
                    }
                    if (savingIsOk) {

                        const int candidate_length = resultData.h_candidate_sequences_lengths[global_candidate_index];
                        const int candidate_shift = resultData.h_alignment_shifts[global_candidate_index];

                        const char* const candidate_data = my_corrected_candidates_data + i * resultData.sequence_pitch;
                        if(transFuncData.correctionOptions.new_columns_to_correct < candidate_shift){
                            std::cerr << "\n" << "readid " << task.readId << " candidate readid " << candidate_read_id << " : "
                                    << candidate_shift << " " << transFuncData.correctionOptions.new_columns_to_correct <<"\n";
                        }
                        assert(transFuncData.correctionOptions.new_columns_to_correct >= candidate_shift);
                        task.corrected_candidates_shifts[i] = candidate_shift;
                        task.corrected_candidates_read_ids[i] = candidate_read_id;
                        task.corrected_candidates[i] = std::move(std::string{candidate_data, candidate_data + candidate_length});

                        const bool originalReadContainsN = transFuncData.readStorage->readContainsN(candidate_read_id);

                        // if(!originalReadContainsN){
                            
                        //     task.corrected_candidate_equals_uncorrected[i] = task.corrected_candidates[i] == uncorrectedCandidate;
                        // }else{
                        //     task.corrected_candidate_equals_uncorrected[i] = false;
                        // }

                       

                        TempCorrectedSequence tmp;

                        if(!originalReadContainsN){
                            const unsigned int* ptr = &resultData.h_candidate_sequences_data[global_candidate_index * encodedSequencePitchInInts];
                            const std::string uncorrectedCandidate = get2BitString((const unsigned int*)ptr, candidate_length);

                            const int maxEdits = candidate_length / 7;
                            int edits = 0;
                            for(int pos = 0; pos < candidate_length && edits <= maxEdits; pos++){
                                if(task.corrected_candidates[i][pos] != uncorrectedCandidate[pos]){
                                    tmp.edits.emplace_back(pos, task.corrected_candidates[i][pos]);
                                    edits++;
                                }
                            }

                            tmp.useEdits = edits <= maxEdits;
                        }else{
                            tmp.useEdits = false;
                        }
                        
                        tmp.type = TempCorrectedSequence::Type::Candidate;
                        tmp.shift = task.corrected_candidates_shifts[i];
                        tmp.readId = candidate_read_id;
                        tmp.sequence = std::move(task.corrected_candidates[i]);

                        task.candidatesoutput.emplace_back(std::move(tmp));
    				}
                }
            }
        };

        auto writeOutput = [resultDataPtr,
                            transFuncDataPtr,
                            id = batch.id](){
            nvtx::push_range("batch "+std::to_string(id)+" writeresultoutputhread", 4);
            int notCorrectedNoCandidates = 0;
            int notCorrected = 0;

            //write result to file
    		for(std::size_t subject_index = 0; subject_index < resultDataPtr->tasks.size(); ++subject_index) {

    			const auto& task = resultDataPtr->tasks[subject_index];
    			//std::cout << task.readId << "result" << std::endl;

    			//std::cout << "finished readId " << task.readId << std::endl;

    			if(task.corrected) {
                    transFuncDataPtr->saveCorrectedSequence(task.anchoroutput, task.anchoroutput.encode());
    			}else{
                    if(task.candidate_read_ids.empty()){
                        notCorrectedNoCandidates++;
                    }

                    notCorrected++;
                }

                for(const auto& tmp : task.candidatesoutput){
                    transFuncDataPtr->saveCorrectedSequence(tmp, tmp.encode());
                }
            }
            
            resultDataPtr->signal();

            //std::cerr << "not corrected "<< " " << notCorrectedNoCandidates << " " << notCorrected << "/" << tasks.size() << "\n";

            nvtx::pop_range();
        };

        auto outputThreadPtr = batch.outputThread;
        
        auto batchPtr = &batch;

        if(!transFuncDataPtr->correctionOptions.correctCandidates){
            batch.backgroundWorker->enqueue([&, batchPtr, resultDataPtr](){

                batchPtr->threadPool->parallelFor(
                    resultDataPtr->pforHandle, 
                    0, 
                    int(resultDataPtr->tasks.size()), 
                    [=](auto begin, auto end, auto /*threadId*/){
                        unpackAnchors(begin, end);
                    }
                );

                outputThreadPtr->enqueue(std::move(writeOutput));

            });            
        }else{
            batch.backgroundWorker->enqueue([&, batchPtr, resultDataPtr](){

                batchPtr->threadPool->parallelFor(
                    resultDataPtr->pforHandle, 
                    0, 
                    int(resultDataPtr->tasks.size()), 
                    [=](auto begin, auto end, auto /*threadId*/){
                        unpackAnchors(begin, end);
                        unpackcandidates(begin, end);
                    }
                );

                outputThreadPtr->enqueue(std::move(writeOutput));

            });
        }

        batch.setState(BatchState::Finished, expectedState);
    }

#endif



    // void unpackClassicResults(Batch& batch){

    //     const auto& transFuncData = *batch.transFuncData;

    //     cudaSetDevice(batch.deviceId); CUERR;

    //     auto& events = batch.events;
    //     auto& streams = batch.streams;

    //     cudaError_t errort = cudaEventQuery(events[correction_finished_event_index]);
    //     if(errort != cudaSuccess){
    //         std::cout << "error cudaEventQuery\n";
    //         std::exit(0);
    //     }
    //     assert(cudaEventQuery(events[correction_finished_event_index]) == cudaSuccess); CUERR;
    //     assert(cudaEventQuery(events[result_transfer_finished_event_index]) == cudaSuccess); CUERR;


    //     assert(transFuncData.correctionOptions.correctionType == CorrectionType::Classic);

    //     Batch* batchptr = &batch;

    //     //batch.dataArrays.copyEverythingToHostForDebugging();

    //     auto unpackAnchors = [batchptr](int begin, int end){
    //         nvtx::push_range("Anchor unpacking", 3);
    //         Batch& batch = *batchptr;
    //         DataArrays& dataArrays = batch.dataArrays;
    //         const auto& transFuncData = *batch.transFuncData;

    //         //std::cerr << "in unpackAnchors " << begin << " - " << end << "\n";

    //         for(int subject_index = begin; subject_index < end; ++subject_index) {
    //             auto& task = batch.tasks[subject_index];
    //             const char* const my_corrected_subject_data = dataArrays.h_corrected_subjects + subject_index * batch.decodedSequencePitchInBytes;
    //             task.corrected = dataArrays.h_subject_is_corrected[subject_index];
    //             task.highQualityAlignment = dataArrays.h_is_high_quality_subject[subject_index].hq();

    //             // if(task.readId == 5383){
    //             //     dataArrays.printActiveDataOfSubject(subject_index, std::cerr);
    //             // }

    //             if(task.corrected) {
    //                 const int subject_length = dataArrays.h_subject_sequences_lengths[subject_index];
    //                 task.corrected_subject = std::move(std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length});

    //                 //task.correctionEqualsOriginal = task.corrected_subject == task.subject_string;

    //                 const int numUncorrectedPositions = dataArrays.h_num_uncorrected_positions_per_subject[subject_index];
    //                 if(numUncorrectedPositions > 0){
    //                     task.uncorrectedPositionsNoConsensus.resize(numUncorrectedPositions);
    //                     std::copy_n(dataArrays.h_uncorrected_positions_per_subject + subject_index * transFuncData.sequenceFileProperties.maxSequenceLength,
    //                                 numUncorrectedPositions,
    //                                 task.uncorrectedPositionsNoConsensus.begin());

    //                 }

    //                 auto isValidSequence = [](const std::string& s){
    //                     return std::all_of(s.begin(), s.end(), [](char c){
    //                         return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
    //                     });
    //                 };

    //                 if(!isValidSequence(task.corrected_subject)){
    //                     std::cout << task.corrected_subject << std::endl;
    //                 }

    //                 if(task.highQualityAlignment){
    //                     transFuncData.correctionStatusFlagsPerRead[task.readId] |= readCorrectedAsHQAnchor;
    //                 }
    //                 //transFuncData->unlock(task.readId);

    //                 const bool originalReadContainsN = transFuncData.readStorage->readContainsN(task.readId);

    //                 if(!originalReadContainsN){
    //                     const int maxEdits = subject_length / 7;
    //                     int edits = 0;
    //                     for(int i = 0; i < subject_length && edits <= maxEdits; i++){
    //                         if(task.corrected_subject[i] != task.subject_string[i]){
    //                             task.anchoroutput.edits.emplace_back(i, task.corrected_subject[i]);
    //                             edits++;
    //                         }
    //                     }
    //                     task.anchoroutput.useEdits = edits <= maxEdits;
    //                 }else{
    //                     task.anchoroutput.useEdits = false;
    //                 }

    //                 task.anchoroutput.hq = task.highQualityAlignment;                    
    //                 task.anchoroutput.type = TempCorrectedSequence::Type::Anchor;
    //                 task.anchoroutput.readId = task.readId;
    //                 task.anchoroutput.sequence = std::move(task.corrected_subject);
    //                 task.anchoroutput.uncorrectedPositionsNoConsensus = std::move(task.uncorrectedPositionsNoConsensus);
    //                 task.encodedAnchoroutput = task.anchoroutput.encode();

    //             }else{

    //                 transFuncData.correctionStatusFlagsPerRead[task.readId] |= readCouldNotBeCorrectedAsAnchor;

    //             }
    //         }

    //         nvtx::pop_range();
    //     };

    //     auto unpackcandidates = [batchptr](int begin, int end){
    //         nvtx::push_range("candidate unpacking", 3);
    //         Batch& batch = *batchptr;
    //         DataArrays& dataArrays = batch.dataArrays;
    //         const auto& transFuncData = *batch.transFuncData;

    //         //std::cerr << "in unpackcandidates " << begin << " - " << end << "\n";

    //         for(int subject_index = begin; subject_index < end; ++subject_index) {
    //             auto& task = batch.tasks[subject_index];

    //             const int n_corrected_candidates = dataArrays.h_num_corrected_candidates[subject_index];
    //             const char* const my_corrected_candidates_data = dataArrays.h_corrected_candidates
    //                                             + dataArrays.h_indices_per_subject_prefixsum[subject_index] * batch.decodedSequencePitchInBytes;
    //             const int* const my_indices_of_corrected_candidates = dataArrays.h_indices_of_corrected_candidates
    //                                             + dataArrays.h_indices_per_subject_prefixsum[subject_index];


    //             task.corrected_candidates_shifts.resize(n_corrected_candidates);
    //             task.corrected_candidates_read_ids.resize(n_corrected_candidates);
    //             task.corrected_candidates.resize(n_corrected_candidates);
    //             task.corrected_candidate_equals_uncorrected.resize(n_corrected_candidates);
    //             task.candidatesoutput.reserve(n_corrected_candidates);
    //             task.encodedCandidatesoutput.reserve(n_corrected_candidates);

    //             // if(task.readId == 10){
    //             //     for(int i = 0; i < n_corrected_candidates; ++i) {
    //             //         std::cerr << my_indices_of_corrected_candidates[i] << " ";
    //             //     }
    //             //     std::cerr << std::endl;
    //             // }

    //             for(int i = 0; i < n_corrected_candidates; ++i) {
    //                 const int global_candidate_index = my_indices_of_corrected_candidates[i];

    //                 const read_number candidate_read_id = dataArrays.h_candidate_read_ids[global_candidate_index];

    //                 bool savingIsOk = false;
    //                 const std::uint8_t mask = transFuncData.correctionStatusFlagsPerRead[candidate_read_id];
    //                 if(!(mask & readCorrectedAsHQAnchor)) {
    //                     savingIsOk = true;
    //                 }
    //                 if (savingIsOk) {

    //                     const int candidate_length = dataArrays.h_candidate_sequences_lengths[global_candidate_index];
    //                     const int candidate_shift = dataArrays.h_alignment_shifts[global_candidate_index];

    //                     const char* const candidate_data = my_corrected_candidates_data + i * batch.decodedSequencePitchInBytes;
    //                     if(transFuncData.correctionOptions.new_columns_to_correct < candidate_shift){
    //                         std::cerr << "\n" << "readid " << task.readId << " candidate readid " << candidate_read_id << " : "
    //                                 << candidate_shift << " " << transFuncData.correctionOptions.new_columns_to_correct <<"\n";
    //                     }
    //                     assert(transFuncData.correctionOptions.new_columns_to_correct >= candidate_shift);
    //                     task.corrected_candidates_shifts[i] = candidate_shift;
    //                     task.corrected_candidates_read_ids[i] = candidate_read_id;
    //                     task.corrected_candidates[i] = std::move(std::string{candidate_data, candidate_data + candidate_length});

    //                     const bool originalReadContainsN = transFuncData.readStorage->readContainsN(candidate_read_id);

    //                     // if(!originalReadContainsN){
                            
    //                     //     task.corrected_candidate_equals_uncorrected[i] = task.corrected_candidates[i] == uncorrectedCandidate;
    //                     // }else{
    //                     //     task.corrected_candidate_equals_uncorrected[i] = false;
    //                     // }

                       

    //                     TempCorrectedSequence tmp;

    //                     if(!originalReadContainsN){
    //                         const unsigned int* ptr = &dataArrays.h_candidate_sequences_data[global_candidate_index * batch.encodedSequencePitchInInts];
    //                         const std::string uncorrectedCandidate = get2BitString((const unsigned int*)ptr, candidate_length);

    //                         const int maxEdits = candidate_length / 7;
    //                         int edits = 0;
    //                         for(int pos = 0; pos < candidate_length && edits <= maxEdits; pos++){
    //                             if(task.corrected_candidates[i][pos] != uncorrectedCandidate[pos]){
    //                                 tmp.edits.emplace_back(pos, task.corrected_candidates[i][pos]);
    //                                 edits++;
    //                             }
    //                         }

    //                         tmp.useEdits = edits <= maxEdits;
    //                     }else{
    //                         tmp.useEdits = false;
    //                     }
                        
    //                     tmp.type = TempCorrectedSequence::Type::Candidate;
    //                     tmp.shift = task.corrected_candidates_shifts[i];
    //                     tmp.readId = candidate_read_id;
    //                     tmp.sequence = std::move(task.corrected_candidates[i]);

    //                     task.encodedCandidatesoutput.emplace_back(tmp.encode());
    //                     task.candidatesoutput.emplace_back(std::move(tmp));
    // 				}
    //             }
    //         }

    //         nvtx::pop_range();
    //     };


    //     if(!transFuncData.correctionOptions.correctCandidates){
    //         batch.threadPool->parallelFor(batch.pforHandle, 0, int(batch.tasks.size()), [=](auto begin, auto end, auto /*threadId*/){
    //             unpackAnchors(begin, end);
    //         });
    //     }else{
    //         batch.threadPool->parallelFor(batch.pforHandle, 0, int(batch.tasks.size()), [=](auto begin, auto end, auto /*threadId*/){
    //             unpackAnchors(begin, end);
    //             unpackcandidates(begin, end);
    //         });
    //     }

    // }








    void unpackClassicResultsContiguousVersion(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;

        cudaSetDevice(batch.deviceId); CUERR;

        auto& events = batch.events;
        auto& streams = batch.streams;

        cudaError_t errort = cudaEventQuery(events[correction_finished_event_index]);
        if(errort != cudaSuccess){
            std::cout << "error cudaEventQuery\n";
            std::exit(0);
        }
        assert(cudaEventQuery(events[correction_finished_event_index]) == cudaSuccess); CUERR;
        assert(cudaEventQuery(events[result_transfer_finished_event_index]) == cudaSuccess); CUERR;


        assert(transFuncData.correctionOptions.correctionType == CorrectionType::Classic);


        auto& outputData = batch.waitableOutputData.data;
        auto& dataArrays = batch.dataArrays;

        Batch* batchptr = &batch;

        auto& subjectIndicesToProcess = outputData.subjectIndicesToProcess;
        auto& candidateIndicesToProcess = outputData.candidateIndicesToProcess;

        subjectIndicesToProcess.clear();
        candidateIndicesToProcess.clear();

        subjectIndicesToProcess.reserve(batch.tasks.size());
        candidateIndicesToProcess.reserve(16 * batch.tasks.size());

#if 1
        for(int subject_index = 0; subject_index < int(batch.tasks.size()); subject_index++){
            const read_number readId = dataArrays.h_subject_read_ids[subject_index];
            const bool isCorrected = dataArrays.h_subject_is_corrected[subject_index];
            const bool isHighQualityAlignment = dataArrays.h_is_high_quality_subject[subject_index].hq();

            //assert(batch.tasks[subject_index].readId == readId);

            if(isHighQualityAlignment){
                transFuncData.correctionStatusFlagsPerRead[readId] |= readCorrectedAsHQAnchor;
            }

            if(isCorrected){
                subjectIndicesToProcess.emplace_back(subject_index);
            }else{
                transFuncData.correctionStatusFlagsPerRead[readId] |= readCouldNotBeCorrectedAsAnchor;
            }
        }
#else

        for(int subject_index = 0; subject_index < int(batch.tasks.size()); subject_index++){
            auto& task = batch.tasks[subject_index];
            task.corrected = dataArrays.h_subject_is_corrected[subject_index];
            task.highQualityAlignment = dataArrays.h_is_high_quality_subject[subject_index].hq();

            if(task.highQualityAlignment){
                transFuncData.correctionStatusFlagsPerRead[task.readId] |= readCorrectedAsHQAnchor;
            }

            if(task.corrected){
                subjectIndicesToProcess.emplace_back(subject_index);
            }else{
                transFuncData.correctionStatusFlagsPerRead[task.readId] |= readCouldNotBeCorrectedAsAnchor;
            }
        }

#endif
        for(int subject_index = 0; subject_index < int(batch.tasks.size()); subject_index++){

            const int n_corrected_candidates = dataArrays.h_num_corrected_candidates[subject_index];
            const int* const my_indices_of_corrected_candidates = dataArrays.h_indices_of_corrected_candidates
                                                + dataArrays.h_indices_per_subject_prefixsum[subject_index];

            for(int i = 0; i < n_corrected_candidates; ++i) {
                const int global_candidate_index = my_indices_of_corrected_candidates[i];

                const read_number candidate_read_id = dataArrays.h_candidate_read_ids[global_candidate_index];

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

        outputData.anchorCorrections.clear();
        outputData.encodedAnchorCorrections.clear();
        outputData.candidateCorrections.clear();
        outputData.encodedCandidateCorrections.clear();

        outputData.anchorCorrections.resize(numCorrectedAnchors);
        outputData.encodedAnchorCorrections.resize(numCorrectedAnchors);
        outputData.candidateCorrections.resize(numCorrectedCandidates);
        outputData.encodedCandidateCorrections.resize(numCorrectedCandidates);

        #if 1
        auto unpackAnchors = [batchptr](int begin, int end){
            nvtx::push_range("Anchor unpacking", 3);
            Batch& batch = *batchptr;
            auto& outputData = batch.waitableOutputData.data;
            DataArrays& dataArrays = batch.dataArrays;
            const auto& transFuncData = *batch.transFuncData;
            const auto& subjectIndicesToProcess = outputData.subjectIndicesToProcess;
            

            //std::cerr << "in unpackAnchors " << begin << " - " << end << "\n";

            for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                const int subject_index = subjectIndicesToProcess[positionInVector];
                auto& task = batch.tasks[subject_index];

                const read_number readId = dataArrays.h_subject_read_ids[subject_index];
                //assert(readId == task.readId);

                auto& tmp = outputData.anchorCorrections[positionInVector];
                auto& tmpencoded = outputData.encodedAnchorCorrections[positionInVector];


                const char* const my_corrected_subject_data = dataArrays.h_corrected_subjects + subject_index * batch.decodedSequencePitchInBytes;
                const int subject_length = dataArrays.h_subject_sequences_lengths[subject_index];

                std::string correctedSubjectString = std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length};
                const auto& originalSubjectString = batch.decodedSubjectStrings[subject_index];

                // if(task.subject_string != originalSubjectString){
                //     std::cerr << "\task.subject_string " << task.subject_string << " : originalSubjectString " << originalSubjectString << "\n";
                // }
                // assert(task.subject_string == originalSubjectString);


                const int numUncorrectedPositions = dataArrays.h_num_uncorrected_positions_per_subject[subject_index];
                if(numUncorrectedPositions > 0){
                    tmp.uncorrectedPositionsNoConsensus.resize(numUncorrectedPositions);
                    std::copy_n(
                        dataArrays.h_uncorrected_positions_per_subject 
                            + subject_index * transFuncData.sequenceFileProperties.maxSequenceLength,
                        numUncorrectedPositions,
                        tmp.uncorrectedPositionsNoConsensus.begin()
                    );
                }

                auto isValidSequence = [](const std::string& s){
                    return std::all_of(
                        s.begin(), 
                        s.end(), 
                        [](char c){
                            return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
                        }
                    );
                };

                if(!isValidSequence(correctedSubjectString)){
                    std::cerr << correctedSubjectString << '\n';
                }

                const bool originalReadContainsN = transFuncData.readStorage->readContainsN(readId);

                if(!originalReadContainsN){
                    const int maxEdits = subject_length / 7;
                    int edits = 0;
                    for(int i = 0; i < subject_length && edits <= maxEdits; i++){
                        if(correctedSubjectString[i] != originalSubjectString[i]){
                            tmp.edits.emplace_back(i, correctedSubjectString[i]);
                            edits++;
                        }
                    }
                    tmp.useEdits = edits <= maxEdits;
                }else{
                    tmp.useEdits = false;
                }

                const bool isHighQualityAlignment = dataArrays.h_is_high_quality_subject[subject_index].hq();

                tmp.hq = isHighQualityAlignment;                    
                tmp.type = TempCorrectedSequence::Type::Anchor;
                tmp.readId = readId;
                tmp.sequence = std::move(correctedSubjectString);

                tmpencoded = tmp.encode();
            }

            nvtx::pop_range();
        };
#else 

        auto unpackAnchors = [batchptr](int begin, int end){
            nvtx::push_range("Anchor unpacking", 3);
            Batch& batch = *batchptr;
            auto& outputData = batch.waitableOutputData.data;
            DataArrays& dataArrays = batch.dataArrays;
            const auto& transFuncData = *batch.transFuncData;
            const auto& subjectIndicesToProcess = outputData.subjectIndicesToProcess;
            

            //std::cerr << "in unpackAnchors " << begin << " - " << end << "\n";

            for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                const int subject_index = subjectIndicesToProcess[positionInVector];

                auto& task = batch.tasks[subject_index];
                const char* const my_corrected_subject_data = dataArrays.h_corrected_subjects + subject_index * batch.decodedSequencePitchInBytes;


                const int subject_length = dataArrays.h_subject_sequences_lengths[subject_index];
                task.corrected_subject = std::move(std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length});

                //task.correctionEqualsOriginal = task.corrected_subject == task.subject_string;

                const int numUncorrectedPositions = dataArrays.h_num_uncorrected_positions_per_subject[subject_index];
                if(numUncorrectedPositions > 0){
                    task.uncorrectedPositionsNoConsensus.resize(numUncorrectedPositions);
                    std::copy_n(dataArrays.h_uncorrected_positions_per_subject + subject_index * transFuncData.sequenceFileProperties.maxSequenceLength,
                                numUncorrectedPositions,
                                task.uncorrectedPositionsNoConsensus.begin());

                }

                auto isValidSequence = [](const std::string& s){
                    return std::all_of(s.begin(), s.end(), [](char c){
                        return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
                    });
                };

                if(!isValidSequence(task.corrected_subject)){
                    std::cout << task.corrected_subject << std::endl;
                }

                const bool originalReadContainsN = transFuncData.readStorage->readContainsN(task.readId);

                auto& tmp = outputData.anchorCorrections[positionInVector];
                auto& tmpencoded = outputData.encodedAnchorCorrections[positionInVector];

                if(!originalReadContainsN){
                    const int maxEdits = subject_length / 7;
                    int edits = 0;
                    for(int i = 0; i < subject_length && edits <= maxEdits; i++){
                        if(task.corrected_subject[i] != task.subject_string[i]){
                            tmp.edits.emplace_back(i, task.corrected_subject[i]);
                            edits++;
                        }
                    }
                    tmp.useEdits = edits <= maxEdits;
                }else{
                    tmp.useEdits = false;
                }

                tmp.hq = task.highQualityAlignment;                    
                tmp.type = TempCorrectedSequence::Type::Anchor;
                tmp.readId = task.readId;
                tmp.sequence = std::move(task.corrected_subject);
                tmp.uncorrectedPositionsNoConsensus = std::move(task.uncorrectedPositionsNoConsensus);

                tmpencoded = tmp.encode();
            }

            nvtx::pop_range();
        };

#endif
        auto unpackcandidates = [batchptr](int begin, int end){
            nvtx::push_range("candidate unpacking" + std::to_string(end-begin), 3);
            Batch& batch = *batchptr;
            auto& outputData = batch.waitableOutputData.data;
            DataArrays& dataArrays = batch.dataArrays;
            const auto& transFuncData = *batch.transFuncData;
            const auto& subjectIndicesToProcess = outputData.subjectIndicesToProcess;
            const auto& candidateIndicesToProcess = outputData.candidateIndicesToProcess;

            //std::cerr << "in unpackcandidates " << begin << " - " << end << "\n";

            for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                const int subject_index = candidateIndicesToProcess[positionInVector].first;
                const int candidateIndex = candidateIndicesToProcess[positionInVector].second;

                auto& task = batch.tasks[subject_index];

                const size_t offset = dataArrays.h_indices_per_subject_prefixsum[subject_index];

                const char* const my_corrected_candidates_data = dataArrays.h_corrected_candidates
                                                + offset * batch.decodedSequencePitchInBytes;
                const int* const my_indices_of_corrected_candidates = dataArrays.h_indices_of_corrected_candidates
                                                + offset;

                auto& tmp = outputData.candidateCorrections[positionInVector];
                auto& tmpencoded = outputData.encodedCandidateCorrections[positionInVector];


                const int global_candidate_index = my_indices_of_corrected_candidates[candidateIndex];

                const read_number candidate_read_id = dataArrays.h_candidate_read_ids[global_candidate_index];

                const int candidate_length = dataArrays.h_candidate_sequences_lengths[global_candidate_index];
                const int candidate_shift = dataArrays.h_alignment_shifts[global_candidate_index];

                const char* const candidate_data = my_corrected_candidates_data + candidateIndex * batch.decodedSequencePitchInBytes;

                if(transFuncData.correctionOptions.new_columns_to_correct < candidate_shift){
                    std::cerr << "\n" << "readid " << task.readId << " candidate readid " << candidate_read_id << " : "
                            << candidate_shift << " " << transFuncData.correctionOptions.new_columns_to_correct <<"\n";
                }
                assert(transFuncData.correctionOptions.new_columns_to_correct >= candidate_shift);

                auto correctedCandidateString = std::string{candidate_data, candidate_data + candidate_length};

                const bool originalReadContainsN = transFuncData.readStorage->readContainsN(candidate_read_id);
                

                if(!originalReadContainsN){
                    const unsigned int* ptr = &dataArrays.h_candidate_sequences_data[global_candidate_index * batch.encodedSequencePitchInInts];
                    const std::string uncorrectedCandidate = get2BitString((const unsigned int*)ptr, candidate_length);

                    const int maxEdits = candidate_length / 7;
                    int edits = 0;
                    for(int pos = 0; pos < candidate_length && edits <= maxEdits; pos++){
                        if(correctedCandidateString[pos] != uncorrectedCandidate[pos]){
                            tmp.edits.emplace_back(pos, correctedCandidateString[pos]);
                            edits++;
                        }
                    }

                    tmp.useEdits = edits <= maxEdits;
                }else{
                    tmp.useEdits = false;
                }
                
                tmp.type = TempCorrectedSequence::Type::Candidate;
                tmp.shift = candidate_shift;
                tmp.readId = candidate_read_id;
                tmp.sequence = std::move(correctedCandidateString);

                tmpencoded = tmp.encode();
            }

            nvtx::pop_range();
        };


        if(!transFuncData.correctionOptions.correctCandidates){
            batch.threadPool->parallelFor(batch.pforHandle, 0, numCorrectedAnchors, [=](auto begin, auto end, auto /*threadId*/){
                unpackAnchors(begin, end);
            });
        }else{
            batch.threadPool->parallelFor(batch.pforHandle, 0, numCorrectedAnchors, [=](auto begin, auto end, auto /*threadId*/){
                unpackAnchors(begin, end);
            });
            batch.threadPool->parallelFor(batch.pforHandle, 0, numCorrectedCandidates, [=](auto begin, auto end, auto /*threadId*/){
                unpackcandidates(begin, end);
            });
        }

    }





	// void saveResults(Batch& batch){

    //     const auto& transFuncData = *batch.transFuncData;

    //     cudaSetDevice(batch.deviceId); CUERR;

    //     /*DataArrays& dataArrays = batch.dataArrays;
	// 	std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
	// 	std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

    //     for(size_t subjectIndex = 0; subjectIndex < batch.tasks.size(); subjectIndex++){
    //         const auto& task = batch.tasks[subjectIndex];
    //         if(task.readId == 207){

    //             cudaDeviceSynchronize(); CUERR;

    //             cudaMemcpyAsync(dataArrays.h_consensus,
    //                             dataArrays.d_consensus,
    //                             dataArrays.d_consensus.sizeInBytes(),
    //                             D2H,
    //                             streams[primary_stream_index]);

    //             cudaMemcpyAsync(dataArrays.h_support,
    //                             dataArrays.d_support,
    //                             dataArrays.d_support.sizeInBytes(),
    //                             D2H,
    //                             streams[primary_stream_index]);

    //             cudaMemcpyAsync(dataArrays.h_coverage,
    //                             dataArrays.d_coverage,
    //                             dataArrays.d_coverage.sizeInBytes(),
    //                             D2H,
    //                             streams[primary_stream_index]);

    //             cudaMemcpyAsync(dataArrays.h_origWeights,
    //                             dataArrays.d_origWeights,
    //                             dataArrays.d_origWeights.sizeInBytes(),
    //                             D2H,
    //                             streams[primary_stream_index]);

    //             cudaMemcpyAsync(dataArrays.h_msa_column_properties,
    //                             dataArrays.d_msa_column_properties,
    //                             dataArrays.d_msa_column_properties.sizeInBytes(),
    //                             D2H,
    //                             streams[primary_stream_index]);

    //             cudaMemcpyAsync(dataArrays.h_counts,
    //                             dataArrays.d_counts,
    //                             dataArrays.d_counts.sizeInBytes(),
    //                             D2H,
    //                             streams[primary_stream_index]);

    //             cudaMemcpyAsync(dataArrays.h_weights,
    //                             dataArrays.d_weights,
    //                             dataArrays.d_weights.sizeInBytes(),
    //                             D2H,
    //                             streams[primary_stream_index]);

    //             cudaMemcpyAsync(dataArrays.msa_data_host,
    //                         dataArrays.msa_data_device,
    //                         dataArrays.msa_data_usable_size,
    //                         D2H,
    //                         streams[primary_stream_index]); CUERR;
    //             cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

    //             //DEBUGGING
    //             cudaMemcpyAsync(dataArrays.alignment_result_data_host,
    //                         dataArrays.alignment_result_data_device,
    //                         dataArrays.alignment_result_data_usable_size,
    //                         D2H,
    //                         streams[primary_stream_index]); CUERR;
    //             cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

    //             //DEBUGGING
    //             cudaMemcpyAsync(dataArrays.subject_indices_data_host,
    //                         dataArrays.subject_indices_data_device,
    //                         dataArrays.subject_indices_data_usable_size,
    //                         D2H,
    //                         streams[primary_stream_index]); CUERR;
    //             cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

    //             cudaMemcpyAsync(dataArrays.indices_transfer_data_host,
    //                         dataArrays.indices_transfer_data_device,
    //                         dataArrays.indices_transfer_data_usable_size,
    //                         D2H,
    //                         streams[primary_stream_index]); CUERR;
    //             cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

    //             cudaMemcpyAsync(dataArrays.qualities_transfer_data_host,
    //                             dataArrays.qualities_transfer_data_device,
    //                             dataArrays.qualities_transfer_data_usable_size,
    //                             D2H,
    //                             streams[primary_stream_index]); CUERR;
    //             cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

    //             cudaDeviceSynchronize(); CUERR;

    //             dataArrays.printActiveDataOfSubject(subjectIndex, std::cerr);

    //         }
    //     }*/

    //     auto function = [tasks = std::move(batch.tasks),
    //                      transFuncData = &transFuncData,
    //                      id = batch.id](){
    //         nvtx::push_range("batch "+std::to_string(id)+" writeresultoutputhread", 4);
    //         int notCorrectedNoCandidates = 0;
    //         int notCorrected = 0;

    //         //write result to file
    // 		for(std::size_t subject_index = 0; subject_index < tasks.size(); ++subject_index) {

    // 			const auto& task = tasks[subject_index];
    // 			//std::cout << task.readId << "result" << std::endl;

    // 			//std::cout << "finished readId " << task.readId << std::endl;

    // 			if(task.corrected) {
    //                 transFuncData->saveCorrectedSequence(task.anchoroutput, task.encodedAnchoroutput);
    // 			}else{
    //                 if(task.candidate_read_ids.empty()){
    //                     notCorrectedNoCandidates++;
    //                 }

    //                 notCorrected++;

    //             }

    //             for(int i = 0; i < int(task.candidatesoutput.size()); i++){
    //                 transFuncData->saveCorrectedSequence(task.candidatesoutput[i], task.encodedCandidatesoutput[i]);
    //             }
    // 		}

    //         //std::cerr << "not corrected "<< " " << notCorrectedNoCandidates << " " << notCorrected << "/" << tasks.size() << "\n";

    //         nvtx::pop_range();
    //     };

	// 	//function();

    //     nvtx::push_range("enqueue to outputthread", 2);
    //     batch.outputThread->enqueue(std::move(function));
    //     nvtx::pop_range();
    //     //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
    // }
    






    void saveResultsContiguous(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;

        cudaSetDevice(batch.deviceId); CUERR;

        auto function = [outputData = std::move(batch.waitableOutputData.data),
                         transFuncData = &transFuncData,
                         id = batch.id](){
            nvtx::push_range("batch "+std::to_string(id)+" writeresultoutputhread", 4);

            const int numA = outputData.anchorCorrections.size();
            const int numC = outputData.candidateCorrections.size();

            for(int i = 0; i < numA; i++){
                transFuncData->saveCorrectedSequence(
                    outputData.anchorCorrections[i], 
                    outputData.encodedAnchorCorrections[i]
                );
            }

            for(int i = 0; i < numC; i++){
                transFuncData->saveCorrectedSequence(
                    outputData.candidateCorrections[i], 
                    outputData.encodedCandidateCorrections[i]
                );
            }

            nvtx::pop_range();
        };

		//function();

        nvtx::push_range("enqueue to outputthread", 2);
        batch.outputThread->enqueue(std::move(function));
        nvtx::pop_range();
        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
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

      int oldNumOMPThreads = 1;
      #pragma omp parallel
      {
          #pragma omp single
          oldNumOMPThreads = omp_get_num_threads();
      }

      omp_set_num_threads(runtimeOptions.nCorrectorThreads);
      //omp_set_num_threads(1);

      std::chrono::time_point<std::chrono::system_clock> timepoint_begin = std::chrono::system_clock::now();
      std::chrono::duration<double> runtime = std::chrono::seconds(0);

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

    const std::size_t availableMemory = getAvailableMemoryInKB();
    const std::size_t memoryForPartialResults = availableMemory - (std::size_t(1) << 30);

    auto heapusageOfTCS = [](const auto& x){
        return x.data.capacity();
    };

    MemoryFile<EncodedTempCorrectedSequence> partialResults(memoryForPartialResults, tmpfiles[0], heapusageOfTCS);

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

      Read readInFile;

      TransitionFunctionData transFuncData;

      const int nParallelBatches = runtimeOptions.gpuParallelBatches;
      const int batchsize = correctionOptions.batchsize;

      std::cerr << "Using " << nParallelBatches << " batches of size " << batchsize << " for correction\n";

      std::vector<Batch> batches(nParallelBatches);
      BackgroundThread outputThread;
      std::vector<BackgroundThread> backgroundWorkers(nParallelBatches);
      const int threadPoolSize = std::max(1, runtimeOptions.threads - nParallelBatches);
      std::cerr << "threadpool size for correction = " << threadPoolSize << "\n";
      ThreadPool threadPool(threadPoolSize);

      int deviceIdIndex = 0;

      for(int i = 0; i < nParallelBatches; i++) {
          const int deviceId = deviceIds[deviceIdIndex];

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

          batches[i].id = i;
          batches[i].deviceId = deviceId;
          batches[i].dataArrays = std::move(dataArrays);
          batches[i].streams = std::move(streams);
          batches[i].events = std::move(events);
          batches[i].kernelLaunchHandle = make_kernel_launch_handle(deviceId);
          batches[i].subjectSequenceGatherHandle2 = readStorage.makeGatherHandleSequences();
          batches[i].candidateSequenceGatherHandle2 = readStorage.makeGatherHandleSequences();
          //batches[i].subjectLengthGatherHandle2 = readStorage.makeGatherHandleLengths();
          //batches[i].candidateLengthGatherHandle2 = readStorage.makeGatherHandleLengths();
          batches[i].subjectQualitiesGatherHandle2 = readStorage.makeGatherHandleQualities();
          batches[i].candidateQualitiesGatherHandle2 = readStorage.makeGatherHandleQualities();
          batches[i].transFuncData = &transFuncData;
          batches[i].outputThread = &outputThread;
          batches[i].backgroundWorker = &backgroundWorkers[i];
          batches[i].threadPool = &threadPool;
          batches[i].threadsInThreadPool = threadPoolSize;
          batches[i].minhashHandles.resize(threadPoolSize);
          batches[i].encodedSequencePitchInInts = getEncodedNumInts2Bit(sequenceFileProperties.maxSequenceLength);
          batches[i].decodedSequencePitchInBytes = SDIV(sequenceFileProperties.maxSequenceLength, 4) * 4;
          batches[i].qualityPitchInBytes = SDIV(sequenceFileProperties.maxSequenceLength, 32) * 32;
          
          initNextIterationData(batches[i].nextIterationData, batches[i].deviceId);

          deviceIdIndex = (deviceIdIndex + 1) % deviceIds.size();
      }

#ifndef DO_PROFILE
        cpu::RangeGenerator<read_number> readIdGenerator(sequenceFileProperties.nReads);
        //cpu::RangeGenerator<read_number> readIdGenerator(1000000);
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

      transFuncData.saveCorrectedSequence = [&](const TempCorrectedSequence& tmp, EncodedTempCorrectedSequence encoded){
          //std::unique_lock<std::mutex> l(outputstreammutex);
          if(!(tmp.hq && tmp.useEdits && tmp.edits.empty())){
              //outputstream << tmp << '\n';
              partialResults.storeElement(std::move(encoded));
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


#if 0

        for(auto& w : backgroundWorkers){
            w.start();
        }

        for(int i = 0; i < nParallelBatches; ++i) {
            batchExecutors.emplace_back([&,i](){
                auto& batchData = batches[i];
                auto& streams = batchData.streams;
                auto& events = batchData.events;

                auto pushrange = [&](const std::string& msg, int color){
                    nvtx::push_range("batch "+std::to_string(batchData.id)+msg, color);
                };

                auto poprange = [&](){
                    nvtx::pop_range();
                };


                while(!(readIdGenerator.empty() 
                        && batchData.nextIterationData.isDone()
                        && batchData.nextIterationData.tasks.empty())) {
                        
                    batchData.reset();

                    pushrange("getNextBatchOfSubjectsAndDetermineCandidateReadIds", 0);
                    
                    getNextBatchOfSubjectsAndDetermineCandidateReadIds(batchData);

                    poprange();

                    //cudaDeviceSynchronize(); CUERR;

                    if(batchData.initialNumberOfCandidates == 0){
                        continue;
                    }

                    pushrange("getCandidateSequenceData", 1);

                    getCandidateSequenceData(batchData, *transFuncData.readStorage);

                    poprange();

                    //cudaDeviceSynchronize(); CUERR;


                    pushrange("getCandidateAlignments", 2);

                    getCandidateAlignments(batchData);

                    poprange();

                    //cudaDeviceSynchronize(); CUERR;

                    //cudaDeviceSynchronize(); CUERR;


                    // cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

                    // if(batchData.dataArrays.h_num_indices[0] == 0){
                    //     continue;
                    // }

                    if(transFuncData.correctionOptions.useQualityScores) {
                        pushrange("getQualities", 4);

                        getQualities(batchData);

                        poprange();
                    }

                    

                    //cudaDeviceSynchronize(); CUERR;

                    pushrange("buildMultipleSequenceAlignment", 5);

                    buildMultipleSequenceAlignment(batchData);

                    poprange();

                    //cudaDeviceSynchronize(); CUERR;


                #ifdef USE_MSA_MINIMIZATION

                    pushrange("removeCandidatesOfDifferentRegionFromMSA", 6);

                    removeCandidatesOfDifferentRegionFromMSA(batchData);

                    poprange();

                    //cudaDeviceSynchronize(); CUERR;

                #endif

                    //cudaEventSynchronize(events[indices_transfer_finished_event_index]); CUERR;

                //cudaDeviceSynchronize(); CUERR;

                    pushrange("correctSubjects", 7);

                    correctSubjects(batchData);

                    poprange();

                    //cudaDeviceSynchronize(); CUERR;


                    if(transFuncData.correctionOptions.correctCandidates) {                        

                        pushrange("correctCandidates", 8);

                        correctCandidates(batchData);

                        poprange();
                    }

                    cudaEventRecord(events[secondary_stream_finished_event_index], streams[secondary_stream_index]); CUERR;
                    cudaStreamWaitEvent(streams[primary_stream_index], events[secondary_stream_finished_event_index], 0); CUERR;            
                    cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

                    //cudaDeviceSynchronize(); CUERR;

                    pushrange("unpackClassicResults", 9);

                    unpackClassicResults(batchData);

                    poprange();

                    //cudaDeviceSynchronize(); CUERR;


                    pushrange("saveResults", 10);

                    saveResults(batchData);

                    poprange();
                    
                }

                batchData.isTerminated = true;
                batchData.state = BatchState::Finished;
            });
        }

        while(
            !(std::all_of(batches.begin(), batches.end(), [](const auto& batch){
                return batch.state == BatchState::Finished;
            }) && readIdGenerator.empty())) {


            const auto now = std::chrono::system_clock::now();
            runtime = now - timepoint_begin;

            #ifndef DO_PROFILE
            if(runtimeOptions.showProgress/* && readIdGenerator.getCurrentUnsafe() - previousprocessedreads > 100000*/){
                printf("Processed %10u of %10lu reads (Runtime: %03d:%02d:%02d)\r",
                    readIdGenerator.getCurrentUnsafe() - readIdGenerator.getBegin(), sequenceFileProperties.nReads,
                    int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
                    int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count()) % 60,
                    int(runtime.count()) % 60);
                //previousprocessedreads = readIdGenerator.getCurrentUnsafe();
            }
            #endif

            std::this_thread::sleep_for(std::chrono::seconds{1});
        }


        for(auto& w : backgroundWorkers){
            w.stopThread(BackgroundThread::StopType::FinishAndStop);
        }

        for(auto& thread : batchExecutors){
            thread.join();
        }
        
        threadPool.wait();

        outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

        runtime = std::chrono::system_clock::now() - timepoint_begin;
        if(runtimeOptions.showProgress){
            printf("Processed %10lu of %10lu reads (Runtime: %03d:%02d:%02d)\n",
                    sequenceFileProperties.nReads, sequenceFileProperties.nReads,
                    int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
                    int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count()) % 60,
                    int(runtime.count()) % 60);
        }

#else 


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
            batchData.subjectSequenceGatherHandle2 = readStorage.makeGatherHandleSequences();
            batchData.candidateSequenceGatherHandle2 = readStorage.makeGatherHandleSequences();
            //batch.subjectLengthGatherHandle2 = readStorage.makeGatherHandleLengths();
            //batch.candidateLengthGatherHandle2 = readStorage.makeGatherHandleLengths();
            batchData.subjectQualitiesGatherHandle2 = readStorage.makeGatherHandleQualities();
            batchData.candidateQualitiesGatherHandle2 = readStorage.makeGatherHandleQualities();
            batchData.transFuncData = &transFuncData;
            batchData.outputThread = &outputThread;
            batchData.backgroundWorker = nullptr;//&backgroundWorkers[i];
            batchData.threadPool = &threadPool;
            batchData.threadsInThreadPool = threadPoolSize;
            batchData.minhashHandles.resize(threadPoolSize);
            batchData.encodedSequencePitchInInts = getEncodedNumInts2Bit(sequenceFileProperties.maxSequenceLength);
            batchData.decodedSequencePitchInBytes = SDIV(sequenceFileProperties.maxSequenceLength, 4) * 4;
            batchData.qualityPitchInBytes = SDIV(sequenceFileProperties.maxSequenceLength, 32) * 32;
            
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

            //std::cerr << "processBatchUntilResultTransferIsInitiated batch " << &batchData << " id " << batchData.id << "\n";
            
            batchData.reuseFlag.wait();
            assert(batchData.reuseFlag.done == true);

            //std::cerr << "processBatchUntilResultTransferIsInitiated batch " << &batchData << " id " << batchData.id << " waited for reuseFlag\n";
                
            pushrange("getNextBatchOfSubjectsAndDetermineCandidateReadIds", 0);
            
            getNextBatchOfSubjectsAndDetermineCandidateReadIds(batchData);

            poprange();

            if(batchData.initialNumberOfCandidates == 0){
                return;
            }

            pushrange("getCandidateSequenceData", 1);

            getCandidateSequenceData(batchData, *transFuncData.readStorage);

            poprange();


            pushrange("getCandidateAlignments", 2);

            getCandidateAlignments(batchData);

            poprange();

            if(transFuncData.correctionOptions.useQualityScores) {
                pushrange("getQualities", 4);

                getQualities(batchData);

                poprange();
            }

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

            batchData.hasUnprocessedResults = true;

            //std::cerr << "processBatchUntilResultTransferIsInitiated batch " << &batchData << " id " << batchData.id << " finished\n";
        };

        auto processBatchResults = [&](auto& batchData){
            auto& streams = batchData.streams;
            auto& events = batchData.events;

            auto pushrange = [&](const std::string& msg, int color){
                nvtx::push_range("batch "+std::to_string(batchData.id)+msg, color);
            };

            auto poprange = [&](){
                nvtx::pop_range();
            };

            cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

#if 1
            pushrange("unpackClassicResults", 9);

            //unpackClassicResults(batchData);
            unpackClassicResultsContiguousVersion(batchData);

            poprange();


            pushrange("saveResults", 10);

            //saveResults(batchData);
            saveResultsContiguous(batchData);

            poprange();
#else 
            pushrange("submit to outputthread", 9);
            processResultsInOutputThread(batchData);
            poprange();
#endif
            batchData.hasUnprocessedResults = false;

    //         auto func = [batchDataPtr = &batchData](){
    //             //std::cerr << "backgroundWorker output batch " << batchDataPtr << " id " << batchDataPtr->id << "\n";
    //             auto& batchData = *batchDataPtr;
    //             auto& streams = batchData.streams;
    //             auto& events = batchData.events;

    //             auto pushrange = [&](const std::string& msg, int color){
    //                 nvtx::push_range("batch "+std::to_string(batchData.id)+msg, color);
    //             };

    //             auto poprange = [&](){
    //                 nvtx::pop_range();
    //             };

    //             cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

    // #if 1
    //             pushrange("unpackClassicResults", 9);

    //             unpackClassicResults(batchData);
    //             //unpackClassicResultsContiguousVersion(batchData);

    //             poprange();


    //             pushrange("saveResults", 10);

    //             saveResults(batchData);
    //             //saveResultsContiguous(batchData);

    //             poprange();
    // #else 
    //             pushrange("submit to outputthread", 9);
    //             processResultsInOutputThread(batchData);
    //             poprange();
    // #endif
    //             batchData.reuseFlag.signal();
    //             batchData.hasUnprocessedResults = false;     
    //             //std::cerr << "backgroundWorker output batch " << batchDataPtr << " id " << batchDataPtr->id << " finished\n";  
                
    //             batchData.reset();
    //         };

    //         batchData.reuseFlag.done = false;

    //         //std::cerr << "enque backgroundWorker output batch " << &batchData << " id " << batchData.id << "\n";
    //         batchData.backgroundWorker->enqueue(func);
            
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
                        && batchDataArray[0].nextIterationData.isDone()
                        && batchDataArray[0].nextIterationData.tasks.empty()
                        && batchDataArray[1].nextIterationData.isDone()
                        && batchDataArray[1].nextIterationData.tasks.empty())) {

                    auto& batchData = batchDataArray[batchIndex];

                    processBatchUntilResultTransferIsInitiated(batchData);

                    if(batchData.initialNumberOfCandidates == 0){
                        batchData.reset();
                        progressThread.addProgress(batchsize);
                        continue;
                    }

                    processBatchResults(batchData);

                    batchData.reset();
                    progressThread.addProgress(batchsize);                    
                }
#else 
                while(!(readIdGenerator.empty() 
                        && batchDataArray[0].nextIterationData.isDone()
                        && batchDataArray[0].nextIterationData.tasks.empty()
                        && !batchDataArray[0].hasUnprocessedResults
                        && batchDataArray[1].nextIterationData.isDone()
                        && batchDataArray[1].nextIterationData.tasks.empty()
                        && !batchDataArray[1].hasUnprocessedResults)) {

                    const int nextBatchIndex = 1 - batchIndex;
                    auto& currentBatchData = batchDataArray[batchIndex];
                    auto& nextBatchData = batchDataArray[nextBatchIndex];

                    if(isFirstIteration){
                        isFirstIteration = false;
//std::cerr << "\nprocessBatchUntilResultTransferIsInitiated batch " << currentBatchData.id << "\n";
                        processBatchUntilResultTransferIsInitiated(currentBatchData);
                    }else{

                        // while(!nextBatchData.waitableOutputData.isDone()){
                        //     std::cerr << "nextBatchIndex " << nextBatchIndex << "not ready\n";
                        // }
                       // std::cerr << "nextBatchIndex " << nextBatchIndex << " wait\n";
                        nextBatchData.waitableOutputData.wait(); // until outputdata can savely be reused
                       // std::cerr << "nextBatchIndex " << nextBatchIndex << " waited\n";

                       // std::cerr << "\nprocessBatchUntilResultTransferIsInitiated batch " << nextBatchData.id << "\n";
                        processBatchUntilResultTransferIsInitiated(nextBatchData);

                        if(currentBatchData.initialNumberOfCandidates == 0){
                            std::cerr << "ZEEERROOO\n";
                            currentBatchData.waitableOutputData.signal();
                            currentBatchData.reset();
                            progressThread.addProgress(batchsize);
                            batchIndex = 1-batchIndex;
                            continue;
                        }
                        //std::cerr << "\processBatchResults batch " << currentBatchData.id << "\n";
                        processBatchResults(currentBatchData);
    
                        
                        progressThread.addProgress(batchsize);    
                        currentBatchData.reset(); 

                        batchIndex = 1-batchIndex;
                    }                
                }

#endif
                batchDataArray[0].isTerminated = true;
                batchDataArray[0].state = BatchState::Finished;
                batchDataArray[1].isTerminated = true;
                batchDataArray[1].state = BatchState::Finished;

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
#endif

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

      for(auto& batch : batches){
          cudaSetDevice(batch.deviceId); CUERR;

          batch.dataArrays.reset();
          destroyNextIterationData(batch.nextIterationData);

          for(auto& stream : batch.streams) {
              cudaStreamDestroy(stream); CUERR;
          }

          for(auto& event : batch.events){
              cudaEventDestroy(event); CUERR;
          }
      }

      correctionStatusFlagsPerRead.reset();



      omp_set_num_threads(oldNumOMPThreads);


      //size_t occupiedMemory = minhasher.numBytes() + cpuReadStorage.size();

      minhasher.destroy();
      readStorage.destroy();

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
