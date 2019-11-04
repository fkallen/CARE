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
#include <nn_classifier.hpp>
#include <minhasher.hpp>
#include <options.hpp>
#include <candidatedistribution.hpp>
#include <sequencefileio.hpp>
#include <rangegenerator.hpp>
#include <threadpool.hpp>

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

    constexpr int nParallelBatches = 4;
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

    std::string nameOf(const BatchState&);

    struct Batch {

		std::vector<CorrectionTask> tasks;
		int initialNumberOfCandidates = 0;
		BatchState state = BatchState::Unprepared;

		int copiedTasks = 0;         // used if state == CandidatesPresent
		int copiedCandidates = 0;         // used if state == CandidatesPresent

        int copiedSubjects = 0;
        bool handledReadIds = false;

        bool combinedStreams = false;

        int initialNumberOfAnchorIds = 0;

		std::vector<char> collectedCandidateReads;
		int numsortedCandidateIds = 0;
		int numsortedCandidateIdTasks = 0;

		DataArrays dataArrays;

        bool doImproveMSA = false;
        int numMinimizations = 0;
        int previousNumIndices = 0;

		std::array<cudaStream_t, nStreamsPerBatch> streams;
		std::array<cudaEvent_t, nEventsPerBatch> events;

        std::array<std::atomic_int, nBatchStates> waitCounts{};
        int activeWaitIndex = 0;
        //std::vector<std::unique_ptr<WaitCallbackData>> callbackDataList;

        TransitionFunctionData* transFuncData;
        BackgroundThread* executor;
        BackgroundThread* outputThread;

        int id = -1;
        int deviceId = 0;

		KernelLaunchHandle kernelLaunchHandle;

        DistributedReadStorage::GatherHandleSequences subjectSequenceGatherHandle2;
        DistributedReadStorage::GatherHandleSequences candidateSequenceGatherHandle2;
        DistributedReadStorage::GatherHandleLengths subjectLengthGatherHandle2;
        DistributedReadStorage::GatherHandleLengths candidateLengthGatherHandle2;
        DistributedReadStorage::GatherHandleQualities subjectQualitiesGatherHandle2;
        DistributedReadStorage::GatherHandleQualities candidateQualitiesGatherHandle2;

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
    		collectedCandidateReads.clear();

    		findcandidateidsDataFrame.reset();
            unpackclassicresultsDataFrame.reset();

            statesInProgress = 0;

    		state = BatchState::Unprepared;
    		copiedTasks = 0;
    		copiedCandidates = 0;
            copiedSubjects = 0;
            handledReadIds = false;

            combinedStreams = false;

            initialNumberOfAnchorIds = 0;

    		numsortedCandidateIds = 0;
    		numsortedCandidateIdTasks = 0;

            //assert(std::all_of(waitCounts.begin(), waitCounts.end(), [](const auto& i){return i == 0;}));

            activeWaitIndex = 0;

            doImproveMSA = false;
            numMinimizations = 0;
            previousNumIndices = 0;
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
        std::function<void(const TempCorrectedSequence&)> saveCorrectedSequence;
		std::function<void(const read_number)> lock;
		std::function<void(const read_number)> unlock;

        std::condition_variable isFinishedCV;
        std::mutex isFinishedMutex;

        std::function<void(const SerializedFeature&)> saveFeature;

        ForestClassifier fc;// = ForestClassifier{"./forests/testforest.so"};
        NN_Correction_Classifier nnClassifier;
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

    void state_unprepared_func(Batch& batch);
    void state_findcandidateids_func(Batch& batch);
	void state_copyreads_func(Batch& batch);
	void state_startalignment_func(Batch& batch);
    void state_rearrangeindices_func(Batch& batch);
	void state_copyqualities_func(Batch& batch);
	void state_buildmsa_func(Batch& batch);
    void state_improvemsa_func(Batch& batch);
	void state_startclassiccorrection_func(Batch& batch);
	void state_startforestcorrection_func(Batch& batch);
    void state_startconvnetcorrection_func(Batch& batch);
    void state_startclassiccandidatecorrection_func(Batch& batch);
    void state_combinestreams_func(Batch& batch);
	void state_unpackclassicresults_func(Batch& batch);
	void state_writeresults_func(Batch& batch);
	void state_writefeatures_func(Batch& batch);
	void state_finished_func(Batch& batch);

    //void* b is a pointer to Batch
    void CUDART_CB nextStep(void* b){
        Batch* const batch = (Batch*)b;
        auto executorPtr = batch->executor;

        auto call = [=](auto f){
            // if(batch->statesInProgress > 0){
            //     std::cerr << "\nbatch " << batch->id << nameOf(batch->state) << " " << batch->statesInProgress << "\n";
            //     assert(false);
            // }

            batch->statesInProgress++;
            //std::cerr << "batch " << batch->id << " " << nameOf(batch->state) << "\n";
            nvtx::push_range("batch "+std::to_string(batch->id)+nameOf(batch->state), int(batch->state));
            f(*batch);
            nvtx::pop_range();

            batch->statesInProgress--;
        };

        switch(BatchState(batch->state)) {
        case BatchState::Unprepared:
            executorPtr->enqueue([=](){
                call(state_unprepared_func);
            });
            break;
        case BatchState::FindCandidateIds:
            executorPtr->enqueue([=](){
                call(state_findcandidateids_func);
            });
            break;
        case BatchState::CopyReads:
            executorPtr->enqueue([=](){
                call(state_copyreads_func);
            });
            break;
        case BatchState::StartAlignment:
            executorPtr->enqueue([=](){
                call(state_startalignment_func);
            });
            break;
        case BatchState::RearrangeIndices:
            executorPtr->enqueue([=](){
                call(state_rearrangeindices_func);
            });
            break;
        case BatchState::CopyQualities:
            executorPtr->enqueue([=](){
                call(state_copyqualities_func);
            });
            break;
        case BatchState::BuildMSA:
            executorPtr->enqueue([=](){
                call(state_buildmsa_func);
            });
            break;
        case BatchState::ImproveMSA:
            executorPtr->enqueue([=](){
                call(state_improvemsa_func);
            });
            break;
        case BatchState::StartClassicCorrection:
            executorPtr->enqueue([=](){
                call(state_startclassiccorrection_func);
            });
            break;
        case BatchState::StartForestCorrection:
            executorPtr->enqueue([=](){
                call(state_startforestcorrection_func);
            });
            break;
        case BatchState::StartConvnetCorrection:
            executorPtr->enqueue([=](){
                call(state_startconvnetcorrection_func);
            });
            break;
        case BatchState::StartClassicCandidateCorrection:
            executorPtr->enqueue([=](){
                call(state_startclassiccandidatecorrection_func);
            });
            break;
        case BatchState::CombineStreams:
            executorPtr->enqueue([=](){
                call(state_combinestreams_func);
            });
            break;
        case BatchState::UnpackClassicResults:
            executorPtr->enqueue([=](){
                call(state_unpackclassicresults_func);
            });
            break;
        case BatchState::WriteResults:
            executorPtr->enqueue([=](){
                call(state_writeresults_func);
            });
            break;
        case BatchState::WriteFeatures:
            executorPtr->enqueue([=](){
                call(state_writefeatures_func);
            });
            break;
        case BatchState::Finished:
            executorPtr->enqueue([=](){
                call(state_finished_func);
            });
            break;
        default: assert(false);
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
                    const int* h_num_indices,
                    const int* d_num_indices,
                    float expectedAffectedIndicesFraction,
                    bool useQualityScores,
                    float desiredAlignmentMaxErrorRate,
                    int maximum_sequence_length,
                    int maxSequenceBytes,
                    size_t encoded_sequence_pitch,
                    size_t quality_pitch,
                    size_t msa_pitch,
                    size_t msa_weights_pitch,
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
                stream,
                kernelLaunchHandle);
#if 1
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
                    h_num_indices,
                    d_num_indices,
                    expectedAffectedIndicesFraction,
                    useQualityScores,
                    desiredAlignmentMaxErrorRate,
                    maximum_sequence_length,
                    maxSequenceBytes,
                    encoded_sequence_pitch,
                    quality_pitch,
                    msa_pitch,
                    msa_weights_pitch,
                    stream,
                    kernelLaunchHandle,
                    false);
#else

        call_msa_add_sequences_implicit_singlecol_kernel_async(
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
                    useQualityScores,
                    desiredAlignmentMaxErrorRate,
                    maximum_sequence_length,
                    maxSequenceBytes,
                    encoded_sequence_pitch,
                    quality_pitch,
                    msa_weights_pitch,
                    stream,
                    kernelLaunchHandle,
                    nullptr,
                    false);

#endif
        call_msa_find_consensus_implicit_kernel_async(
                    d_msapointers,
                    d_sequencePointers,
                    d_indices_per_subject,
                    n_subjects,
                    encoded_sequence_pitch,
                    msa_pitch,
                    msa_weights_pitch,
                    stream,
                    kernelLaunchHandle);
    };


    void state_unprepared_func(Batch& batch){

        constexpr BatchState expectedState = BatchState::Unprepared;

        assert(batch.state == expectedState);
        assert(batch.initialNumberOfCandidates == 0 && batch.tasks.empty());

        cudaSetDevice(batch.deviceId); CUERR;

        //get next anchor read ids to correct. then begin fetching sequences and length of those anchors

        const auto& transFuncData = *batch.transFuncData;

        DataArrays& dataArrays = batch.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        //std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        const int maximumSequenceBytes = sizeof(unsigned int) * getEncodedNumInts2BitHiLo(transFuncData.sequenceFileProperties.maxSequenceLength);

        const auto batchsize = transFuncData.correctionOptions.batchsize;

        dataArrays.resizeAnchorSequenceData(batchsize, maximumSequenceBytes);

        read_number* const readIdsBegin = dataArrays.h_subject_read_ids.get();
        read_number* const readIdsEnd = transFuncData.readIdGenerator->next_n_into_buffer(batchsize, readIdsBegin);
        batch.initialNumberOfAnchorIds = std::distance(readIdsBegin, readIdsEnd);

        if(batch.initialNumberOfAnchorIds == 0) {
            batch.setState(BatchState::Finished, expectedState);
            cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
            return;
        };

        cudaMemcpyAsync(dataArrays.d_subject_read_ids,
                        dataArrays.h_subject_read_ids,
                        dataArrays.h_subject_read_ids.sizeInBytes(),
                        H2D,
                        streams[primary_stream_index]); CUERR;

        transFuncData.readStorage->gatherSequenceDataToGpuBufferAsync(
                                    batch.subjectSequenceGatherHandle2,
                                    dataArrays.d_subject_sequences_data.get(),
                                    maximumSequenceBytes,
                                    dataArrays.h_subject_read_ids,
                                    dataArrays.d_subject_read_ids,
                                    batch.initialNumberOfAnchorIds,
                                    batch.deviceId,
                                    streams[primary_stream_index],
                                    transFuncData.runtimeOptions.nCorrectorThreads);

        // transFuncData.readStorage->gatherSequenceLengthsToGpuBufferAsync(
        //                             batch.subjectLengthGatherHandle2,
        //                             dataArrays.d_subject_sequences_lengths.get(),
        //                             dataArrays.h_subject_read_ids.get(),
        //                             dataArrays.d_subject_read_ids.get(),
        //                             batch.initialNumberOfAnchorIds,
        //                             batch.deviceId,
        //                             streams[primary_stream_index],
        //                             transFuncData.runtimeOptions.nCorrectorThreads);

        transFuncData.readStorage->gatherSequenceLengthsToGpuBufferAsyncNew(
                                    dataArrays.d_subject_sequences_lengths.get(),
                                    batch.deviceId,
                                    dataArrays.d_subject_read_ids.get(),
                                    batch.initialNumberOfAnchorIds,            
                                    streams[primary_stream_index]);

        cudaMemcpyAsync(dataArrays.h_subject_sequences_data,
                        dataArrays.d_subject_sequences_data,
                        dataArrays.d_subject_sequences_data.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_subject_sequences_lengths,
                        dataArrays.d_subject_sequences_lengths,
                        dataArrays.d_subject_sequences_lengths.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        batch.tasks.resize(batch.initialNumberOfAnchorIds);

        batch.setState(BatchState::FindCandidateIds, expectedState);

        cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
    }



    void state_findcandidateids_func(Batch& batch){

        constexpr BatchState expectedState = BatchState::FindCandidateIds;

        assert(batch.state == expectedState);
        assert(batch.initialNumberOfAnchorIds > 0);

        cudaSetDevice(batch.deviceId); CUERR;

        //for each anchor, get candidates

        Batch* batchptr = &batch;

        auto maketasks = [batchptr](int begin, int end){

            const auto& transFuncData = *(batchptr->transFuncData);

            const int hits_per_candidate = transFuncData.correctionOptions.hits_per_candidate;

            DataArrays& dataArrays = batchptr->dataArrays;

            const auto& minhasher = transFuncData.minhasher;

            const int maximumSequenceBytes = sizeof(unsigned int) * getEncodedNumInts2BitHiLo(transFuncData.sequenceFileProperties.maxSequenceLength);

            int initialNumberOfCandidates = 0;

            for(int i = begin; i < end; i++){
                auto& task = batchptr->tasks[i];

                const read_number readId = batchptr->dataArrays.h_subject_read_ids[i];
                //TIMERSTARTCPU(CorrectionTask);
                task = CorrectionTask(readId);
                //TIMERSTOPCPU(CorrectionTask);

                // bool ok = false;
                // if ((*transFuncData.correctionStatusFlagsPerRead)[readId] == 0) {
                //     ok = true;
                // }
                const bool ok = true;

                if(ok){

                    const char* sequenceptr = dataArrays.h_subject_sequences_data.get() + i * maximumSequenceBytes;
                    const int sequencelength = dataArrays.h_subject_sequences_lengths[i];

                    //TIMERSTARTCPU(get2BitHiLoString);
                    task.subject_string = get2BitHiLoString((const unsigned int*)sequenceptr, sequencelength);
                    //TIMERSTOPCPU(get2BitHiLoString);

                    //TIMERSTARTCPU(getCandidates);
                    task.candidate_read_ids = minhasher->getCandidates(task.subject_string,
                                                                        hits_per_candidate,
                                                                        transFuncData.runtimeOptions.max_candidates);
                    //TIMERSTOPCPU(getCandidates);

                    //TIMERSTARTCPU(lower_bound);
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
                }else{
                    task.active = false;
                }

                const int myNumCandidates = int(task.candidate_read_ids.size());
                initialNumberOfCandidates += myNumCandidates;
            }

            batchptr->findcandidateidsDataFrame.initialNumberOfCandidates += initialNumberOfCandidates;
        };

        auto allChunksFinished = [batchptr](){

            std::unique_lock<std::mutex> l(batchptr->findcandidateidsDataFrame.m);

            batchptr->findcandidateidsDataFrame.cv.wait(l, [batchptr](){
                return batchptr->findcandidateidsDataFrame.finishedChunks == batchptr->findcandidateidsDataFrame.chunksToWaitFor;
            });

            const auto& transFuncData = *(batchptr->transFuncData);

            DataArrays& dataArrays = batchptr->dataArrays;
            std::array<cudaStream_t, nStreamsPerBatch>& streams = batchptr->streams;
            const int initialNumberOfCandidates = batchptr->findcandidateidsDataFrame.initialNumberOfCandidates;

            auto it = std::remove_if(batchptr->tasks.begin(), batchptr->tasks.end(), [](const auto& t){return !t.active;});
            batchptr->tasks.erase(it, batchptr->tasks.end());

            if(initialNumberOfCandidates == 0) {
                batchptr->setState(BatchState::Finished, expectedState);
                cudaLaunchHostFunc(streams[primary_stream_index], nextStep, batchptr); CUERR;
                return;
            }else{

                //assert(batch.initialNumberOfCandidates < transFuncData.correctionOptions.batchsize + transFuncData.runtimeOptions.max_candidates);

                //allocate data arrays
                nvtx::push_range("set_problem_dimensions", 4);
                dataArrays.set_problem_dimensions(int(batchptr->tasks.size()),
                            initialNumberOfCandidates,
                            transFuncData.sequenceFileProperties.maxSequenceLength,
                            sizeof(unsigned int) * getEncodedNumInts2BitHiLo(transFuncData.sequenceFileProperties.maxSequenceLength),
                            transFuncData.goodAlignmentProperties.min_overlap,
                            transFuncData.goodAlignmentProperties.min_overlap_ratio,
                            transFuncData.correctionOptions.useQualityScores); CUERR;
                nvtx::pop_range();

                std::size_t temp_storage_bytes = 0;
                std::size_t max_temp_storage_bytes = 0;
                cub::DeviceHistogram::HistogramRange((void*)nullptr, temp_storage_bytes,
                            (int*)nullptr, (int*)nullptr,
                            dataArrays.n_subjects+1,
                            (int*)nullptr,
                            dataArrays.n_queries,
                            streams[primary_stream_index]); CUERR;

                max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

                cub::DeviceSelect::Flagged((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                            (bool*)nullptr, (int*)nullptr, (int*)nullptr,
                            batchptr->initialNumberOfCandidates,
                            streams[primary_stream_index]); CUERR;

                max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

                cub::DeviceScan::ExclusiveSum((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                            (int*)nullptr,
                            dataArrays.n_subjects,
                            streams[primary_stream_index]); CUERR;

                cub::DeviceScan::InclusiveSum((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                            (int*)nullptr,
                            dataArrays.n_subjects,
                            streams[primary_stream_index]); CUERR;

                cub::DeviceSegmentedRadixSort::SortPairs((void*)nullptr,
                                                        temp_storage_bytes,
                                                        (const char*) nullptr,
                                                        (char*)nullptr,
                                                        (const int*)nullptr,
                                                        (int*)nullptr,
                                                        batchptr->initialNumberOfCandidates,
                                                        dataArrays.n_subjects,
                                                        (const int*)nullptr,
                                                        (const int*)nullptr,
                                                        0,
                                                        3,
                                                        streams[primary_stream_index]);

                max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

                temp_storage_bytes = max_temp_storage_bytes;

                dataArrays.set_cub_temp_storage_size(max_temp_storage_bytes);
                dataArrays.zero_gpu(streams[primary_stream_index]);

                auto roundToNextMultiple = [](int num, int factor){
                    return SDIV(num, factor) * factor;
                };

                batchptr->setState(BatchState::CopyReads, expectedState);
                cudaLaunchHostFunc(streams[primary_stream_index], nextStep, batchptr); CUERR;
                return;
            }
        };

        threadpool.parallelFor(0, batch.initialNumberOfAnchorIds, [=](auto begin, auto end, auto /*threadId*/){
            maketasks(begin, end);
        });

        allChunksFinished();
    }


    void state_copyreads_func(Batch& batch){

        constexpr BatchState expectedState = BatchState::CopyReads;

        assert(batch.state == expectedState);
        assert(batch.copiedTasks <= int(batch.tasks.size()));

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        DataArrays& dataArrays = batch.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;



        const bool handledReadIds = batch.handledReadIds;

        if(!handledReadIds){

            dataArrays.h_candidates_per_subject_prefixsum[0] = 0;
            for(size_t i = 0; i < batch.tasks.size(); i++){
                const size_t num = batch.tasks[i].candidate_read_ids.size();
                dataArrays.h_candidates_per_subject[i] = num;
                dataArrays.h_candidates_per_subject_prefixsum[i+1] = dataArrays.h_candidates_per_subject_prefixsum[i] + num;
            }

            for(size_t i = 0; i < batch.tasks.size(); i++){
                const auto& task = batch.tasks[i];
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

            batch.handledReadIds = true;
        }

        // transFuncData.readStorage->gatherSequenceLengthsToGpuBufferAsync(batch.subjectLengthGatherHandle2,
        //                                                              dataArrays.d_subject_sequences_lengths,
        //                                                              dataArrays.h_subject_read_ids,
        //                                                              dataArrays.d_subject_read_ids,
        //                                                              dataArrays.n_subjects,
        //                                                              batch.deviceId,
        //                                                              streams[primary_stream_index],
        //                                                              transFuncData.runtimeOptions.nCorrectorThreads);

        // transFuncData.readStorage->gatherSequenceLengthsToGpuBufferAsync(batch.candidateLengthGatherHandle2,
        //                                                           dataArrays.d_candidate_sequences_lengths,
        //                                                           dataArrays.h_candidate_read_ids,
        //                                                           dataArrays.d_candidate_read_ids,
        //                                                           dataArrays.n_queries,
        //                                                           batch.deviceId,
        //                                                           streams[primary_stream_index],
        //                                                           transFuncData.runtimeOptions.nCorrectorThreads);

        transFuncData.readStorage->gatherSequenceLengthsToGpuBufferAsyncNew(
                                        dataArrays.d_subject_sequences_lengths.get(),
                                        batch.deviceId,
                                        dataArrays.d_subject_read_ids.get(),
                                        dataArrays.n_subjects,   
                                        streams[primary_stream_index]);

        transFuncData.readStorage->gatherSequenceLengthsToGpuBufferAsyncNew(
                                        dataArrays.d_candidate_sequences_lengths.get(),
                                        batch.deviceId,
                                        dataArrays.d_candidate_read_ids.get(),
                                        dataArrays.n_queries,            
                                        streams[primary_stream_index]);

        transFuncData.readStorage->gatherSequenceDataToGpuBufferAsync(batch.subjectSequenceGatherHandle2,
                                                                         dataArrays.d_subject_sequences_data,
                                                                         dataArrays.encoded_sequence_pitch,
                                                                         dataArrays.h_subject_read_ids,
                                                                         dataArrays.d_subject_read_ids,
                                                                         dataArrays.n_subjects,
                                                                         batch.deviceId,
                                                                         streams[primary_stream_index],
                                                                         transFuncData.runtimeOptions.nCorrectorThreads);

        transFuncData.readStorage->gatherSequenceDataToGpuBufferAsync(batch.candidateSequenceGatherHandle2,
                                                                          dataArrays.d_candidate_sequences_data,
                                                                          dataArrays.encoded_sequence_pitch,
                                                                          dataArrays.h_candidate_read_ids,
                                                                          dataArrays.d_candidate_read_ids,
                                                                          dataArrays.n_queries,
                                                                          batch.deviceId,
                                                                          streams[primary_stream_index],
                                                                          transFuncData.runtimeOptions.nCorrectorThreads);

        assert(dataArrays.encoded_sequence_pitch % sizeof(int) == 0);

        call_transpose_kernel((int*)dataArrays.d_subject_sequences_data_transposed.get(),
                         (const int*)dataArrays.d_subject_sequences_data.get(),
                         dataArrays.n_subjects,
                         getEncodedNumInts2BitHiLo(transFuncData.sequenceFileProperties.maxSequenceLength),
                         dataArrays.encoded_sequence_pitch / sizeof(int),
                         streams[primary_stream_index]);

        call_transpose_kernel((int*)dataArrays.d_candidate_sequences_data_transposed.get(),
                         (const int*)dataArrays.d_candidate_sequences_data.get(),
                         dataArrays.n_queries,
                         getEncodedNumInts2BitHiLo(transFuncData.sequenceFileProperties.maxSequenceLength),
                         dataArrays.encoded_sequence_pitch / sizeof(int),
                         streams[primary_stream_index]);

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

        batch.copiedTasks = 0;
        batch.copiedCandidates = 0;
        batch.copiedSubjects = 0;
        batch.handledReadIds = false;

        batch.setState(BatchState::StartAlignment, expectedState);
        cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
    }



	void state_startalignment_func(Batch& batch){

        constexpr BatchState expectedState = BatchState::StartAlignment;

		assert(batch.state == expectedState);

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

		DataArrays& dataArrays = batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

		//cudaStreamWaitEvent(streams[primary_stream_index], events[alignment_data_transfer_h2d_finished_event_index], 0); CUERR;

        call_cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel_async(
                    dataArrays.getDeviceAlignmentResultPointers(),
                    dataArrays.getDeviceSequencePointers(),
                    dataArrays.d_candidates_per_subject_prefixsum,
                    dataArrays.h_candidates_per_subject,
                    dataArrays.d_candidates_per_subject,
                    dataArrays.n_subjects,
                    dataArrays.n_queries,
                    dataArrays.encoded_sequence_pitch,
                    sizeof(unsigned int) * getEncodedNumInts2BitHiLo(dataArrays.maximum_sequence_length),
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
            std::string s; s.resize(128);
            decode2BitHiLoSequence(&s[0], (const unsigned int*)dataArrays.h_subject_sequences_data.get() + i * dataArrays.encoded_sequence_pitch, 100, identity);
            std::cout << "Subject  : " << s << " " << batch.tasks[i].readId << std::endl;

            if(dataArrays.n_queries > 0){
                for(int j = 0; j < dataArrays.n_queries; j++){
                    //std::string s; s.resize(128);
                    //decode2BitHiLoSequence(&s[0], (const unsigned int*)dataArrays.h_candidate_sequences_data.get() + j * dataArrays.encoded_sequence_pitch, 100, identity);
                    const char* hostptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(batch.tasks[i].candidate_read_ids[j]);
                    std::string hostsequence = get2BitHiLoString((const unsigned int*)hostptr, 100, identity);
                    std::string s = get2BitHiLoString((const unsigned int*)(dataArrays.h_candidate_sequences_data.get() + j * dataArrays.encoded_sequence_pitch), 100, identity);
                    if(hostsequence != s){
                        std::cout << "host " << hostsequence << std::endl;
                        std::cout << "device " << s << std::endl;
                    }
                    std::cout << "Candidate  : " << s << " " << batch.tasks[i].candidate_read_ids[j] << std::endl;
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
		//Step 5. Compare each forward alignment with the correspoding reverse complement alignment and keep the best, if any.
		//    If reverse complement is the best, it is copied into the first half, replacing the forward alignment

        call_cuda_find_best_alignment_kernel_async_exp(
                    dataArrays.getDeviceAlignmentResultPointers(),
                    dataArrays.getDeviceSequencePointers(),
                    dataArrays.d_candidates_per_subject_prefixsum.get(),
                    dataArrays.n_subjects,
					dataArrays.n_queries,
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
					dataArrays.n_subjects,
					dataArrays.n_queries,
					transFuncData.correctionOptions.estimatedErrorrate,
					transFuncData.correctionOptions.estimatedCoverage * transFuncData.correctionOptions.m_coverage,
					streams[primary_stream_index],
					batch.kernelLaunchHandle);


        //initialize indices with -1. this allows to calculate the histrogram later on
        //without knowing the number of valid indices
        call_fill_kernel_async(dataArrays.d_indices.get(), dataArrays.n_queries, -1, streams[primary_stream_index]);

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
                                    dataArrays.n_queries,
                                    streams[primary_stream_index]); CUERR;

        //calculate indices_per_subject
        cub::DeviceHistogram::HistogramRange(dataArrays.d_cub_temp_storage.get(),
                    cubTempSize,
                    dataArrays.d_indices.get(),
                    dataArrays.d_indices_per_subject.get(),
                    dataArrays.n_subjects+1,
                    dataArrays.d_candidates_per_subject_prefixsum.get(),
                    dataArrays.n_queries,
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
                    dataArrays.n_subjects,
                    streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_num_indices,
                        dataArrays.d_num_indices,
                        sizeof(int),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaEventRecord(events[num_indices_transfered_event_index], streams[primary_stream_index]); CUERR;

        batch.setState(BatchState::RearrangeIndices, expectedState);
        cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
	}

    void state_rearrangeindices_func(Batch& batch){

        constexpr BatchState expectedState = BatchState::RearrangeIndices;

        assert(batch.state == expectedState);

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        DataArrays& dataArrays = batch.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

#ifdef REARRANGE_INDICES



        int* d_indices_segmented_partitioned;
        BestAlignment_t* d_alignment_best_alignment_flags_compact;
        BestAlignment_t* d_alignment_best_alignment_flags_discardedoutput;

        cubCachingAllocator.DeviceAllocate((void**)&d_indices_segmented_partitioned,
                                            sizeof(int) * dataArrays.n_queries,
                                            streams[primary_stream_index]); CUERR;

        cubCachingAllocator.DeviceAllocate((void**)&d_alignment_best_alignment_flags_compact,
                                            sizeof(BestAlignment_t) * dataArrays.n_queries,
                                            streams[primary_stream_index]); CUERR;

        cubCachingAllocator.DeviceAllocate((void**)&d_alignment_best_alignment_flags_discardedoutput,
                                            sizeof(BestAlignment_t) * dataArrays.n_queries,
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
                                                dataArrays.n_subjects,
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

        if(transFuncData.correctionOptions.useQualityScores) {
            batch.setState(BatchState::CopyQualities, expectedState);
        }else{
            batch.setState(BatchState::BuildMSA, expectedState);
        }

        cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
    }


    void state_copyqualities_func(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        constexpr BatchState expectedState = BatchState::CopyQualities;

		assert(batch.state == expectedState);

        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        DataArrays& dataArrays = batch.dataArrays;

        //if there are no good candidates, clean up batch and discard reads
        if(dataArrays.h_num_indices[0] == 0){
            batch.setState(BatchState::Finished, expectedState);
            cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
            return;
        }

		std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;

        const auto* gpuReadStorage = transFuncData.readStorage;

		if(transFuncData.correctionOptions.useQualityScores) {

            gpuReadStorage->gatherQualitiesToGpuBufferAsync(batch.subjectQualitiesGatherHandle2,
                                                              dataArrays.d_subject_qualities,
                                                              dataArrays.quality_pitch,
                                                              dataArrays.h_subject_read_ids,
                                                              dataArrays.d_subject_read_ids,
                                                              dataArrays.n_subjects,
                                                              batch.deviceId,
                                                              streams[primary_stream_index],
                                                              transFuncData.runtimeOptions.nCorrectorThreads);

            read_number* d_tmp_read_ids = nullptr;
            cubCachingAllocator.DeviceAllocate((void**)&d_tmp_read_ids, dataArrays.n_queries * sizeof(read_number), streams[primary_stream_index]); CUERR;

            call_compact_kernel_async(d_tmp_read_ids,
                                        dataArrays.d_candidate_read_ids.get(),
                                        dataArrays.d_indices,
                                        dataArrays.h_num_indices[0],
                                        streams[primary_stream_index]);

            std::vector<read_number> h_tmp_read_ids(dataArrays.h_num_indices[0]);
            for(int i = 0; i < dataArrays.h_num_indices[0]; i++){
                h_tmp_read_ids[i] = dataArrays.h_candidate_read_ids[dataArrays.h_indices[i]];
            }

            gpuReadStorage->gatherQualitiesToGpuBufferAsync(batch.candidateQualitiesGatherHandle2,
                                                              dataArrays.d_candidate_qualities,
                                                              dataArrays.quality_pitch,
                                                              h_tmp_read_ids.data(),
                                                              d_tmp_read_ids,
                                                              dataArrays.h_num_indices[0],
                                                              batch.deviceId,
                                                              streams[primary_stream_index],
                                                              transFuncData.runtimeOptions.nCorrectorThreads);

            cubCachingAllocator.DeviceFree(d_tmp_read_ids); CUERR;

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

            batch.setState(BatchState::BuildMSA, expectedState);
            cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
        }else{
            batch.setState(BatchState::BuildMSA, expectedState);
            cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
        }
	}


    void state_buildmsa_func(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        constexpr BatchState expectedState = BatchState::BuildMSA;

        assert(batch.state == expectedState);

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        DataArrays& dataArrays = batch.dataArrays;

        //if there are no good candidates, clean up batch and discard reads
        if(dataArrays.h_num_indices[0] == 0){
            //std::cerr << "buildmsa *h_num_indices = " << dataArrays.h_num_indices[0] << '\n';
            batch.setState(BatchState::Finished, expectedState);
            cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
            return;
        }

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
                        dataArrays.n_subjects,
                        dataArrays.n_queries,
                        dataArrays.h_num_indices,
                        dataArrays.d_num_indices,
                        1.0f,
                        transFuncData.correctionOptions.useQualityScores,
                        desiredAlignmentMaxErrorRate,
                        dataArrays.maximum_sequence_length,
                        sizeof(unsigned int) * getEncodedNumInts2BitHiLo(dataArrays.maximum_sequence_length),
                        dataArrays.encoded_sequence_pitch,
                        dataArrays.quality_pitch,
                        dataArrays.msa_pitch,
                        dataArrays.msa_weights_pitch,
                        streams[primary_stream_index],
                        batch.kernelLaunchHandle);

        //batch.dataArrays.copyEverythingToHostForDebugging();

        //At this point the msa is built
        cudaEventRecord(events[msa_build_finished_event_index], streams[primary_stream_index]); CUERR;

        batch.setState(BatchState::ImproveMSA, expectedState);
        cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
	}





    void state_improvemsa_func(Batch& batch){

        cudaSetDevice(batch.deviceId); CUERR;

        const auto& transFuncData = *batch.transFuncData;

        constexpr BatchState expectedState = BatchState::ImproveMSA;

        assert(batch.state == expectedState);

        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        DataArrays& dataArrays = batch.dataArrays;

        //if there are no good candidates, clean up batch and discard reads
        if(dataArrays.h_num_indices[0] == 0){
            std::cerr << "improvemsa *h_num_indices = " << dataArrays.h_num_indices[0] << '\n';
            batch.setState(BatchState::Finished, expectedState);
            cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
            return;
        }


#ifdef USE_MSA_MINIMIZATION

        constexpr int max_num_minimizations = 5;
        const float desiredAlignmentMaxErrorRate = transFuncData.goodAlignmentProperties.maxErrorRate;
        //const float desiredAlignmentMaxErrorRate = transFuncData.correctionOptions.estimatedErrorrate * 4.0f;

        if(max_num_minimizations > 0){
            if(batch.numMinimizations < max_num_minimizations && !(batch.numMinimizations > 0 && batch.previousNumIndices == dataArrays.h_num_indices[0])){

                const int currentNumIndices = dataArrays.h_num_indices[0];

                //std::cerr << batch.numMinimizations << " " << currentNumIndices << "\n";

                bool* d_shouldBeKept;


                cubCachingAllocator.DeviceAllocate((void**)&d_shouldBeKept, sizeof(bool) * dataArrays.n_queries, streams[primary_stream_index]); CUERR;
                call_fill_kernel_async(d_shouldBeKept, dataArrays.n_queries, true, streams[primary_stream_index]);

                //select candidates which are to be removed
                call_msa_findCandidatesOfDifferentRegion_kernel_async(
                            dataArrays.getDeviceMSAPointers(),
                            dataArrays.getDeviceAlignmentResultPointers(),
                            dataArrays.getDeviceSequencePointers(),
                            d_shouldBeKept,
                            dataArrays.d_candidates_per_subject_prefixsum,
                            dataArrays.n_subjects,
                            dataArrays.n_queries,
                            sizeof(unsigned int) * getEncodedNumInts2BitHiLo(dataArrays.maximum_sequence_length),
                            dataArrays.encoded_sequence_pitch,
                            dataArrays.msa_pitch,
                            dataArrays.msa_weights_pitch,
                            dataArrays.d_indices,
                            dataArrays.d_indices_per_subject,
                            dataArrays.d_indices_per_subject_prefixsum,
                            desiredAlignmentMaxErrorRate,
                            transFuncData.correctionOptions.estimatedCoverage,
                            streams[primary_stream_index],
                            batch.kernelLaunchHandle,
                            dataArrays.d_subject_read_ids,
                            false);  CUERR;


                int* d_shouldBeKept_positions = nullptr;
                cubCachingAllocator.DeviceAllocate((void**)&d_shouldBeKept_positions, sizeof(int) * dataArrays.n_queries, streams[primary_stream_index]); CUERR;

                int* d_newIndices = nullptr;
                cubCachingAllocator.DeviceAllocate((void**)&d_newIndices, sizeof(int) * dataArrays.n_queries, streams[primary_stream_index]); CUERR;

                int* d_indices_per_subject_tmp = nullptr;
                cubCachingAllocator.DeviceAllocate((void**)&d_indices_per_subject_tmp, sizeof(int) * dataArrays.n_subjects, streams[primary_stream_index]); CUERR;

                //save current indices_per_subject
                cudaMemcpyAsync(d_indices_per_subject_tmp,
                    dataArrays.d_indices_per_subject,
                    sizeof(int) * dataArrays.n_subjects,
                    D2D,
                    streams[primary_stream_index]); CUERR;

                call_fill_kernel_async(d_newIndices, dataArrays.n_queries, -1, streams[primary_stream_index]);

                size_t cubTempSize = dataArrays.d_cub_temp_storage.sizeInBytes();

                cub::DeviceSelect::Flagged(dataArrays.d_cub_temp_storage.get(),
                            cubTempSize,
                            cub::CountingInputIterator<int>{0},
                            d_shouldBeKept,
                            d_shouldBeKept_positions,
                            dataArrays.d_num_indices.get(),
                            currentNumIndices,
                            streams[primary_stream_index]); CUERR;

                call_compact_kernel_async(d_newIndices,
                    dataArrays.d_indices.get(),
                    d_shouldBeKept_positions,
                    dataArrays.d_num_indices.get(),
                    currentNumIndices,
                    streams[primary_stream_index]);

                cudaMemcpyAsync(dataArrays.d_indices,
                    d_newIndices,
                    sizeof(int) * dataArrays.n_queries,
                    D2D,
                    streams[primary_stream_index]); CUERR;

                //calculate indices per subject
                cub::DeviceHistogram::HistogramRange(dataArrays.d_cub_temp_storage.get(),
                            cubTempSize,
                            d_newIndices,
                            dataArrays.d_indices_per_subject.get(),
                            dataArrays.n_subjects+1,
                            dataArrays.d_candidates_per_subject_prefixsum.get(),
                            dataArrays.n_queries,
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
                            dataArrays.n_subjects,
                            streams[primary_stream_index]); CUERR;


                //compact the quality scores
                if(transFuncData.correctionOptions.useQualityScores){
#if 1                    
                    dim3 block(128,1,1);
                    dim3 grid(SDIV(currentNumIndices, block.x),1,1);

                    char* const d_candidate_qualities = dataArrays.d_candidate_qualities;
                    char* const d_candidate_qualities_tmp = dataArrays.d_candidate_qualities_tmp;

                    const size_t quality_pitch = dataArrays.quality_pitch;
                    //const int maximum_sequence_length = dataArrays.maximum_sequence_length;
                    const int* const d_num_indices = dataArrays.d_num_indices;
                    //const int numCandidates = dataArrays.n_queries;

                    cudaMemsetAsync(d_candidate_qualities_tmp, currentNumIndices * quality_pitch, 0, streams[primary_stream_index]); CUERR;

                    generic_kernel<<<grid, block,0, streams[primary_stream_index]>>>([=] __device__ (){
                        const int numIndices = *d_num_indices;

                        for(int targetIndex = blockIdx.x; targetIndex < numIndices; targetIndex += gridDim.x){
                            const int srcIndex = d_shouldBeKept_positions[targetIndex];
                            const char* const srcPtr = &d_candidate_qualities[srcIndex * quality_pitch];
                            char* const destPtr = &d_candidate_qualities_tmp[targetIndex * quality_pitch];

                            for(int pos = threadIdx.x; pos < quality_pitch; pos += blockDim.x){
                                destPtr[pos] = srcPtr[pos];
                            }
                        }
                    }); CUERR;
#else 

                    constexpr int blocksize = 128;
                    dim3 block(blocksize,1,1);
                    dim3 grid(SDIV(quality_pitch * currentNumIndices, blocksize),1,1);

                    char* const d_candidate_qualities = dataArrays.d_candidate_qualities;
                    char* const d_candidate_qualities_tmp = dataArrays.d_candidate_qualities_tmp;

                    const size_t quality_pitch = dataArrays.quality_pitch;
                    //const int maximum_sequence_length = dataArrays.maximum_sequence_length;
                    const int* const d_num_indices = dataArrays.d_num_indices;
                    //const int numCandidates = dataArrays.n_queries;

                    cudaMemsetAsync(d_candidate_qualities_tmp, currentNumIndices * quality_pitch, 0, streams[primary_stream_index]); CUERR;

                    generic_kernel<<<grid, block,0, streams[primary_stream_index]>>>([=] __device__ (){
                        const int numIndices = *d_num_indices;

                        for(int idx = blockIdx.x * blocksize + threadIdx.x; idx < quality_pitch * numIndices; idx += blocksize * gridDim.x){
                            const int targetIndex = idx / numIndices;
                            const int qualPos = idx % numIndices;
                            const int srcIndex = d_shouldBeKept_positions[targetIndex];
                            const char* const srcPtr = &d_candidate_qualities[srcIndex * quality_pitch];
                            char* const destPtr = &d_candidate_qualities_tmp[targetIndex * quality_pitch];
                            destPtr[qualPos] = srcPtr[qualPos];
                        }
                    }); CUERR;


#endif


                   /* char* d_candidate_qualities_tmp2;
                    cubCachingAllocator.DeviceAllocate((void**)&d_candidate_qualities_tmp2, currentNumIndices * quality_pitch, streams[primary_stream_index]); CUERR;
                    cudaMemsetAsync(d_candidate_qualities_tmp2, currentNumIndices * quality_pitch, 0, streams[primary_stream_index]); CUERR;

                    char* const d_candidate_qualities_transposed = dataArrays.d_candidate_qualities_transposed;
                    dim3 block2(32,8,1);
                    dim3 grid2(SDIV(currentNumIndices, block.x), SDIV(maximum_sequence_length, block.y),1);

                    generic_kernel<<<grid2, block2,0, streams[primary_stream_index]>>>([=] __device__ (){
                        const int numIndices = *d_num_indices;

                        for(int targetIndex = threadIdx.x + blockIdx.x * blockDim.x; targetIndex < numIndices; targetIndex += blockDim.x * gridDim.x){
                            const int srcIndex = d_shouldBeKept_positions[targetIndex];
                            const char* const srcPtr = &d_candidate_qualities_transposed[srcIndex];
                            char* const destPtr = &d_candidate_qualities_tmp2[targetIndex];

                            for(int pos = threadIdx.y + blockIdx.y * blockDim.y; pos < maximum_sequence_length; pos += blockDim.y * gridDim.y){
                                destPtr[pos * numIndices] = srcPtr[pos * numIndices];
                            }
                        }
                    }); CUERR;

                    char* d_candidate_qualities_tmp3;
                    cubCachingAllocator.DeviceAllocate((void**)&d_candidate_qualities_tmp3, currentNumIndices * quality_pitch, streams[primary_stream_index]); CUERR;
                    cudaMemsetAsync(d_candidate_qualities_tmp3, currentNumIndices * quality_pitch, 0, streams[primary_stream_index]); CUERR;

                    call_transpose_kernel(d_candidate_qualities_tmp3, d_candidate_qualities_tmp, currentNumIndices, maximum_sequence_length, quality_pitch, streams[primary_stream_index]);

                    generic_kernel<<<1,1,0,streams[primary_stream_index]>>>([=]__device__(){
                        for(int row = 0; row < maximum_sequence_length; row++){
                            for(int col = 0; col < currentNumIndices; col++){
                                if(d_candidate_qualities_tmp3[row * currentNumIndices + col] != d_candidate_qualities_tmp2[row * currentNumIndices + col]){
                                    //printf("error at row %d, col %d\n", row, col);
                                    //assert(false);
                                }
                                printf("%c", d_candidate_qualities_tmp3[row * currentNumIndices + col]);
                            }
                            printf("\n");
                        }
                        printf("\n");

                        for(int row = 0; row < maximum_sequence_length; row++){
                            for(int col = 0; col < currentNumIndices; col++){
                                printf("%c", d_candidate_qualities_tmp2[row * currentNumIndices + col]);
                            }
                            printf("\n");
                        }
                        printf("\n");

                        for(int row = 0; row < maximum_sequence_length; row++){
                            for(int col = 0; col < currentNumIndices; col++){
                                if(d_candidate_qualities_tmp3[row * currentNumIndices + col] != d_candidate_qualities_tmp2[row * currentNumIndices + col]){
                                    printf("error at row %d, col %d\n", row, col);
                                    assert(false);
                                }
                            }
                        }
                    }); CUERR;*/
                    /*generic_kernel<<<1,1,0,streams[primary_stream_index]>>>([=]__device__(){
                        bool error = false;
                        int r = 0;
                        int c = 0;
                        const int numIndices = *d_num_indices;

                        for(int row = 0; row < maximum_sequence_length && !error; row++){
                            for(int col = 0; col < numIndices && !error; col++){
                                if(d_candidate_qualities_tmp3[row * numIndices + col] != d_candidate_qualities_tmp2[row * numIndices + col]){
                                    error = true;
                                    r = row;
                                    c = col;
                                }
                            }
                        }

                        if(error){
                            printf("d_candidate_qualities_tmp3\n");

                            for(int row = 0; row < maximum_sequence_length; row++){
                                for(int col = 0; col < numIndices; col++){
                                    printf("%c", d_candidate_qualities_tmp3[row * numIndices + col]);
                                }
                                printf("\n");
                            }
                            printf("\n");

                            printf("d_candidate_qualities_tmp2\n");

                            for(int row = 0; row < maximum_sequence_length; row++){
                                for(int col = 0; col < numIndices; col++){
                                    printf("%c", d_candidate_qualities_tmp2[row * numIndices + col]);
                                }
                                printf("\n");
                            }
                            printf("\n");

                            printf("d_candidate_qualities_tmp\n");

                            for(int row = 0; row < numIndices; row++){
                                for(int col = 0; col < quality_pitch; col++){
                                    printf("%c", d_candidate_qualities_tmp[row * quality_pitch + col]);
                                }
                                printf("\n");
                            }
                            printf("\n");

                            printf("error at row %d, col %d\n", r, c);

                            assert(false);
                        }
                    }); CUERR;

                    cudaDeviceSynchronize(); CUERR;

                    cubCachingAllocator.DeviceFree(d_candidate_qualities_tmp2); CUERR;
                    cubCachingAllocator.DeviceFree(d_candidate_qualities_tmp3); CUERR;*/

                    //std::exit(0);

                    std::swap(dataArrays.d_candidate_qualities, dataArrays.d_candidate_qualities_tmp);
                }

                {
                    /*
                        compare old indices_per_subject , which are stored in indices_per_subject_tmp
                        to the new indices_per_subject.
                        set value in indices_per_subject_tmp to 0 if old value and new value are equal, else the value is set to the new value.
                        this prevents rebuilding the MSAs of subjects whose indices where not changed by minimization (all indices are kept)
                    */

                    const int* d_indices_per_subject = dataArrays.d_indices_per_subject;
                    const int n_subjects = dataArrays.n_subjects;
                    cudaStream_t stream = streams[primary_stream_index];

                    dim3 block(128,1,1);
                    dim3 grid(SDIV(dataArrays.n_subjects, block.x),1,1);
                    generic_kernel<<<grid, block, 0, stream>>>([=] __device__ (){
                        for(int i = threadIdx.x + blockDim.x * blockIdx.x; i < n_subjects; i += blockDim.x * gridDim.x){
                            if(d_indices_per_subject[i] == d_indices_per_subject_tmp[i]){
                                d_indices_per_subject_tmp[i] = 0;
                            }else{
                                d_indices_per_subject_tmp[i] = d_indices_per_subject[i];
                            }
                        }
                    }); CUERR;

                    /*const int* d_num_indices = dataArrays.d_num_indices;
                    generic_kernel<<<1, 1, 0, stream>>>([=] __device__ (){
                        int sum = 0;
                        for(int i = 0; i < n_subjects; i++){
                            sum += d_indices_per_subject_tmp[i];
                        }
                        printf("sum = %d, totalindices %d\n", sum, *d_num_indices);
                    }); CUERR;*/
                }

                cudaMemcpyAsync(dataArrays.h_num_indices, dataArrays.d_num_indices, sizeof(int), D2H, streams[primary_stream_index]);  CUERR;

                cudaEventRecord(events[num_indices_transfered_event_index], streams[primary_stream_index]); CUERR;

                const float desiredAlignmentMaxErrorRate = transFuncData.goodAlignmentProperties.maxErrorRate;
                //const float desiredAlignmentMaxErrorRate = transFuncData.correctionOptions.estimatedErrorrate * 4.0f;

                build_msa_async(dataArrays.getDeviceMSAPointers(),
                                dataArrays.getDeviceAlignmentResultPointers(),
                                dataArrays.getDeviceSequencePointers(),
                                dataArrays.getDeviceQualityPointers(),
                                dataArrays.d_candidates_per_subject_prefixsum,
                                dataArrays.d_indices,
                                d_indices_per_subject_tmp,
                                dataArrays.d_indices_per_subject_prefixsum,
                                dataArrays.n_subjects,
                                dataArrays.n_queries,
                                dataArrays.h_num_indices,
                                dataArrays.d_num_indices,
                                0.05f, //
                                transFuncData.correctionOptions.useQualityScores,
                                desiredAlignmentMaxErrorRate,
                                dataArrays.maximum_sequence_length,
                                sizeof(unsigned int) * getEncodedNumInts2BitHiLo(dataArrays.maximum_sequence_length),
                                dataArrays.encoded_sequence_pitch,
                                dataArrays.quality_pitch,
                                dataArrays.msa_pitch,
                                dataArrays.msa_weights_pitch,
                                streams[primary_stream_index],
                                batch.kernelLaunchHandle);

                //batch.dataArrays.copyEverythingToHostForDebugging();

                

                cubCachingAllocator.DeviceFree(d_shouldBeKept); CUERR;
                cubCachingAllocator.DeviceFree(d_newIndices); CUERR;
                cubCachingAllocator.DeviceFree(d_indices_per_subject_tmp); CUERR;
                cubCachingAllocator.DeviceFree(d_shouldBeKept_positions); CUERR;

                batch.numMinimizations++;
                batch.previousNumIndices = currentNumIndices;

                //repeat state
                batch.setState(BatchState::ImproveMSA, expectedState);
                cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
                return;
            }else{
                //std::cerr << "minimization finished\n";
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
        }
#endif


        //At this point the msa is built, maybe minimized, and is ready to be used for correction

        cudaEventRecord(events[msa_build_finished_event_index], streams[primary_stream_index]); CUERR;

        if(transFuncData.correctionOptions.extractFeatures || transFuncData.correctionOptions.correctionType != CorrectionType::Classic) {

            cudaStreamWaitEvent(streams[secondary_stream_index], events[msa_build_finished_event_index], 0); CUERR;

            cudaMemcpyAsync(dataArrays.h_consensus,
                        dataArrays.d_consensus,
                        dataArrays.n_subjects * dataArrays.msa_pitch,
                        D2H,
                        streams[secondary_stream_index]); CUERR;
            cudaMemcpyAsync(dataArrays.h_support,
                        dataArrays.d_support,
                        dataArrays.n_subjects * dataArrays.msa_weights_pitch,
                        D2H,
                        streams[secondary_stream_index]); CUERR;
            cudaMemcpyAsync(dataArrays.h_coverage,
                        dataArrays.d_coverage,
                        dataArrays.n_subjects * dataArrays.msa_weights_pitch,
                        D2H,
                        streams[secondary_stream_index]); CUERR;
            cudaMemcpyAsync(dataArrays.h_origCoverages,
                        dataArrays.d_origCoverages,
                        dataArrays.n_subjects * dataArrays.msa_weights_pitch,
                        D2H,
                        streams[secondary_stream_index]); CUERR;
            cudaMemcpyAsync(dataArrays.h_msa_column_properties,
                        dataArrays.d_msa_column_properties,
                        dataArrays.n_subjects * sizeof(MSAColumnProperties),
                        D2H,
                        streams[secondary_stream_index]); CUERR;
            cudaMemcpyAsync(dataArrays.h_counts,
                        dataArrays.d_counts,
                        dataArrays.n_subjects * dataArrays.msa_weights_pitch * 4,
                        D2H,
                        streams[secondary_stream_index]); CUERR;
            cudaMemcpyAsync(dataArrays.h_weights,
                        dataArrays.d_weights,
                        dataArrays.n_subjects * dataArrays.msa_weights_pitch * 4,
                        D2H,
                        streams[secondary_stream_index]); CUERR;

            cudaEventRecord(events[msadata_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;
        }

        if(transFuncData.correctionOptions.extractFeatures){
            cudaStreamWaitEvent(streams[primary_stream_index], events[msadata_transfer_finished_event_index], 0); CUERR;
            batch.setState(BatchState::WriteFeatures, expectedState);
            cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
            return;
        }else{
            switch(transFuncData.correctionOptions.correctionType){
            case CorrectionType::Classic:
                //cudaStreamWaitEvent(streams[primary_stream_index], events[indices_transfer_finished_event_index], 0); CUERR;
                batch.setState(BatchState::StartClassicCorrection, expectedState);
                cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
                return;
            case CorrectionType::Forest:
                cudaStreamWaitEvent(streams[primary_stream_index], events[msadata_transfer_finished_event_index], 0); CUERR;
                batch.setState(BatchState::StartForestCorrection, expectedState);
                cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
                return;
            case CorrectionType::Convnet:
                cudaStreamWaitEvent(streams[primary_stream_index], events[msadata_transfer_finished_event_index], 0); CUERR;
                batch.setState(BatchState::StartConvnetCorrection, expectedState);
                cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
                return;
            default:
                batch.setState(BatchState::StartClassicCorrection, expectedState);
                cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
                return;
            }
        }


    }


	void state_startclassiccorrection_func(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;
        constexpr BatchState expectedState = BatchState::StartClassicCorrection;

        assert(batch.state == expectedState);
		assert(transFuncData.correctionOptions.correctionType == CorrectionType::Classic);

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

                    const char* candidateSequencePtr = dataArrays.h_candidate_sequences_data.get() + index * dataArrays.encoded_sequence_pitch;

                    assert(dataArrays.h_alignment_best_alignment_flags[index] != BestAlignment_t::None);

                    std::string candidatestring = get2BitHiLoString((unsigned int*)candidateSequencePtr, dataArrays.h_candidate_sequences_lengths[index]);
                    if(dataArrays.h_alignment_best_alignment_flags[index] == BestAlignment_t::ReverseComplement){
                        candidatestring = reverseComplementString(candidatestring.c_str(), candidatestring.length());
                    }

                    std::copy(candidatestring.begin(), candidatestring.end(), dst);
                    //decode2BitHiLoSequence(dst, (const unsigned int*)candidateSequencePtr, 100, identity);
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
                 std::string consensus = std::string{dataArrays.h_consensus + i * dataArrays.msa_pitch, dataArrays.h_consensus + (i+1) * dataArrays.msa_pitch};
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
                    dataArrays.n_subjects,
                    dataArrays.encoded_sequence_pitch,
                    dataArrays.sequence_pitch,
                    dataArrays.msa_pitch,
                    dataArrays.msa_weights_pitch,
                    transFuncData.sequenceFileProperties.maxSequenceLength,
                    transFuncData.correctionOptions.estimatedErrorrate,
                    transFuncData.goodAlignmentProperties.maxErrorRate,
                    avg_support_threshold,
                    min_support_threshold,
                    min_coverage_threshold,
                    max_coverage_threshold,
                    transFuncData.correctionOptions.kmerlength,
                    dataArrays.maximum_sequence_length,
                    streams[primary_stream_index],
                    batch.kernelLaunchHandle);

        cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_corrected_subjects,
                        dataArrays.d_corrected_subjects,
                        dataArrays.d_corrected_subjects.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_subject_is_corrected,
                        dataArrays.d_subject_is_corrected,
                        dataArrays.d_subject_is_corrected.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaMemcpyAsync(dataArrays.h_is_high_quality_subject,
                        dataArrays.d_is_high_quality_subject,
                        dataArrays.d_is_high_quality_subject.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_num_uncorrected_positions_per_subject,
                        dataArrays.d_num_uncorrected_positions_per_subject,
                        dataArrays.d_num_uncorrected_positions_per_subject.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaMemcpyAsync(dataArrays.h_uncorrected_positions_per_subject,
                        dataArrays.d_uncorrected_positions_per_subject,
                        dataArrays.d_uncorrected_positions_per_subject.sizeInBytes(),
                        D2H,
                        streams[primary_stream_index]); CUERR;

		cudaEventRecord(events[result_transfer_finished_event_index], streams[primary_stream_index]); CUERR;

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
                        dataArrays.n_subjects,
                        streams[primary_stream_index]); CUERR;

            cudaMemcpyAsync(dataArrays.h_high_quality_subject_indices,
                            dataArrays.d_high_quality_subject_indices,
                            dataArrays.d_high_quality_subject_indices.sizeInBytes(),
                            D2H,
                            streams[primary_stream_index]); CUERR;

            cudaMemcpyAsync(dataArrays.h_num_high_quality_subject_indices,
                            dataArrays.d_num_high_quality_subject_indices,
                            dataArrays.d_num_high_quality_subject_indices.sizeInBytes(),
                            D2H,
                            streams[primary_stream_index]); CUERR;

            cudaStreamWaitEvent(streams[primary_stream_index], events[indices_transfer_finished_event_index], 0); CUERR;


            batch.setState(BatchState::StartClassicCandidateCorrection, expectedState);
		}else{
            batch.setState(BatchState::UnpackClassicResults, expectedState);
        }

        cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
	}



    void state_startclassiccandidatecorrection_func(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;
        constexpr BatchState expectedState = BatchState::StartClassicCandidateCorrection;

        assert(batch.state == expectedState);
        assert(transFuncData.correctionOptions.correctionType == CorrectionType::Classic);

        cudaSetDevice(batch.deviceId); CUERR;

        DataArrays& dataArrays = batch.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        const float min_support_threshold = 1.0f-3.0f*transFuncData.correctionOptions.estimatedErrorrate;
        // coverage is always >= 1
        const float min_coverage_threshold = std::max(1.0f,
                    transFuncData.correctionOptions.m_coverage / 6.0f * transFuncData.correctionOptions.estimatedCoverage);
        const int new_columns_to_correct = transFuncData.correctionOptions.new_columns_to_correct;


        if(transFuncData.correctionOptions.correctCandidates) {

#if 1
            call_msa_correct_candidates_kernel_async_experimental(
                    dataArrays.getDeviceMSAPointers(),
                    dataArrays.getDeviceAlignmentResultPointers(),
                    dataArrays.getDeviceSequencePointers(),
                    dataArrays.getDeviceCorrectionResultPointers(),
                    dataArrays.getHostCorrectionResultPointers(),
                    dataArrays.d_indices,
                    dataArrays.d_indices_per_subject,
                    dataArrays.d_indices_per_subject_prefixsum,
                    dataArrays.h_indices_per_subject,
                    dataArrays.n_subjects,
                    dataArrays.n_queries,
                    dataArrays.d_num_indices,
                    dataArrays.encoded_sequence_pitch,
                    dataArrays.sequence_pitch,
                    dataArrays.msa_pitch,
                    dataArrays.msa_weights_pitch,
                    min_support_threshold,
                    min_coverage_threshold,
                    new_columns_to_correct,
                    dataArrays.maximum_sequence_length,
                    streams[primary_stream_index],
                    batch.kernelLaunchHandle);

            // correct candidates
#else
            call_msa_correct_candidates_kernel_async(
                    dataArrays.getDeviceMSAPointers(),
                    dataArrays.getDeviceAlignmentResultPointers(),
                    dataArrays.getDeviceSequencePointers(),
                    dataArrays.getDeviceCorrectionResultPointers(),
                    dataArrays.d_indices,
                    dataArrays.d_indices_per_subject,
                    dataArrays.d_indices_per_subject_prefixsum,
                    dataArrays.n_subjects,
                    dataArrays.n_queries,
                    dataArrays.d_num_indices,
                    dataArrays.encoded_sequence_pitch,
                    dataArrays.sequence_pitch,
                    dataArrays.msa_pitch,
                    dataArrays.msa_weights_pitch,
                    min_support_threshold,
                    min_coverage_threshold,
                    new_columns_to_correct,
                    dataArrays.maximum_sequence_length,
                    streams[primary_stream_index],
                    batch.kernelLaunchHandle);
#endif

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
        }
        cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;

        batch.setState(BatchState::UnpackClassicResults, expectedState);
        cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
        return;
    }

    void state_startconvnetcorrection_func(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;

        constexpr BatchState expectedState = BatchState::StartConvnetCorrection;

        assert(batch.state == expectedState);
        assert(transFuncData.correctionOptions.correctionType == CorrectionType::Convnet);

        cudaSetDevice(batch.deviceId); CUERR;

        DataArrays& dataArrays = batch.dataArrays;

        std::vector<MSAFeature3> MSAFeatures;
        std::vector<int> MSAFeaturesPerSubject(batch.tasks.size());
        std::vector<int> MSAFeaturesPerSubjectPrefixSum(batch.tasks.size()+1);
        MSAFeaturesPerSubjectPrefixSum[0] = 0;

        for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {

            auto& task = batch.tasks[subject_index];

            const auto& columnProperties = dataArrays.h_msa_column_properties[subject_index];

            //std::cout << task.readId << "feature" << std::endl;

            const std::size_t msa_weights_pitch_floats = dataArrays.msa_weights_pitch / sizeof(float);

            const int msa_rows = 1 + dataArrays.h_indices_per_subject[subject_index];
#if 1
            const std::size_t countsOffset = subject_index * msa_weights_pitch_floats * 4;
            const std::size_t weightsOffset = subject_index * msa_weights_pitch_floats * 4;
            const int* const countsA = &dataArrays.h_counts[countsOffset + 0 * msa_weights_pitch_floats];
            const int* const countsC = &dataArrays.h_counts[countsOffset + 1 * msa_weights_pitch_floats];
            const int* const countsG = &dataArrays.h_counts[countsOffset + 2 * msa_weights_pitch_floats];
            const int* const countsT = &dataArrays.h_counts[countsOffset + 3 * msa_weights_pitch_floats];
            const float* const weightsA = &dataArrays.h_weights[weightsOffset + 0 * msa_weights_pitch_floats];
            const float* const weightsC = &dataArrays.h_weights[weightsOffset + 1 * msa_weights_pitch_floats];
            const float* const weightsG = &dataArrays.h_weights[weightsOffset + 2 * msa_weights_pitch_floats];
            const float* const weightsT = &dataArrays.h_weights[weightsOffset + 3 * msa_weights_pitch_floats];

            std::vector<MSAFeature3> tmpfeatures
                    = extractFeatures3_2(
                                        countsA,
                                        countsC,
                                        countsG,
                                        countsT,
                                        weightsA,
                                        weightsC,
                                        weightsG,
                                        weightsT,
                                        msa_rows,
                                        columnProperties.lastColumn_excl,
                                        dataArrays.h_consensus + subject_index * dataArrays.msa_pitch,
                                        dataArrays.h_support + subject_index * msa_weights_pitch_floats,
                                        dataArrays.h_coverage + subject_index * msa_weights_pitch_floats,
                                        dataArrays.h_origCoverages + subject_index * msa_weights_pitch_floats,
                                        columnProperties.subjectColumnsBegin_incl,
                                        columnProperties.subjectColumnsEnd_excl,
                                        task.subject_string,
                                        transFuncData.correctionOptions.estimatedCoverage);
#else
            const unsigned offset1 = dataArrays.msa_pitch * (subject_index +  dataArrays.h_indices_per_subject_prefixsum[subject_index]);
            const unsigned offset2 = msa_weights_pitch_floats * (subject_index +  dataArrays.h_indices_per_subject_prefixsum[subject_index]);

            const char* const my_multiple_sequence_alignment = dataArrays.h_multiple_sequence_alignments + offset1;
            const float* const my_multiple_sequence_alignment_weight = dataArrays.h_multiple_sequence_alignment_weights + offset2;

            std::vector<MSAFeature3> tmpfeatures = extractFeatures3(
                                        my_multiple_sequence_alignment,
                                        my_multiple_sequence_alignment_weight,
                                        msa_rows,
                                        columnProperties.columnsToCheck,
                                        transFuncData.correctionOptions.useQualityScores,
                                        dataArrays.h_consensus + subject_index * dataArrays.msa_pitch,
                                        dataArrays.h_support + subject_index * msa_weights_pitch_floats,
                                        dataArrays.h_coverage + subject_index * msa_weights_pitch_floats,
                                        dataArrays.h_origCoverages + subject_index * msa_weights_pitch_floats,
                                        columnProperties.subjectColumnsBegin_incl,
                                        columnProperties.subjectColumnsEnd_excl,
                                        task.subject_string,
                                        transFuncData.correctionOptions.estimatedCoverage,
                                        true,
                                        dataArrays.msa_pitch,
                                        msa_weights_pitch_floats);
#endif
            MSAFeatures.insert(MSAFeatures.end(), tmpfeatures.begin(), tmpfeatures.end());
            MSAFeaturesPerSubject[subject_index] = tmpfeatures.size();
        }

        std::partial_sum(MSAFeaturesPerSubject.begin(), MSAFeaturesPerSubject.end(),MSAFeaturesPerSubjectPrefixSum.begin()+1);

        std::vector<float> predictions = transFuncData.nnClassifier.infer(MSAFeatures);
        assert(predictions.size() == MSAFeatures.size());

        for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {
            auto& task = batch.tasks[subject_index];
            task.corrected_subject = task.subject_string;

            const int offset = MSAFeaturesPerSubjectPrefixSum[subject_index];
            const int end_index = offset + MSAFeaturesPerSubject[subject_index];

            const char* const consensus = &dataArrays.h_consensus[subject_index * dataArrays.msa_pitch];
            const auto& columnProperties = dataArrays.h_msa_column_properties[subject_index];

            for(int index = offset; index < end_index; index++){
                constexpr float threshold = 0.95;
                const auto& msafeature = MSAFeatures[index];

                if(predictions[index] >= threshold){
                    task.corrected = true;

                    const int globalIndex = columnProperties.subjectColumnsBegin_incl + msafeature.position;
                    task.corrected_subject[msafeature.position] = consensus[globalIndex];
                }
            }

        }

        batch.setState(BatchState::WriteResults, expectedState);
        cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
        return;
    }

    void state_startforestcorrection_func(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;

        constexpr BatchState expectedState = BatchState::StartForestCorrection;

        assert(batch.state == expectedState);
        assert(transFuncData.correctionOptions.correctionType == CorrectionType::Forest);

        cudaSetDevice(batch.deviceId); CUERR;

		DataArrays& dataArrays = batch.dataArrays;

        std::vector<MSAFeature> MSAFeatures;
        std::vector<int> MSAFeaturesPerSubject(batch.tasks.size());
        std::vector<int> MSAFeaturesPerSubjectPrefixSum(batch.tasks.size()+1);

        for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {
            auto& task = batch.tasks[subject_index];

            const auto& columnProperties = dataArrays.h_msa_column_properties[subject_index];
            const std::size_t msa_weights_pitch_floats = dataArrays.msa_weights_pitch / sizeof(float);

            const char* const consensus = dataArrays.h_consensus + subject_index * dataArrays.msa_pitch;

            std::vector<MSAFeature> tmpfeatures = extractFeatures(consensus,
                                                dataArrays.h_support + subject_index * msa_weights_pitch_floats,
                                                dataArrays.h_coverage + subject_index * msa_weights_pitch_floats,
                                                dataArrays.h_origCoverages + subject_index * msa_weights_pitch_floats,
                                                columnProperties.lastColumn_excl,
                                                columnProperties.subjectColumnsBegin_incl,
                                                columnProperties.subjectColumnsEnd_excl,
                                                task.subject_string,
                                                transFuncData.correctionOptions.kmerlength, 0.0f,
                                                transFuncData.correctionOptions.estimatedCoverage);

            MSAFeatures.insert(MSAFeatures.end(), tmpfeatures.begin(), tmpfeatures.end());
            MSAFeaturesPerSubject[subject_index] = tmpfeatures.size();

        }

        MSAFeaturesPerSubjectPrefixSum[0] = 0;
        std::partial_sum(MSAFeaturesPerSubject.begin(), MSAFeaturesPerSubject.end(), MSAFeaturesPerSubjectPrefixSum.begin()+1);

        constexpr float maxgini = 0.05f;
        constexpr float forest_correction_fraction = 0.5f;

        //#pragma omp parallel for schedule(dynamic,2)
        for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {
            auto& task = batch.tasks[subject_index];
            task.corrected_subject = task.subject_string;

            const int offset = MSAFeaturesPerSubjectPrefixSum[subject_index];
            const int end_index = offset + MSAFeaturesPerSubject[subject_index];

            const char* const consensus = &dataArrays.h_consensus[subject_index * dataArrays.msa_pitch];
            const auto& columnProperties = dataArrays.h_msa_column_properties[subject_index];

            
            for(int index = offset; index < end_index; index++){
                const auto& msafeature = MSAFeatures[index];

                const bool doCorrect = transFuncData.fc.shouldCorrect(msafeature.position_support,
                                                                    msafeature.position_coverage,
                                                                    msafeature.alignment_coverage,
                                                                    msafeature.dataset_coverage,
                                                                    msafeature.min_support,
                                                                    msafeature.min_coverage,
                                                                    msafeature.max_support,
                                                                    msafeature.max_coverage,
                                                                    msafeature.mean_support,
                                                                    msafeature.mean_coverage,
                                                                    msafeature.median_support,
                                                                    msafeature.median_coverage,
                                                                    maxgini,
                                                                    forest_correction_fraction);

				if(doCorrect) {
					task.corrected = true;

					const int globalIndex = columnProperties.subjectColumnsBegin_incl + msafeature.position;
                    task.corrected_subject[msafeature.position] = consensus[globalIndex];
                }
            }

            if(task.corrected){
                    
                const int subject_length = dataArrays.h_subject_sequences_lengths[subject_index];

                task.correctionEqualsOriginal = task.corrected_subject == task.subject_string;

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
            }
        }

        batch.setState(BatchState::WriteResults, expectedState);
        cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
        return;
	}

    void state_combinestreams_func(Batch& batch){
        constexpr BatchState expectedState = BatchState::CombineStreams;

        assert(batch.state == expectedState);

        cudaSetDevice(batch.deviceId); CUERR;

        auto& events = batch.events;
        auto& streams = batch.streams;

        cudaEventRecord(events[secondary_stream_finished_event_index], streams[secondary_stream_index]); CUERR;

        cudaStreamWaitEvent(streams[primary_stream_index], events[secondary_stream_finished_event_index], 0); CUERR;
    }

#if 0
void state_unpackclassicresults_func(Batch& batch){

    const auto& transFuncData = *batch.transFuncData;

    constexpr BatchState expectedState = BatchState::UnpackClassicResults;

    assert(batch.state == expectedState);

    cudaSetDevice(batch.deviceId); CUERR;

    if(!batch.combinedStreams){
        auto& events = batch.events;
        auto& streams = batch.streams;

        cudaEventRecord(events[secondary_stream_finished_event_index], streams[secondary_stream_index]); CUERR;

        cudaStreamWaitEvent(streams[primary_stream_index], events[secondary_stream_finished_event_index], 0); CUERR;
        batch.combinedStreams = true;

        cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
        return;
    }

    std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

    DataArrays& dataArrays = batch.dataArrays;
    //std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;

    cudaError_t errort = cudaEventQuery(events[correction_finished_event_index]);
    if(errort != cudaSuccess){
        std::cout << "error cudaEventQuery\n";
        std::exit(0);
    }
    assert(cudaEventQuery(events[correction_finished_event_index]) == cudaSuccess); CUERR;
    assert(cudaEventQuery(events[result_transfer_finished_event_index]) == cudaSuccess); CUERR;


    assert(transFuncData.correctionOptions.correctionType == CorrectionType::Classic);

    //#pragma omp parallel for
    for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {
        auto& task = batch.tasks[subject_index];
        const char* const my_corrected_subject_data = dataArrays.h_corrected_subjects + subject_index * dataArrays.sequence_pitch;
        task.corrected = dataArrays.h_subject_is_corrected[subject_index];
        task.highQualityAlignment = dataArrays.h_is_high_quality_subject[subject_index];
        //if(task.readId == 207){
        //    std::cerr << "\n\ncorrected: " << task.corrected << "\n";
        //}
        if(task.corrected) {
            const int subject_length = dataArrays.h_subject_sequences_lengths[subject_index];
            task.corrected_subject = std::move(std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length});

            const int numUncorrectedPositions = dataArrays.h_num_uncorrected_positions_per_subject[subject_index];
            if(numUncorrectedPositions > 0){
                task.uncorrectedPositionsNoConsensus.resize(numUncorrectedPositions);
                std::copy_n(dataArrays.h_uncorrected_positions_per_subject + subject_index * dataArrays.maximum_sequence_length,
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


            //if(task.readId == 207){
            //    std::cerr << "\n\ncorrected sequence: " << task.corrected_subject << "\n";
            //}
        }
    }

    if(transFuncData.correctionOptions.correctCandidates) {

        #pragma omp parallel for schedule(dynamic, 4)
        for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {
            auto& task = batch.tasks[subject_index];
            const int n_corrected_candidates = dataArrays.h_num_corrected_candidates[subject_index];
            const char* const my_corrected_candidates_data = dataArrays.h_corrected_candidates
                                            + dataArrays.h_indices_per_subject_prefixsum[subject_index] * dataArrays.sequence_pitch;
            const int* const my_indices_of_corrected_candidates = dataArrays.h_indices_of_corrected_candidates
                                            + dataArrays.h_indices_per_subject_prefixsum[subject_index];


            task.corrected_candidates_shifts.resize(n_corrected_candidates);
            task.corrected_candidates_read_ids.resize(n_corrected_candidates);
            task.corrected_candidates.resize(n_corrected_candidates);
            task.corrected_candidate_equals_uncorrected.resize(n_corrected_candidates);

            for(int i = 0; i < n_corrected_candidates; ++i) {
                const int global_candidate_index = my_indices_of_corrected_candidates[i];
                //const int local_candidate_index = global_candidate_index - dataArrays.h_candidates_per_subject_prefixsum[subject_index];

                //const read_number candidate_read_id = task.candidate_read_ids[local_candidate_index];
                //const read_number candidate_read_id = task.candidate_read_ids_begin[local_candidate_index];

                const read_number candidate_read_id = dataArrays.h_candidate_read_ids[global_candidate_index];
                const int candidate_length = dataArrays.h_candidate_sequences_lengths[global_candidate_index];//transFuncData.gpuReadStorage->fetchSequenceLength(candidate_read_id);
                const int candidate_shift = dataArrays.h_alignment_shifts[global_candidate_index];

                const char* const candidate_data = my_corrected_candidates_data + i * dataArrays.sequence_pitch;
                if(transFuncData.correctionOptions.new_columns_to_correct < candidate_shift){
                    std::cerr << "\n" << "readid " << task.readId << " candidate readid " << candidate_read_id << " : "
                            << candidate_shift << " " << transFuncData.correctionOptions.new_columns_to_correct <<"\n";
                }
                assert(transFuncData.correctionOptions.new_columns_to_correct >= candidate_shift);
                task.corrected_candidates_shifts[i] = candidate_shift;
                task.corrected_candidates_read_ids[i] = candidate_read_id;
                task.corrected_candidates[i] = std::move(std::string{candidate_data, candidate_data + candidate_length});

                const bool originalReadContainsN = transFuncData.readStorage->readContainsN(candidate_read_id);

                if(!originalReadContainsN){
                    const char* ptr = &dataArrays.h_candidate_sequences_data[global_candidate_index * dataArrays.encoded_sequence_pitch];
                    const std::string uncorrectedCandidate = get2BitHiLoString((const unsigned int*)ptr, candidate_length);
                    task.corrected_candidate_equals_uncorrected[i] = task.corrected_candidates[i] == uncorrectedCandidate;
                }else{
                    task.corrected_candidate_equals_uncorrected[i] = false;
                }



                //task.corrected_candidates_read_ids.emplace_back(candidate_read_id);
                //task.corrected_candidates.emplace_back(std::move(std::string{candidate_data, candidate_data + candidate_length}));
            }
        }
    }

    batch.setState(BatchState::WriteResults, expectedState);
    cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
    return;
}
#else

    void state_unpackclassicresults_func(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;

        constexpr BatchState expectedState = BatchState::UnpackClassicResults;

        assert(batch.state == expectedState);

        cudaSetDevice(batch.deviceId); CUERR;

        if(!batch.combinedStreams){
            auto& events = batch.events;
            auto& streams = batch.streams;

            cudaEventRecord(events[secondary_stream_finished_event_index], streams[secondary_stream_index]); CUERR;

            cudaStreamWaitEvent(streams[primary_stream_index], events[secondary_stream_finished_event_index], 0); CUERR;
            batch.combinedStreams = true;

            cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
            //std::cerr << "\nbatch " << batch.id << " finished " << nameOf(batch.state) << " " << batch.statesInProgress << "\n";
            return;
        }

        std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;
        cudaError_t errort = cudaEventQuery(events[correction_finished_event_index]);
        if(errort != cudaSuccess){
            std::cout << "error cudaEventQuery\n";
            std::exit(0);
        }
        assert(cudaEventQuery(events[correction_finished_event_index]) == cudaSuccess); CUERR;
        assert(cudaEventQuery(events[result_transfer_finished_event_index]) == cudaSuccess); CUERR;


        assert(transFuncData.correctionOptions.correctionType == CorrectionType::Classic);

        Batch* batchptr = &batch;

        //batch.dataArrays.copyEverythingToHostForDebugging();

        auto unpackAnchors = [batchptr](int begin, int end){
            Batch& batch = *batchptr;
            DataArrays& dataArrays = batch.dataArrays;
            const auto& transFuncData = *batch.transFuncData;

            //std::cerr << "in unpackAnchors " << begin << " - " << end << "\n";

            for(int subject_index = begin; subject_index < end; ++subject_index) {
                auto& task = batch.tasks[subject_index];
                const char* const my_corrected_subject_data = dataArrays.h_corrected_subjects + subject_index * dataArrays.sequence_pitch;
                task.corrected = dataArrays.h_subject_is_corrected[subject_index];
                task.highQualityAlignment = dataArrays.h_is_high_quality_subject[subject_index].hq();

                // if(task.readId == 5383){
                //     dataArrays.printActiveDataOfSubject(subject_index, std::cerr);
                // }

                if(task.corrected) {
                    const int subject_length = dataArrays.h_subject_sequences_lengths[subject_index];
                    task.corrected_subject = std::move(std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length});

                    //task.correctionEqualsOriginal = task.corrected_subject == task.subject_string;

                    const int numUncorrectedPositions = dataArrays.h_num_uncorrected_positions_per_subject[subject_index];
                    if(numUncorrectedPositions > 0){
                        task.uncorrectedPositionsNoConsensus.resize(numUncorrectedPositions);
                        std::copy_n(dataArrays.h_uncorrected_positions_per_subject + subject_index * dataArrays.maximum_sequence_length,
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

        auto unpackcandidates = [batchptr](int begin, int end){
            Batch& batch = *batchptr;
            DataArrays& dataArrays = batch.dataArrays;
            const auto& transFuncData = *batch.transFuncData;

            //std::cerr << "in unpackcandidates " << begin << " - " << end << "\n";

            for(int subject_index = begin; subject_index < end; ++subject_index) {
                auto& task = batch.tasks[subject_index];

                const int n_corrected_candidates = dataArrays.h_num_corrected_candidates[subject_index];
                const char* const my_corrected_candidates_data = dataArrays.h_corrected_candidates
                                                + dataArrays.h_indices_per_subject_prefixsum[subject_index] * dataArrays.sequence_pitch;
                const int* const my_indices_of_corrected_candidates = dataArrays.h_indices_of_corrected_candidates
                                                + dataArrays.h_indices_per_subject_prefixsum[subject_index];


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

                    const read_number candidate_read_id = dataArrays.h_candidate_read_ids[global_candidate_index];

                    bool savingIsOk = false;
                    const std::uint8_t mask = transFuncData.correctionStatusFlagsPerRead[candidate_read_id];
                    if(!(mask & readCorrectedAsHQAnchor)) {
                        savingIsOk = true;
                    }
                    if (savingIsOk) {

                        const int candidate_length = dataArrays.h_candidate_sequences_lengths[global_candidate_index];
                        const int candidate_shift = dataArrays.h_alignment_shifts[global_candidate_index];

                        const char* const candidate_data = my_corrected_candidates_data + i * dataArrays.sequence_pitch;
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
                            const char* ptr = &dataArrays.h_candidate_sequences_data[global_candidate_index * dataArrays.encoded_sequence_pitch];
                            const std::string uncorrectedCandidate = get2BitHiLoString((const unsigned int*)ptr, candidate_length);

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

        auto allChunksFinished = [batchptr](){

            std::array<cudaStream_t, nStreamsPerBatch>& streams = batchptr->streams;

            batchptr->setState(BatchState::WriteResults, expectedState);
            //std::cerr << "\nbatch " << batchptr->id << " finished " << nameOf(batchptr->state) << " " << batchptr->statesInProgress << "\n";
            cudaLaunchHostFunc(streams[primary_stream_index], nextStep, batchptr); CUERR;
        };

        if(!transFuncData.correctionOptions.correctCandidates){
            threadpool.parallelFor(0, int(batch.tasks.size()), [=](auto begin, auto end, auto /*threadId*/){
                unpackAnchors(begin, end);
            });
        }else{
            threadpool.parallelFor(0, int(batch.tasks.size()), [=](auto begin, auto end, auto /*threadId*/){
                unpackAnchors(begin, end);
                unpackcandidates(begin, end);
            });
        }

        allChunksFinished();
    }
#endif


	void state_writeresults_func(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;

        constexpr BatchState expectedState = BatchState::WriteResults;

		assert(batch.state == expectedState);

        cudaSetDevice(batch.deviceId); CUERR;

        if(!batch.combinedStreams){
            auto& events = batch.events;
            auto& streams = batch.streams;

            cudaEventRecord(events[secondary_stream_finished_event_index], streams[secondary_stream_index]); CUERR;

            cudaStreamWaitEvent(streams[primary_stream_index], events[secondary_stream_finished_event_index], 0); CUERR;
            batch.combinedStreams = true;

            cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
            return;
        }

        /*DataArrays& dataArrays = batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = batch.events;

        for(size_t subjectIndex = 0; subjectIndex < batch.tasks.size(); subjectIndex++){
            const auto& task = batch.tasks[subjectIndex];
            if(task.readId == 207){

                cudaDeviceSynchronize(); CUERR;

                cudaMemcpyAsync(dataArrays.h_consensus,
                                dataArrays.d_consensus,
                                dataArrays.d_consensus.sizeInBytes(),
                                D2H,
                                streams[primary_stream_index]);

                cudaMemcpyAsync(dataArrays.h_support,
                                dataArrays.d_support,
                                dataArrays.d_support.sizeInBytes(),
                                D2H,
                                streams[primary_stream_index]);

                cudaMemcpyAsync(dataArrays.h_coverage,
                                dataArrays.d_coverage,
                                dataArrays.d_coverage.sizeInBytes(),
                                D2H,
                                streams[primary_stream_index]);

                cudaMemcpyAsync(dataArrays.h_origWeights,
                                dataArrays.d_origWeights,
                                dataArrays.d_origWeights.sizeInBytes(),
                                D2H,
                                streams[primary_stream_index]);

                cudaMemcpyAsync(dataArrays.h_msa_column_properties,
                                dataArrays.d_msa_column_properties,
                                dataArrays.d_msa_column_properties.sizeInBytes(),
                                D2H,
                                streams[primary_stream_index]);

                cudaMemcpyAsync(dataArrays.h_counts,
                                dataArrays.d_counts,
                                dataArrays.d_counts.sizeInBytes(),
                                D2H,
                                streams[primary_stream_index]);

                cudaMemcpyAsync(dataArrays.h_weights,
                                dataArrays.d_weights,
                                dataArrays.d_weights.sizeInBytes(),
                                D2H,
                                streams[primary_stream_index]);

                cudaMemcpyAsync(dataArrays.msa_data_host,
                            dataArrays.msa_data_device,
                            dataArrays.msa_data_usable_size,
                            D2H,
                            streams[primary_stream_index]); CUERR;
                cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

                //DEBUGGING
                cudaMemcpyAsync(dataArrays.alignment_result_data_host,
                            dataArrays.alignment_result_data_device,
                            dataArrays.alignment_result_data_usable_size,
                            D2H,
                            streams[primary_stream_index]); CUERR;
                cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

                //DEBUGGING
                cudaMemcpyAsync(dataArrays.subject_indices_data_host,
                            dataArrays.subject_indices_data_device,
                            dataArrays.subject_indices_data_usable_size,
                            D2H,
                            streams[primary_stream_index]); CUERR;
                cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

                cudaMemcpyAsync(dataArrays.indices_transfer_data_host,
                            dataArrays.indices_transfer_data_device,
                            dataArrays.indices_transfer_data_usable_size,
                            D2H,
                            streams[primary_stream_index]); CUERR;
                cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

                cudaMemcpyAsync(dataArrays.qualities_transfer_data_host,
                                dataArrays.qualities_transfer_data_device,
                                dataArrays.qualities_transfer_data_usable_size,
                                D2H,
                                streams[primary_stream_index]); CUERR;
                cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

                cudaDeviceSynchronize(); CUERR;

                dataArrays.printActiveDataOfSubject(subjectIndex, std::cerr);

            }
        }*/

        auto function = [tasks = std::move(batch.tasks),
                         transFuncData = &transFuncData,
                         id = batch.id](){
            nvtx::push_range("batch "+std::to_string(id)+" writeresultoutputhread", 4);
            int notCorrectedNoCandidates = 0;
            int notCorrected = 0;

            //write result to file
    		for(std::size_t subject_index = 0; subject_index < tasks.size(); ++subject_index) {

    			const auto& task = tasks[subject_index];
    			//std::cout << task.readId << "result" << std::endl;

    			//std::cout << "finished readId " << task.readId << std::endl;

    			if(task.corrected) {
                    transFuncData->saveCorrectedSequence(task.anchoroutput);
    			}else{
                    if(task.candidate_read_ids.empty()){
                        notCorrectedNoCandidates++;
                    }

                    notCorrected++;

                }

                for(const auto& tmp : task.candidatesoutput){
                    transFuncData->saveCorrectedSequence(tmp);
                }
    		}

            //std::cerr << "not corrected "<< " " << notCorrectedNoCandidates << " " << notCorrected << "/" << tasks.size() << "\n";

            nvtx::pop_range();
        };

		//function();

        batch.outputThread->enqueue(std::move(function));

        batch.setState(BatchState::Finished, expectedState);
        cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
        return;
	}

	void state_writefeatures_func(Batch& batch){

        const auto& transFuncData = *batch.transFuncData;

        constexpr BatchState expectedState = BatchState::WriteFeatures;

        assert(batch.state == expectedState);

        cudaSetDevice(batch.deviceId); CUERR;

        if(!batch.combinedStreams){
            auto& events = batch.events;
            auto& streams = batch.streams;

            cudaEventRecord(events[secondary_stream_finished_event_index], streams[secondary_stream_index]); CUERR;

            cudaStreamWaitEvent(streams[primary_stream_index], events[secondary_stream_finished_event_index], 0); CUERR;
            batch.combinedStreams = true;

            cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
            return;
        }

        DataArrays& dataArrays = batch.dataArrays;
        
        std::vector<SerializedFeature> featuresToSave;

#if 1
		//auto& featurestream = *transFuncData.featurestream;

		for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {

			const auto& task = batch.tasks[subject_index];
			const auto& columnProperties = dataArrays.h_msa_column_properties[subject_index];

			//std::cout << task.readId << "feature" << std::endl;

			const std::size_t msa_weights_pitch_floats = dataArrays.msa_weights_pitch / sizeof(float);
#if 1
			std::vector<MSAFeature> MSAFeatures = extractFeatures(dataArrays.h_consensus + subject_index * dataArrays.msa_pitch,
                                                                    dataArrays.h_support + subject_index * msa_weights_pitch_floats,
                                                                    dataArrays.h_coverage + subject_index * msa_weights_pitch_floats,
                                                                    dataArrays.h_origCoverages + subject_index * msa_weights_pitch_floats,
                                                                    columnProperties.lastColumn_excl,
                                                                    columnProperties.subjectColumnsBegin_incl,
                                                                    columnProperties.subjectColumnsEnd_excl,
                                                                    task.subject_string,
                                                                    transFuncData.correctionOptions.kmerlength, 0.0f,
                                                                    transFuncData.correctionOptions.estimatedCoverage);
#else
            //const size_t msa_weights_pitch_floats = dataarrays.msa_weights_pitch / sizeof(float);
            //const unsigned offset1 = dataArrays.msa_pitch * (subject_index +  dataArrays.h_indices_per_subject_prefixsum[subject_index]);
            //const unsigned offset2 = msa_weights_pitch_floats * (subject_index +  dataArrays.h_indices_per_subject_prefixsum[subject_index]);

            //const char* const my_multiple_sequence_alignment = dataArrays.h_multiple_sequence_alignments + offset1;
            //const float* const my_multiple_sequence_alignment_weight = dataArrays.h_multiple_sequence_alignment_weights + offset2;
            const int msa_rows = 1 + dataArrays.h_indices_per_subject[subject_index];

            const std::size_t countsOffset = subject_index * msa_weights_pitch_floats * 4;
            const std::size_t weightsOffset = subject_index * msa_weights_pitch_floats * 4;
            const int* countsA = &dataArrays.h_counts[countsOffset + 0 * msa_weights_pitch_floats];
            const int* countsC = &dataArrays.h_counts[countsOffset + 1 * msa_weights_pitch_floats];
            const int* countsG = &dataArrays.h_counts[countsOffset + 2 * msa_weights_pitch_floats];
            const int* countsT = &dataArrays.h_counts[countsOffset + 3 * msa_weights_pitch_floats];
            const float* weightsA = &dataArrays.h_weights[weightsOffset + 0 * msa_weights_pitch_floats];
            const float* weightsC = &dataArrays.h_weights[weightsOffset + 1 * msa_weights_pitch_floats];
            const float* weightsG = &dataArrays.h_weights[weightsOffset + 2 * msa_weights_pitch_floats];
            const float* weightsT = &dataArrays.h_weights[weightsOffset + 3 * msa_weights_pitch_floats];

            std::vector<MSAFeature3> MSAFeatures = extractFeatures3_2(
                                        countsA,
                                        countsC,
                                        countsG,
                                        countsT,
                                        weightsA,
                                        weightsC,
                                        weightsG,
                                        weightsT,
                                        msa_rows,
                                        columnProperties.columnsToCheck,
                                        dataArrays.h_consensus + subject_index * dataArrays.msa_pitch,
                                        dataArrays.h_support + subject_index * msa_weights_pitch_floats,
                						dataArrays.h_coverage + subject_index * msa_weights_pitch_floats,
                						dataArrays.h_origCoverages + subject_index * msa_weights_pitch_floats,
                                        columnProperties.subjectColumnsBegin_incl,
                						columnProperties.subjectColumnsEnd_excl,
                                        task.subject_string,
                                        transFuncData.correctionOptions.estimatedCoverage);

#endif
            
            
			for(const auto& msafeature : MSAFeatures) {
                std::stringstream ss;
                ss << msafeature;
                featuresToSave.emplace_back(task.readId, msafeature.position, msafeature.consensus, ss.str());
				//featurestream << task.readId << '\t' << msafeature.position << '\t' << msafeature.consensus << '\n';
				//featurestream << msafeature << '\n';
			}
		}
#endif

        auto func = [featuresToSave = std::move(featuresToSave), t = batch.transFuncData](){
            auto& featurestream = *(t->featurestream);
            for(const auto& f : featuresToSave){
                featurestream << f.readId << '\t' << f.position << '\t' << f.consensus << '\n';
				featurestream << f.featureString << '\n';
            }
        };

        batch.outputThread->enqueue(std::move(func));

        batch.setState(BatchState::Finished, expectedState);
        cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
        return;
	}

	void state_finished_func(Batch& batch){

        auto& transFuncData = *batch.transFuncData;

		assert(batch.state == BatchState::Finished);

        cudaSetDevice(batch.deviceId); CUERR;

        if(!(transFuncData.readIdGenerator->empty())) {
            //there are reads left to correct, so this batch can be reused again
            batch.reset();
            cudaLaunchHostFunc(batch.streams[primary_stream_index], nextStep, &batch); CUERR;
        }else{
            transFuncData.isFinishedCV.notify_one();
        }
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

      std::ofstream outputstream;
      std::unique_ptr<SequenceFileWriter> writer;

      //if candidate correction is not enabled, it is possible to write directly into the result file
      // if(!correctionOptions.correctCandidates){
      //     //writer = std::move(makeSequenceWriter(fileOptions.outputfile, FileFormat::FASTQGZ));
      //     outputstream = std::move(std::ofstream(fileOptions.outputfile));
      //     if(!outputstream){
      //         throw std::runtime_error("Could not open output file " + tmpfiles[0]);
      //     }
      // }else{
          outputstream = std::move(std::ofstream(tmpfiles[0]));
          if(!outputstream){
              throw std::runtime_error("Could not open output file " + tmpfiles[0]);
          }
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

      std::array<Batch, nParallelBatches> batches;
      std::array<Batch*, nParallelBatches> batchPointers;
      std::array<BackgroundThread, nParallelBatches> batchExecutors;

      BackgroundThread outputThread;

      int deviceIdIndex = 0;

      for(int i = 0; i < nParallelBatches; i++) {
          const int deviceId = deviceIds[deviceIdIndex];

          cudaSetDevice(deviceId); CUERR;

          DataArrays dataArrays(deviceId);

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
          batches[i].executor = &batchExecutors[i];
          batchPointers[i] = &batches[i];

          deviceIdIndex = (deviceIdIndex + 1) % deviceIds.size();
      }

#ifndef DO_PROFILE
        cpu::RangeGenerator<read_number> readIdGenerator(sequenceFileProperties.nReads);
#else
        cpu::RangeGenerator<read_number> readIdGenerator(num_reads_to_profile);
#endif



        // constexpr int maxCachedResults = 2000000;
        // static_assert(maxCachedResults > 0, "");
        //
        // std::vector<TempCorrectedSequence> cachedResults;
        //
        // cachedResults.reserve(maxCachedResults);
        //
        // std::vector<std::string> usedOutputfileNames;//{fileOptions.outputfile + "_tmp"};
        //
        // auto sortByReadId = [](const auto& l, const auto& r){
        //     return l.readId < r.readId;
        // };
        //
        //
        // //std::set<TempCorrectedSequence, decltype(sortByReadId)> cachedResultsSet(sortByReadId);
        //
        // auto flushCachedResults = [&](){
        //     std::vector<int> indices(cachedResults.size());
        //     std::iota(indices.begin(), indices.end(), 0);
        //     std::sort(indices.begin(), indices.end(), [&](int l, int r){
        //         return sortByReadId(cachedResults[l], cachedResults[r]);
        //     });
        //
        //     std::string filename = fileOptions.outputfile + "_tmp" + std::to_string(usedOutputfileNames.size());
        //     std::ofstream outputstream(filename);
        //     if(!outputstream){
        //         throw std::runtime_error("Could not open output file " + filename);
        //     }
        //
        //     usedOutputfileNames.emplace_back(std::move(filename));
        //
        //     for(int i : indices){
        //         outputstream << cachedResults[i] << '\n';
        //     }
        //
        //     outputstream.flush();
        //     outputstream.close();
        //
        //     cachedResults.clear();
        //
        //     // std::sort(cachedResults.begin(), cachedResults.end(), sortByReadId);
        //     // std::copy(cachedResults.begin(), cachedResults.end(), std::ostream_iterator<TempCorrectedSequence>(outputstream, "\n"));
        //     // cachedResults.clear();
        //
        //     //std::copy(cachedResultsSet.begin(), cachedResultsSet.end(), std::ostream_iterator<TempCorrectedSequence>(outputstream, "\n"));
        //     //cachedResultsSet.clear();
        // };


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

      transFuncData.saveCorrectedSequence = [&](const TempCorrectedSequence& tmp){
          // auto isValidSequence = [](const std::string& s){
          //     return std::all_of(s.begin(), s.end(), [](char c){
          //         return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
          //     });
          // };
          //
          // assert(isValidSequence(tmp.sequence));
          //
          // cachedResults.emplace_back(std::move(tmp));
          // //auto insertposition = std::upper_bound(cachedResults.begin(), cachedResults.end(), tmp, sortByReadId);
          // //cachedResults.insert(insertposition, std::move(tmp));
          //
          // //cachedResultsSet.emplace(std::move(tmp));
          //
          // if(int(cachedResults.size()) == maxCachedResults){
          //     flushCachedResults();
          // }
          //std::cout << tmp << '\n';
          //std::unique_lock<std::mutex> l(outputstreammutex);
          if(!(tmp.hq && tmp.useEdits && tmp.edits.empty())){
            outputstream << tmp << '\n';
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

      NN_Correction_Classifier_Base nnClassifierBase;
      NN_Correction_Classifier nnClassifier;
      if(correctionOptions.correctionType == CorrectionType::Convnet){
          nnClassifierBase = std::move(NN_Correction_Classifier_Base{"./nn_sources", fileOptions.nnmodelfilename});
          nnClassifier = std::move(NN_Correction_Classifier{&nnClassifierBase});
      }

      // BEGIN CORRECTION

    for(auto& executor : batchExecutors){
        executor.start();
    }

      outputThread.start();


      #ifdef DO_PROFILE
          cudaProfilerStart();
      #endif

        for(int i = 0; i < nParallelBatches; ++i) {
            nextStep(&batches[i]);
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

        threadpool.wait();

        for(auto& executor : batchExecutors){
            executor.stopThread(BackgroundThread::StopType::FinishAndStop);
        }

        outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);


      //flushCachedResults();
      outputstream.flush();
      featurestream.flush();

      #ifdef DO_PROFILE
          cudaProfilerStop();
      #endif

      runtime = std::chrono::system_clock::now() - timepoint_begin;
      if(runtimeOptions.showProgress){
          printf("Processed %10lu of %10lu reads (Runtime: %03d:%02d:%02d)\n",
                  sequenceFileProperties.nReads, sequenceFileProperties.nReads,
                  int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
                  int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count()) % 60,
                  int(runtime.count()) % 60);
      }

      for(const auto& batch : batches){
          std::cout << "size elements: " << batch.dataArrays.h_candidate_read_ids.size() << ", capacity elements " << batch.dataArrays.h_candidate_read_ids.capacity() << std::endl;
      
        }

        for(const auto& batch : batches){
            std::cerr << "Memory usage: \n";
            batch.dataArrays.printMemoryUsage();
            std::cerr << "Total: " << batch.dataArrays.getMemoryUsageInBytes() << " bytes\n";
            std::cerr << '\n';
        }

      for(auto& batch : batches){
          cudaSetDevice(batch.deviceId); CUERR;

          batch.dataArrays.reset();

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
                                tmpfiles, 
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




#endif
