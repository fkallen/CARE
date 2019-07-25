#if 1

#include <gpu/correct_gpu.hpp>
#include <gpu/distributedreadstorage.hpp>
#include <gpu/qualityscoreweights.hpp>
#include <gpu/nvvptimelinemarkers.hpp>
#include <gpu/kernels.hpp>
#include <gpu/dataarrays.hpp>

#include <config.hpp>
#include <qualityscoreweights.hpp>
#include <tasktiming.hpp>
#include <sequence.hpp>
#include <featureextractor.hpp>
#include <forestclassifier.hpp>
#include <nn_classifier.hpp>
#include <minhasher.hpp>
#include <options.hpp>
#include <candidatedistribution.hpp>
#include <sequencefileio.hpp>
#include <rangegenerator.hpp>

#include <hpc_helpers.cuh>

#include <cuda_profiler_api.h>

#include <memory>
#include <sstream>
#include <cstdlib>
#include <thread>
#include <vector>
#include <string>
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
#include <cub/util_allocator.cuh>

#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>

//#define CARE_GPU_DEBUG
//#define CARE_GPU_DEBUG_MEMCOPY
//#define CARE_GPU_DEBUG_PRINT_ARRAYS
//#define CARE_GPU_DEBUG_PRINT_MSA

#define MSA_IMPLICIT

//#define REARRANGE_INDICES
#define USE_MSA_MINIMIZATION

#define USE_WAIT_FLAGS

//#define DO_PROFILE

#ifdef DO_PROFILE
    constexpr size_t num_reads_to_profile = 100000;
#endif


namespace care{
namespace gpu{

    constexpr int nParallelBatches = 4;
    constexpr int sideBatchStepsPerWaitIter = 1;

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
    constexpr int nEventsPerBatch = 10;


    struct CorrectionTask {
        CorrectionTask(){
        }

        CorrectionTask(read_number readId)
            :   active(true),
            corrected(false),
            readId(readId)
        {
        }

        CorrectionTask(const CorrectionTask& other)
            : active(other.active),
            corrected(other.corrected),
            readId(other.readId),
            subject_string(other.subject_string),
            candidate_read_ids(other.candidate_read_ids),
            corrected_subject(other.corrected_subject),
            corrected_candidates(other.corrected_candidates),
            corrected_candidates_read_ids(other.corrected_candidates_read_ids){
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
            swap(l.readId, r.readId);
            swap(l.subject_string, r.subject_string);
            swap(l.candidate_read_ids, r.candidate_read_ids);
            swap(l.corrected_subject, r.corrected_subject);
            swap(l.corrected_candidates_read_ids, r.corrected_candidates_read_ids);
        }

        bool active;
        bool corrected;
        read_number readId;

        std::vector<read_number> candidate_read_ids;

        std::string subject_string;

        std::string corrected_subject;
        std::vector<std::string> corrected_candidates;
        std::vector<read_number> corrected_candidates_read_ids;
    };

    enum class BatchState : int{
		Unprepared,
		CopyReads,
		StartAlignment,
        RearrangeIndices,
		CopyQualities,
		BuildMSA,
        ImproveMSA,
		StartClassicCorrection,
		StartForestCorrection,
        StartConvnetCorrection,
		UnpackClassicResults,
		WriteResults,
		WriteFeatures,
		Finished,
		Aborted,
	};

    static constexpr int nBatchStates = static_cast<int>(BatchState::Aborted)+1;

    struct Batch {
        struct WaitCallbackData{
            Batch* b{};
            int index = -1;
            WaitCallbackData(){}
            WaitCallbackData(Batch* ptr, int i) : b(ptr), index(i){}
        };
		std::vector<CorrectionTask> tasks;
		int initialNumberOfCandidates = 0;
		BatchState state = BatchState::Unprepared;

		int copiedTasks = 0;         // used if state == CandidatesPresent
		int copiedCandidates = 0;         // used if state == CandidatesPresent

        int copiedSubjects = 0;
        bool handledReadIds = false;


		std::vector<read_number> allReadIdsOfTasks;
		std::vector<read_number> allReadIdsOfTasks_tmp;
		std::vector<char> collectedCandidateReads;
		int numsortedCandidateIds = 0;
		int numsortedCandidateIdTasks = 0;

		DataArrays* dataArrays;

        bool doImproveMSA = false;
        int numMinimizations = 0;
        int previousNumIndices = 0;

		std::array<cudaStream_t, nStreamsPerBatch>* streams;
		std::array<cudaEvent_t, nEventsPerBatch>* events;

        std::array<std::atomic_int, nBatchStates> waitCounts{};
        int activeWaitIndex = 0;
        //std::vector<std::unique_ptr<WaitCallbackData>> callbackDataList;

        int id = -1;

		KernelLaunchHandle kernelLaunchHandle;

        DistributedReadStorage::GatherHandleSequences subjectSequenceGatherHandle2;
        DistributedReadStorage::GatherHandleSequences candidateSequenceGatherHandle2;
        DistributedReadStorage::GatherHandleLengths subjectLengthGatherHandle2;
        DistributedReadStorage::GatherHandleLengths candidateLengthGatherHandle2;
        DistributedReadStorage::GatherHandleQualities subjectQualitiesGatherHandle2;
        DistributedReadStorage::GatherHandleQualities candidateQualitiesGatherHandle2;

        bool isWaiting() const{
        #ifdef USE_WAIT_FLAGS
            return 0 != waitCounts[activeWaitIndex].load();
        #else
            return cudaEventQuery((*events)[activeWaitIndex]) == cudaErrorNotReady;
        #endif
        }

        void addWaitSignal(BatchState state, cudaStream_t stream){
            const int wait_index = static_cast<int>(state);
            waitCounts[wait_index]++;

            //std::cout << "batch " << id << ". wait_index " << wait_index << ", count increased to " << waitCounts[wait_index] << std::endl;

            #define handlethis(s) {\
                auto waitsuccessfunc = [](void* batch){ \
                    Batch* b = static_cast<Batch*>(batch); \
                    int old = b->waitCounts[static_cast<int>((s))]--; \
                }; \
                cudaLaunchHostFunc(stream, waitsuccessfunc, (void*)this); CUERR; \
            }

            #define mycase(s) case (s): handlethis((s)); break;

            //assert(old > 0);
            //std::cout << "batch " << b->id << ". wait_index " << static_cast<int>((s)) << ", count decreased to " << b->waitCounts[static_cast<int>((s))] << std::endl;

            switch(state) {
            mycase(BatchState::Unprepared)
            mycase(BatchState::CopyReads)
            mycase(BatchState::StartAlignment)
            mycase(BatchState::RearrangeIndices)
            mycase(BatchState::CopyQualities)
            mycase(BatchState::BuildMSA)
            mycase(BatchState::ImproveMSA)
            mycase(BatchState::StartClassicCorrection)
            mycase(BatchState::StartForestCorrection)
            mycase(BatchState::StartConvnetCorrection)
            mycase(BatchState::UnpackClassicResults)
            mycase(BatchState::WriteResults)
            mycase(BatchState::WriteFeatures)
            mycase(BatchState::Finished)
            mycase(BatchState::Aborted)
            default: assert(false);
            }

            #undef mycase
            #undef handlethis
        }

		void reset(){
            tasks.clear();
    		allReadIdsOfTasks.clear();
    		allReadIdsOfTasks_tmp.clear();
    		collectedCandidateReads.clear();

    		initialNumberOfCandidates = 0;
    		state = BatchState::Unprepared;
    		copiedTasks = 0;
    		copiedCandidates = 0;
            copiedSubjects = 0;
            handledReadIds = false;

    		numsortedCandidateIds = 0;
    		numsortedCandidateIdTasks = 0;

            //assert(std::all_of(waitCounts.begin(), waitCounts.end(), [](const auto& i){return i == 0;}));

            activeWaitIndex = 0;

            doImproveMSA = false;
            numMinimizations = 0;
            previousNumIndices = 0;
        }

        void waitUntilAllCallbacksFinished() const{
            assert(std::any_of(waitCounts.begin(), waitCounts.end(), [](const auto& i){return i >= 0;}));

            while(std::any_of(waitCounts.begin(), waitCounts.end(), [](const auto& i){return i > 0;})){
                ;
            }
        }
	};

	struct AdvanceResult {
		BatchState oldState = BatchState::Unprepared;
		BatchState newState = BatchState::Unprepared;
		bool noProgressBlocking = false;
		bool noProgressLaunching = false;
	};

    struct TransitionFunctionData {
        int deviceId;
		cpu::RangeGenerator<read_number>* readIdGenerator;
		std::vector<read_number>* readIdBuffer;
        std::vector<CorrectionTask>* tmptasksBuffer;
		const Minhasher* minhasher;
        const DistributedReadStorage* readStorage;
		std::mutex* locksForProcessedFlags;
		std::size_t nLocksForProcessedFlags;
		CorrectionOptions correctionOptions;
        GoodAlignmentProperties goodAlignmentProperties;
        SequenceFileProperties sequenceFileProperties;
        RuntimeOptions runtimeOptions;
        MinhashOptions minhashOptions;
        AlignmentOptions alignmentOptions;
        FileOptions fileOptions;
		std::vector<char>* readIsCorrectedVector;
		std::ofstream* featurestream;
		std::function<void(const read_number, const std::string&)> write_read_to_stream;
		std::function<void(const read_number)> lock;
		std::function<void(const read_number)> unlock;

        ForestClassifier fc;// = ForestClassifier{"./forests/testforest.so"};
        NN_Correction_Classifier nnClassifier;

        std::vector<char>* sequenceDataBuffer;
        std::vector<int>* sequenceLengthsBuffer;
	};

    BatchState state_unprepared_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

    BatchState state_unprepared_func2(Batch& batch,
                bool isPausable,
                TransitionFunctionData& transFuncData);

	BatchState state_copyreads_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

	BatchState state_startalignment_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

    BatchState state_rearrangeindices_func(Batch& batch,
            bool isPausable,
            TransitionFunctionData& transFuncData);

	BatchState state_copyqualities_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

	BatchState state_buildmsa_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

    BatchState state_improvemsa_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

	BatchState state_startclassiccorrection_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

	BatchState state_startforestcorrection_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

    BatchState state_startconvnetcorrection_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

	BatchState state_unpackclassicresults_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

	BatchState state_writeresults_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

	BatchState state_writefeatures_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

	BatchState state_finished_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

	BatchState state_aborted_func(Batch& batch,
				bool isPausable,
				TransitionFunctionData& transFuncData);

    using FuncTableEntry = BatchState (*)(Batch&,
				bool,
				TransitionFunctionData&);

    std::string nameOf(const BatchState& state){
		switch(state) {
		case BatchState::Unprepared: return "Unprepared";
		case BatchState::CopyReads: return "CopyReads";
		case BatchState::StartAlignment: return "StartAlignment";
        case BatchState::RearrangeIndices: return "RearrangeIndices";
		case BatchState::CopyQualities: return "CopyQualities";
		case BatchState::BuildMSA: return "BuildMSA";
        case BatchState::ImproveMSA: return "ImproveMSA";
		case BatchState::StartClassicCorrection: return "StartClassicCorrection";
		case BatchState::StartForestCorrection: return "StartForestCorrection";
        case BatchState::StartConvnetCorrection: return "StartConvnetCorrection";
		case BatchState::UnpackClassicResults: return "UnpackClassicResults";
		case BatchState::WriteResults: return "WriteResults";
		case BatchState::WriteFeatures: return "WriteFeatures";
		case BatchState::Finished: return "Finished";
		case BatchState::Aborted: return "Aborted";
		default: assert(false); return "None";
		}
	}

    std::unordered_map<BatchState, FuncTableEntry>
    makeTransitionFunctionTable(){
        std::unordered_map<BatchState, FuncTableEntry> transitionFunctionTable;

		transitionFunctionTable[BatchState::Unprepared] = state_unprepared_func;
		transitionFunctionTable[BatchState::CopyReads] = state_copyreads_func;
		transitionFunctionTable[BatchState::StartAlignment] = state_startalignment_func;
        transitionFunctionTable[BatchState::RearrangeIndices] = state_rearrangeindices_func;
		transitionFunctionTable[BatchState::CopyQualities] = state_copyqualities_func;
		transitionFunctionTable[BatchState::BuildMSA] = state_buildmsa_func;
        transitionFunctionTable[BatchState::ImproveMSA] = state_improvemsa_func;
		transitionFunctionTable[BatchState::StartClassicCorrection] = state_startclassiccorrection_func;
		transitionFunctionTable[BatchState::StartForestCorrection] = state_startforestcorrection_func;
        transitionFunctionTable[BatchState::StartConvnetCorrection] = state_startconvnetcorrection_func;
		transitionFunctionTable[BatchState::UnpackClassicResults] = state_unpackclassicresults_func;
		transitionFunctionTable[BatchState::WriteResults] = state_writeresults_func;
		transitionFunctionTable[BatchState::WriteFeatures] = state_writefeatures_func;
		transitionFunctionTable[BatchState::Finished] = state_finished_func;
		transitionFunctionTable[BatchState::Aborted] = state_aborted_func;

        return transitionFunctionTable;
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


    // Caching allocator for device memory
    cub::CachingDeviceAllocator cubCachingAllocator(
                8,                                                  ///< Geometric growth factor for bin-sizes
                3,                                                  ///< Minimum bin (default is bin_growth ^ 1)
                cub::CachingDeviceAllocator::INVALID_BIN,           ///< Maximum bin (default is no max bin)
                cub::CachingDeviceAllocator::INVALID_SIZE,          ///< Maximum aggregate cached bytes per device (default is no limit)
                true,                                               ///< Whether or not to skip a call to \p FreeAllCached() when the destructor is called (default is to deallocate)
                false);                                             ///< Whether or not to print (de)allocation events to stdout (default is no stderr output)


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


    BatchState state_unprepared_func(Batch& batch,
                          bool isPausable,
                TransitionFunctionData& transFuncData){

        assert(batch.state == BatchState::Unprepared);
        assert((batch.initialNumberOfCandidates == 0 && batch.tasks.empty()) || batch.initialNumberOfCandidates > 0);

        auto identity = [](auto i){return i;};

        const int hits_per_candidate = transFuncData.correctionOptions.hits_per_candidate;

        DataArrays& dataArrays = *batch.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
        //std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

        std::vector<read_number>* readIdBuffer = transFuncData.readIdBuffer;
        std::vector<CorrectionTask>* tmptasksBuffer = transFuncData.tmptasksBuffer;
        std::vector<char>* sequenceDataBuffer = transFuncData.sequenceDataBuffer;
        std::vector<int>* sequenceLengthsBuffer = transFuncData.sequenceLengthsBuffer;

        auto erase_from_range = [](auto begin, auto end, auto position_to_erase){
                        auto copybegin = position_to_erase;
                        std::advance(copybegin, 1);
                        return std::copy(copybegin, end, position_to_erase);
                    };

        const auto& minhasher = transFuncData.minhasher;

        const size_t maxNumResultsPerMapQuery = transFuncData.correctionOptions.estimatedCoverage * 2.5;

        constexpr int num_simultaneous_tasks = 64;
        const size_t seqpitch = getEncodedNumInts2BitHiLo(transFuncData.sequenceFileProperties.maxSequenceLength) * sizeof(int);

        std::vector<CorrectionTask> tmptasks;
        //std::vector<bool> tmpokflags(num_simultaneous_tasks);

        while(batch.initialNumberOfCandidates < transFuncData.correctionOptions.batchsize
              && !(transFuncData.readIdGenerator->empty() && readIdBuffer->empty() && tmptasksBuffer->empty())) {



            if(tmptasksBuffer->empty()){

                if(readIdBuffer->empty()){
                    *readIdBuffer = transFuncData.readIdGenerator->next_n(1000);

                    sequenceDataBuffer->resize(readIdBuffer->size() * seqpitch);
                    sequenceLengthsBuffer->resize(readIdBuffer->size());

                    transFuncData.readStorage->gatherSequenceDataToHostBuffer(
                                                batch.candidateSequenceGatherHandle2,
                                                sequenceDataBuffer->data(),
                                                seqpitch,
                                                readIdBuffer->data(),
                                                readIdBuffer->size(),
                                                transFuncData.runtimeOptions.nCorrectorThreads);

                    transFuncData.readStorage->gatherSequenceLengthsToHostBuffer(
                                                batch.candidateLengthGatherHandle2,
                                                sequenceLengthsBuffer->data(),
                                                readIdBuffer->data(),
                                                readIdBuffer->size(),
                                                transFuncData.runtimeOptions.nCorrectorThreads);
                }

                if(readIdBuffer->empty())
                    continue;

                const int readIdsInBuffer = readIdBuffer->size();
                const int max_tmp_tasks = std::min(readIdsInBuffer, num_simultaneous_tasks);

                tmptasks.resize(max_tmp_tasks);

                #pragma omp parallel for
                for(int tmptaskindex = 0; tmptaskindex < max_tmp_tasks; tmptaskindex++){
                    auto& task = tmptasks[tmptaskindex];

                    const read_number readId = (*readIdBuffer)[tmptaskindex];
                    task = CorrectionTask(readId);

                    bool ok = false;
                    if ((*transFuncData.readIsCorrectedVector)[readId] == 0) {
                        ok = true;
                    }

                    if(ok){
                        const char* sequenceptr = sequenceDataBuffer->data() + tmptaskindex * seqpitch;
                        const int sequencelength = (*sequenceLengthsBuffer)[tmptaskindex];

                        task.subject_string = get2BitHiLoString((const unsigned int*)sequenceptr, sequencelength);

                        task.candidate_read_ids = minhasher->getCandidates(task.subject_string,
                                                                            hits_per_candidate,
                                                                            transFuncData.runtimeOptions.max_candidates,
                                                                            maxNumResultsPerMapQuery);

                        auto readIdPos = std::lower_bound(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.readId);

                        if(readIdPos != task.candidate_read_ids.end() && *readIdPos == task.readId) {
                            task.candidate_read_ids.erase(readIdPos);
                        }

                        std::size_t myNumCandidates = task.candidate_read_ids.size();

                        assert(myNumCandidates <= std::size_t(transFuncData.runtimeOptions.max_candidates));

                        if(myNumCandidates == 0) {
                            task.active = false;
                        }
                    }else{
                        task.active = false;
                    }
                }

                readIdBuffer->erase(readIdBuffer->begin(), readIdBuffer->begin() + max_tmp_tasks);
                sequenceDataBuffer->erase(sequenceDataBuffer->begin(), sequenceDataBuffer->begin() + max_tmp_tasks * seqpitch);
                sequenceLengthsBuffer->erase(sequenceLengthsBuffer->begin(), sequenceLengthsBuffer->begin() + max_tmp_tasks);

                std::swap(*tmptasksBuffer, tmptasks);

                //only perform one iteration if pausable
                if(isPausable)
                    break;
            }

            while(batch.initialNumberOfCandidates < transFuncData.correctionOptions.batchsize
                    && !tmptasksBuffer->empty()){

                auto& task = tmptasksBuffer->back();

                if(task.active){

                    const read_number id = task.readId;
                    bool ok = false;
                    transFuncData.lock(id);
                    if ((*transFuncData.readIsCorrectedVector)[id] == 0) {
                        (*transFuncData.readIsCorrectedVector)[id] = 1;
                        ok = true;
                    }else{
                    }
                    transFuncData.unlock(id);

                    if(ok){
                        const size_t myNumCandidates = task.candidate_read_ids.size();

                        batch.tasks.emplace_back(task);
                        batch.initialNumberOfCandidates += int(myNumCandidates);

                        #ifdef CARE_GPU_DEBUG

                        if(task.readId == 999013){
                            std::cout << "add task" << std::endl;
                        }

                        #endif
                    }
                }

                tmptasksBuffer->pop_back();
            }

            //only perform one iteration if pausable
            if(isPausable)
                break;
        }


        if(batch.initialNumberOfCandidates < transFuncData.correctionOptions.batchsize
            && !(transFuncData.readIdGenerator->empty() && readIdBuffer->empty())) {
            //still more read ids to add

            return BatchState::Unprepared;
        }else{

            if(batch.initialNumberOfCandidates == 0) {
                return BatchState::Aborted;
            }else{

                assert(batch.initialNumberOfCandidates < transFuncData.correctionOptions.batchsize + transFuncData.runtimeOptions.max_candidates);

                //allocate data arrays

                dataArrays.set_problem_dimensions(int(batch.tasks.size()),
                            batch.initialNumberOfCandidates,
                            transFuncData.sequenceFileProperties.maxSequenceLength,
                            sizeof(unsigned int) * getEncodedNumInts2BitHiLo(transFuncData.sequenceFileProperties.maxSequenceLength),
                            transFuncData.goodAlignmentProperties.min_overlap,
                            transFuncData.goodAlignmentProperties.min_overlap_ratio,
                            transFuncData.correctionOptions.useQualityScores); CUERR;

                //std::cout << "batch.initialNumberOfCandidates " << batch.initialNumberOfCandidates << std::endl;

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
                            batch.initialNumberOfCandidates,
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
                                                		batch.initialNumberOfCandidates,
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

                batch.initialNumberOfCandidates = 0;

                return BatchState::CopyReads;
            }
        }
    }


    BatchState state_copyreads_func(Batch& batch,
                bool isPausable,
                TransitionFunctionData& transFuncData){

        assert(batch.state == BatchState::CopyReads);
        assert(batch.copiedTasks <= int(batch.tasks.size()));

        DataArrays& dataArrays = *batch.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;



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

        transFuncData.readStorage->gatherSequenceLengthsToGpuBufferAsync(batch.subjectLengthGatherHandle2,
                                                                     dataArrays.d_subject_sequences_lengths,
                                                                     dataArrays.h_subject_read_ids,
                                                                     dataArrays.d_subject_read_ids,
                                                                     dataArrays.n_subjects,
                                                                     transFuncData.deviceId,
                                                                     streams[primary_stream_index],
                                                                     transFuncData.runtimeOptions.nCorrectorThreads);

        transFuncData.readStorage->gatherSequenceLengthsToGpuBufferAsync(batch.candidateLengthGatherHandle2,
                                                                  dataArrays.d_candidate_sequences_lengths,
                                                                  dataArrays.h_candidate_read_ids,
                                                                  dataArrays.d_candidate_read_ids,
                                                                  dataArrays.n_queries,
                                                                  transFuncData.deviceId,
                                                                  streams[primary_stream_index],
                                                                  transFuncData.runtimeOptions.nCorrectorThreads);

        transFuncData.readStorage->gatherSequenceDataToGpuBufferAsync(batch.subjectSequenceGatherHandle2,
                                                                         dataArrays.d_subject_sequences_data,
                                                                         dataArrays.encoded_sequence_pitch,
                                                                         dataArrays.h_subject_read_ids,
                                                                         dataArrays.d_subject_read_ids,
                                                                         dataArrays.n_subjects,
                                                                         transFuncData.deviceId,
                                                                         streams[primary_stream_index],
                                                                         transFuncData.runtimeOptions.nCorrectorThreads);

        transFuncData.readStorage->gatherSequenceDataToGpuBufferAsync(batch.candidateSequenceGatherHandle2,
                                                                          dataArrays.d_candidate_sequences_data,
                                                                          dataArrays.encoded_sequence_pitch,
                                                                          dataArrays.h_candidate_read_ids,
                                                                          dataArrays.d_candidate_read_ids,
                                                                          dataArrays.n_queries,
                                                                          transFuncData.deviceId,
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

        return BatchState::StartAlignment;
    }



	BatchState state_startalignment_func(Batch& batch,
												bool isPausable,
				TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::StartAlignment);

		DataArrays& dataArrays = *batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

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
					batch.kernelLaunchHandle);

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
#ifdef USE_WAIT_FLAGS
        batch.addWaitSignal(BatchState::RearrangeIndices, streams[primary_stream_index]);
#endif
        cudaEventRecord(events[num_indices_transfered_event_index], streams[primary_stream_index]); CUERR;

        return BatchState::RearrangeIndices;
	}

    BatchState state_rearrangeindices_func(Batch& batch,
												bool isPausable,
				TransitionFunctionData& transFuncData){

        DataArrays& dataArrays = *batch.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

#ifdef REARRANGE_INDICES

        constexpr BatchState expectedState = BatchState::RearrangeIndices;
#ifdef USE_WAIT_FLAGS
        constexpr int wait_index = static_cast<int>(expectedState);
#endif
        assert(batch.state == expectedState);


#ifdef USE_WAIT_FLAGS
        if(batch.waitCounts[wait_index] != 0){
            batch.activeWaitIndex = wait_index;
            return expectedState;
        }
#else
        cudaError_t status = cudaEventQuery(events[num_indices_transfered_event_index]); CUERR;
        if(status == cudaErrorNotReady){
            batch.activeWaitIndex = num_indices_transfered_event_index;
            return expectedState;
        }
#endif

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

        assert(cudaSuccess == cudaEventQuery(events[indices_transfer_finished_event_index])); CUERR;

        cudaEventRecord(events[indices_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;

#ifdef USE_WAIT_FLAGS
        batch.addWaitSignal(BatchState::BuildMSA, streams[secondary_stream_index]);
        batch.addWaitSignal(BatchState::CopyQualities, streams[secondary_stream_index]);
        batch.addWaitSignal(BatchState::UnpackClassicResults, streams[secondary_stream_index]);
#endif

        if(transFuncData.correctionOptions.useQualityScores) {
            return BatchState::CopyQualities;
        }else{
            return BatchState::BuildMSA;
        }

    }


    BatchState state_copyqualities_func(Batch& batch,
												bool isPausable,
				TransitionFunctionData& transFuncData){

        constexpr BatchState expectedState = BatchState::CopyQualities;
#ifdef USE_WAIT_FLAGS
        constexpr int wait_index = static_cast<int>(expectedState);
#endif
		assert(batch.state == expectedState);

        std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

#ifdef USE_WAIT_FLAGS
        if(batch.waitCounts[wait_index] != 0){
            batch.activeWaitIndex = wait_index;
            return expectedState;
        }
#else
        cudaError_t status = cudaEventQuery(events[indices_transfer_finished_event_index]); CUERR;
        if(status == cudaErrorNotReady){
            batch.activeWaitIndex = indices_transfer_finished_event_index;
            return expectedState;
        }
#endif




        DataArrays& dataArrays = *batch.dataArrays;

        //if there are no good candidates, clean up batch and discard reads
        if(dataArrays.h_num_indices[0] == 0){
            return BatchState::WriteResults;
        }

		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;

        const auto* gpuReadStorage = transFuncData.readStorage;

		if(transFuncData.correctionOptions.useQualityScores) {

            gpuReadStorage->gatherQualitiesToGpuBufferAsync(batch.subjectQualitiesGatherHandle2,
                                                              dataArrays.d_subject_qualities,
                                                              dataArrays.quality_pitch,
                                                              dataArrays.h_subject_read_ids,
                                                              dataArrays.d_subject_read_ids,
                                                              dataArrays.n_subjects,
                                                              transFuncData.deviceId,
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
                                                              transFuncData.deviceId,
                                                              streams[primary_stream_index],
                                                              transFuncData.runtimeOptions.nCorrectorThreads);

            cubCachingAllocator.DeviceFree(d_tmp_read_ids); CUERR;

            assert(cudaSuccess == cudaEventQuery(events[quality_transfer_finished_event_index])); CUERR;

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

            return BatchState::BuildMSA;


        }else{
            return BatchState::BuildMSA;
        }
	}


    BatchState state_buildmsa_func(Batch& batch,
												bool isPausable,
				TransitionFunctionData& transFuncData){

        constexpr BatchState expectedState = BatchState::BuildMSA;

#ifdef USE_WAIT_FLAGS
        constexpr int wait_index = static_cast<int>(expectedState);
#endif

        assert(batch.state == expectedState);

        std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

#ifdef USE_WAIT_FLAGS
        if(batch.waitCounts[wait_index] != 0){
            batch.activeWaitIndex = wait_index;
            return expectedState;
        }
#else
        cudaError_t status = cudaEventQuery(events[num_indices_transfered_event_index]); CUERR;
        if(status == cudaErrorNotReady){
            batch.activeWaitIndex = num_indices_transfered_event_index;
            return expectedState;
        }
#endif

        DataArrays& dataArrays = *batch.dataArrays;

        //if there are no good candidates, clean up batch and discard reads
        if(dataArrays.h_num_indices[0] == 0){
            std::cerr << "buildmsa *h_num_indices = " << dataArrays.h_num_indices[0] << '\n';
            return BatchState::WriteResults;
        }

        if(transFuncData.correctionOptions.useQualityScores){
		     cudaStreamWaitEvent(streams[primary_stream_index], events[quality_transfer_finished_event_index], 0); CUERR;
        }

		const float desiredAlignmentMaxErrorRate = transFuncData.goodAlignmentProperties.maxErrorRate;

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

        //At this point the msa is built
        cudaEventRecord(events[msa_build_finished_event_index], streams[primary_stream_index]); CUERR;

        return BatchState::ImproveMSA;
	}





    BatchState state_improvemsa_func(Batch& batch,
                          bool isPausable,
                TransitionFunctionData& transFuncData){

        constexpr BatchState expectedState = BatchState::ImproveMSA;

    #ifdef USE_WAIT_FLAGS
        constexpr int wait_index = static_cast<int>(expectedState);
    #endif

        assert(batch.state == expectedState);

        std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
        std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

    #ifdef USE_WAIT_FLAGS
        if(batch.waitCounts[wait_index] != 0){
            batch.activeWaitIndex = wait_index;
            return expectedState;
        }
    #else
        cudaError_t status = cudaEventQuery(events[num_indices_transfered_event_index]); CUERR;
        if(status == cudaErrorNotReady){
            batch.activeWaitIndex = num_indices_transfered_event_index;
            return expectedState;
        }
    #endif

        DataArrays& dataArrays = *batch.dataArrays;

        //if there are no good candidates, clean up batch and discard reads
        if(dataArrays.h_num_indices[0] == 0){
            std::cerr << "improvemsa *h_num_indices = " << dataArrays.h_num_indices[0] << '\n';
            return BatchState::WriteResults;
        }


#ifdef USE_MSA_MINIMIZATION

        constexpr int max_num_minimizations = 5;

        if(max_num_minimizations > 0){
            if(batch.numMinimizations < max_num_minimizations && !(batch.numMinimizations > 0 && batch.previousNumIndices == dataArrays.h_num_indices[0])){

                const int currentNumIndices = dataArrays.h_num_indices[0];

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

                const float desiredAlignmentMaxErrorRate = transFuncData.goodAlignmentProperties.maxErrorRate;

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

                cudaMemcpyAsync(dataArrays.h_num_indices, dataArrays.d_num_indices, sizeof(int), D2H, streams[primary_stream_index]);  CUERR;

    #ifdef USE_WAIT_FLAGS
                batch.addWaitSignal(expectedState, streams[primary_stream_index]);
    #endif

                cudaEventRecord(events[num_indices_transfered_event_index], streams[primary_stream_index]); CUERR;

                cubCachingAllocator.DeviceFree(d_shouldBeKept); CUERR;
                cubCachingAllocator.DeviceFree(d_newIndices); CUERR;
                cubCachingAllocator.DeviceFree(d_indices_per_subject_tmp); CUERR;
                cubCachingAllocator.DeviceFree(d_shouldBeKept_positions); CUERR;

                batch.numMinimizations++;
                batch.previousNumIndices = currentNumIndices;

                //repeat state_buildmsa_func to rebuild the msa using the new index list
                return expectedState;

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

        		assert(cudaSuccess == cudaEventQuery(events[indices_transfer_finished_event_index])); CUERR;

        		cudaEventRecord(events[indices_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;

    #ifdef USE_WAIT_FLAGS
                batch.addWaitSignal(BatchState::UnpackClassicResults, streams[secondary_stream_index]);
    #endif
            }
        }
#endif


        //At this point the msa is built, maybe minimized, and is ready to be used for correction

        //assert(cudaSuccess == cudaEventQuery(events[msa_build_finished_event_index])); CUERR;

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

        #ifdef USE_WAIT_FLAGS
            batch.addWaitSignal(BatchState::StartForestCorrection, streams[secondary_stream_index]);
            batch.addWaitSignal(BatchState::StartConvnetCorrection, streams[secondary_stream_index]);
            batch.addWaitSignal(BatchState::WriteFeatures, streams[secondary_stream_index]);
        #endif

        }

        if(transFuncData.correctionOptions.extractFeatures){
            return BatchState::WriteFeatures;
        }else{
            switch(transFuncData.correctionOptions.correctionType){
            case CorrectionType::Classic:
                return BatchState::StartClassicCorrection;
            case CorrectionType::Forest:
                return BatchState::StartForestCorrection;
            case CorrectionType::Convnet:
                return BatchState::StartConvnetCorrection;
            default:
                return BatchState::StartClassicCorrection;
            }
        }


    }





	BatchState state_startclassiccorrection_func(Batch& batch,
												bool isPausable,
				TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::StartClassicCorrection);
		assert(transFuncData.correctionOptions.correctionType == CorrectionType::Classic);

		DataArrays& dataArrays = *batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		const float avg_support_threshold = 1.0f-1.0f*transFuncData.correctionOptions.estimatedErrorrate;
		const float min_support_threshold = 1.0f-3.0f*transFuncData.correctionOptions.estimatedErrorrate;
		// coverage is always >= 1
		const float min_coverage_threshold = std::max(1.0f,
					transFuncData.correctionOptions.m_coverage / 6.0f * transFuncData.correctionOptions.estimatedCoverage);
        const float max_coverage_threshold = 0.5 * transFuncData.correctionOptions.estimatedCoverage;
		const int new_columns_to_correct = transFuncData.correctionOptions.new_columns_to_correct;

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
        cudaDeviceSynchronize(); CUERR;

        auto identity = [](auto i){return i;};


        for(int i = 0; i < dataArrays.n_subjects; i++){
            std::string s; s.resize(128);
            decode2BitHiLoSequence(&s[0], (const unsigned int*)dataArrays.h_subject_sequences_data.get() + i * dataArrays.encoded_sequence_pitch, 100, identity);
            std::cout << "Subject  : " << s << ", subject id " <<  batch.tasks[i].readId << std::endl;
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
                    decode2BitHiLoSequence(dst, (const unsigned int*)candidateSequencePtr, 100, identity);
                    //std::cout << "Candidate: " << s << std::endl;
                }

                printSequencesInMSA(std::cout,
                                         s.c_str(),
                                         dataArrays.h_subject_sequences_lengths[i],
                                         cands.data(),
                                         candlengths.data(),
                                         numind,
                                         candshifts.data(),
                                         dataArrays.h_msa_column_properties[i].subjectColumnsBegin_incl,
                                         dataArrays.h_msa_column_properties[i].subjectColumnsEnd_excl,
                                         dataArrays.h_msa_column_properties[i].lastColumn_excl - dataArrays.h_msa_column_properties[i].firstColumn_incl,
                                         128);

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

		if(transFuncData.correctionOptions.correctCandidates) {


			// find subject ids of subjects with high quality multiple sequence alignment

            size_t cubTempSize = dataArrays.d_cub_temp_storage.sizeInBytes();

            cub::DeviceSelect::Flagged(dataArrays.d_cub_temp_storage.get(),
                        cubTempSize,
                        cub::CountingInputIterator<int>(0),
                        dataArrays.d_is_high_quality_subject.get(),
                        dataArrays.d_high_quality_subject_indices.get(),
                        dataArrays.d_num_high_quality_subject_indices.get(),
                        dataArrays.n_subjects,
                        streams[primary_stream_index]); CUERR;

			// correct candidates
            call_msa_correct_candidates_kernel_async_exp(
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
                    dataArrays.sequence_pitch,
                    dataArrays.msa_pitch,
                    dataArrays.msa_weights_pitch,
                    min_support_threshold,
                    min_coverage_threshold,
                    new_columns_to_correct,
                    dataArrays.maximum_sequence_length,
                    streams[primary_stream_index],
                    batch.kernelLaunchHandle);
		}

		assert(cudaSuccess == cudaEventQuery(events[correction_finished_event_index])); CUERR;

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

        if(transFuncData.correctionOptions.correctCandidates){
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
        }

		assert(cudaSuccess == cudaEventQuery(events[result_transfer_finished_event_index])); CUERR;

		cudaEventRecord(events[result_transfer_finished_event_index], streams[primary_stream_index]); CUERR;

#ifdef USE_WAIT_FLAGS
        batch.addWaitSignal(BatchState::UnpackClassicResults, streams[primary_stream_index]);
#endif

        return BatchState::UnpackClassicResults;
	}

    BatchState state_startconvnetcorrection_func(Batch& batch,
                          bool isPausable,
                TransitionFunctionData& transFuncData){

        constexpr BatchState expectedState = BatchState::StartConvnetCorrection;
#ifdef USE_WAIT_FLAGS
        constexpr int wait_index = static_cast<int>(expectedState);
#endif
        assert(batch.state == expectedState);



#ifdef USE_WAIT_FLAGS
        if(batch.waitCounts[wait_index] != 0){
            batch.activeWaitIndex = wait_index;
            return expectedState;
        }
#else
        std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

        cudaError_t status = cudaEventQuery(events[msadata_transfer_finished_event_index]); CUERR;
        if(status == cudaErrorNotReady){
            batch.activeWaitIndex = msadata_transfer_finished_event_index;
            return expectedState;
        }
#endif

        assert(transFuncData.correctionOptions.correctionType == CorrectionType::Convnet);

        DataArrays& dataArrays = *batch.dataArrays;

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


        return BatchState::WriteResults;
    }

    BatchState state_startforestcorrection_func(Batch& batch,
												bool isPausable,
				TransitionFunctionData& transFuncData){

        constexpr BatchState expectedState = BatchState::StartForestCorrection;
#ifdef USE_WAIT_FLAGS
        constexpr int wait_index = static_cast<int>(expectedState);
#endif
        assert(batch.state == expectedState);



#ifdef USE_WAIT_FLAGS
        if(batch.waitCounts[wait_index] != 0){
            batch.activeWaitIndex = wait_index;
            return expectedState;
        }
#else
        std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

        cudaError_t status = cudaEventQuery(events[msadata_transfer_finished_event_index]); CUERR;
        if(status == cudaErrorNotReady){
            batch.activeWaitIndex = msadata_transfer_finished_event_index;
            return expectedState;
        }
#endif

        assert(transFuncData.correctionOptions.correctionType == CorrectionType::Forest);

		DataArrays& dataArrays = *batch.dataArrays;

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

        #pragma omp parallel for schedule(dynamic,2)
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

        }

		return BatchState::WriteResults;
	}

    BatchState state_unpackclassicresults_func(Batch& batch,
                          bool isPausable,
                TransitionFunctionData& transFuncData){

        constexpr BatchState expectedState = BatchState::UnpackClassicResults;
    #ifdef USE_WAIT_FLAGS
        constexpr int wait_index = static_cast<int>(expectedState);
    #endif

        assert(batch.state == expectedState);

        std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

    #ifdef USE_WAIT_FLAGS
        if(batch.waitCounts[wait_index] != 0){
            batch.activeWaitIndex = wait_index;
            return expectedState;
        }
    #else
        cudaError_t status = cudaEventQuery(events[indices_transfer_finished_event_index]); CUERR;
        if(status == cudaErrorNotReady){
            batch.activeWaitIndex = indices_transfer_finished_event_index;
            return expectedState;
        }

        status = cudaEventQuery(events[result_transfer_finished_event_index]); CUERR;
        if(status == cudaErrorNotReady){
            batch.activeWaitIndex = result_transfer_finished_event_index;
            return expectedState;
        }
    #endif



        DataArrays& dataArrays = *batch.dataArrays;
        //std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;

        cudaError_t errort = cudaEventQuery(events[correction_finished_event_index]);
        if(errort != cudaSuccess){
            std::cout << "error cudaEventQuery\n";
            std::exit(0);
        }
        assert(cudaEventQuery(events[correction_finished_event_index]) == cudaSuccess); CUERR;

        assert(transFuncData.correctionOptions.correctionType == CorrectionType::Classic);

        //#pragma omp parallel for
        for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {
            auto& task = batch.tasks[subject_index];
            const char* const my_corrected_subject_data = dataArrays.h_corrected_subjects + subject_index * dataArrays.sequence_pitch;
            task.corrected = dataArrays.h_subject_is_corrected[subject_index];
            //if(task.readId == 207){
            //    std::cerr << "\n\ncorrected: " << task.corrected << "\n";
            //}
            if(task.corrected) {
                const int subject_length = dataArrays.h_subject_sequences_lengths[subject_index];//task.subject_string.length();//dataArrays.h_subject_sequences_lengths[subject_index];
                task.corrected_subject = std::move(std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length});

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


                task.corrected_candidates_read_ids.resize(n_corrected_candidates);
                task.corrected_candidates.resize(n_corrected_candidates);

                for(int i = 0; i < n_corrected_candidates; ++i) {
                    const int global_candidate_index = my_indices_of_corrected_candidates[i];
                    //const int local_candidate_index = global_candidate_index - dataArrays.h_candidates_per_subject_prefixsum[subject_index];

					//const read_number candidate_read_id = task.candidate_read_ids[local_candidate_index];
					//const read_number candidate_read_id = task.candidate_read_ids_begin[local_candidate_index];

                    const read_number candidate_read_id = dataArrays.h_candidate_read_ids[global_candidate_index];
                    const int candidate_length = dataArrays.h_candidate_sequences_lengths[global_candidate_index];//transFuncData.gpuReadStorage->fetchSequenceLength(candidate_read_id);

                    const char* const candidate_data = my_corrected_candidates_data + i * dataArrays.sequence_pitch;

                    task.corrected_candidates_read_ids[i] = candidate_read_id;
                    task.corrected_candidates[i] = std::move(std::string{candidate_data, candidate_data + candidate_length});

                    //task.corrected_candidates_read_ids.emplace_back(candidate_read_id);
                    //task.corrected_candidates.emplace_back(std::move(std::string{candidate_data, candidate_data + candidate_length}));
                }
            }
        }

        return BatchState::WriteResults;
    }



	BatchState state_writeresults_func(Batch& batch,
												bool isPausable,
				TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::WriteResults);

        /*DataArrays& dataArrays = *batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

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

		//write result to file
		for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {

			const auto& task = batch.tasks[subject_index];
			//std::cout << task.readId << "result" << std::endl;

			//std::cout << "finished readId " << task.readId << std::endl;

			if(task.corrected/* && task.corrected_subject != task.subject_string*/) {
				push_range("write_subject", 4);
				//std::cout << task.readId << "\n" << task.corrected_subject << std::endl;
				transFuncData.write_read_to_stream(task.readId, task.corrected_subject);
				//transFuncData.lock(task.readId);
				//(*transFuncData.readIsCorrectedVector)[task.readId] = 1;
				//transFuncData.unlock(task.readId);
				pop_range();
			}else{
				push_range("subject_not_corrected", 5);
				//mark read as not corrected
				if((*transFuncData.readIsCorrectedVector)[task.readId] == 1) {
					transFuncData.lock(task.readId);
					if((*transFuncData.readIsCorrectedVector)[task.readId] == 1) {
						(*transFuncData.readIsCorrectedVector)[task.readId] = 0;
					}
					transFuncData.unlock(task.readId);
				}
                pop_range();
			}
			push_range("correctedcandidates", 6);
			for(std::size_t corrected_candidate_index = 0; corrected_candidate_index < task.corrected_candidates.size(); ++corrected_candidate_index) {

				read_number candidateId = task.corrected_candidates_read_ids[corrected_candidate_index];
				const std::string& corrected_candidate = task.corrected_candidates[corrected_candidate_index];

                //const char* sequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(candidateId);
				//const int sequencelength = transFuncData.gpuReadStorage->fetchSequenceLength(candidateId);
				//const std::string original_candidate = Sequence_t::Impl_t::toString((const std::uint8_t*)sequenceptr, sequencelength);

                //if(corrected_candidate == original_candidate){
                    bool savingIsOk = false;
    				if((*transFuncData.readIsCorrectedVector)[candidateId] == 0) {
    					transFuncData.lock(candidateId);
    					if((*transFuncData.readIsCorrectedVector)[candidateId]== 0) {
    						(*transFuncData.readIsCorrectedVector)[candidateId] = 1;         // we will process this read
    						savingIsOk = true;
    						//nCorrectedCandidates++;
    					}
    					transFuncData.unlock(candidateId);
    				}
    				if (savingIsOk) {
    					transFuncData.write_read_to_stream(candidateId, corrected_candidate);
    				}
                //}


			}
			pop_range();
		}

		return BatchState::Finished;
	}

	BatchState state_writefeatures_func(Batch& batch,
												bool isPausable,
				TransitionFunctionData& transFuncData){

        constexpr BatchState expectedState = BatchState::WriteFeatures;
#ifdef USE_WAIT_FLAGS
        constexpr int wait_index = static_cast<int>(expectedState);
#endif
        assert(batch.state == expectedState);



#ifdef USE_WAIT_FLAGS
        if(batch.waitCounts[wait_index] != 0){
            batch.activeWaitIndex = wait_index;
            return expectedState;
        }
#else
        std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

        cudaError_t status = cudaEventQuery(events[msadata_transfer_finished_event_index]); CUERR;
        if(status == cudaErrorNotReady){
            batch.activeWaitIndex = msadata_transfer_finished_event_index;
            return expectedState;
        }
#endif

		DataArrays& dataArrays = *batch.dataArrays;

		auto& featurestream = *transFuncData.featurestream;

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
				featurestream << task.readId << '\t' << msafeature.position << '\t' << msafeature.consensus << '\n';
				featurestream << msafeature << '\n';
			}
		}

		return BatchState::Finished;
	}

	BatchState state_finished_func(Batch& batch,
												bool isPausable,
				TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::Finished);

		assert(false);         //Finished is end node

		return BatchState::Finished;
	}

	BatchState state_aborted_func(Batch& batch,
												bool isPausable,
				TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::Aborted);

		assert(false);         //Aborted is end node

		return BatchState::Aborted;
	}



AdvanceResult advance_one_step(Batch& batch,
			bool isPausable,
			TransitionFunctionData& transFuncData,
            const std::unordered_map<BatchState, FuncTableEntry>& transitionFunctionTable){

	AdvanceResult advanceResult;

	advanceResult.oldState = batch.state;
	advanceResult.noProgressBlocking = false;
	advanceResult.noProgressLaunching = false;

	auto iter = transitionFunctionTable.find(batch.state);
	if(iter != transitionFunctionTable.end()) {
		batch.state = iter->second(batch, isPausable, transFuncData);
	}else{
		std::cout << nameOf(batch.state) << std::endl;
		assert(false); // Every State should be handled above
	}

	advanceResult.newState = batch.state;

	return advanceResult;
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
                  std::uint64_t maxCandidatesPerRead,
                  std::vector<char>& readIsCorrectedVector,
                  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
                  std::size_t nLocksForProcessedFlags){

      assert(runtimeOptions.canUseGpu);
      assert(runtimeOptions.max_candidates > 0);
      assert(runtimeOptions.deviceIds.size() > 0);

      int oldNumOMPThreads = 1;
      #pragma omp parallel
      {
          #pragma omp single
          oldNumOMPThreads = omp_get_num_threads();
      }

      omp_set_num_threads(runtimeOptions.nCorrectorThreads);
      std::cerr << "omp_set_num_threads " << runtimeOptions.nCorrectorThreads << "\n";

      std::chrono::time_point<std::chrono::system_clock> timepoint_begin = std::chrono::system_clock::now();
      std::chrono::duration<double> runtime = std::chrono::seconds(0);

      const auto& deviceIds = runtimeOptions.deviceIds;

      std::vector<std::string> tmpfiles{fileOptions.outputfile + "_tmp"};

      std::unique_ptr<SequenceFileReader> reader = makeSequenceReader(fileOptions.inputfile, fileOptions.format);

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
          featurestream = std::move(std::ofstream(tmpfiles[0] + "_features"));
          if(!featurestream && correctionOptions.extractFeatures){
              throw std::runtime_error("Could not open output feature file");
          }
      //}

      //std::mutex outputstreamlock;

      Read readInFile;

      cudaSetDevice(deviceIds[0]); CUERR;

      gpu::init_weights(deviceIds);


      auto transitionFunctionTable = makeTransitionFunctionTable();


      std::vector<DataArrays > dataArrays;
      std::array<std::array<cudaStream_t, nStreamsPerBatch>, nParallelBatches> streams;
      std::array<std::array<cudaEvent_t, nEventsPerBatch>, nParallelBatches> cudaevents;

      for(int i = 0; i < nParallelBatches; i++) {
          dataArrays.emplace_back(deviceIds[0]);

          for(int j = 0; j < nStreamsPerBatch; ++j) {
              cudaStreamCreate(&streams[i][j]); CUERR;
          }

          for(int j = 0; j < nEventsPerBatch; ++j) {
              cudaEventCreateWithFlags(&cudaevents[i][j], cudaEventDisableTiming); CUERR;
          }
      }

      auto kernelLaunchHandle = make_kernel_launch_handle(deviceIds[0]);


      std::vector<read_number> readIdBuffer;
      std::vector<CorrectionTask> tmptasksBuffer;
      std::vector<char> sequenceDataBuffer;
      std::vector<int> sequenceLengthsBuffer;

#ifndef DO_PROFILE
        cpu::RangeGenerator<read_number> readIdGenerator(sequenceFileProperties.nReads);
#else
        cpu::RangeGenerator<read_number> readIdGenerator(num_reads_to_profile);
#endif

      TransitionFunctionData transFuncData;

      //transFuncData.mybatchgen = &mybatchgen;
      transFuncData.deviceId = deviceIds[0];
      transFuncData.goodAlignmentProperties = goodAlignmentProperties;
      transFuncData.correctionOptions = correctionOptions;
      transFuncData.runtimeOptions = runtimeOptions;
      transFuncData.minhashOptions = minhashOptions;
      transFuncData.alignmentOptions = alignmentOptions;
      transFuncData.fileOptions = fileOptions;
      transFuncData.sequenceFileProperties = sequenceFileProperties;

      transFuncData.readIdGenerator = &readIdGenerator;
      transFuncData.readIdBuffer = &readIdBuffer;
      transFuncData.tmptasksBuffer = &tmptasksBuffer;
      transFuncData.sequenceDataBuffer = &sequenceDataBuffer;
      transFuncData.sequenceLengthsBuffer = &sequenceLengthsBuffer;
      transFuncData.minhasher = &minhasher;
      transFuncData.readStorage = &readStorage;
      transFuncData.locksForProcessedFlags = locksForProcessedFlags.get();
      transFuncData.nLocksForProcessedFlags = nLocksForProcessedFlags;
      transFuncData.readIsCorrectedVector = &readIsCorrectedVector;
      transFuncData.featurestream = &featurestream;
      transFuncData.write_read_to_stream = [&](const read_number readId, const std::string& sequence){
                               //std::cout << readId << " " << sequence << std::endl;
                               auto& stream = outputstream;
  #if 1
                               stream << readId << ' ' << sequence << '\n';
  #else
                               stream << readId << '\n';
                               stream << sequence << '\n';
  #endif
                           };
      transFuncData.lock = [&](read_number readId){
                       read_number index = readId % transFuncData.nLocksForProcessedFlags;
                       transFuncData.locksForProcessedFlags[index].lock();
                   };
      transFuncData.unlock = [&](read_number readId){
                         read_number index = readId % transFuncData.nLocksForProcessedFlags;
                         transFuncData.locksForProcessedFlags[index].unlock();
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

      std::array<Batch, nParallelBatches> batches;
      std::array<Batch*, nParallelBatches> batchPointers;

        for(int i = 0; i < nParallelBatches; ++i) {
            batches[i].id = i;
            batches[i].dataArrays = &dataArrays[i];
            batches[i].streams = &streams[i];
            batches[i].events = &cudaevents[i];
            batches[i].kernelLaunchHandle = make_kernel_launch_handle(deviceIds[0]);
            batches[i].subjectSequenceGatherHandle2 = readStorage.makeGatherHandleSequences();
            batches[i].candidateSequenceGatherHandle2 = readStorage.makeGatherHandleSequences();
            batches[i].subjectLengthGatherHandle2 = readStorage.makeGatherHandleLengths();
            batches[i].candidateLengthGatherHandle2 = readStorage.makeGatherHandleLengths();
            batches[i].subjectQualitiesGatherHandle2 = readStorage.makeGatherHandleQualities();
            batches[i].candidateQualitiesGatherHandle2 = readStorage.makeGatherHandleQualities();
            batchPointers[i] = &batches[i];
        }

      auto nextBatchIndex = [](int currentBatchIndex, int nParallelBatches){
                        return (currentBatchIndex + 1) % nParallelBatches;
                    };


      #ifdef DO_PROFILE
          cudaProfilerStart();
      #endif




      int stacksize = 0;
      //auto previousprocessedreads = readIdGenerator.getCurrentUnsafe();
      while(
            !(std::all_of(batches.begin(), batches.end(), [](const auto& batch){
                  return batch.state == BatchState::Finished;
              })
              && readIdBuffer.empty()
              && readIdGenerator.empty())) {

          if(stacksize != 0)
              assert(stacksize == 0);

          Batch& mainBatch = *batchPointers[0];

          // size_t hostSizeBytes = mainBatch.dataArrays->hostArraysSizeInBytes();
          // size_t deviceSizeBytes = mainBatch.dataArrays->deviceArraysSizeInBytes();
          // size_t hostCapacityBytes = mainBatch.dataArrays->hostArraysCapacityInBytes();
          // size_t deviceCapacityBytes = mainBatch.dataArrays->deviceArraysCapacityInBytes();
          //
          // auto MB = [](auto bytes){
          //     return bytes / 1024. / 1024;
          // };
          //
          // std::cerr << "Resize: Host " << MB(hostSizeBytes) << " " << MB(hostCapacityBytes);
          // std::cerr << " Device " << MB(deviceSizeBytes) << " " << MB(deviceCapacityBytes) << '\n';

          AdvanceResult mainBatchAdvanceResult;
          bool popMain = false;

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



          assert(popMain == false);
          push_range("mainBatch"+nameOf(mainBatch.state)+"first", int(mainBatch.state));
          ++stacksize;
          popMain = true;

          while(!(mainBatch.state == BatchState::Finished || mainBatch.state == BatchState::Aborted)) {

              mainBatchAdvanceResult = advance_one_step(mainBatch,
                          false,                   //cannot be paused
                          transFuncData,
                            transitionFunctionTable);

              if((mainBatchAdvanceResult.oldState != mainBatchAdvanceResult.newState)) {
                  pop_range("main inner");
                  popMain = false;
                  --stacksize;

                  assert(popMain == false);
                  push_range("mainBatch"+nameOf(mainBatchAdvanceResult.newState), int(mainBatchAdvanceResult.newState));
                  ++stacksize;
                  popMain = true;
              }


      #if 1

              while(mainBatch.isWaiting()) {
                  /*
                      Prepare next batch while waiting for the mainBatch gpu work to finish
                   */
                  if(nParallelBatches > 1) {

                      // 0 <= sideBatchIndex < nParallelBatches - 1
                      // index in batches array is sideBatchIndex + 1;
                      int localSideBatchIndex = 0;
                      const int nSideBatches = nParallelBatches - 1;

                      bool popSide = false;
                      bool firstSideIter = true;

                      AdvanceResult sideBatchAdvanceResult;

                      while(mainBatch.isWaiting()) {
                          const int globalBatchIndex = localSideBatchIndex + 1;

                          Batch& sideBatch = *batchPointers[globalBatchIndex];

                          if(sideBatch.state == BatchState::Finished || sideBatch.state == BatchState::Aborted) {
                              continue;
                          }

                          for(int i = 0; i < sideBatchStepsPerWaitIter; ++i) {
                              if(sideBatch.state == BatchState::Finished || sideBatch.state == BatchState::Aborted) {
                                  break;
                              }

                              if(firstSideIter) {
                                  assert(popSide == false);
                                  push_range("sideBatch"+std::to_string(localSideBatchIndex)+nameOf(sideBatch.state)+"first", int(sideBatch.state));
                                  ++stacksize;
                                  popSide = true;
                              }else{
                                  if(sideBatchAdvanceResult.oldState != sideBatchAdvanceResult.newState) {
                                      assert(popSide == false);
                                      push_range("sideBatch"+std::to_string(localSideBatchIndex)+nameOf(sideBatchAdvanceResult.newState), int(sideBatchAdvanceResult.newState));
                                      ++stacksize;
                                      popSide = true;
                                  }
                              }

                              //auto curstate = sideBatch.state;
                              //if(curstate == BatchState::Unprepared){
                              //    push_range("sideBatch"+std::to_string(localSideBatchIndex)+nameOf(curstate), int(sideBatch.state));
                              //}
                              sideBatchAdvanceResult = advance_one_step(sideBatch,
                                          true,                   //can be paused
                                          transFuncData,
                                            transitionFunctionTable);

                              //if(curstate == BatchState::Unprepared){
                              //    pop_range();
                              //}

                              if(sideBatchAdvanceResult.oldState != sideBatchAdvanceResult.newState) {
                                  pop_range("side inner");
                                  popSide = false;
                                  --stacksize;
                              }



                              firstSideIter = false;
                          }

                          if(sideBatch.isWaiting()) {
                              //current side batch is waiting, move to next side batch
                              localSideBatchIndex = nextBatchIndex(localSideBatchIndex, nSideBatches);

                              if(popSide) {
                                  pop_range("switch sidebatch");
                                  popSide = false;
                                  --stacksize;
                              }

                              firstSideIter = true;
                              sideBatchAdvanceResult = AdvanceResult{};
                          }
                      }

                      if(popSide) {
                          pop_range("side outer");
                          popSide = false;
                          --stacksize;
                      }

                      //assert(eventquerystatus == cudaSuccess);
                  }else{

                  }


              }
      #endif


          }

          if(popMain) {
              pop_range("main outer");
              popMain = false;
              --stacksize;
          }

          assert(stacksize == 0);

          assert(mainBatch.state == BatchState::Finished || mainBatch.state == BatchState::Aborted);

          if(!(readIdGenerator.empty() && readIdBuffer.empty() && tmptasksBuffer.empty())) {
              //there are reads left to correct, so this batch can be reused again
              mainBatch.reset();
          }else{
              mainBatch.state = BatchState::Finished;
          }

          //nProcessedReads = threadOpts.readIdGenerator->.currentId - mybatchgen.firstId;

          //rotate left to position next batch at index 0
          std::rotate(batchPointers.begin(), batchPointers.begin()+1, batchPointers.end());



      } // end batch processing


      outputstream.flush();
      featurestream.flush();

      for(auto& batch : batches){
          batch.waitUntilAllCallbacksFinished();
      }

      assert(tmptasksBuffer.empty());











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

      for(auto& array : dataArrays) {
          array.reset();
      }

      for(auto& streamarray : streams) {
          for(int i = 1; i < nStreamsPerBatch; ++i)
              cudaStreamDestroy(streamarray[i]); CUERR;
      }

      for(auto& eventarray : cudaevents) {
          for(auto& event : eventarray)
              cudaEventDestroy(event); CUERR;
      }

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

              mergeResultFiles(sequenceFileProperties.nReads, fileOptions.inputfile, fileOptions.format, tmpfiles, fileOptions.outputfile);
              //mergeResultFiles2(sequenceFileProperties.nReads, fileOptions.inputfile, fileOptions.format, tmpfiles, fileOptions.outputfile, occupiedMemory);

              TIMERSTOPCPU(merge);

              std::cout << "end merging reads" << std::endl;

          }

          deleteFiles(tmpfiles);
      }

      std::vector<std::string> featureFiles(tmpfiles);
      for(auto& s : featureFiles)
          s = s + "_features";

      //concatenate feature files of each thread into one file

      if(correctionOptions.extractFeatures){
          std::cout << "begin merging features" << std::endl;

          std::stringstream commandbuilder;

          commandbuilder << "cat";

          for(const auto& featureFile : featureFiles){
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
              for(const auto& s : featureFiles)
                  std::cerr << s << '\n';
          }else{
              deleteFiles(featureFiles);
          }

          std::cout << "end merging features" << std::endl;
      }else{
          deleteFiles(featureFiles);
      }

      std::cout << "end merge" << std::endl;

      #endif



}






















}
}




#endif
