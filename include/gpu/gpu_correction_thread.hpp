#ifndef CARE_GPU_CORRECTION_THREAD_HPP
#define CARE_GPU_CORRECTION_THREAD_HPP


#include "../hpc_helpers.cuh"
#include "../options.hpp"
#include "../tasktiming.hpp"
#include "../sequence.hpp"
#include "../featureextractor.hpp"

#include <forestclassifier.hpp>
#include <nn_classifier.hpp>
#include <minhasher.hpp>
#include <rangegenerator.hpp>

#include <gpu/kernels.hpp>
#include <gpu/dataarrays.hpp>
#include <gpu/readstorage.hpp>

#include <config.hpp>

#include <thread>
#include <vector>
#include <string>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <numeric>
#include <algorithm>

#include <unordered_map>

namespace care {
namespace gpu {


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
		candidate_read_ids_begin(other.candidate_read_ids_begin),
		candidate_read_ids_end(other.candidate_read_ids_end),
		corrected_subject(other.corrected_subject),
		corrected_candidates(other.corrected_candidates),
		corrected_candidates_read_ids(other.corrected_candidates_read_ids){

		candidate_read_ids_begin = &(candidate_read_ids[0]);
		candidate_read_ids_end = &(candidate_read_ids[candidate_read_ids.size()]);

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
		swap(l.candidate_read_ids_begin, r.candidate_read_ids_begin);
		swap(l.candidate_read_ids_end, r.candidate_read_ids_end);
		swap(l.corrected_subject, r.corrected_subject);
		swap(l.corrected_candidates_read_ids, r.corrected_candidates_read_ids);
	}

	bool active;
	bool corrected;
	read_number readId;

	std::vector<read_number> candidate_read_ids;
	read_number* candidate_read_ids_begin;
	read_number* candidate_read_ids_end;         // exclusive

	std::string subject_string;

	std::string corrected_subject;
	std::vector<std::string> corrected_candidates;
	std::vector<read_number> corrected_candidates_read_ids;
};

#ifdef __NVCC__
struct ErrorCorrectionThreadOnlyGPU {
    using Minhasher_t = Minhasher;
    using GPUReadStorage_t = gpu::ContiguousReadStorage;

	using CorrectionTask_t = CorrectionTask;
	//using ReadIdGenerator_t = readIdGenerator_t;
    using ReadIdGenerator_t = cpu::RangeGenerator<read_number>;


	static constexpr int primary_stream_index = 0;
	static constexpr int secondary_stream_index = 1;
	static constexpr int h2d_stream_index = 2;
	static constexpr int d2h_stream_index = 3;
	static constexpr int nStreamsPerBatch = 4;

	static constexpr int alignments_finished_event_index = 0;
	static constexpr int quality_transfer_finished_event_index = 1;
	static constexpr int indices_transfer_finished_event_index = 2;
	static constexpr int correction_finished_event_index = 3;
	static constexpr int result_transfer_finished_event_index = 4;
	static constexpr int msadata_transfer_finished_event_index = 5;
	static constexpr int alignment_data_transfer_h2d_finished_event_index = 6;
	static constexpr int msa_build_finished_event_index = 7;
    static constexpr int indices_calculated_event_index = 8;
    static constexpr int num_indices_transfered_event_index = 9;
	static constexpr int nEventsPerBatch = 10;

    static constexpr int wait_before_copyqualites_index = 0;
    static constexpr int wait_before_unpackclassicresults_index = 1;
    static constexpr int wait_before_startforestcorrection_index = 2;
    static constexpr int nWaitCountPerBatch = 3;

	enum class BatchState : int{
		Unprepared,
		CopyReads,
		StartAlignment,
        RearrangeIndices,
		CopyQualities,
		BuildMSA,
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

	enum class BatchStatePriority {
		Low,
		Medium,
		High,
	};

	struct Batch {
        struct WaitCallbackData{
            Batch* b{};
            int index = -1;
            WaitCallbackData(){}
            WaitCallbackData(Batch* ptr, int i) : b(ptr), index(i){}
        };
		std::vector<CorrectionTask_t> tasks;
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

        BatchDataHost batchDataHost;
        BatchDataDevice batchDataDevice;

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

        bool isWaiting() const;

        void addWaitSignal(BatchState state, cudaStream_t stream);

		void reset();

        void waitUntilAllCallbacksFinished() const;
	};

	struct AdvanceResult {
		BatchState oldState = BatchState::Unprepared;
		BatchState newState = BatchState::Unprepared;
		bool noProgressBlocking = false;
		bool noProgressLaunching = false;
	};

    struct CorrectionThreadOptions {
        int threadId;
        int deviceId;
        bool canUseGpu;

        std::string outputfile;
        ReadIdGenerator_t* readIdGenerator;
        const Minhasher_t* minhasher;
        GPUReadStorage_t* gpuReadStorage;
        std::mutex* coutLock;
        std::vector<char>* readIsCorrectedVector;
        std::mutex* locksForProcessedFlags;
        std::size_t nLocksForProcessedFlags;
    };

	struct TransitionFunctionData {
        CorrectionThreadOptions threadOpts;
		ReadIdGenerator_t* readIdGenerator;
		std::vector<read_number>* readIdBuffer;
        std::vector<CorrectionTask_t>* tmptasksBuffer;
		float min_overlap_ratio;
		int min_overlap;
		float estimatedErrorrate;
		float maxErrorRate;
		float estimatedCoverage;
		float m_coverage;
		int new_columns_to_correct;
		bool correctCandidates;
		bool useQualityScores;
		int kmerlength;
		int num_ids_per_add_tasks;
		int minimum_candidates_per_batch;
		int max_candidates;
		int maxSequenceLength;
		const Minhasher_t* minhasher;
		const GPUReadStorage_t* gpuReadStorage;
		typename GPUReadStorage_t::GPUData readStorageGpuData;
		std::mutex* locksForProcessedFlags;
		std::size_t nLocksForProcessedFlags;
		CorrectionOptions correctionOptions;
		std::vector<char>* readIsCorrectedVector;
		std::ofstream* featurestream;
		std::function<void(const read_number, const std::string&)> write_read_to_stream;
		std::function<void(const read_number)> lock;
		std::function<void(const read_number)> unlock;

        ForestClassifier fc;// = ForestClassifier{"./forests/testforest.so"};
        NN_Correction_Classifier nnClassifier;
	};



	AlignmentOptions alignmentOptions;
	GoodAlignmentProperties goodAlignmentProperties;
	CorrectionOptions correctionOptions;
	CorrectionThreadOptions threadOpts;
    FileOptions fileOptions;
	SequenceFileProperties fileProperties;

    NN_Correction_Classifier_Base* classifierBase;

	std::uint64_t max_candidates = 0;

	std::uint64_t nProcessedReads = 0;

	std::uint64_t minhashcandidates = 0;
	std::uint64_t duplicates = 0;
	int nProcessedQueries = 0;
	int nCorrectedCandidates = 0; // candidates which were corrected in addition to query correction.

	int avgsupportfail = 0;
	int minsupportfail = 0;
	int mincoveragefail = 0;
	int sobadcouldnotcorrect = 0;
	int verygoodalignment = 0;

	std::chrono::duration<double> getCandidatesTimeTotal;
	std::chrono::duration<double> mapMinhashResultsToSequencesTimeTotal;
	std::chrono::duration<double> getAlignmentsTimeTotal;
	std::chrono::duration<double> determinegoodalignmentsTime;
	std::chrono::duration<double> fetchgoodcandidatesTime;
	std::chrono::duration<double> majorityvotetime;
	std::chrono::duration<double> basecorrectiontime;
	std::chrono::duration<double> readcorrectionTimeTotal;
	std::chrono::duration<double> mapminhashresultsdedup;
	std::chrono::duration<double> mapminhashresultsfetch;
	std::chrono::duration<double> graphbuildtime;
	std::chrono::duration<double> graphcorrectiontime;


	std::chrono::duration<double> initIdTimeTotal;

	std::chrono::duration<double> da, db, dc;

	TaskTimings detailedCorrectionTimings;

	int num_ids_per_add_tasks = 2;
	int minimum_candidates_per_batch = 1000;

	using FuncTableEntry = BatchState (*)(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData&);

	std::unordered_map<BatchState, FuncTableEntry> transitionFunctionTable;

	std::thread thread;
	bool isRunning = false;
	volatile bool stopAndAbort = false;

	void run();

	void join();

public:

	std::string nameOf(const BatchState& state) const;

	void makeTransitionFunctionTable();

	static BatchState state_unprepared_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

    static BatchState state_unprepared_func2(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

	static BatchState state_copyreads_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

	static BatchState state_startalignment_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

    static BatchState state_rearrangeindices_func(Batch& batch,
            bool canBlock,
            bool canLaunchKernel,
            bool isPausable,
            const TransitionFunctionData& transFuncData);

	static BatchState state_copyqualities_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

	static BatchState state_buildmsa_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

    static BatchState state_improvemsa_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

	static BatchState state_startclassiccorrection_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

	static BatchState state_startforestcorrection_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

    static BatchState state_startconvnetcorrection_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

	static BatchState state_unpackclassicresults_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

	static BatchState state_writeresults_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

	static BatchState state_writefeatures_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

	static BatchState state_finished_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);

	static BatchState state_aborted_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);


	AdvanceResult advance_one_step(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData);


	void execute();
};


#endif

}
}

#if 0

#if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_MEMCOPY
//DEBUGGING
cudaMemcpyAsync(dataArrays.msa_data_host,
			dataArrays.msa_data_device,
			dataArrays.msa_data_usable_size,
			D2H,
			streams[batchIndex][0]); CUERR;
cudaStreamSynchronize(streams[batchIndex][0]); CUERR;

//DEBUGGING
cudaMemcpyAsync(dataArrays.alignment_result_data_host,
			dataArrays.alignment_result_data_device,
			dataArrays.alignment_result_data_usable_size,
			D2H,
			streams[batchIndex][0]); CUERR;
cudaStreamSynchronize(streams[batchIndex][0]); CUERR;

//DEBUGGING
cudaMemcpyAsync(dataArrays.subject_indices_data_host,
			dataArrays.subject_indices_data_device,
			dataArrays.subject_indices_data_usable_size,
			D2H,
			streams[batchIndex][0]); CUERR;
cudaStreamSynchronize(streams[batchIndex][0]); CUERR;
#endif

/*std::cout << "h_is_high_quality_subject" << std::endl;
   for(int i = 0; i< dataArrays.n_subjects; i++){
    std::cout << dataArrays.h_is_high_quality_subject[i] << "\t";
   }
   std::cout << std::endl;*/

#if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_PRINT_ARRAYS
//DEBUGGING
std::cout << "alignment scores" << std::endl;
for(int i = 0; i< dataArrays.n_queries * 2; i++) {
	std::cout << dataArrays.h_alignment_scores[i] << "\t";
}
std::cout << std::endl;
//DEBUGGING
std::cout << "alignment overlaps" << std::endl;
for(int i = 0; i< dataArrays.n_queries * 2; i++) {
	std::cout << dataArrays.h_alignment_overlaps[i] << "\t";
}
std::cout << std::endl;
//DEBUGGING
std::cout << "alignment shifts" << std::endl;
for(int i = 0; i< dataArrays.n_queries * 2; i++) {
	std::cout << dataArrays.h_alignment_shifts[i] << "\t";
}
std::cout << std::endl;
//DEBUGGING
std::cout << "alignment nOps" << std::endl;
for(int i = 0; i< dataArrays.n_queries * 2; i++) {
	std::cout << dataArrays.h_alignment_nOps[i] << "\t";
}
std::cout << std::endl;
//DEBUGGING
std::cout << "alignment isvalid" << std::endl;
for(int i = 0; i< dataArrays.n_queries * 2; i++) {
	std::cout << dataArrays.h_alignment_isValid[i] << "\t";
}
std::cout << std::endl;
//DEBUGGING
std::cout << "alignment flags" << std::endl;
for(int i = 0; i< dataArrays.n_queries * 2; i++) {
	std::cout << int(dataArrays.h_alignment_best_alignment_flags[i]) << "\t";
}
std::cout << std::endl;

//DEBUGGING
std::cout << "h_candidates_per_subject_prefixsum" << std::endl;
for(int i = 0; i< dataArrays.n_subjects +1; i++) {
	std::cout << dataArrays.h_candidates_per_subject_prefixsum[i] << "\t";
}
std::cout << std::endl;

//DEBUGGING
std::cout << "h_num_indices" << std::endl;
for(int i = 0; i< 1; i++) {
	std::cout << dataArrays.h_num_indices[i] << "\t";
}
std::cout << std::endl;

//DEBUGGING
std::cout << "h_indices" << std::endl;
for(int i = 0; i< *dataArrays.h_num_indices; i++) {
	std::cout << dataArrays.h_indices[i] << "\t";
}
std::cout << std::endl;

//DEBUGGING
std::cout << "h_indices_per_subject" << std::endl;
for(int i = 0; i< dataArrays.n_subjects; i++) {
	std::cout << dataArrays.h_indices_per_subject[i] << "\t";
}
std::cout << std::endl;

//DEBUGGING
std::cout << "h_indices_per_subject_prefixsum" << std::endl;
for(int i = 0; i< dataArrays.n_subjects; i++) {
	std::cout << dataArrays.h_indices_per_subject_prefixsum[i] << "\t";
}
std::cout << std::endl;

//DEBUGGING
std::cout << "h_high_quality_subject_indices" << std::endl;
for(int i = 0; i< dataArrays.n_subjects; i++) {
	std::cout << dataArrays.h_high_quality_subject_indices[i] << "\t";
}
std::cout << std::endl;

//DEBUGGING
std::cout << "h_is_high_quality_subject" << std::endl;
for(int i = 0; i< dataArrays.n_subjects; i++) {
	std::cout << dataArrays.h_is_high_quality_subject[i] << "\t";
}
std::cout << std::endl;

//DEBUGGING
std::cout << "h_num_high_quality_subject_indices" << std::endl;
for(int i = 0; i< 1; i++) {
	std::cout << dataArrays.h_num_high_quality_subject_indices[i] << "\t";
}
std::cout << std::endl;

//DEBUGGING
std::cout << "h_num_corrected_candidates" << std::endl;
for(int i = 0; i< dataArrays.n_subjects; i++) {
	std::cout << dataArrays.h_num_corrected_candidates[i] << "\t";
}
std::cout << std::endl;

//DEBUGGING
std::cout << "h_subject_is_corrected" << std::endl;
for(int i = 0; i< dataArrays.n_subjects; i++) {
	std::cout << dataArrays.h_subject_is_corrected[i] << "\t";
}
std::cout << std::endl;

//DEBUGGING
std::cout << "h_indices_of_corrected_candidates" << std::endl;
for(int i = 0; i< *dataArrays.h_num_indices; i++) {
	std::cout << dataArrays.h_indices_of_corrected_candidates[i] << "\t";
}
std::cout << std::endl;

#if 0
{
	auto& arrays = dataArrays[batchIndex];
	for(int row = 0; row < *arrays[batchIndex].h_num_indices+1 && row < 50; ++row) {
		for(int col = 0; col < arrays.msa_pitch; col++) {
			char c = arrays.h_multiple_sequence_alignments[row * arrays.msa_pitch + col];
			std::cout << (c == '\0' ? '0' : c);
		}
		std::cout << std::endl;
	}
}
#endif

#endif


#if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_PRINT_MSA

//DEBUGGING
for(std::size_t subject_index = 0; subject_index < mainBatch.tasks.size(); ++subject_index) {
	auto& task = mainBatch.tasks[subject_index];
	auto& arrays = dataArrays[batchIndex];

	const size_t msa_weights_pitch_floats = arrays.msa_weights_pitch / sizeof(float);

	const unsigned offset1 = arrays.msa_pitch * (subject_index + arrays.h_indices_per_subject_prefixsum[subject_index]);
	const unsigned offset2 = msa_weights_pitch_floats * (subject_index + arrays.h_indices_per_subject_prefixsum[subject_index]);

	const char* const my_multiple_sequence_alignment = arrays.h_multiple_sequence_alignments + offset1;
	const float* const my_multiple_sequence_alignment_weight = arrays.h_multiple_sequence_alignment_weights + offset2;

	char* const my_consensus = arrays.h_consensus + subject_index * arrays.msa_pitch;
	float* const my_support = arrays.h_support + subject_index * msa_weights_pitch_floats;
	int* const my_coverage = arrays.h_coverage + subject_index * msa_weights_pitch_floats;

	float* const my_orig_weights = arrays.h_origWeights + subject_index * msa_weights_pitch_floats;
	int* const my_orig_coverage = arrays.h_origCoverages + subject_index * msa_weights_pitch_floats;

	const int subjectColumnsBegin_incl = arrays.h_msa_column_properties[subject_index].subjectColumnsBegin_incl;
	const int subjectColumnsEnd_excl = arrays.h_msa_column_properties[subject_index].subjectColumnsEnd_excl;
	const int columnsToCheck = arrays.h_msa_column_properties[subject_index].columnsToCheck;

	// const unsigned offset1 = msa_row_pitch * (subjectIndex + candidates_before_this_subject);
	// char* const multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;

	const int msa_rows = 1 + arrays.h_indices_per_subject[subject_index];

	const int* const indices_for_this_subject = arrays.h_indices + arrays.h_indices_per_subject_prefixsum[subject_index];

	std::cout << "ReadId " << task.readId << ": msa rows = " << msa_rows << ", columnsToCheck = " << columnsToCheck << ", subjectColumnsBegin_incl = " << subjectColumnsBegin_incl << ", subjectColumnsEnd_excl = " << subjectColumnsEnd_excl << std::endl;
	std::cout << "MSA:" << std::endl;
	for(int row = 0; row < msa_rows; row++) {
		for(int col = 0; col < columnsToCheck; col++) {
			//multiple_sequence_alignment[row * msa_row_pitch + globalIndex]
			char c = my_multiple_sequence_alignment[row * arrays.msa_pitch + col];
			assert(c != 'F');
			std::cout << (c == '\0' ? '0' : c);
			if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
				std::cout << " ";
		}
		if(row > 0) {
			const int queryIndex = indices_for_this_subject[row-1];
			const int shift = arrays.h_alignment_shifts[queryIndex];

			std::cout << " shift " << shift;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Consensus: "<< std::endl;
	for(int col = 0; col < columnsToCheck; col++) {
		char c = my_consensus[col];
		std::cout << (c == '\0' ? '0' : c);
		if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
			std::cout << " ";
	}
	std::cout << std::endl;

	std::cout << "MSA weights:" << std::endl;
	for(int row = 0; row < msa_rows; row++) {
		for(int col = 0; col < columnsToCheck; col++) {
			float f = my_multiple_sequence_alignment_weight[row * msa_weights_pitch_floats + col];
			std::cout << f << " ";
			if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
				std::cout << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Support: "<< std::endl;
	for(int col = 0; col < columnsToCheck; col++) {
		std::cout << my_support[col] << " ";
	}
	std::cout << std::endl;

	std::cout << "Coverage: "<< std::endl;
	for(int col = 0; col < columnsToCheck; col++) {
		std::cout << my_coverage[col] << " ";
	}
	std::cout << std::endl;

	std::cout << "Orig weights: "<< std::endl;
	for(int col = 0; col < columnsToCheck; col++) {
		std::cout << my_orig_weights[col] << " ";
	}
	std::cout << std::endl;

	std::cout << "Orig coverage: "<< std::endl;
	for(int col = 0; col < columnsToCheck; col++) {
		std::cout << my_orig_coverage[col] << " ";
	}
	std::cout << std::endl;


}
#endif
#endif





#ifdef MSA_IMPLICIT
#undef MSA_IMPLICIT
#endif


#endif
