#ifndef CARE_GPU_CORRECTION_THREAD_HPP
#define CARE_GPU_CORRECTION_THREAD_HPP

//#define USE_NVTX2


#if defined USE_NVTX2 && defined __NVCC__
#include <nvToolsExt.h>

const uint32_t colors_[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff, 0xdeadbeef, 0x12345678 };
const int num_colors_ = sizeof(colors_)/sizeof(uint32_t);

#define PUSH_RANGE_2(name,cid) { \
		int color_id = cid; \
		color_id = color_id%num_colors_; \
		nvtxEventAttributes_t eventAttrib = {0}; \
		eventAttrib.version = NVTX_VERSION; \
		eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
		eventAttrib.colorType = NVTX_COLOR_ARGB; \
		eventAttrib.color = colors_[color_id]; \
		eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
		eventAttrib.message.ascii = name; \
		nvtxRangePushEx(&eventAttrib); \
}



#define POP_RANGE_2 nvtxRangePop();
#else
#define PUSH_RANGE_2(name,cid)
#define POP_RANGE_2
#endif


#ifdef __NVCC__
#include <nvToolsExt.h>
#endif

#ifdef __NVCC__

__inline__
void push_range(const std::string& name, int cid){
	const uint32_t colors_[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff, 0xdeadbeef, 0x12345678, 0xabcdef42 };
	const int num_colors_ = sizeof(colors_)/sizeof(uint32_t);

	int color_id = cid;
	color_id = color_id%num_colors_;
	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = colors_[color_id];
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = name.c_str();
	nvtxRangePushEx(&eventAttrib);
	//std::cout << "push " << name << std::endl;
}

__inline__
void pop_range(const std::string& name){
	nvtxRangePop();
	//std::cout << "pop " << name << std::endl;
}

void pop_range(){
	nvtxRangePop();
	//std::cout << "pop " << std::endl;
}

#else

__inline__
void push_range(const std::string& name, int cid){
}

__inline__
void pop_range(){
}


#endif

#include "../hpc_helpers.cuh"
#include "../options.hpp"
#include "../tasktiming.hpp"
#include "../sequence.hpp"
#include "../featureextractor.hpp"

#include "../forestclassifier.hpp"

#include "kernels.hpp"
#include "kernel_selection.hpp"
#include "dataarrays.hpp"

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

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif

//#define CARE_GPU_DEBUG
//#define CARE_GPU_DEBUG_MEMCOPY
//#define CARE_GPU_DEBUG_PRINT_ARRAYS
//#define CARE_GPU_DEBUG_PRINT_MSA



namespace care {
namespace gpu {

template<class ReadId_t>
struct BatchGenerator {
	BatchGenerator(){
	}

	BatchGenerator(ReadId_t firstId, ReadId_t lastIdExcl)
		: firstId(firstId), lastIdExcl(lastIdExcl), currentId(firstId){
		if(firstId >= lastIdExcl) throw std::runtime_error("BatchGenerator: firstId >= lastIdExcl");
	}

	std::vector<ReadId_t> getNextReadIds(int maxnumreadIds){
		std::vector<ReadId_t> result;
		while(int(result.size()) < maxnumreadIds && currentId < lastIdExcl) {
			result.push_back(currentId);
			currentId++;
		}
		return result;
	}

	bool empty() const {
		return currentId == lastIdExcl;
	}

	ReadId_t firstId;
	ReadId_t lastIdExcl;
	ReadId_t currentId;
};











template<class Sequence_t, class ReadId_t>
struct CorrectionTask {
	CorrectionTask(){
	}

	CorrectionTask(ReadId_t readId)
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
	ReadId_t readId;

	std::vector<ReadId_t> candidate_read_ids;
	ReadId_t* candidate_read_ids_begin;
	ReadId_t* candidate_read_ids_end;         // exclusive

	std::string subject_string;

	std::string corrected_subject;
	std::vector<std::string> corrected_candidates;
	std::vector<ReadId_t> corrected_candidates_read_ids;
};

#ifdef __NVCC__
template<class minhasher_t,
         class gpureadStorage_t,
         class readIdGenerator_t>
struct ErrorCorrectionThreadOnlyGPU {
	using Minhasher_t = minhasher_t;
	using GPUReadStorage_t = gpureadStorage_t;
	using Sequence_t = typename GPUReadStorage_t::Sequence_t;
	using ReadId_t = typename GPUReadStorage_t::ReadId_t;
	using CorrectionTask_t = CorrectionTask<Sequence_t, ReadId_t>;
	using ReadIdGenerator_t = readIdGenerator_t;


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
	static constexpr int nEventsPerBatch = 8;

	enum class BatchState {
		Unprepared,
		CopyReads,
		StartAlignment,
		WaitForIndices,
		CopyQualities,
		BuildMSA,
		StartClassicCorrection,
		StartForestCorrection,
		WaitForClassicResults,
		UnpackClassicResults,
		WriteResults,
		WriteFeatures,
		WaitForMSAData,
		Finished,
		Aborted,
	};

	enum class BatchStatePriority {
		Low,
		Medium,
		High,
	};

	struct Batch {
		std::vector<CorrectionTask_t> tasks;
		int initialNumberOfCandidates = 0;
		BatchState state = BatchState::Unprepared;

		int copiedTasks = 0;         // used if state == CandidatesPresent
		int copiedCandidates = 0;         // used if state == CandidatesPresent


		std::vector<ReadId_t> allReadIdsOfTasks;
		std::vector<ReadId_t> allReadIdsOfTasks_tmp;
		std::vector<char> collectedCandidateReads;
		int numsortedCandidateIds = 0;
		int numsortedCandidateIdTasks = 0;

		DataArrays<Sequence_t, ReadId_t>* dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>* streams;
		std::array<cudaEvent_t, nEventsPerBatch>* events;

		KernelLaunchHandle kernelLaunchHandle;

		void reset(){
			tasks.clear();
			allReadIdsOfTasks.clear();
			allReadIdsOfTasks_tmp.clear();
			collectedCandidateReads.clear();

			initialNumberOfCandidates = 0;
			state = BatchState::Unprepared;
			copiedTasks = 0;
			copiedCandidates = 0;
			numsortedCandidateIds = 0;
			numsortedCandidateIdTasks = 0;
		}
	};

	struct AdvanceResult {
		BatchState oldState = BatchState::Unprepared;
		BatchState newState = BatchState::Unprepared;
		bool noProgressBlocking = false;
		bool noProgressLaunching = false;
	};

	struct IsWaitingResult {
		bool isWaiting;
		int eventIndexToWaitFor;
	};

	struct TransitionFunctionData {
		//BatchGenerator<ReadId_t>* mybatchgen;
		ReadIdGenerator_t* readIdGenerator;
		std::vector<ReadId_t>* readIdBuffer;
		double min_overlap_ratio;
		int min_overlap;
		double estimatedErrorrate;
		double maxErrorRate;
		double estimatedCoverage;
		double m_coverage;
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
		std::function<void(const ReadId_t, const std::string&)> write_read_to_stream;
		std::function<void(const ReadId_t)> lock;
		std::function<void(const ReadId_t)> unlock;

        ForestClassifier fc;// = ForestClassifier{"./forests/testforest.so"};
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
		std::vector<char>* readIsProcessedVector;
		std::vector<char>* readIsCorrectedVector;
		std::mutex* locksForProcessedFlags;
		std::size_t nLocksForProcessedFlags;
	};

	AlignmentOptions alignmentOptions;
	GoodAlignmentProperties goodAlignmentProperties;
	CorrectionOptions correctionOptions;
	CorrectionThreadOptions threadOpts;
    FileOptions fileOptions;
	SequenceFileProperties fileProperties;

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

	//BatchGenerator<ReadId_t> mybatchgen;
	//int num_ids_per_add_tasks = 2;
	//int minimum_candidates_per_batch = 1000;
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

	void run(){
		if(isRunning) throw std::runtime_error("ErrorCorrectionThreadOnlyGPU::run: Is already running.");
		isRunning = true;
		thread = std::move(std::thread(&ErrorCorrectionThreadOnlyGPU::execute, this));
	}

	void join(){
		thread.join();
		isRunning = false;
	}

public:

	std::string nameOf(BatchState state) const {
		switch(state) {
		case BatchState::Unprepared: return "Unprepared";
		case BatchState::CopyReads: return "CopyReads";
		case BatchState::StartAlignment: return "StartAlignment";
		case BatchState::WaitForIndices: return "WaitForIndices";
		case BatchState::CopyQualities: return "CopyQualities";
		case BatchState::BuildMSA: return "BuildMSA";
		case BatchState::StartClassicCorrection: return "StartClassicCorrection";
		case BatchState::StartForestCorrection: return "StartForestCorrection";
		case BatchState::WaitForClassicResults: return "WaitForClassicResults";
		case BatchState::UnpackClassicResults: return "UnpackClassicResults";
		case BatchState::WriteResults: return "WriteResults";
		case BatchState::WriteFeatures: return "WriteFeatures";
		case BatchState::WaitForMSAData: return "WaitForMSAData";
		case BatchState::Finished: return "Finished";
		case BatchState::Aborted: return "Aborted";
		default: assert(false); return "None";
		}
	}

	IsWaitingResult isWaiting(BatchState state) const {
		switch(state) {
		case BatchState::Unprepared: return {false, -1};
		case BatchState::CopyReads: return {false, -1};
		case BatchState::StartAlignment: return {false, -1};
		case BatchState::WaitForIndices: return {true, indices_transfer_finished_event_index};
		case BatchState::CopyQualities: return {false, -1};
		case BatchState::BuildMSA: return {false, -1};
		case BatchState::StartClassicCorrection: return {false, -1};
		case BatchState::StartForestCorrection: return {false, -1};
		case BatchState::WaitForClassicResults: return {true, result_transfer_finished_event_index};
		case BatchState::UnpackClassicResults: return {false, -1};
		case BatchState::WriteResults: return {false, -1};
		case BatchState::WriteFeatures: return {false, -1};
		case BatchState::WaitForMSAData: return {true, msadata_transfer_finished_event_index};
		case BatchState::Finished: return {false, -1};
		case BatchState::Aborted: return {false, -1};
		default: assert(false); return {false, -1};
		}
	}

	void makeTransitionFunctionTable(){
		transitionFunctionTable[BatchState::Unprepared] = state_unprepared_func;
		transitionFunctionTable[BatchState::CopyReads] = state_copyreads_func;
		transitionFunctionTable[BatchState::StartAlignment] = state_startalignment_func;
		transitionFunctionTable[BatchState::WaitForIndices] = state_waitforindices_func;
		transitionFunctionTable[BatchState::CopyQualities] = state_copyqualities_func;
		transitionFunctionTable[BatchState::BuildMSA] = state_buildmsa_func;
		transitionFunctionTable[BatchState::StartClassicCorrection] = state_startclassiccorrection_func;
		transitionFunctionTable[BatchState::StartForestCorrection] = state_startforestcorrection_func;
		transitionFunctionTable[BatchState::WaitForClassicResults] = state_waitforclassicresults_func;
		transitionFunctionTable[BatchState::UnpackClassicResults] = state_unpackclassicresults_func;
		transitionFunctionTable[BatchState::WriteResults] = state_writeresults_func;
		transitionFunctionTable[BatchState::WriteFeatures] = state_writefeatures_func;
		transitionFunctionTable[BatchState::WaitForMSAData] = state_waitformsadata_func;
		transitionFunctionTable[BatchState::Finished] = state_finished_func;
		transitionFunctionTable[BatchState::Aborted] = state_aborted_func;
	}

	static BatchState state_unprepared_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::Unprepared);
		assert((batch.initialNumberOfCandidates == 0 && batch.tasks.empty()) || batch.initialNumberOfCandidates > 0);

		const int hits_per_candidate = transFuncData.correctionOptions.hits_per_candidate;

		DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		//std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		std::vector<ReadId_t>* readIdBuffer = transFuncData.readIdBuffer;

		auto erase_from_range = [](auto begin, auto end, auto position_to_erase){
						auto copybegin = position_to_erase;
						std::advance(copybegin, 1);
						return std::copy(copybegin, end, position_to_erase);
					};

		dataArrays.allocCandidateIds(transFuncData.minimum_candidates_per_batch + transFuncData.max_candidates);

		//dataArrays.h_candidates_per_subject_prefixsum[0] = 0;

		while(batch.initialNumberOfCandidates < transFuncData.minimum_candidates_per_batch
		      && !(transFuncData.readIdGenerator->empty() && readIdBuffer->empty())) {

			const auto* gpuReadStorage = transFuncData.gpuReadStorage;
			const auto& minhasher = transFuncData.minhasher;


			if(readIdBuffer->empty())
				*readIdBuffer = transFuncData.readIdGenerator->next_n(1000);

			if(readIdBuffer->empty())
				continue;

			ReadId_t id = readIdBuffer->back();
			readIdBuffer->pop_back();

			bool ok = false;
			transFuncData.lock(id);
			if ((*transFuncData.readIsCorrectedVector)[id] == 0) {
				(*transFuncData.readIsCorrectedVector)[id] = 1;
				ok = true;
			}else{
			}
			transFuncData.unlock(id);

			if(ok) {
				const char* sequenceptr = gpuReadStorage->fetchSequenceData_ptr(id);
				const int sequencelength = gpuReadStorage->fetchSequenceLength(id);

				//batch.tasks.emplace_back(id);

				CorrectionTask_t task(id);

				//auto& task = batch.tasks.back();

				task.subject_string = Sequence_t::Impl_t::toString((const std::uint8_t*)sequenceptr, sequencelength);
				task.candidate_read_ids = minhasher->getCandidates(task.subject_string, hits_per_candidate, transFuncData.max_candidates);

				//task.candidate_read_ids.resize(transFuncData.max_candidates);
				//auto vecend = minhasher->getCandidates(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.subject_string, transFuncData.max_candidates);
				//task.candidate_read_ids.resize(std::distance(task.candidate_read_ids.begin(), vecend));

				task.candidate_read_ids_begin = &(task.candidate_read_ids[0]);
				task.candidate_read_ids_end = &(task.candidate_read_ids[task.candidate_read_ids.size()]);


				//auto idsend = minhasher->getCandidates(dataArrays.h_candidate_read_ids + batch.initialNumberOfCandidates, dataArrays.h_candidate_read_ids + batch.initialNumberOfCandidates + transFuncData.max_candidates, task.subject_string, transFuncData.max_candidates);

				//task.candidate_read_ids_begin = dataArrays.h_candidate_read_ids + batch.initialNumberOfCandidates; //&(task.candidate_read_ids[0]);
				//task.candidate_read_ids_end = idsend; //&(task.candidate_read_ids[task.candidate_read_ids.size()]);

				//task.candidate_read_ids.resize(std::distance(dataArrays.h_candidate_read_ids + batch.initialNumberOfCandidates, idsend));
				//std::copy(dataArrays.h_candidate_read_ids + batch.initialNumberOfCandidates, dataArrays.h_candidate_read_ids + batch.initialNumberOfCandidates + transFuncData.max_candidates, task.candidate_read_ids.begin());



				//remove self from candidates
				//read ids are sorted
#if 0
				auto readIdPos = std::lower_bound(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.readId);
				if(readIdPos != task.candidate_read_ids.end() && *readIdPos == task.readId)
					task.candidate_read_ids.erase(readIdPos);

				if(task.candidate_read_ids.size() == 0) {
					//no need for further processing without candidates
					task.active = false;
				}

				batch.initialNumberOfCandidates += int(task.candidate_read_ids.size());
				assert(!task.active || batch.initialNumberOfCandidates > 0);
				assert(!task.active || int(task.candidate_read_ids.size()) > 0);
#else


				auto readIdPos = std::lower_bound(task.candidate_read_ids_begin, task.candidate_read_ids_end, task.readId);

				if(readIdPos != task.candidate_read_ids_end && *readIdPos == task.readId) {

					task.candidate_read_ids_end = erase_from_range(task.candidate_read_ids_begin, task.candidate_read_ids_end, readIdPos);
					task.candidate_read_ids.resize(std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end));
				}

				//assert(task.candidate_read_ids_begin == &(task.candidate_read_ids[0]));
				//assert(task.candidate_read_ids_end == &(task.candidate_read_ids[task.candidate_read_ids.size()]));

				std::size_t myNumCandidates = std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end);

				assert(myNumCandidates <= std::size_t(transFuncData.max_candidates));

				if(myNumCandidates > 0) {
					batch.tasks.emplace_back(task);
					batch.initialNumberOfCandidates += int(myNumCandidates);

					//dataArrays.h_candidates_per_subject_prefixsum[batch.tasks.size()]
					//		= dataArrays.h_candidates_per_subject_prefixsum[batch.tasks.size() - 1]
					//			+ int(myNumCandidates);
				}else{
					task.active = false;
				}

				assert(!task.active || batch.initialNumberOfCandidates > 0);
				assert(!task.active || int(myNumCandidates) > 0);
#endif

			}

			//only perform one iteration if pausable
			if(isPausable)
				break;
		}


		if(batch.initialNumberOfCandidates < transFuncData.minimum_candidates_per_batch
		   && !(transFuncData.readIdGenerator->empty() && readIdBuffer->empty())) {
			//still more read ids to add

			return BatchState::Unprepared;
		}else{

			if(batch.initialNumberOfCandidates == 0) {
				return BatchState::Aborted;
			}else{

				assert(batch.initialNumberOfCandidates < transFuncData.minimum_candidates_per_batch + transFuncData.max_candidates);

				//allocate data arrays

				dataArrays.set_problem_dimensions(int(batch.tasks.size()),
							batch.initialNumberOfCandidates,
							transFuncData.maxSequenceLength,
							transFuncData.min_overlap,
							transFuncData.min_overlap_ratio,
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

				max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);
				temp_storage_bytes = max_temp_storage_bytes;

				dataArrays.set_tmp_storage_size(max_temp_storage_bytes);
				dataArrays.zero_gpu(streams[primary_stream_index]);

				batch.initialNumberOfCandidates = 0;

				return BatchState::CopyReads;
			}
		}
	}

	static BatchState state_copyreads_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::CopyReads);
		assert(batch.copiedTasks <= int(batch.tasks.size()));

		DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		dataArrays.h_candidates_per_subject_prefixsum[0] = 0;
#if 0
		if(batch.copiedTasks == 0) {
			push_range("newsortids", 0);


			std::size_t totalNumIds = 0;
			for(int i = 0; i < int(batch.tasks.size()); i++) {
				const auto& task = batch.tasks[i];
				totalNumIds += std::distance(task.candidate_read_ids_begin,
							task.candidate_read_ids_end);
			}

			batch.allReadIdsOfTasks.reserve(totalNumIds);
			batch.allReadIdsOfTasks_tmp.reserve(totalNumIds);

			while(batch.numsortedCandidateIdTasks < int(batch.tasks.size())) {
				const auto& task = batch.tasks[batch.numsortedCandidateIdTasks];
				const std::size_t numIds = std::distance(task.candidate_read_ids_begin,
							task.candidate_read_ids_end);

				std::copy(task.candidate_read_ids_begin,
							task.candidate_read_ids_end,
							batch.allReadIdsOfTasks.begin() + batch.numsortedCandidateIds);

				batch.numsortedCandidateIds += numIds;
				++batch.numsortedCandidateIdTasks;
			}

			std::vector<ReadId_t> indexList(batch.numsortedCandidateIds);
			std::iota(indexList.begin(), indexList.end(), ReadId_t(0));
			std::sort(indexList.begin(), indexList.end(),
						[&](auto index1, auto index2){
						return batch.allReadIdsOfTasks[index1] < batch.allReadIdsOfTasks[index2];
					});

			std::vector<char> readsSortedById(dataArrays.encoded_sequence_pitch * batch.numsortedCandidateIds, 0);

			constexpr std::size_t prefetch_distance = 4;

			for(int i = 0; i < batch.numsortedCandidateIds && i < prefetch_distance; ++i) {
				const ReadId_t next_candidate_read_id = batch.allReadIdsOfTasks[indexList[i]];
				const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
				__builtin_prefetch(nextsequenceptr, 0, 0);
			}

			const std::size_t candidatesequencedatabytes = dataArrays.memQueries / sizeof(char);

			for(int i = 0; i < batch.numsortedCandidateIds; ++i) {
				if(i + prefetch_distance < batch.numsortedCandidateIds) {
					const ReadId_t next_candidate_read_id = batch.allReadIdsOfTasks[indexList[i + prefetch_distance]];
					const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
					__builtin_prefetch(nextsequenceptr, 0, 0);
				}

				const ReadId_t candidate_read_id = batch.allReadIdsOfTasks[indexList[i]];
				const char* sequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(candidate_read_id);
				const int sequencelength = transFuncData.gpuReadStorage->fetchSequenceLength(candidate_read_id);

				assert(i * dataArrays.encoded_sequence_pitch + Sequence_t::getNumBytes(sequencelength) <= candidatesequencedatabytes);

				std::memcpy(readsSortedById.data()
							+ i * dataArrays.encoded_sequence_pitch,
							sequenceptr,
							Sequence_t::getNumBytes(sequencelength));
			}

			batch.collectedCandidateReads.resize(dataArrays.encoded_sequence_pitch * batch.numsortedCandidateIds);
			for(int i = 0; i < batch.numsortedCandidateIds; ++i) {
				std::memcpy(batch.collectedCandidateReads.data()
							+ indexList[i] * dataArrays.encoded_sequence_pitch,
							readsSortedById.data()
							+ i * dataArrays.encoded_sequence_pitch,
							dataArrays.encoded_sequence_pitch);
			}

			pop_range();
		}
#endif

		/*while(batch.sortedTaskCandidateIds < int(batch.tasks.size())){
		    const auto& task = batch.tasks[batch.sortedTaskCandidateIds];
		    std::size_t oldSize = batch.allReadIdsOfTasks.size();
		    std::size_t numCandidatesOfTask = std::distance(task.candidate_read_ids_begin,
		                                                    task.candidate_read_ids_end);
		    batch.allReadIdsOfTasks_tmp.resize(oldSize + numCandidatesOfTask);
		    std::merge(batch.allReadIdsOfTasks.begin(),
		                                batch.allReadIdsOfTasks.end(),
		                                task.candidate_read_ids_begin,
		                                task.candidate_read_ids_end,
		                                batch.allReadIdsOfTasks_tmp.begin());

		    std::swap(batch.allReadIdsOfTasks, batch.allReadIdsOfTasks_tmp);

		 ++batch.sortedTaskCandidateIds;
		   }*/

		//copy one task
		while(batch.copiedTasks < int(batch.tasks.size())) {
			const auto& task = batch.tasks[batch.copiedTasks];
			auto& arrays = dataArrays;

			if(transFuncData.readStorageGpuData.isValidSequenceData()) {
				//copy read Ids
				arrays.h_subject_read_ids[batch.copiedTasks] = task.readId;
#if 0
				std::copy(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), arrays.h_candidate_read_ids + batch.copiedCandidates);

				batch.copiedCandidates += task.candidate_read_ids.size();

				//update prefix sum
				arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks+1]
				        = arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks]
				          + int(task.candidate_read_ids.size());
#else

				//assert(task.candidate_read_ids.size() == std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end));
				//assert(task.candidate_read_ids_begin == &(task.candidate_read_ids[0]));
				//assert(task.candidate_read_ids_end == &(task.candidate_read_ids[task.candidate_read_ids.size()]));

				const std::size_t h_candidate_read_ids_size = arrays.memCandidateIds / sizeof(ReadId_t);

				assert(std::size_t(batch.copiedCandidates) + std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end) <= h_candidate_read_ids_size);

				std::copy(task.candidate_read_ids_begin, task.candidate_read_ids_end, arrays.h_candidate_read_ids + batch.copiedCandidates);

				batch.copiedCandidates += std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end);

				//update prefix sum
				arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks+1]
				        = arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks]
				          + int(std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end));
#endif
			}else{
				//copy subject data
				const char* sequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(task.readId);
				const int sequencelength = transFuncData.gpuReadStorage->fetchSequenceLength(task.readId);

				const std::size_t maxbytes = arrays.memSubjects / sizeof(char);
				assert(batch.copiedTasks * arrays.encoded_sequence_pitch + Sequence_t::getNumBytes(sequencelength) <= maxbytes);
#if 1
				std::memcpy(arrays.h_subject_sequences_data + batch.copiedTasks * arrays.encoded_sequence_pitch,
							sequenceptr,
							Sequence_t::getNumBytes(sequencelength));
#else

				cudaMemcpyAsync(arrays.d_subject_sequences_data + batch.copiedTasks * arrays.encoded_sequence_pitch,
							sequenceptr,
							Sequence_t::getNumBytes(sequencelength),
							H2D,
							streams[primary_stream_index]);
#endif

				//copy subject length
				arrays.h_subject_sequences_lengths[batch.copiedTasks] = task.subject_string.length();

#if 1
				constexpr std::size_t prefetch_distance = 4;

				for(std::size_t i = 0; i < task.candidate_read_ids.size() && i < prefetch_distance; ++i) {
					const ReadId_t next_candidate_read_id = task.candidate_read_ids[i];
					const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
					__builtin_prefetch(nextsequenceptr, 0, 0);
				}

				const std::size_t candidatesequencedatabytes = arrays.memQueries / sizeof(char);

				for(std::size_t i = 0; i < task.candidate_read_ids.size(); ++i) {
					if(i + prefetch_distance < task.candidate_read_ids.size()) {
						const ReadId_t next_candidate_read_id = task.candidate_read_ids[i + prefetch_distance];
						const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
						__builtin_prefetch(nextsequenceptr, 0, 0);
					}

					const ReadId_t candidate_read_id = task.candidate_read_ids[i];
					const char* sequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(candidate_read_id);
					const int sequencelength = transFuncData.gpuReadStorage->fetchSequenceLength(candidate_read_id);

					assert(batch.copiedCandidates * arrays.encoded_sequence_pitch + Sequence_t::getNumBytes(sequencelength) <= candidatesequencedatabytes);

					std::memcpy(arrays.h_candidate_sequences_data
								+ batch.copiedCandidates * arrays.encoded_sequence_pitch,
								sequenceptr,
								Sequence_t::getNumBytes(sequencelength));

					arrays.h_candidate_sequences_lengths[batch.copiedCandidates] = sequencelength;

					++batch.copiedCandidates;
				}
#else

				constexpr std::size_t prefetch_distance = 4;

				for(std::size_t i = 0; i < task.candidate_read_ids.size() && i < prefetch_distance; ++i) {
					const ReadId_t next_candidate_read_id = task.candidate_read_ids[i];
					const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
					__builtin_prefetch(nextsequenceptr, 0, 0);
				}

				const std::size_t candidatesequencedatabytes = arrays.memQueries / sizeof(char);

				for(std::size_t i = 0; i < task.candidate_read_ids.size(); ++i) {
					if(i + prefetch_distance < task.candidate_read_ids.size()) {
						const ReadId_t next_candidate_read_id = task.candidate_read_ids[i + prefetch_distance];
						const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
						__builtin_prefetch(nextsequenceptr, 0, 0);
					}

					const ReadId_t candidate_read_id = task.candidate_read_ids[i];
					const char* sequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(candidate_read_id);
					const int sequencelength = transFuncData.gpuReadStorage->fetchSequenceLength(candidate_read_id);

					assert(batch.copiedCandidates * arrays.encoded_sequence_pitch + Sequence_t::getNumBytes(sequencelength) <= candidatesequencedatabytes);

					cudaMemcpyAsync(arrays.d_candidate_sequences_data
								+ batch.copiedCandidates * arrays.encoded_sequence_pitch,
								sequenceptr,
								Sequence_t::getNumBytes(sequencelength),
								H2D,
								streams[primary_stream_index]);

					arrays.h_candidate_sequences_lengths[batch.copiedCandidates] = sequencelength;

					++batch.copiedCandidates;
				}
#endif
				//update prefix sum
#if 0
				arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks+1]
				        = arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks]
				          + int(task.candidate_read_ids.size());
#else
				arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks+1]
				        = arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks]
				          + int(std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end));
#endif
			}

			++batch.copiedTasks;

			if(isPausable)
				break;
		}

		//if batch is fully copied, transfer to gpu
		if(batch.copiedTasks == int(batch.tasks.size())) {
			assert(batch.copiedTasks == int(batch.tasks.size()));

			if(transFuncData.readStorageGpuData.isValidSequenceData()) {
				dataArrays.h_candidates_per_subject_prefixsum[0] = 0;

				cudaMemcpyAsync(dataArrays.d_subject_read_ids,
							dataArrays.h_subject_read_ids,
							dataArrays.memSubjectIds,
							H2D,
							streams[primary_stream_index]); CUERR;

				cudaMemcpyAsync(dataArrays.d_candidate_read_ids,
							dataArrays.h_candidate_read_ids,
							dataArrays.memCandidateIds,
							H2D,
							streams[primary_stream_index]); CUERR;
				cudaMemcpyAsync(dataArrays.d_candidates_per_subject_prefixsum,
							dataArrays.h_candidates_per_subject_prefixsum,
							dataArrays.memNqueriesPrefixSum,
							H2D,
							streams[primary_stream_index]); CUERR;
			}else{

				/*int memcmpresult = std::memcmp(batch.collectedCandidateReads.data(),
				                                dataArrays.h_candidate_sequences_data,
				                                dataArrays.encoded_sequence_pitch * batch.numsortedCandidateIds);
				   if(memcmpresult == 0){
				    std::cerr << "ok\n";
				   }else{
				    std::cerr << "not ok\n";
				   }*/

#if 1
				cudaMemcpyAsync(dataArrays.alignment_transfer_data_device,
							dataArrays.alignment_transfer_data_host,
							dataArrays.alignment_transfer_data_usable_size,
							H2D,
							streams[primary_stream_index]); CUERR;
#else
				dataArrays.h_candidates_per_subject_prefixsum[0] = 0;

				/*cudaMemcpyAsync(dataArrays.d_subject_read_ids,
				                dataArrays.h_subject_read_ids,
				                dataArrays.memSubjectIds,
				                H2D,
				                streams[primary_stream_index]); CUERR;*/

				/*cudaMemcpyAsync(dataArrays.d_candidate_read_ids,
				                dataArrays.h_candidate_read_ids,
				                dataArrays.memCandidateIds,
				                H2D,
				                streams[primary_stream_index]); CUERR;*/
				cudaMemcpyAsync(dataArrays.d_subject_sequences_lengths,
							dataArrays.h_subject_sequences_lengths,
							dataArrays.memSubjectLengths,
							H2D,
							streams[primary_stream_index]); CUERR;
				cudaMemcpyAsync(dataArrays.d_candidate_sequences_lengths,
							dataArrays.h_candidate_sequences_lengths,
							dataArrays.memQueryLengths,
							H2D,
							streams[primary_stream_index]); CUERR;
				cudaMemcpyAsync(dataArrays.d_candidates_per_subject_prefixsum,
							dataArrays.h_candidates_per_subject_prefixsum,
							dataArrays.memNqueriesPrefixSum,
							H2D,
							streams[primary_stream_index]); CUERR;

#endif
			}

			/*cudaMemcpyAsync(dataArrays.d_candidate_read_ids,
			                                                dataArrays.h_candidate_read_ids,
			                                                dataArrays.candidate_ids_usable_size,
			                                                H2D,
			                                                streams[primary_stream_index]); CUERR;*/

			cudaEventRecord(events[alignment_data_transfer_h2d_finished_event_index], streams[primary_stream_index]); CUERR;

			batch.copiedTasks = 0;
			batch.copiedCandidates = 0;
			return BatchState::StartAlignment;
		}else{
			return BatchState::CopyReads;
		}
	}

	static BatchState state_startalignment_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::StartAlignment);

		DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		if(!canLaunchKernel) {

			return BatchState::StartAlignment;
		}

		//cudaStreamWaitEvent(streams[primary_stream_index], events[alignment_data_transfer_h2d_finished_event_index], 0); CUERR;

		auto select_alignment_op = [] __device__ (const BestAlignment_t& flag){
			return flag != BestAlignment_t::None;
		};

		/*
		    Writes indices of candidates with alignmentflag != None to dataArrays.d_indices
		    Writes number of writen indices to dataArrays.d_num_indices
		    Elements after d_indices[d_num_indices - 1] are set to -1;
		 */
		auto select_alignments_by_flag = [&](auto dataArray, cudaStream_t stream){
							 dataArray.fill_d_indices(-1, stream);

							 cub::TransformInputIterator<bool,decltype(select_alignment_op), BestAlignment_t*>
							 d_isGoodAlignment(dataArray.d_alignment_best_alignment_flags,
							                   select_alignment_op);

							 cub::DeviceSelect::Flagged(dataArray.d_temp_storage,
										 dataArray.tmp_storage_allocation_size,
										 cub::CountingInputIterator<int>(0),
										 d_isGoodAlignment,
										 dataArray.d_indices,
										 dataArray.d_num_indices,
										 //nTotalCandidates,
										 dataArray.n_queries,
										 stream); CUERR;
						 };

		ShiftedHammingDistanceChooserExp<Sequence_t, ReadId_t>::callKernelAsync(
					dataArrays.d_alignment_scores,
					dataArrays.d_alignment_overlaps,
					dataArrays.d_alignment_shifts,
					dataArrays.d_alignment_nOps,
					dataArrays.d_alignment_isValid,
					dataArrays.d_subject_read_ids,
					dataArrays.d_candidate_read_ids,
					dataArrays.d_subject_sequences_data,
					dataArrays.d_candidate_sequences_data,
					dataArrays.d_subject_sequences_lengths,
					dataArrays.d_candidate_sequences_lengths,
					dataArrays.d_candidates_per_subject_prefixsum,
					dataArrays.n_subjects,
					dataArrays.n_queries,
					Sequence_t::getNumBytes(dataArrays.maximum_sequence_length),
					dataArrays.encoded_sequence_pitch,
					transFuncData.min_overlap,
					transFuncData.maxErrorRate, //correctionOptions.estimatedErrorrate * 4.0,
					transFuncData.min_overlap_ratio,
					//batch.maxSubjectLength,
					//batch.maxQueryLength,
					transFuncData.gpuReadStorage,
					transFuncData.readStorageGpuData,
					true,
					streams[primary_stream_index],
					batch.kernelLaunchHandle);

		//Step 5. Compare each forward alignment with the correspoding reverse complement alignment and keep the best, if any.
		//    If reverse complement is the best, it is copied into the first half, replacing the forward alignment

		FindBestAlignmentChooserExp<ReadId_t>::callKernelAsync(
					dataArrays.d_alignment_best_alignment_flags,
					dataArrays.d_alignment_scores,
					dataArrays.d_alignment_overlaps,
					dataArrays.d_alignment_shifts,
					dataArrays.d_alignment_nOps,
					dataArrays.d_alignment_isValid,
					dataArrays.d_candidates_per_subject_prefixsum,
					transFuncData.min_overlap_ratio,
					transFuncData.min_overlap,
					transFuncData.maxErrorRate,
					transFuncData.estimatedErrorrate,
					dataArrays.d_subject_read_ids,
					dataArrays.d_candidate_read_ids,
					dataArrays.d_subject_sequences_lengths,
					dataArrays.d_candidate_sequences_lengths,
					dataArrays.n_subjects,
					dataArrays.n_queries,
					transFuncData.gpuReadStorage,
					transFuncData.readStorageGpuData,
					true,
					streams[primary_stream_index],
					batch.kernelLaunchHandle);



		//Determine indices where d_alignment_best_alignment_flags[i] != BestAlignment_t::None. this selects all good alignments
		//select_alignments_by_flag(dataArrays, streams[primary_stream_index]);

		//Get number of indices per subject by creating histrogram.
		//The elements d_indices[d_num_indices] to d_indices[n_queries - 1] will be -1.
		//Thus, they will not be accounted for by the histrogram, since the histrogram bins (d_candidates_per_subject_prefixsum) are >= 0.
		/*cub::DeviceHistogram::HistogramRange(dataArrays.d_temp_storage,
		                                    dataArrays.tmp_storage_allocation_size,
		                                    dataArrays.d_indices,
		                                    dataArrays.d_indices_per_subject,
		                                    dataArrays.n_subjects+1,
		                                    dataArrays.d_candidates_per_subject_prefixsum,
		                                    dataArrays.n_queries,
		                                    streams[primary_stream_index]); CUERR;

		   cub::DeviceScan::ExclusiveSum(dataArrays.d_temp_storage,
		                                dataArrays.tmp_storage_allocation_size,
		                                dataArrays.d_indices_per_subject,
		                                dataArrays.d_indices_per_subject_prefixsum,
		                                dataArrays.n_subjects,
		                                streams[primary_stream_index]); CUERR;*/

		//choose the most appropriate subset of alignments from the good alignments.
		//This sets d_alignment_best_alignment_flags[i] = BestAlignment_t::None for all non-appropriate alignments
		call_cuda_filter_alignments_by_mismatchratio_kernel_async(
					dataArrays.d_alignment_best_alignment_flags,
					dataArrays.d_alignment_overlaps,
					dataArrays.d_alignment_nOps,
					//dataArrays.d_indices,
					cub::CountingInputIterator<int>(0),
					dataArrays.d_indices_per_subject,
					//dataArrays.d_indices_per_subject_prefixsum,
					dataArrays.d_candidates_per_subject_prefixsum,
					dataArrays.n_subjects,
					dataArrays.n_queries,
					dataArrays.d_num_indices,
					transFuncData.estimatedErrorrate,
					transFuncData.estimatedCoverage * transFuncData.m_coverage,
					streams[primary_stream_index],
					batch.kernelLaunchHandle);

		//determine indices of remaining alignments
		select_alignments_by_flag(dataArrays, streams[primary_stream_index]);

		//update indices_per_subject
		cub::DeviceHistogram::HistogramRange(dataArrays.d_temp_storage,
					dataArrays.tmp_storage_allocation_size,
					dataArrays.d_indices,
					dataArrays.d_indices_per_subject,
					dataArrays.n_subjects+1,
					dataArrays.d_candidates_per_subject_prefixsum,
					// *dataArrays.h_num_indices,
					dataArrays.n_queries,
					streams[primary_stream_index]); CUERR;

		//Make indices_per_subject_prefixsum
		cub::DeviceScan::ExclusiveSum(dataArrays.d_temp_storage,
					dataArrays.tmp_storage_allocation_size,
					dataArrays.d_indices_per_subject,
					dataArrays.d_indices_per_subject_prefixsum,
					dataArrays.n_subjects,
					streams[primary_stream_index]); CUERR;

		//cudaMemcpyAsync(dataArrays.h_num_indices, dataArrays.d_num_indices, sizeof(int), D2H, streams[primary_stream_index]); CUERR;

		assert(cudaSuccess == cudaEventQuery(events[alignments_finished_event_index])); CUERR;

		cudaEventRecord(events[alignments_finished_event_index], streams[primary_stream_index]); CUERR;

		//copy indices of usable candidates. these are required on the host for coping quality scores, and for creating results.
		cudaStreamWaitEvent(streams[secondary_stream_index], events[alignments_finished_event_index], 0); CUERR;

		cudaMemcpyAsync(dataArrays.indices_transfer_data_host,
					dataArrays.indices_transfer_data_device,
					dataArrays.indices_transfer_data_usable_size,
					D2H,
					streams[secondary_stream_index]); CUERR;

		assert(cudaSuccess == cudaEventQuery(events[indices_transfer_finished_event_index])); CUERR;

		cudaEventRecord(events[indices_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;

		if(transFuncData.correctionOptions.useQualityScores) {
			if(transFuncData.readStorageGpuData.isValidQualityData()) {
				return BatchState::BuildMSA;
			}else{
				// need indices to copy individual quality scores
				return BatchState::WaitForIndices;
			}
		}else{
			return BatchState::BuildMSA;
		}
	}

	static BatchState state_waitforindices_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::WaitForIndices);

		//DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		//std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		cudaError_t querystatus = cudaEventQuery(events[indices_transfer_finished_event_index]); CUERR;

		assert(querystatus == cudaSuccess || querystatus == cudaErrorNotReady);

		if(querystatus == cudaSuccess) {
			return BatchState::CopyQualities;
		}else{
			return BatchState::WaitForIndices;
		}
	}

	static BatchState state_copyqualities_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::CopyQualities);

		DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;
		const auto* gpuReadStorage = transFuncData.gpuReadStorage;

		if(transFuncData.correctionOptions.useQualityScores) {

			//if(transFuncData.useGpuReadStorage && transFuncData.gpuReadStorage->type == GPUReadStorageType::SequencesAndQualities){
			if(transFuncData.readStorageGpuData.isValidQualityData()) {
				//we don't need to copy any quality strings. they are already present in the gpu read storage
				return BatchState::StartClassicCorrection;
			}else{

				assert(batch.copiedTasks <= int(batch.tasks.size()));

				const std::size_t maxsubjectqualitychars = dataArrays.n_subjects * dataArrays.quality_pitch;
				const std::size_t maxcandidatequalitychars = dataArrays.n_queries * dataArrays.quality_pitch;

				while(batch.copiedTasks < int(batch.tasks.size())) {

					const auto& task = batch.tasks[batch.copiedTasks];
					auto& arrays = dataArrays;

					//copy subject quality

					const int sequencelength = transFuncData.gpuReadStorage->fetchSequenceLength(task.readId);
					const char* subject_quality = gpuReadStorage->fetchQuality_ptr(task.readId);
					assert(batch.copiedTasks * arrays.quality_pitch + sequencelength <= maxsubjectqualitychars);
					std::memcpy(arrays.h_subject_qualities + batch.copiedTasks * arrays.quality_pitch,
								subject_quality,
								sequencelength);

					//copy qualities of candidates identified by indices

					const int my_num_indices = dataArrays.h_indices_per_subject[batch.copiedTasks];
					const int* my_indices = dataArrays.h_indices + dataArrays.h_indices_per_subject_prefixsum[batch.copiedTasks];
					const int candidatesOfPreviousTasks = dataArrays.h_candidates_per_subject_prefixsum[batch.copiedTasks];

					constexpr int prefetch_distance = 4;

					for(int i = 0; i < my_num_indices && i < prefetch_distance; ++i) {
						const int next_candidate_index = my_indices[i];
						const int next_local_candidate_index = next_candidate_index - candidatesOfPreviousTasks;

						const char* next_qual = gpuReadStorage->fetchQuality_ptr(task.candidate_read_ids_begin[next_local_candidate_index]);
						__builtin_prefetch(next_qual, 0, 0);
					}


					for(int i = 0; i < my_num_indices; ++i) {
						if(i+prefetch_distance < my_num_indices) {
							const int next_candidate_index = my_indices[i+prefetch_distance];
							const int next_local_candidate_index = next_candidate_index - candidatesOfPreviousTasks;
							const char* next_qual = gpuReadStorage->fetchQuality_ptr(task.candidate_read_ids_begin[next_local_candidate_index]);
							__builtin_prefetch(next_qual, 0, 0);
						}
						const int candidate_index = my_indices[i];
						const int local_candidate_index = candidate_index - candidatesOfPreviousTasks;

						//assert(batch.copiedCandidates * arrays.quality_pitch + qual->length() <= maxcandidatequalitychars);

						const int sequencelength = transFuncData.gpuReadStorage->fetchSequenceLength(task.candidate_read_ids_begin[local_candidate_index]);
						const char* candidate_quality = gpuReadStorage->fetchQuality_ptr(task.candidate_read_ids_begin[local_candidate_index]);
						assert(batch.copiedCandidates * arrays.quality_pitch + sequencelength <= maxcandidatequalitychars);
						std::memcpy(arrays.h_candidate_qualities + batch.copiedCandidates * arrays.quality_pitch,
									candidate_quality,
									sequencelength);

						++batch.copiedCandidates;
					}

					++batch.copiedTasks;

					if(isPausable)
						break;
				}

				//gather_quality_scores_of_next_task(batch, dataArrays);

				//if batch is fully copied to pinned memory, transfer to gpu
				if(batch.copiedTasks == int(batch.tasks.size())) {
					cudaMemcpyAsync(dataArrays.qualities_transfer_data_device,
								dataArrays.qualities_transfer_data_host,
								dataArrays.qualities_transfer_data_usable_size,
								H2D,
								streams[secondary_stream_index]); CUERR;

					assert(cudaSuccess == cudaEventQuery(events[quality_transfer_finished_event_index])); CUERR;

					cudaEventRecord(events[quality_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;
					batch.copiedTasks = 0;
					batch.copiedCandidates = 0;
					return BatchState::BuildMSA;
				}else{
					return BatchState::CopyQualities;
				}
			}
		}else{
			return BatchState::BuildMSA;
		}
	}

	static BatchState state_buildmsa_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::BuildMSA);

		DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		if(!canLaunchKernel) {
			return BatchState::BuildMSA;
		}else{

			cudaStreamWaitEvent(streams[primary_stream_index], events[quality_transfer_finished_event_index], 0); CUERR;

			// coverage is always >= 1
			const double min_coverage_threshold = std::max(1.0,
						transFuncData.m_coverage / 6.0 * transFuncData.estimatedCoverage);

			const float desiredAlignmentMaxErrorRate = transFuncData.maxErrorRate;

			//Determine multiple sequence alignment properties

			MSAInitChooserExp<ReadId_t>::callKernelAsync(
						dataArrays.d_msa_column_properties,
						dataArrays.d_alignment_shifts,
						dataArrays.d_alignment_best_alignment_flags,
						dataArrays.d_subject_read_ids,
						dataArrays.d_candidate_read_ids,
						dataArrays.d_subject_sequences_lengths,
						dataArrays.d_candidate_sequences_lengths,
						dataArrays.d_indices,
						dataArrays.d_indices_per_subject,
						dataArrays.d_indices_per_subject_prefixsum,
						dataArrays.n_subjects,
						dataArrays.n_queries,
						transFuncData.gpuReadStorage,
						transFuncData.readStorageGpuData,
						true,
						streams[primary_stream_index],
						batch.kernelLaunchHandle);

			MSAAddSequencesChooserExp<Sequence_t, ReadId_t>::callKernelAsync(
						dataArrays.d_multiple_sequence_alignments,
						dataArrays.d_multiple_sequence_alignment_weights,
						dataArrays.d_alignment_shifts,
						dataArrays.d_alignment_best_alignment_flags,
						dataArrays.d_subject_read_ids,
						dataArrays.d_candidate_read_ids,
						dataArrays.d_subject_sequences_data,
						dataArrays.d_candidate_sequences_data,
						dataArrays.d_subject_sequences_lengths,
						dataArrays.d_candidate_sequences_lengths,
						dataArrays.d_subject_qualities,
						dataArrays.d_candidate_qualities,
						dataArrays.d_alignment_overlaps,
						dataArrays.d_alignment_nOps,
						dataArrays.d_msa_column_properties,
						dataArrays.d_candidates_per_subject_prefixsum,
						dataArrays.d_indices,
						dataArrays.d_indices_per_subject,
						dataArrays.d_indices_per_subject_prefixsum,
						dataArrays.n_subjects,
						dataArrays.n_queries,
						dataArrays.d_num_indices,
						transFuncData.correctionOptions.useQualityScores,
						desiredAlignmentMaxErrorRate,
						dataArrays.maximum_sequence_length,
						Sequence_t::getNumBytes(dataArrays.maximum_sequence_length),
						dataArrays.encoded_sequence_pitch,
						dataArrays.quality_pitch,
						dataArrays.msa_pitch,
						dataArrays.msa_weights_pitch,
						//true,
						transFuncData.gpuReadStorage,
						transFuncData.readStorageGpuData,
						true,
						streams[primary_stream_index],
						batch.kernelLaunchHandle);

			call_msa_find_consensus_kernel_async(
						dataArrays.d_consensus,
						dataArrays.d_support,
						dataArrays.d_coverage,
						dataArrays.d_origWeights,
						dataArrays.d_origCoverages,
						dataArrays.d_multiple_sequence_alignments,
						dataArrays.d_multiple_sequence_alignment_weights,
						dataArrays.d_msa_column_properties,
						dataArrays.d_candidates_per_subject_prefixsum,
						dataArrays.d_indices_per_subject,
						dataArrays.d_indices_per_subject_prefixsum,
						dataArrays.n_subjects,
						dataArrays.n_queries,
						dataArrays.d_num_indices,
						dataArrays.msa_pitch,
						dataArrays.msa_weights_pitch,
						3*dataArrays.maximum_sequence_length - 2*transFuncData.min_overlap,
						streams[primary_stream_index],
						batch.kernelLaunchHandle);

			assert(cudaSuccess == cudaEventQuery(events[msa_build_finished_event_index])); CUERR;

			cudaEventRecord(events[msa_build_finished_event_index], streams[primary_stream_index]); CUERR;

			if(transFuncData.correctionOptions.extractFeatures || !transFuncData.correctionOptions.classicMode) {

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

				cudaEventRecord(events[msadata_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;
			}

			if(transFuncData.correctionOptions.classicMode) {
				return BatchState::StartClassicCorrection;
			}else{
				return BatchState::StartForestCorrection;
			}
		}
	}

	static BatchState state_startclassiccorrection_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::StartClassicCorrection);
		assert(transFuncData.correctionOptions.classicMode);

		DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		if(!canLaunchKernel) {
			return BatchState::StartClassicCorrection;
		}else{

			const double avg_support_threshold = 1.0-1.0*transFuncData.estimatedErrorrate;
			const double min_support_threshold = 1.0-3.0*transFuncData.estimatedErrorrate;
			// coverage is always >= 1
			const double min_coverage_threshold = std::max(1.0,
						transFuncData.m_coverage / 6.0 * transFuncData.estimatedCoverage);
			const int new_columns_to_correct = transFuncData.new_columns_to_correct;

			// Step 14. Correction
			if(transFuncData.correctionOptions.classicMode) {

				// correct subjects
				call_msa_correct_subject_kernel_async(
							dataArrays.d_consensus,
							dataArrays.d_support,
							dataArrays.d_coverage,
							dataArrays.d_origCoverages,
							dataArrays.d_multiple_sequence_alignments,
							dataArrays.d_msa_column_properties,
							dataArrays.d_indices_per_subject_prefixsum,
							dataArrays.d_is_high_quality_subject,
							dataArrays.d_corrected_subjects,
							dataArrays.d_subject_is_corrected,
							dataArrays.n_subjects,
							dataArrays.n_queries,
							dataArrays.d_num_indices,
							dataArrays.sequence_pitch,
							dataArrays.msa_pitch,
							dataArrays.msa_weights_pitch,
							transFuncData.estimatedErrorrate,
							avg_support_threshold,
							min_support_threshold,
							min_coverage_threshold,
							transFuncData.kmerlength,
							dataArrays.maximum_sequence_length,
							streams[primary_stream_index],
							batch.kernelLaunchHandle);

				if(transFuncData.correctionOptions.correctCandidates) {


					// find subject ids of subjects with high quality multiple sequence alignment

					cub::DeviceSelect::Flagged(dataArrays.d_temp_storage,
								dataArrays.tmp_storage_allocation_size,
								cub::CountingInputIterator<int>(0),
								dataArrays.d_is_high_quality_subject,
								dataArrays.d_high_quality_subject_indices,
								dataArrays.d_num_high_quality_subject_indices,
								dataArrays.n_subjects,
								streams[primary_stream_index]); CUERR;

					// correct candidates

					MSACorrectCandidatesChooserExp<ReadId_t>::callKernelAsync(
								dataArrays.d_consensus,
								dataArrays.d_support,
								dataArrays.d_coverage,
								dataArrays.d_origCoverages,
								dataArrays.d_multiple_sequence_alignments,
								dataArrays.d_msa_column_properties,
								dataArrays.d_indices,
								dataArrays.d_indices_per_subject,
								dataArrays.d_indices_per_subject_prefixsum,
								dataArrays.d_high_quality_subject_indices,
								dataArrays.d_num_high_quality_subject_indices,
								dataArrays.d_alignment_shifts,
								dataArrays.d_alignment_best_alignment_flags,
								dataArrays.d_candidate_read_ids,
								dataArrays.d_candidate_sequences_lengths,
								dataArrays.d_num_corrected_candidates,
								dataArrays.d_corrected_candidates,
								dataArrays.d_indices_of_corrected_candidates,
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
								transFuncData.gpuReadStorage,
								transFuncData.readStorageGpuData,
								true,
								streams[primary_stream_index],
								batch.kernelLaunchHandle);
				}
			}
			assert(cudaSuccess == cudaEventQuery(events[correction_finished_event_index])); CUERR;

			cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;

			cudaMemcpyAsync(dataArrays.correction_results_transfer_data_host,
						dataArrays.correction_results_transfer_data_device,
						dataArrays.correction_results_transfer_data_usable_size,
						D2H,
						streams[primary_stream_index]); CUERR;

			assert(cudaSuccess == cudaEventQuery(events[result_transfer_finished_event_index])); CUERR;

			cudaEventRecord(events[result_transfer_finished_event_index], streams[primary_stream_index]); CUERR;

			return BatchState::WaitForClassicResults;
		}
	}

	static BatchState state_startforestcorrection_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::StartForestCorrection);
		assert(!transFuncData.correctionOptions.classicMode);

		DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		//std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		cudaError_t querystatus = cudaEventQuery(events[msadata_transfer_finished_event_index]); CUERR;
		assert(querystatus == cudaSuccess || querystatus == cudaErrorNotReady);

		if(querystatus != cudaSuccess)
			return BatchState::WaitForMSAData;

		if(!canLaunchKernel) {
			return BatchState::StartForestCorrection;
		}else{

			for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {
				auto& task = batch.tasks[subject_index];
				const auto& columnProperties = dataArrays.h_msa_column_properties[subject_index];
				const std::size_t msa_weights_pitch_floats = dataArrays.msa_weights_pitch / sizeof(float);

				task.corrected_subject = task.subject_string;

				const char* cons = dataArrays.h_consensus + subject_index * dataArrays.msa_pitch;

				std::vector<MSAFeature> MSAFeatures = extractFeatures(cons,
							dataArrays.h_support + subject_index * msa_weights_pitch_floats,
							dataArrays.h_coverage + subject_index * msa_weights_pitch_floats,
							dataArrays.h_origCoverages + subject_index * msa_weights_pitch_floats,
							columnProperties.columnsToCheck,
							columnProperties.subjectColumnsBegin_incl,
							columnProperties.subjectColumnsEnd_excl,
							task.subject_string,
							transFuncData.minhasher->minparams.k, 0.0,
							transFuncData.estimatedCoverage);

				for(const auto& msafeature : MSAFeatures) {
					constexpr double maxgini = 0.05;
					constexpr double forest_correction_fraction = 0.5;
    //care::ForestClassifier fc("./forests/testforest.so");
#if 0
					const bool doCorrect = care::forestclassifier::shouldCorrect(
								//care::forestclassifier::Mode::CombinedAlignCov,
								//care::forestclassifier::Mode::CombinedDataCov,
								care::forestclassifier::Mode::Species,
								msafeature.position_support,
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
#else
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
#endif
					if(doCorrect) {
						task.corrected = true;

						const int globalIndex = columnProperties.subjectColumnsBegin_incl + msafeature.position;
						task.corrected_subject[msafeature.position] = cons[globalIndex];
					}
				}
			}

			return BatchState::WriteResults;
		}
	}

	static BatchState state_waitforclassicresults_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::WaitForClassicResults);

		//DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		//std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		cudaError_t querystatus1 = cudaEventQuery(events[result_transfer_finished_event_index]); CUERR;
		assert(querystatus1 == cudaSuccess || querystatus1 == cudaErrorNotReady);

		cudaError_t querystatus2 = cudaSuccess;

		if(transFuncData.correctionOptions.useQualityScores) {
			if(transFuncData.readStorageGpuData.isValidQualityData()) {
				querystatus2 = cudaEventQuery(events[indices_transfer_finished_event_index]); CUERR;
			}else{
				// if we needed to copy individual quality scores, we already waited for indices to copy the quality scores. no need to wait again
				querystatus2 = cudaSuccess;
			}
		}else{
			querystatus2 = cudaEventQuery(events[indices_transfer_finished_event_index]); CUERR;
		}

		assert(querystatus2 == cudaSuccess || querystatus2 == cudaErrorNotReady);

		if(querystatus1 == cudaSuccess && querystatus2 == cudaSuccess) {
			return BatchState::UnpackClassicResults;
		}else{
			return BatchState::WaitForClassicResults;
		}
	}

	static BatchState state_waitformsadata_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::WaitForMSAData);

		//DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		//std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		cudaError_t querystatus = cudaEventQuery(events[msadata_transfer_finished_event_index]); CUERR;
		assert(querystatus == cudaSuccess || querystatus == cudaErrorNotReady);

		if(querystatus == cudaSuccess) {
			return BatchState::StartForestCorrection;
		}else{
			return BatchState::WaitForMSAData;
		}
	}

	static BatchState state_unpackclassicresults_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::UnpackClassicResults);

		DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		//std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		assert(cudaEventQuery(events[correction_finished_event_index]) == cudaSuccess); CUERR;

	    #if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_MEMCOPY
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		//DEBUGGING
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


	    #endif

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
	    #endif

	    #if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_PRINT_MSA

		//DEBUGGING
		for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {
			auto& task = batch.tasks[subject_index];
			auto& arrays = dataArrays;

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

		assert(transFuncData.correctionOptions.classicMode);

		for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {
			auto& task = batch.tasks[subject_index];
			auto& arrays = dataArrays;

			const char* const my_corrected_subject_data = arrays.h_corrected_subjects + subject_index * arrays.sequence_pitch;
			const char* const my_corrected_candidates_data = arrays.h_corrected_candidates + arrays.h_indices_per_subject_prefixsum[subject_index] * arrays.sequence_pitch;
			const int* const my_indices_of_corrected_candidates = arrays.h_indices_of_corrected_candidates + arrays.h_indices_per_subject_prefixsum[subject_index];

			const bool any_correction_candidates = arrays.h_indices_per_subject[subject_index] > 0;

			//has subject been corrected ?
			task.corrected = arrays.h_subject_is_corrected[subject_index];
			//if corrected, copy corrected subject to task
			if(task.corrected) {

				const int subject_length = task.subject_string.length();
				//const int subject_length = if(transFuncData.useGpuReadStorage && transFuncData.gpuReadStorage->hasSequences()) task.subject_sequence->length();

				task.corrected_subject = std::move(std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length});
			}

			if(transFuncData.correctCandidates) {
				const int n_corrected_candidates = arrays.h_num_corrected_candidates[subject_index];

				assert((!any_correction_candidates && n_corrected_candidates == 0) || any_correction_candidates);

				for(int i = 0; i < n_corrected_candidates; ++i) {
					const int global_candidate_index = my_indices_of_corrected_candidates[i];
					const int local_candidate_index = global_candidate_index - arrays.h_candidates_per_subject_prefixsum[subject_index];

					//const ReadId_t candidate_read_id = task.candidate_read_ids[local_candidate_index];
					const ReadId_t candidate_read_id = task.candidate_read_ids_begin[local_candidate_index];
					const int candidate_length = transFuncData.gpuReadStorage->fetchSequenceLength(candidate_read_id);
					//const int candidate_length = task.candidate_sequences[local_candidate_index]->length();
					const char* const candidate_data = my_corrected_candidates_data + i * arrays.sequence_pitch;

					//task.corrected_candidates_read_ids.emplace_back(task.candidate_read_ids[local_candidate_index]);
					task.corrected_candidates_read_ids.emplace_back(task.candidate_read_ids_begin[local_candidate_index]);

					task.corrected_candidates.emplace_back(std::move(std::string{candidate_data, candidate_data + candidate_length}));
				}
			}
		}

		return BatchState::WriteResults;
	}

	static BatchState state_writeresults_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::WriteResults);

		//DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		//std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		//std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		//write result to file
		for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {

			const auto& task = batch.tasks[subject_index];
			//std::cout << task.readId << "result" << std::endl;

			//std::cout << "finished readId " << task.readId << std::endl;

			if(task.corrected) {
				push_range("write_subject", 4);
				//std::cout << task.readId << " " << task.corrected_subject << std::endl;
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
					pop_range();
				}
			}
			push_range("correctedcandidates", 6);
			for(std::size_t corrected_candidate_index = 0; corrected_candidate_index < task.corrected_candidates.size(); ++corrected_candidate_index) {

				ReadId_t candidateId = task.corrected_candidates_read_ids[corrected_candidate_index];
				const std::string& corrected_candidate = task.corrected_candidates[corrected_candidate_index];

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
			}
			pop_range();
		}

		if(transFuncData.correctionOptions.extractFeatures)
			return BatchState::WriteFeatures;
		else
			return BatchState::Finished;
	}

	static BatchState state_writefeatures_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::WriteFeatures);

		DataArrays<Sequence_t, ReadId_t>& dataArrays = *batch.dataArrays;
		//std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		cudaEventSynchronize(events[msadata_transfer_finished_event_index]); CUERR;

		auto& featurestream = *transFuncData.featurestream;

		for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {

			const auto& task = batch.tasks[subject_index];
			const auto& columnProperties = dataArrays.h_msa_column_properties[subject_index];

			//std::cout << task.readId << "feature" << std::endl;

			const std::size_t msa_weights_pitch_floats = dataArrays.msa_weights_pitch / sizeof(float);

			std::vector<MSAFeature> MSAFeatures = extractFeatures(dataArrays.h_consensus + subject_index * dataArrays.msa_pitch,
						dataArrays.h_support + subject_index * msa_weights_pitch_floats,
						dataArrays.h_coverage + subject_index * msa_weights_pitch_floats,
						dataArrays.h_origCoverages + subject_index * msa_weights_pitch_floats,
						columnProperties.columnsToCheck,
						columnProperties.subjectColumnsBegin_incl,
						columnProperties.subjectColumnsEnd_excl,
						task.subject_string,
						transFuncData.minhasher->minparams.k, 0.0,
						transFuncData.estimatedCoverage);

			for(const auto& msafeature : MSAFeatures) {
				featurestream << task.readId << '\t' << msafeature.position << '\n';
				featurestream << msafeature << '\n';
			}
		}

		return BatchState::Finished;
	}

	static BatchState state_finished_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::Finished);

		assert(false);         //Finished is end node

		return BatchState::Finished;
	}

	static BatchState state_aborted_func(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::Aborted);

		assert(false);         //Aborted is end node

		return BatchState::Aborted;
	}


	AdvanceResult advance_one_step(Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const TransitionFunctionData& transFuncData){

		AdvanceResult advanceResult;

		advanceResult.oldState = batch.state;
		advanceResult.noProgressBlocking = false;
		advanceResult.noProgressLaunching = false;

		auto iter = transitionFunctionTable.find(batch.state);
		if(iter != transitionFunctionTable.end()) {
			batch.state = iter->second(batch, canBlock, canLaunchKernel, isPausable, transFuncData);
		}else{
			std::cout << nameOf(batch.state) << std::endl;
			assert(false); // Every State should be handled above
		}

		advanceResult.newState = batch.state;

		return advanceResult;
	}


	void execute() {
		isRunning = true;

		assert(threadOpts.canUseGpu);
		assert(max_candidates > 0);

		//mybatchgen = BatchGenerator<ReadId_t>(threadOpts.batchGen->firstId, threadOpts.batchGen->lastIdExcl);
		makeTransitionFunctionTable();

		//std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

		std::ofstream outputstream(threadOpts.outputfile);
		if(!outputstream)
			throw std::runtime_error("Could not open output file");

		std::ofstream featurestream(threadOpts.outputfile + "_features");
		if(!featurestream)
			throw std::runtime_error("Could not open output feature file");


		constexpr int nParallelBatches = 4;
		constexpr int sideBatchStepsPerWaitIter = 1;

		cudaSetDevice(threadOpts.deviceId); CUERR;

		//std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();

		std::vector<DataArrays<Sequence_t, ReadId_t> > dataArrays;
		//std::array<Batch, nParallelBatches> batches;
		std::array<std::array<cudaStream_t, nStreamsPerBatch>, nParallelBatches> streams;
		std::array<std::array<cudaEvent_t, nEventsPerBatch>, nParallelBatches> cudaevents;

		std::queue<Batch> batchQueue;
		std::queue<DataArrays<Sequence_t, ReadId_t>*> freeDataArraysQueue;
		std::queue<std::array<cudaStream_t, nStreamsPerBatch>*> freeStreamsQueue;
		std::queue<std::array<cudaEvent_t, nEventsPerBatch>*> freeEventsQueue;

		cudaStream_t computeStream;
		cudaStreamCreate(&computeStream);

		for(int i = 0; i < nParallelBatches; i++) {
			dataArrays.emplace_back(threadOpts.deviceId);

			//streams[i][0] = computeStream;

			for(int j = 0; j < nStreamsPerBatch; ++j) {
				cudaStreamCreate(&streams[i][j]); CUERR;
			}

			for(int j = 0; j < nEventsPerBatch; ++j) {
				cudaEventCreateWithFlags(&cudaevents[i][j], cudaEventDisableTiming); CUERR;
			}

			/*dataArrays[i].set_problem_dimensions(readIds.size(),
			                                            max_candidates * readIds.size(),
			                                            fileProperties.maxSequenceLength,
			                                            goodAlignmentProperties.min_overlap,
			                                            goodAlignmentProperties.min_overlap_ratio,
			                                            correctionOptions.useQualityScores);*/

		}

		for(auto& array : dataArrays)
			freeDataArraysQueue.push(&array);
		for(auto& streamArray : streams)
			freeStreamsQueue.push(&streamArray);
		for(auto& eventArray : cudaevents)
			freeEventsQueue.push(&eventArray);

		std::vector<ReadId_t> readIdBuffer;

		TransitionFunctionData transFuncData;

		//transFuncData.mybatchgen = &mybatchgen;
		transFuncData.readIdGenerator = threadOpts.readIdGenerator;
		transFuncData.readIdBuffer = &readIdBuffer;
		transFuncData.minhasher = threadOpts.minhasher;
		transFuncData.gpuReadStorage = threadOpts.gpuReadStorage;
		transFuncData.readStorageGpuData = threadOpts.gpuReadStorage->getGPUData(threadOpts.deviceId);
		transFuncData.min_overlap_ratio = goodAlignmentProperties.min_overlap_ratio;
		transFuncData.min_overlap = goodAlignmentProperties.min_overlap;
		transFuncData.estimatedErrorrate = correctionOptions.estimatedErrorrate;
		transFuncData.maxErrorRate = goodAlignmentProperties.maxErrorRate;
		transFuncData.estimatedCoverage = correctionOptions.estimatedCoverage;
		transFuncData.m_coverage = correctionOptions.m_coverage;
		transFuncData.new_columns_to_correct = correctionOptions.new_columns_to_correct;
		transFuncData.correctCandidates = correctionOptions.correctCandidates;
		transFuncData.correctionOptions.useQualityScores = correctionOptions.useQualityScores;
		transFuncData.kmerlength = correctionOptions.kmerlength;
		transFuncData.num_ids_per_add_tasks = num_ids_per_add_tasks;
		transFuncData.minimum_candidates_per_batch = correctionOptions.batchsize;
		transFuncData.max_candidates = max_candidates;
		transFuncData.correctionOptions = correctionOptions;
		transFuncData.maxSequenceLength = fileProperties.maxSequenceLength;
		transFuncData.locksForProcessedFlags = threadOpts.locksForProcessedFlags;
		transFuncData.nLocksForProcessedFlags = threadOpts.nLocksForProcessedFlags;
		transFuncData.readIsCorrectedVector = threadOpts.readIsCorrectedVector;
		transFuncData.featurestream = &featurestream;
		transFuncData.write_read_to_stream = [&](const ReadId_t readId, const std::string& sequence){
							     //std::cout << readId << " " << sequence << std::endl;
							     auto& stream = outputstream;
    #if 1
							     stream << readId << ' ' << sequence << '\n';
    #else
							     stream << readId << '\n';
							     stream << sequence << '\n';
    #endif
						     };
		transFuncData.lock = [&](ReadId_t readId){
					     ReadId_t index = readId % transFuncData.nLocksForProcessedFlags;
					     transFuncData.locksForProcessedFlags[index].lock();
				     };
		transFuncData.unlock = [&](ReadId_t readId){
					       ReadId_t index = readId % transFuncData.nLocksForProcessedFlags;
					       transFuncData.locksForProcessedFlags[index].unlock();
				       };

        if(!correctionOptions.classicMode){
           transFuncData.fc = ForestClassifier{fileOptions.forestfilename};
        }


		std::array<Batch, nParallelBatches> batches;

		for(int i = 0; i < nParallelBatches; ++i) {
			batches[i].dataArrays = &dataArrays[i];
			batches[i].streams = &streams[i];
			batches[i].events = &cudaevents[i];
			batches[i].kernelLaunchHandle = make_kernel_launch_handle(threadOpts.deviceId);
		}

		auto nextBatchIndex = [](int currentBatchIndex, int nParallelBatches){
					      return (currentBatchIndex + 1) % nParallelBatches;
				      };

		//int num_finished_batches = 0;

		int stacksize = 0;

		//while(!stopAndAbort && !(num_finished_batches == nParallelBatches && readIds.empty())){
		while(!stopAndAbort &&
		      !(std::all_of(batches.begin(), batches.end(), [](const auto& batch){
					return batch.state == BatchState::Finished;
				})
		        && readIdBuffer.empty()
		        && threadOpts.readIdGenerator->empty())) {

			if(stacksize != 0)
				assert(stacksize == 0);

			Batch& mainBatch = batches[0];

			AdvanceResult mainBatchAdvanceResult;
			bool popMain = false;



			assert(popMain == false);
			push_range("mainBatch"+nameOf(mainBatch.state)+"first", int(mainBatch.state));
			++stacksize;
			popMain = true;

			while(!(mainBatch.state == BatchState::Finished || mainBatch.state == BatchState::Aborted)) {

				mainBatchAdvanceResult = advance_one_step(mainBatch,
							true,                   //can block
							true,                   //can launch kernels
							false,                   //cannot be paused
							transFuncData);

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
				IsWaitingResult isWaitingResult = isWaiting(mainBatch.state);
				if(isWaitingResult.isWaiting) {
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
						cudaError_t eventquerystatus = cudaSuccess;
						cudaEvent_t eventToWaitFor = (*mainBatch.events)[isWaitingResult.eventIndexToWaitFor];

						while((eventquerystatus = cudaEventQuery(eventToWaitFor)) == cudaErrorNotReady) {
							const int globalBatchIndex = localSideBatchIndex + 1;

							Batch& sideBatch = batches[globalBatchIndex];

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
											false,                                                                 //must not block
											true,                                                                 //can launch kernels
											true,                   //can be paused
											transFuncData);

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

							IsWaitingResult isWaitingResultSideBatch = isWaiting(sideBatch.state);
							if(isWaitingResultSideBatch.isWaiting) {
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

						assert(eventquerystatus == cudaSuccess);
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

			if(!(threadOpts.readIdGenerator->empty() && readIdBuffer.empty())) {
				//there are reads left to correct, so this batch can be reused again
				mainBatch.reset();
			}else{
				mainBatch.state = BatchState::Finished;
			}

			//nProcessedReads = threadOpts.readIdGenerator->.currentId - mybatchgen.firstId;

			//rotate left to position next batch index 0
			std::rotate(batches.begin(), batches.begin()+1, batches.end());






//#ifdef CARE_GPU_DEBUG
			//stopAndAbort = true; //remove
//#endif


		} // end batch processing


		outputstream.flush();
		featurestream.flush();

		std::cout << "GPU worker (device " << threadOpts.deviceId << ") finished" << std::endl;

	#if 0
		{
			std::lock_guard < std::mutex > lg(*threadOpts.coutLock);

			std::cout << "thread " << threadOpts.threadId << " processed " << nProcessedQueries
			          << " queries" << std::endl;
			std::cout << "thread " << threadOpts.threadId << " corrected "
			          << nCorrectedCandidates << " candidates" << std::endl;
			std::cout << "thread " << threadOpts.threadId << " avgsupportfail "
			          << avgsupportfail << std::endl;
			std::cout << "thread " << threadOpts.threadId << " minsupportfail "
			          << minsupportfail << std::endl;
			std::cout << "thread " << threadOpts.threadId << " mincoveragefail "
			          << mincoveragefail << std::endl;
			std::cout << "thread " << threadOpts.threadId << " sobadcouldnotcorrect "
			          << sobadcouldnotcorrect << std::endl;
			std::cout << "thread " << threadOpts.threadId << " verygoodalignment "
			          << verygoodalignment << std::endl;
			/*std::cout << "thread " << threadOpts.threadId
			                << " CPU alignments " << cpuAlignments
			                << " GPU alignments " << gpuAlignments << std::endl;*/

			//   std::cout << "thread " << threadOpts.threadId << " savedAlignments "
			//           << savedAlignments << " performedAlignments " << performedAlignments << std::endl;
		}
	#endif

	#if 0
		{
			TaskTimings tmp;
			for(std::uint64_t i = 0; i < threadOpts.batchGen->batchsize; i++)
				tmp += batchElems[i].findCandidatesTiming;

			std::lock_guard < std::mutex > lg(*threadOpts.coutLock);
			std::cout << "thread " << threadOpts.threadId << " findCandidatesTiming:\n";
			std::cout << tmp << std::endl;
		}
	#endif

	#if 0
		{
			std::lock_guard < std::mutex > lg(*threadOpts.coutLock);

			std::cout << "thread " << threadOpts.threadId
			          << " : preparation timings detail "
			          << da.count() << " " << db.count() << " " << dc.count()<< '\n';


			std::cout << "thread " << threadOpts.threadId
			          << " : init batch elems "
			          << initIdTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId
			          << " : find candidates time "
			          << mapMinhashResultsToSequencesTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment time "
			          << getAlignmentsTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId
			          << " : determine good alignments time "
			          << determinegoodalignmentsTime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId
			          << " : fetch good candidates time "
			          << fetchgoodcandidatesTime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : correction time "
			          << readcorrectionTimeTotal.count() << '\n';

			std::cout << "thread " << threadOpts.threadId << " : detailed correction time " << '\n'
			          << detailedCorrectionTimings << '\n';


	#if 0
			if (correctionOptions.correctionMode == CorrectionMode::Hamming) {
				std::cout << "thread " << threadOpts.threadId << " : pileup vote "
				          << pileupImage.timings.findconsensustime.count() << '\n';
				std::cout << "thread " << threadOpts.threadId << " : pileup correct "
				          << pileupImage.timings.correctiontime.count() << '\n';

			} else if (correctionOptions.correctionMode == CorrectionMode::Graph) {
				std::cout << "thread " << threadOpts.threadId << " : graph build "
				          << graphbuildtime.count() << '\n';
				std::cout << "thread " << threadOpts.threadId << " : graph correct "
				          << graphcorrectiontime.count() << '\n';
			}
	#endif
		}
	#endif

		for(auto& array : dataArrays) {
			array.reset();
		}

		cudaStreamDestroy(computeStream); CUERR;

		for(auto& streamarray : streams) {
			for(int i = 1; i < nStreamsPerBatch; ++i)
				cudaStreamDestroy(streamarray[i]); CUERR;
		}

		for(auto& eventarray : cudaevents) {
			for(auto& event : eventarray)
				cudaEventDestroy(event); CUERR;
		}
	}
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








#endif
