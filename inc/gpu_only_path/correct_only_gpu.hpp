#ifndef CARE_CORRECT_ONLY_GPU_HPP
#define CARE_CORRECT_ONLY_GPU_HPP

//#define USE_NVTX2


#if defined USE_NVTX2 && defined __NVCC__
#include <nvToolsExt.h>

const uint32_t colors_[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff, 0xdeadbeef, 0x12345678 };
const int num_colors_ = sizeof(colors_)/sizeof(uint32_t);

#define PUSH_RANGE_2(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors_;\
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

#include "kernels.hpp"
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



namespace care{
namespace gpu{

template<class ReadId_t>
struct BatchGenerator{
    BatchGenerator(){}
    
    BatchGenerator(ReadId_t firstId, ReadId_t lastIdExcl)
            : firstId(firstId), lastIdExcl(lastIdExcl), currentId(firstId){
                if(firstId >= lastIdExcl) throw std::runtime_error("BatchGenerator: firstId >= lastIdExcl");
	}

    std::vector<ReadId_t> getNextReadIds(int maxnumreadIds){
        std::vector<ReadId_t> result;
    	while(int(result.size()) < maxnumreadIds && currentId < lastIdExcl){
    		result.push_back(currentId);
    		currentId++;
    	}
        return result;
    }
    
    bool empty() const{
		return currentId == lastIdExcl;
	}

    ReadId_t firstId;
    ReadId_t lastIdExcl;
    ReadId_t currentId;
};


    /*
        Correct reads using shifted hamming distance and multple sequence alignment.
        Shifted hamming distance as well as multiple sequence alignment are performed on gpu.

        Batch size = N
        Total number of candidates for N batches = M
        maximum sequence length in N batches = L

        1. Get subject sequence and its candidate read ids from hash map

            Sequence_t* subject_sequences
            ReadId_t* candidate_read_ids
            std::string* subject_quality

        2. Get candidate sequences from read storage
            Sequence_t* candidates_sequences
            std::vector<std::string*> subject_quality

        3. Copy subject sequences, subject sequence lengths, candidate sequences, candidate sequence lengths, candidates_per_subject_prefixsum to GPU
            h_subject_sequences_data[N * compressedlength(L)]
            h_candidate_sequences_data[M * compressedlength(L)]
            h_subject_sequences_lengths[N]
            h_candidate_sequences_lengths[M]
            h_candidates_per_subject_prefixsum[N+1]

            d_subject_sequences_data[N * compressedlength(L)]
            d_candidate_sequences_data[M * compressedlength(L)]
            d_subject_sequences_lengths[N]
            d_candidate_sequences_lengths[M]
            d_candidates_per_subject_prefixsum[N+1]

        4. Perform Alignment. Produces 2*M alignments, M alignments for forward sequences, M alignments for reverse complement sequences
            d_alignment_scores[2*M]
            d_alignment_overlaps[2*M]
            d_alignment_shifts[2*M]
            d_alignment_nOps[2*M]
            d_alignment_isValid[2*M]

            ALIGNMENT_KERNEL(   d_alignment_scores,
                                d_alignment_overlaps,
                                d_alignment_shifts,
                                d_alignment_nOps,
                                d_alignment_isValid,
                                d_subject_sequences_data,
                                d_candidate_sequences_data,
                                d_subject_sequences_lengths,
                                d_candidate_sequences_lengths,
                                d_candidates_per_subject_prefixsum)

        5. Compare each forward alignment with the correspoding reverse complement alignment and keep the best, if any.
            If reverse complement is the best, it is copied into the first half, replacing the forward alignment

            d_alignment_best_alignment_flags[M]

            COMPARE_ALIGNMENTS_KERNEL(
                                        d_alignment_best_alignment_flags,
                                        d_alignment_scores,
                                        d_alignment_overlaps,
                                        d_alignment_shifts,
                                        d_alignment_nOps,
                                        d_alignment_isValid,
                                        d_subject_sequences_lengths,
                                        d_candidate_sequences_lengths,
                                        d_candidates_per_subject_prefixsum)



        6. Determine alignments which should be used for correction

            d_indices[M]

            auto end = thrust::copy_if(thrust::counting_iterator<int>(0),
					thrust::counting_iterator<int>(0) + M,
					d_alignment_best_alignment_flags,
					d_indices,
					[]__device__(BestAlignment_t flag){return flag != BestAlignment_t::None;});
            num_indices = std::distance(d_indices, end)

        7. Determine number of indices per subject
            d_indices_per_subject[N]
            d_indices_per_subject_prefixsum[N]
            d_temp_storage[max(temp_storage_bytes, other_temp_storage_bytes)]

            cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes,
                    d_indices, d_indices_per_subject, N+1, d_candidates_per_subject_prefixsum, num_indices);

            cub::DeviceScan::ExclusiveSum(d_temp_storage, other_temp_storage_bytes, d_indices_per_subject, d_indices_per_subject_prefixsum, N);

        8. Copy d_indices, d_indices_per_subject, d_indices_per_subject_prefixsum to host
            h_indices[num_indices]
            h_indices_per_subject[N]
            h_indices_per_subject_prefixsum[N]

        9. Allocate quality score data and correction data
            h_candidate_qualities[num_indices]
            h_subject_qualities[N]
            d_candidate_qualities[num_indices]
            d_subject_qualities[N]
            d_multiple_sequence_alignment[(3*L-2) * (num_indices + N)];
            d_multiple_sequence_alignment_weights[L * (num_indices + N)];
            d_consensus[(3*L-2) * N]
            d_support[(3*L-2) * N]
            d_coverage[(3*L-2) * N]
            d_origWeights[(3*L-2) * N]
            d_origCoverages[(3*L-2) * N]
            d_msa_column_properties[N]

        10. Copy quality scores of candidates referenced by h_indices to gpu

        11. Determine multiple sequence alignment properties

                INIT_MSA_KERNEL(d_msa_column_properties,
                                d_alignment_shifts,
                                d_indices,
                                d_indices_per_subject_prefixsum,
                                d_subject_sequences_lengths,
                                d_candidate_sequences_lengths,
                                d_candidates_per_subject_prefixsum)

        12. Fill multiple sequence alignment
                MSA_FILL_KERNEL(
                                d_multiple_sequence_alignment
                                d_multiple_sequence_alignment_weights
                                d_msa_column_properties,
                                d_alignment_shifts,
                                d_indices,
                                d_indices_per_subject_prefixsum,
                                d_subject_sequences_data
                                d_candidate_sequences_data
                                d_subject_sequences_lengths,
                                d_candidate_sequences_lengths,
                                d_candidates_per_subject_prefixsum)

        13. Determine consensus in multiple sequence alignment

                MSA_FIND_CONSENSUS_KERNEL(
                        d_consensus,
                        d_support,
                        d_coverage,
                        d_origWeights,
                        d_origCoverages,
                        d_multiple_sequence_alignment
                        d_multiple_sequence_alignment_weights
                        d_msa_column_properties,
                        d_indices,
                        d_indices_per_subject_prefixsum,
                        d_subject_sequences_data
                        d_candidates_per_subject_prefixsum)

        14. Correct







    */
#ifdef __NVCC__

    template<class Sequence_t, class ReadId_t>
    struct CorrectionTask{
        CorrectionTask(){}

        CorrectionTask(ReadId_t readId)
            : CorrectionTask(readId, nullptr, nullptr){}

        CorrectionTask(ReadId_t readId, const Sequence_t* subject_sequence)
            : CorrectionTask(readId, subject_sequence, nullptr){}

        CorrectionTask(ReadId_t readId, const Sequence_t* subject_sequence, const std::string* subject_quality)
            :   active(true),
                corrected(false),
                readId(readId),
                subject_sequence(subject_sequence),
                subject_quality(subject_quality){}

        CorrectionTask(const CorrectionTask& other)
            : active(other.active),
            corrected(other.corrected),
            readId(other.readId),
            subject_sequence(other.subject_sequence),
            subject_quality(other.subject_quality),
            candidate_read_ids(other.candidate_read_ids),
            candidate_sequences(other.candidate_sequences),
            candidate_qualities(other.candidate_qualities),
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
            swap(l.subject_sequence, r.subject_sequence);
            swap(l.candidate_read_ids, r.candidate_read_ids);
            swap(l.candidate_sequences, r.candidate_sequences);
            swap(l.subject_quality, r.subject_quality);
            swap(l.candidate_qualities, r.candidate_qualities);
            swap(l.corrected_subject, r.corrected_subject);
            swap(l.corrected_candidates_read_ids, r.corrected_candidates_read_ids);
        }

        bool active;
        bool corrected;
        ReadId_t readId;
        const Sequence_t* subject_sequence;
        std::vector<ReadId_t> candidate_read_ids;
        std::vector<const Sequence_t*> candidate_sequences;

        const std::string* subject_quality;
        std::vector<const std::string*> candidate_qualities;

        std::string corrected_subject;
        std::vector<std::string> corrected_candidates;
        std::vector<ReadId_t> corrected_candidates_read_ids;
    };

    template<class minhasher_t,
    		 class readStorage_t,
			 class batchgenerator_t>
    struct ErrorCorrectionThreadOnlyGPU{
    using Minhasher_t = minhasher_t;
    	using ReadStorage_t = readStorage_t;
    	using Sequence_t = typename ReadStorage_t::Sequence_t;
    	using ReadId_t = typename ReadStorage_t::ReadId_t;
        using CorrectionTask_t = CorrectionTask<Sequence_t, ReadId_t>;
		using BatchGenerator_t = batchgenerator_t;

		enum class BatchState : int {
			Unprepared = 0,
			CopyReads = 1,
			TransferReads = 2,
			StartAlignment = 3,
			WaitForAlignment = 4,
            TransferIndices = 5,
			WaitForIndices = 6,
			CopyQualities = 7,
			TransferQualities = 8,
			WaitForQualities = 9,
			StartCorrection = 10,
			WaitForCorrection = 11,
			UnpackResults = 12,
			WriteResults = 13,
			Finished = 14,
			Aborted = 15,
		};

		struct Batch{
			std::vector<CorrectionTask_t> tasks;
			int maxSubjectLength = 0;
			int maxQueryLength = 0;
			int initialNumberOfCandidates = 0;
			BatchState state = BatchState::Unprepared;

			int copiedTasks = 0; // used if state == CandidatesPresent
			int copiedCandidates = 0; // used if state == CandidatesPresent
		};
		
		struct AdvanceResult{
			BatchState oldState = BatchState::Unprepared;
			BatchState newState = BatchState::Unprepared;
			bool noProgressBlocking = false;
			bool noProgressLaunching = false;
		};
		
		struct IsWaitingResult{
			bool isWaiting;
			int eventIndexToWaitFor;
		};
				
		struct TransitionFunctionData{
			BatchGenerator<ReadId_t>* mybatchgen;
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
			int maxSequenceLength;
			const ReadStorage_t* readStorage;
			const Minhasher_t* minhasher;
			std::mutex* locksForProcessedFlags;
    		std::size_t nLocksForProcessedFlags;
			std::vector<char>* readIsCorrectedVector;
			std::function<void(const ReadId_t, const std::string&)> write_read_to_stream;
			std::function<void(const ReadId_t)> lock;
			std::function<void(const ReadId_t)> unlock;
		};

    	struct CorrectionThreadOptions{
    		int threadId;
    		int deviceId;
			bool canUseGpu;

    		std::string outputfile;
    		BatchGenerator_t* batchGen;
    		const Minhasher_t* minhasher;
    		const ReadStorage_t* readStorage;
    		std::mutex* coutLock;
    		std::vector<char>* readIsProcessedVector;
    		std::vector<char>* readIsCorrectedVector;
    		std::mutex* locksForProcessedFlags;
    		std::size_t nLocksForProcessedFlags;
    	};

        static constexpr int nStreamsPerBatch = 2;
        static constexpr int primary_stream_index = 0;
        static constexpr int secondary_stream_index = 1;

        static constexpr int nEventsPerBatch = 4;
        static constexpr int alignments_finished_event_index = 0;
        static constexpr int quality_transfer_finished_event_index = 1;
        static constexpr int indices_transfer_finished_event_index = 2;
        static constexpr int correction_finished_event_index = 3;

        AlignmentOptions alignmentOptions;
        GoodAlignmentProperties goodAlignmentProperties;
        CorrectionOptions correctionOptions;
        CorrectionThreadOptions threadOpts;

        SequenceFileProperties fileProperties;

        std::uint64_t max_candidates;

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
		
		BatchGenerator<ReadId_t> mybatchgen;		
		int num_ids_per_add_tasks = 30;
		int minimum_candidates_per_batch = 25000;
		

		using FuncTableEntry = BatchState (ErrorCorrectionThreadOnlyGPU::*)(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
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
		
		std::string nameOf(BatchState state) const{
			switch(state){
				case BatchState::Unprepared: return "Unprepared";
				case BatchState::CopyReads: return "CopyReads";
				case BatchState::TransferReads: return "TransferReads";
				case BatchState::StartAlignment: return "StartAlignment";
				case BatchState::WaitForAlignment: return "WaitForAlignment";
				case BatchState::TransferIndices: return "TransferIndices";
				case BatchState::WaitForIndices: return "WaitForIndices";
				case BatchState::CopyQualities: return "CopyQualities";
				case BatchState::TransferQualities: return "TransferQualities";
				case BatchState::WaitForQualities: return "WaitForQualities";
				case BatchState::StartCorrection: return "StartCorrection";
				case BatchState::WaitForCorrection: return "WaitForCorrection";
				case BatchState::UnpackResults: return "UnpackResults";
				case BatchState::WriteResults: return "WriteResults";
				case BatchState::Finished: return "Finished";
				case BatchState::Aborted: return "Aborted";
				default: return "None";
			}		
		}
				
		IsWaitingResult isWaiting(BatchState state) const{
			switch(state){
				case BatchState::Unprepared: return {false, -1};
				case BatchState::CopyReads: return {false, -1};
				case BatchState::TransferReads: return {false, -1};
				case BatchState::StartAlignment: return {false, -1};
				case BatchState::WaitForAlignment: return {true, alignments_finished_event_index};
				case BatchState::TransferIndices: return {false, -1};
				case BatchState::WaitForIndices: return {true, indices_transfer_finished_event_index};
				case BatchState::CopyQualities: return {false, -1};
				case BatchState::TransferQualities: return {false, -1};
				case BatchState::WaitForQualities: return {true, quality_transfer_finished_event_index};
				case BatchState::StartCorrection: return {false, -1};
				case BatchState::WaitForCorrection: return {true, correction_finished_event_index};
				case BatchState::UnpackResults: return {false, -1};
				case BatchState::WriteResults: return {false, -1};
				case BatchState::Finished: return {false, -1};
				case BatchState::Aborted: return {false, -1};
				default: assert(false); return {false, -1};
			}
		}
		
		void makeTransitionFunctionTable(){
			transitionFunctionTable[BatchState::Unprepared] = &ErrorCorrectionThreadOnlyGPU::state_unprepared_func;
			transitionFunctionTable[BatchState::CopyReads] = &ErrorCorrectionThreadOnlyGPU::state_copyreads_func;
			transitionFunctionTable[BatchState::TransferReads] = &ErrorCorrectionThreadOnlyGPU::state_transferreads_func;
			transitionFunctionTable[BatchState::StartAlignment] = &ErrorCorrectionThreadOnlyGPU::state_startalignment_func;
			transitionFunctionTable[BatchState::WaitForAlignment] = &ErrorCorrectionThreadOnlyGPU::state_waitforalignment_func;
			transitionFunctionTable[BatchState::TransferIndices] = &ErrorCorrectionThreadOnlyGPU::state_transferindices_func;
			transitionFunctionTable[BatchState::WaitForIndices] = &ErrorCorrectionThreadOnlyGPU::state_waitforindices_func;
			transitionFunctionTable[BatchState::CopyQualities] = &ErrorCorrectionThreadOnlyGPU::state_copyqualities_func;
			transitionFunctionTable[BatchState::TransferQualities] = &ErrorCorrectionThreadOnlyGPU::state_transferqualities_func;
			transitionFunctionTable[BatchState::WaitForQualities] = &ErrorCorrectionThreadOnlyGPU::state_waitforqualities_func;
			transitionFunctionTable[BatchState::StartCorrection] = &ErrorCorrectionThreadOnlyGPU::state_startcorrection_func;
			transitionFunctionTable[BatchState::WaitForCorrection] = &ErrorCorrectionThreadOnlyGPU::state_waitforcorrection_func;
			transitionFunctionTable[BatchState::UnpackResults] = &ErrorCorrectionThreadOnlyGPU::state_unpackresults_func;
			transitionFunctionTable[BatchState::WriteResults] = &ErrorCorrectionThreadOnlyGPU::state_writeresults_func;
			transitionFunctionTable[BatchState::Finished] = &ErrorCorrectionThreadOnlyGPU::state_finished_func;
			transitionFunctionTable[BatchState::Aborted] = &ErrorCorrectionThreadOnlyGPU::state_aborted_func;
		}		
		
		BatchState state_unprepared_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::Unprepared);
			assert(batch.tasks.empty());
			
			while(batch.initialNumberOfCandidates < transFuncData.minimum_candidates_per_batch && !transFuncData.mybatchgen->empty()){
				
				const auto& readStorage = transFuncData.readStorage;
				const auto& minhasher = transFuncData.minhasher;
				const auto readIds = transFuncData.mybatchgen->getNextReadIds(transFuncData.num_ids_per_add_tasks);
				
				for(ReadId_t id : readIds){
					bool ok = false;
					transFuncData.lock(id);
					if ((*transFuncData.readIsCorrectedVector)[id] == 0) {
						(*transFuncData.readIsCorrectedVector)[id] = 1;
						ok = true;
					}else{
					}
					transFuncData.unlock(id);

					if(ok){
						const Sequence_t* sequenceptr = readStorage->fetchSequence_ptr(id);
						const std::string* qualityptr = nullptr;

						if(correctionOptions.useQualityScores)
							qualityptr = readStorage->fetchQuality_ptr(id);

						batch.tasks.emplace_back(id, sequenceptr, qualityptr);
						
						auto& task = batch.tasks.back();
						
						const std::string sequencestring = task.subject_sequence->toString();
						task.candidate_read_ids = minhasher->getCandidates(sequencestring, max_candidates);

						//remove self from candidates
						auto readIdPos = std::find(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.readId);
						if(readIdPos != task.candidate_read_ids.end())
							task.candidate_read_ids.erase(readIdPos);

						if(task.candidate_read_ids.size() == 0){
							//no need for further processing without candidates
							task.active = false;
						}else{
							for(auto candidate_read_id : task.candidate_read_ids){
								task.candidate_sequences.emplace_back(readStorage->fetchSequence_ptr(candidate_read_id));
								if(correctionOptions.useQualityScores)
									task.candidate_qualities.emplace_back(readStorage->fetchQuality_ptr(candidate_read_id));
							}
						}
						
						batch.initialNumberOfCandidates += int(task.candidate_read_ids.size());
					}
				}
			}
			
			//if(mybatchgen.empty()){
			//	std::cout << "EMPTY!!! " << batch.initialNumberOfCandidates << ", canLaunchKernel : " << canLaunchKernel << std::endl;
			//}
			
			if(batch.initialNumberOfCandidates == 0){
				return BatchState::Aborted;
			}else{					
				std::remove_if(batch.tasks.begin(),
								batch.tasks.end(),
								[](const auto& t){return !t.active;});
				
				//allocate data arrays

				dataArrays.set_problem_dimensions(int(batch.tasks.size()),
											batch.initialNumberOfCandidates,
											transFuncData.maxSequenceLength,
											transFuncData.min_overlap,
											transFuncData.min_overlap_ratio,
											transFuncData.useQualityScores); CUERR;

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
				
				return BatchState::CopyReads;
			}
		}
		
		BatchState state_copyreads_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::CopyReads);
			assert(batch.copiedTasks <= int(batch.tasks.size()));
			
			dataArrays.h_candidates_per_subject_prefixsum[0] = 0;

			//copy one task
            if(batch.copiedTasks < int(batch.tasks.size())){
                const auto& task = batch.tasks[batch.copiedTasks];
                auto& arrays = dataArrays;

                //fill subject
                std::memcpy(arrays.h_subject_sequences_data + batch.copiedTasks * arrays.encoded_sequence_pitch,
                            task.subject_sequence->begin(),
                            task.subject_sequence->getNumBytes());
                arrays.h_subject_sequences_lengths[batch.copiedTasks] = task.subject_sequence->length();
                batch.maxSubjectLength = std::max(int(task.subject_sequence->length()),
                                                                batch.maxSubjectLength);

                //fill candidates
                for(const Sequence_t* candidate_sequence : task.candidate_sequences){

                    std::memcpy(arrays.h_candidate_sequences_data
                                    + batch.copiedCandidates * arrays.encoded_sequence_pitch,
                                candidate_sequence->begin(),
                                candidate_sequence->getNumBytes());

                    arrays.h_candidate_sequences_lengths[batch.copiedCandidates] = candidate_sequence->length();
                    batch.maxQueryLength = std::max(int(candidate_sequence->length()),
                                                                batch.maxQueryLength);

                    ++batch.copiedCandidates;
                }

                //make prefix sum
                arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks+1]
                                = arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks]
                                    + int(task.candidate_read_ids.size());

                ++batch.copiedTasks;

            }

			//if batch is fully copied, transfer to gpu
			if(batch.copiedTasks == int(batch.tasks.size())){	
				return BatchState::TransferReads;
			}else{
				return BatchState::CopyReads;
			}
		}
		
		BatchState state_transferreads_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::TransferReads);
			assert(batch.copiedTasks == int(batch.tasks.size()));
			
			cudaMemcpyAsync(dataArrays.alignment_transfer_data_device,
							dataArrays.alignment_transfer_data_host,
							dataArrays.alignment_transfer_data_usable_size,
							H2D,
							streams[primary_stream_index]); CUERR;

			batch.copiedTasks = 0;
			batch.copiedCandidates = 0;

			return BatchState::StartAlignment;
		}
		
		BatchState state_startalignment_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::StartAlignment);
			
			if(!canLaunchKernel){
				//since gathering quality scores does not depend on the batch state being RunningAlignmentWork
				//we can gather the quality scores of one task, instead of doing nothing
				//gather_quality_scores_of_next_task(batch, dataArrays);

				return BatchState::StartAlignment;
			}

			auto accessor = [] __device__ (const char* data, int length, int index){
                return Sequence_t::get(data, length, index);
            };

            auto make_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
                return Sequence_t::make_reverse_complement_inplace(sequence, sequencelength);
            };

            auto best_alignment_comp = [=] __device__ (int fwd_alignment_overlap,
                                        int revc_alignment_overlap,
                                        int fwd_alignment_nops,
                                        int revc_alignment_nops,
                                        bool fwd_alignment_isvalid,
                                        bool revc_alignment_isvalid,
                                        int subjectlength,
                                        int querylength) -> BestAlignment_t{

                return choose_best_alignment(fwd_alignment_overlap,
                                            revc_alignment_overlap,
                                            fwd_alignment_nops,
                                            revc_alignment_nops,
                                            fwd_alignment_isvalid,
                                            revc_alignment_isvalid,
                                            subjectlength,
                                            querylength,
                                            transFuncData.min_overlap_ratio,
                                            transFuncData.min_overlap,
                                            transFuncData.estimatedErrorrate * 4.0);
            };

            auto select_alignment_op = [] __device__ (const BestAlignment_t& flag){
                    return flag != BestAlignment_t::None;
            };

            /*
                Writes indices of candidates with alignmentflag != None to dataArrays[batchIndex].d_indices
                Writes number of writen indices to dataArrays[batchIndex].d_num_indices
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

            call_shd_with_revcompl_kernel_async(
                                    dataArrays.d_alignment_scores,
                                    dataArrays.d_alignment_overlaps,
                                    dataArrays.d_alignment_shifts,
                                    dataArrays.d_alignment_nOps,
                                    dataArrays.d_alignment_isValid,
                                    dataArrays.d_subject_sequences_data,
                                    dataArrays.d_candidate_sequences_data,
                                    dataArrays.d_subject_sequences_lengths,
                                    dataArrays.d_candidate_sequences_lengths,
                                    dataArrays.d_candidates_per_subject_prefixsum,
                                    dataArrays.n_subjects,
                                    Sequence_t::getNumBytes(dataArrays.maximum_sequence_length),
                                    dataArrays.encoded_sequence_pitch,
                                    transFuncData.min_overlap,
                                    transFuncData.maxErrorRate, //correctionOptions.estimatedErrorrate * 4.0,
                                    transFuncData.min_overlap_ratio,
                                    accessor,
                                    make_reverse_complement_inplace,
                                    dataArrays.n_queries,
                                    batch.maxSubjectLength,
                                    batch.maxQueryLength,
                                    streams[primary_stream_index]);

            //Step 5. Compare each forward alignment with the correspoding reverse complement alignment and keep the best, if any.
            //    If reverse complement is the best, it is copied into the first half, replacing the forward alignment

            call_cuda_find_best_alignment_kernel_async(
                                                dataArrays.d_alignment_best_alignment_flags,
                                                dataArrays.d_alignment_scores,
                                                dataArrays.d_alignment_overlaps,
                                                dataArrays.d_alignment_shifts,
                                                dataArrays.d_alignment_nOps,
                                                dataArrays.d_alignment_isValid,
                                                dataArrays.d_subject_sequences_lengths,
                                                dataArrays.d_candidate_sequences_lengths,
                                                dataArrays.d_candidates_per_subject_prefixsum,
                                                dataArrays.n_subjects,
                                                transFuncData.min_overlap_ratio,
                                                transFuncData.min_overlap,
                                                transFuncData.maxErrorRate, //transFuncData.estimatedErrorrate * 4.0,
                                                best_alignment_comp,
                                                dataArrays.n_queries,
                                                streams[primary_stream_index]);

            //Determine indices where d_alignment_best_alignment_flags[i] != BestAlignment_t::None. this selects all good alignments
            select_alignments_by_flag(dataArrays, streams[primary_stream_index]);

            //Get number of indices per subject by creating histrogram.
            //The elements d_indices[d_num_indices] to d_indices[n_queries - 1] will be -1.
            //Thus, they will not be accounted for by the histrogram, since the histrogram bins (d_candidates_per_subject_prefixsum) are >= 0.
            cub::DeviceHistogram::HistogramRange(dataArrays.d_temp_storage,
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
                                            streams[primary_stream_index]); CUERR;

            //choose the most appropriate subset of alignments from the good alignments.
            //This sets d_alignment_best_alignment_flags[i] = BestAlignment_t::None for all non-appropriate alignments
            call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                                    dataArrays.d_alignment_best_alignment_flags,
                                    dataArrays.d_alignment_overlaps,
                                    dataArrays.d_alignment_nOps,
                                    dataArrays.d_indices,
                                    dataArrays.d_indices_per_subject,
                                    dataArrays.d_indices_per_subject_prefixsum,
                                    dataArrays.n_subjects,
                                    dataArrays.n_queries,
                                    dataArrays.d_num_indices,
                                    transFuncData.estimatedErrorrate,
                                    transFuncData.estimatedCoverage * transFuncData.m_coverage,
                                    streams[primary_stream_index]);

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

            cudaMemcpyAsync(dataArrays.h_num_indices, dataArrays.d_num_indices, sizeof(int), D2H, streams[primary_stream_index]); CUERR;

            assert(cudaSuccess == cudaEventQuery(events[alignments_finished_event_index])); CUERR;

            cudaEventRecord(events[alignments_finished_event_index], streams[primary_stream_index]); CUERR;
			
            //Determine multiple sequence alignment properties
            call_msa_init_kernel_async(
                            dataArrays.d_msa_column_properties,
                            dataArrays.d_alignment_shifts,
                            dataArrays.d_alignment_best_alignment_flags,
                            dataArrays.d_subject_sequences_lengths,
                            dataArrays.d_candidate_sequences_lengths,
                            dataArrays.d_indices,
                            dataArrays.d_indices_per_subject,
                            dataArrays.d_indices_per_subject_prefixsum,
                            dataArrays.n_subjects,
                            dataArrays.n_queries,
                            streams[primary_stream_index]);

			return BatchState::WaitForAlignment;
		}
		
		BatchState state_waitforalignment_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::WaitForAlignment);
			
			cudaError_t querystatus = cudaEventQuery(events[alignments_finished_event_index]); CUERR;
			
			assert(querystatus == cudaSuccess || querystatus == cudaErrorNotReady);
			
			if(querystatus == cudaSuccess){
				return BatchState::TransferIndices;
			}else{
				return BatchState::WaitForAlignment;
			}
		}
		
		BatchState state_transferindices_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::TransferIndices);
			
			cudaMemcpyAsync(dataArrays.indices_transfer_data_host,
							dataArrays.indices_transfer_data_device,
							dataArrays.indices_transfer_data_usable_size,
							D2H,
							streams[secondary_stream_index]); CUERR;
							
			assert(cudaSuccess == cudaEventQuery(events[indices_transfer_finished_event_index])); CUERR;
									
			cudaEventRecord(events[indices_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;
			
			return BatchState::WaitForIndices;
		}
		
		BatchState state_waitforindices_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::WaitForIndices);
			
			cudaError_t querystatus = cudaEventQuery(events[indices_transfer_finished_event_index]); CUERR;
			
			assert(querystatus == cudaSuccess || querystatus == cudaErrorNotReady);
			
			if(querystatus == cudaSuccess){
				if(*dataArrays.h_num_indices == 0){
                        if(canBlock){
							//wait for outstanding work before reusing buffers, i.e. resizing
							cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
							cudaStreamSynchronize(streams[secondary_stream_index]); CUERR;
							return BatchState::Aborted;
						}else{
							return BatchState::WaitForIndices;
						}
				}else{
					return BatchState::CopyQualities;
				}
			}else{
				return BatchState::WaitForIndices;
			}
		}
		
		BatchState state_copyqualities_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::CopyQualities);
			
			if(transFuncData.useQualityScores){
				
				assert(batch.copiedTasks <= int(batch.tasks.size()));
				
				if(batch.copiedTasks < int(batch.tasks.size())){
							
					const int my_num_indices = dataArrays.h_indices_per_subject[batch.copiedTasks];
					const int* my_indices = dataArrays.h_indices + dataArrays.h_indices_per_subject_prefixsum[batch.copiedTasks];
					const int candidatesOfPreviousTasks = dataArrays.h_candidates_per_subject_prefixsum[batch.copiedTasks];
					
					const auto& task = batch.tasks[batch.copiedTasks];
					auto& arrays = dataArrays;

					//fill subject
					std::memcpy(arrays.h_subject_qualities + batch.copiedTasks * arrays.quality_pitch,
								task.subject_quality->c_str(),
								task.subject_quality->length());
					
					for(int i = 0; i < my_num_indices; ++i){
						const int candidate_index = my_indices[i];
						const int local_candidate_index = candidate_index - candidatesOfPreviousTasks;
						const std::string* qual = task.candidate_qualities[local_candidate_index];
						
						std::memcpy(arrays.h_candidate_qualities + batch.copiedCandidates * arrays.quality_pitch,
									qual->c_str(),
									qual->length());
						++batch.copiedCandidates;
					}

					++batch.copiedTasks;
				}

				//gather_quality_scores_of_next_task(batch, dataArrays);

				//if batch is fully copied, transfer to gpu
				if(batch.copiedTasks == int(batch.tasks.size())){
					return BatchState::TransferQualities;
				}else{
					return BatchState::CopyQualities;
				}

			}else{
				return BatchState::StartCorrection;
			}
		}
		
		BatchState state_transferqualities_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::TransferQualities);
			assert(correctionOptions.useQualityScores);
			assert(batch.copiedTasks == int(batch.tasks.size()));
			
			cudaMemcpyAsync(dataArrays.qualities_transfer_data_device,
							dataArrays.qualities_transfer_data_host,
							dataArrays.qualities_transfer_data_usable_size,
							H2D,
							streams[secondary_stream_index]); CUERR;

			assert(cudaSuccess == cudaEventQuery(events[quality_transfer_finished_event_index])); CUERR;

			cudaEventRecord(events[quality_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;

			batch.copiedTasks = 0;
			batch.copiedCandidates = 0;

			return BatchState::WaitForQualities;
		}
		
		BatchState state_waitforqualities_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::WaitForQualities);
			
			cudaError_t querystatus = cudaEventQuery(events[indices_transfer_finished_event_index]); CUERR;
			
			assert(querystatus == cudaSuccess || querystatus == cudaErrorNotReady);
			
			if(querystatus == cudaSuccess){
				return BatchState::StartCorrection;
			}else{
				return BatchState::WaitForQualities;
			}
		}
		
		BatchState state_startcorrection_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::StartCorrection);
			
			if(!canLaunchKernel){
				return BatchState::StartCorrection;
			}else{
				
				auto nucleotide_accessor = [] __device__ (const char* data, int length, int index){
					return Sequence_t::get_as_nucleotide(data, length, index);
				};

				auto make_unpacked_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
					return care::SequenceString::make_reverse_complement_inplace(sequence, sequencelength);
				};
			
				const double avg_support_threshold = 1.0-1.0*transFuncData.estimatedErrorrate;
				const double min_support_threshold = 1.0-3.0*transFuncData.estimatedErrorrate;
				// coverage is always >= 1
				const double min_coverage_threshold = std::max(1.0,
															transFuncData.m_coverage / 6.0 * transFuncData.estimatedCoverage);
				const int new_columns_to_correct = transFuncData.new_columns_to_correct;
				const float desiredAlignmentMaxErrorRate = transFuncData.maxErrorRate;

				call_msa_add_sequences_kernel_async(
								dataArrays.d_multiple_sequence_alignments,
								dataArrays.d_multiple_sequence_alignment_weights,
								dataArrays.d_alignment_shifts,
								dataArrays.d_alignment_best_alignment_flags,
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
								*dataArrays.h_num_indices,
								transFuncData.useQualityScores,
								desiredAlignmentMaxErrorRate,
								dataArrays.maximum_sequence_length,
								dataArrays.encoded_sequence_pitch,
								dataArrays.quality_pitch,
								dataArrays.msa_pitch,
								dataArrays.msa_weights_pitch,
								nucleotide_accessor,
								make_unpacked_reverse_complement_inplace,
								streams[primary_stream_index]);

				//Step 13. Determine consensus in multiple sequence alignment

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
								*dataArrays.h_num_indices,
								dataArrays.msa_pitch,
								dataArrays.msa_weights_pitch,
								3*dataArrays.maximum_sequence_length - 2*transFuncData.min_overlap,
								streams[primary_stream_index]);

				// Step 14. Correction

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
								*dataArrays.h_num_indices,
								dataArrays.sequence_pitch,
								dataArrays.msa_pitch,
								dataArrays.msa_weights_pitch,
								transFuncData.estimatedErrorrate,
								avg_support_threshold,
								min_support_threshold,
								min_coverage_threshold,
								transFuncData.kmerlength,
								dataArrays.maximum_sequence_length,
								streams[primary_stream_index]);

				if(transFuncData.correctCandidates){


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
					call_msa_correct_candidates_kernel_async(
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
									dataArrays.d_candidate_sequences_lengths,
									dataArrays.d_num_corrected_candidates,
									dataArrays.d_corrected_candidates,
									dataArrays.d_indices_of_corrected_candidates,
									dataArrays.n_subjects,
									dataArrays.n_queries,
									//dataArrays.n_indices,
									*dataArrays.h_num_indices,
									dataArrays.sequence_pitch,
									dataArrays.msa_pitch,
									dataArrays.msa_weights_pitch,
									min_support_threshold,
									min_coverage_threshold,
									new_columns_to_correct,
									make_unpacked_reverse_complement_inplace,
									dataArrays.maximum_sequence_length,
									streams[primary_stream_index]);

				}

				//copy correction results to host
				cudaMemcpyAsync(dataArrays.correction_results_transfer_data_host,
								dataArrays.correction_results_transfer_data_device,
								dataArrays.correction_results_transfer_data_usable_size,
								D2H,
								streams[primary_stream_index]); CUERR;

				assert(cudaSuccess == cudaEventQuery(events[correction_finished_event_index])); CUERR;

				cudaEventRecord(events[correction_finished_event_index], streams[primary_stream_index]); CUERR;		

				return BatchState::WaitForCorrection;
			}
		}
		
		BatchState state_waitforcorrection_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::WaitForCorrection);
			
			cudaError_t querystatus = cudaEventQuery(events[correction_finished_event_index]); CUERR;
			
			assert(querystatus == cudaSuccess || querystatus == cudaErrorNotReady);
			
			if(querystatus == cudaSuccess){
				return BatchState::UnpackResults;
			}else{
				return BatchState::WaitForCorrection;
			}
		}
		
		BatchState state_unpackresults_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::UnpackResults);			
			assert(cudaEventQuery(events[correction_finished_event_index]) == cudaSuccess); CUERR;
			
			for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index){
				auto& task = batch.tasks[subject_index];
				auto& arrays = dataArrays;

				const char* const my_corrected_subject_data = arrays.h_corrected_subjects + subject_index * arrays.sequence_pitch;
				const char* const my_corrected_candidates_data = arrays.h_corrected_candidates + arrays.h_indices_per_subject_prefixsum[subject_index] * arrays.sequence_pitch;
				const int* const my_indices_of_corrected_candidates = arrays.h_indices_of_corrected_candidates + arrays.h_indices_per_subject_prefixsum[subject_index];

				const bool any_correction_candidates = arrays.h_indices_per_subject[subject_index] > 0;

				//has subject been corrected ?
				task.corrected = arrays.h_subject_is_corrected[subject_index];
				//if corrected, copy corrected subject to task
				if(task.corrected){

					const int subject_length = task.subject_sequence->length();

					task.corrected_subject = std::move(std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length});
				}

				if(transFuncData.correctCandidates){
					const int n_corrected_candidates = arrays.h_num_corrected_candidates[subject_index];

					assert((!any_correction_candidates && n_corrected_candidates == 0) || any_correction_candidates);

					for(int i = 0; i < n_corrected_candidates; ++i){
						const int global_candidate_index = my_indices_of_corrected_candidates[i];
						const int local_candidate_index = global_candidate_index - arrays.h_candidates_per_subject_prefixsum[subject_index];

						const int candidate_length = task.candidate_sequences[local_candidate_index]->length();
						const char* const candidate_data = my_corrected_candidates_data + i * arrays.sequence_pitch;

						task.corrected_candidates_read_ids.emplace_back(task.candidate_read_ids[local_candidate_index]);

						task.corrected_candidates.emplace_back(std::move(std::string{candidate_data, candidate_data + candidate_length}));
					}
				}
			}		

			return BatchState::WriteResults;
		}
		
		BatchState state_writeresults_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::WriteResults);			
			
			//write result to file
			for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index){

				const auto& task = batch.tasks[subject_index];

				//std::cout << "finished readId " << task.readId << std::endl;

				if(task.corrected){
					//std::cout << task.readId << " " << task.corrected_subject << std::endl;
					transFuncData.write_read_to_stream(task.readId, task.corrected_subject);
					transFuncData.lock(task.readId);
					(*transFuncData.readIsCorrectedVector)[task.readId] = 1;
					transFuncData.unlock(task.readId);
				}

				for(std::size_t corrected_candidate_index = 0; corrected_candidate_index < task.corrected_candidates.size(); ++corrected_candidate_index){

					ReadId_t candidateId = task.corrected_candidates_read_ids[corrected_candidate_index];
					const std::string& corrected_candidate = task.corrected_candidates[corrected_candidate_index];

					bool savingIsOk = false;
					if((*transFuncData.readIsCorrectedVector)[candidateId] == 0){
						transFuncData.lock(candidateId);
						if((*transFuncData.readIsCorrectedVector)[candidateId]== 0) {
							(*transFuncData.readIsCorrectedVector)[candidateId] = 1; // we will process this read
							savingIsOk = true;
							nCorrectedCandidates++;
						}
						transFuncData.unlock(candidateId);
					}
					if (savingIsOk) {
						transFuncData.write_read_to_stream(candidateId, corrected_candidate);
					}
				}
			}

			return BatchState::Finished;
		}
		
		BatchState state_finished_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::Finished);			
			
			assert(false); //Finished is end node

			return BatchState::Finished;
		}
		
		BatchState state_aborted_func(Batch& batch,
										DataArrays<Sequence_t>& dataArrays,
										std::array<cudaStream_t, nStreamsPerBatch>& streams,
										std::array<cudaEvent_t, nEventsPerBatch>& events,
										bool canBlock,
										bool canLaunchKernel,
										const TransitionFunctionData& transFuncData){
			
			assert(batch.state == BatchState::Aborted);			
			
			assert(false); //Aborted is end node

			return BatchState::Aborted;
		}
		

        AdvanceResult advance_one_step(Batch& batch,
                                DataArrays<Sequence_t>& dataArrays,
                                std::array<cudaStream_t, nStreamsPerBatch>& streams,
                                std::array<cudaEvent_t, nEventsPerBatch>& events,
                                bool canBlock,
                                bool canLaunchKernel,
								const TransitionFunctionData& transFuncData){
			
			

			AdvanceResult advanceResult;
			
			advanceResult.oldState = batch.state;
			advanceResult.noProgressBlocking = false;
			advanceResult.noProgressLaunching = false;
#if 1			
			#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))
			
			auto iter = transitionFunctionTable.find(batch.state);
			if(iter != transitionFunctionTable.end()){
				batch.state = CALL_MEMBER_FN(*this, iter->second)(batch, dataArrays, streams, events, canBlock, canLaunchKernel, transFuncData);
			}else{
					std::cout << nameOf(batch.state) << std::endl;
                    assert(false); // Every State should be handled above	
			}
			
			#undef CALL_MEMBER_FN
			
			
#else
            switch(batch.state){

				case BatchState::Unprepared: {
					
					batch.state = state_unprepared_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
				}

                case BatchState::CopyReads:{
                    batch.state = state_copyreads_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);

                    break;
                }

                case BatchState::TransferReads:{
                    batch.state = state_transferreads_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }
                
                case BatchState::StartAlignment:{
                    batch.state = state_startalignment_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }
                
                case BatchState::WaitForAlignment:{
                    batch.state = state_waitforalignment_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }
                
                case BatchState::TransferIndices:{
                    batch.state = state_transferindices_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }
                
                case BatchState::WaitForIndices:{
                    batch.state = state_waitforindices_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }
                
                case BatchState::CopyQualities:{
                    batch.state = state_copyqualities_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }
                
                case BatchState::TransferQualities:{
                    batch.state = state_transferqualities_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }
                
                case BatchState::WaitForQualities:{
                    batch.state = state_waitforqualities_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }
                
                case BatchState::StartCorrection:{
                    batch.state = state_startcorrection_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }
                
                case BatchState::WaitForCorrection:{
                    batch.state = state_waitforcorrection_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }
                
                case BatchState::UnpackResults:{
                    batch.state = state_unpackresults_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }
                
                case BatchState::WriteResults:{
                    batch.state = state_writeresults_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }
                
                case BatchState::Finished:{
                    batch.state = state_finished_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
					break;
                }                
                
				case BatchState::Aborted:{
					batch.state = state_aborted_func(batch, dataArrays, streams, events, canBlock, canLaunchKernel);
					
                    break;
				}

                default:
					std::cout << nameOf(batch.state) << std::endl;
                    assert(false); // Every State should be handled above
                    break;
            }
#endif            
            advanceResult.newState = batch.state;

            return advanceResult;
        }

        /*void get_subjects(Batch& batch, const std::vector<ReadId_t>& readIds){

            const auto& readStorage = threadOpts.readStorage;

            batch.tasks.clear();
            // Get subject sequence
            for(ReadId_t id : readIds){
                bool ok = false;
                lock(id);
                if ((*threadOpts.readIsCorrectedVector)[id] == 0) {
                    (*threadOpts.readIsCorrectedVector)[id] = 1;
                    ok = true;
                }else{
                }
                unlock(id);

                if(ok){
                    const Sequence_t* sequenceptr = readStorage->fetchSequence_ptr(id);
                    const std::string* qualityptr = nullptr;

                    if(correctionOptions.useQualityScores)
                        qualityptr = readStorage->fetchQuality_ptr(id);

                    batch.tasks.emplace_back(id, sequenceptr, qualityptr);
                }
            }
        }

        void get_candidates(Batch& batch) const{

            const auto& readStorage = threadOpts.readStorage;
            const auto& minhasher = threadOpts.minhasher;

            for(auto& task : batch.tasks){
                const std::string sequencestring = task.subject_sequence->toString();
                task.candidate_read_ids = minhasher->getCandidates(sequencestring, max_candidates);

                //remove self from candidates
                auto readIdPos = std::find(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.readId);
                if(readIdPos != task.candidate_read_ids.end())
                    task.candidate_read_ids.erase(readIdPos);

                if(task.candidate_read_ids.size() == 0){
                    //no need for further processing without candidates
                    task.active = false;
                }else{
                    for(auto candidate_read_id : task.candidate_read_ids){
                        task.candidate_sequences.emplace_back(readStorage->fetchSequence_ptr(candidate_read_id));
                        if(correctionOptions.useQualityScores)
                            task.candidate_qualities.emplace_back(readStorage->fetchQuality_ptr(candidate_read_id));
                    }
                }
            }
        }*/
#if 0        
        void add_tasks(Batch& batch, const std::vector<ReadId_t>& readIds){
			const auto& readStorage = threadOpts.readStorage;
			const auto& minhasher = threadOpts.minhasher;
			
			for(ReadId_t id : readIds){
                bool ok = false;
                lock(id);
                if ((*threadOpts.readIsCorrectedVector)[id] == 0) {
                    (*threadOpts.readIsCorrectedVector)[id] = 1;
                    ok = true;
                }else{
                }
                unlock(id);

                if(ok){
                    const Sequence_t* sequenceptr = readStorage->fetchSequence_ptr(id);
                    const std::string* qualityptr = nullptr;

                    if(correctionOptions.useQualityScores)
                        qualityptr = readStorage->fetchQuality_ptr(id);

                    batch.tasks.emplace_back(id, sequenceptr, qualityptr);
					
					auto& task = batch.tasks.back();
					
					const std::string sequencestring = task.subject_sequence->toString();
					task.candidate_read_ids = minhasher->getCandidates(sequencestring, max_candidates);

					//remove self from candidates
					auto readIdPos = std::find(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.readId);
					if(readIdPos != task.candidate_read_ids.end())
						task.candidate_read_ids.erase(readIdPos);

					if(task.candidate_read_ids.size() == 0){
						//no need for further processing without candidates
						task.active = false;
					}else{
						for(auto candidate_read_id : task.candidate_read_ids){
							task.candidate_sequences.emplace_back(readStorage->fetchSequence_ptr(candidate_read_id));
							if(correctionOptions.useQualityScores)
								task.candidate_qualities.emplace_back(readStorage->fetchQuality_ptr(candidate_read_id));
						}
					}
					
					batch.initialNumberOfCandidates += int(task.candidate_read_ids.size());
                }
            }

		}
#endif		
#if 0
        void allocate_data(Batch& batch, DataArrays<Sequence_t>& dataArrays, int nTotalCandidates, cudaStream_t stream){
            dataArrays.set_problem_dimensions(int(batch.tasks.size()),
                                                        nTotalCandidates,
                                                        fileProperties.maxSequenceLength,
                                                        goodAlignmentProperties.min_overlap,
                                                        goodAlignmentProperties.min_overlap_ratio,
                                                        correctionOptions.useQualityScores); CUERR;

            std::size_t temp_storage_bytes = 0;
            std::size_t max_temp_storage_bytes = 0;
            cub::DeviceHistogram::HistogramRange((void*)nullptr, temp_storage_bytes,
                                                (int*)nullptr, (int*)nullptr,
                                                dataArrays.n_subjects+1,
                                                (int*)nullptr,
                                                dataArrays.n_queries,
                                                stream); CUERR;

            max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

            cub::DeviceSelect::Flagged((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                                        (bool*)nullptr, (int*)nullptr, (int*)nullptr,
                                        nTotalCandidates,
                                        stream); CUERR;

            max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

            cub::DeviceScan::ExclusiveSum((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                                            (int*)nullptr,
                                            dataArrays.n_subjects,
                                            stream); CUERR;

            max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);
            temp_storage_bytes = max_temp_storage_bytes;

            dataArrays.set_tmp_storage_size(max_temp_storage_bytes);
            dataArrays.zero_gpu(stream);
        }
#endif
#if 0
        bool gather_candidates_of_next_task(Batch& batch, DataArrays<Sequence_t>& dataArrays) const{
            dataArrays.h_candidates_per_subject_prefixsum[0] = 0;

            assert(batch.copiedTasks <= int(batch.tasks.size()));
			
			bool success = false;

			//copy one task
            if(batch.copiedTasks < int(batch.tasks.size())){
                const auto& task = batch.tasks[batch.copiedTasks];
                auto& arrays = dataArrays;

                //fill subject
                std::memcpy(arrays.h_subject_sequences_data + batch.copiedTasks * arrays.encoded_sequence_pitch,
                            task.subject_sequence->begin(),
                            task.subject_sequence->getNumBytes());
                arrays.h_subject_sequences_lengths[batch.copiedTasks] = task.subject_sequence->length();
                batch.maxSubjectLength = std::max(int(task.subject_sequence->length()),
                                                                batch.maxSubjectLength);

                //fill candidates
                for(const Sequence_t* candidate_sequence : task.candidate_sequences){

                    std::memcpy(arrays.h_candidate_sequences_data
                                    + batch.copiedCandidates * arrays.encoded_sequence_pitch,
                                candidate_sequence->begin(),
                                candidate_sequence->getNumBytes());

                    arrays.h_candidate_sequences_lengths[batch.copiedCandidates] = candidate_sequence->length();
                    batch.maxQueryLength = std::max(int(candidate_sequence->length()),
                                                                batch.maxQueryLength);

                    ++batch.copiedCandidates;
                }

                //make prefix sum
                arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks+1]
                                = arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks]
                                    + int(task.candidate_read_ids.size());

                ++batch.copiedTasks;
				
				success = true;
            }
            
            return success;
        }
#endif

    	void execute() {
    		isRunning = true;

			assert(threadOpts.canUseGpu);
			
			mybatchgen = BatchGenerator<ReadId_t>(threadOpts.batchGen->firstId, threadOpts.batchGen->lastIdExcl);
			makeTransitionFunctionTable();

    		//std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

    		std::ofstream outputstream(threadOpts.outputfile);
            if(!outputstream)
                throw std::runtime_error("Could not open output file");

            std::ofstream featurestream(threadOpts.outputfile + "_features");
            if(!featurestream)
                throw std::runtime_error("Could not open output feature file");


    		constexpr int nParallelBatches = 1;

			cudaSetDevice(threadOpts.deviceId); CUERR;

            //std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();

    		std::vector<DataArrays<Sequence_t>> dataArrays;
            //std::array<Batch, nParallelBatches> batches;
            std::array<std::array<cudaStream_t, nStreamsPerBatch>, nParallelBatches> streams;
			std::array<std::array<cudaEvent_t, nEventsPerBatch>, nParallelBatches> cudaevents;

			std::queue<Batch> batchQueue;

            for(int i = 0; i < nParallelBatches; i++){
                dataArrays.emplace_back(threadOpts.deviceId);

                for(int j = 0; j < nStreamsPerBatch; ++j){
                    cudaStreamCreate(&streams[i][j]); CUERR;
                }

                for(int j = 0; j < nEventsPerBatch; ++j){
                    cudaEventCreateWithFlags(&cudaevents[i][j], cudaEventDisableTiming); CUERR;
                }

                /*dataArrays[i].set_problem_dimensions(readIds.size(),
                                                            max_candidates * readIds.size(),
                                                            fileProperties.maxSequenceLength,
                                                            goodAlignmentProperties.min_overlap,
                                                            goodAlignmentProperties.min_overlap_ratio,
                                                            correctionOptions.useQualityScores);*/

            }

            auto nextBatchIndex = [](int currentBatchIndex, int nParallelBatches){
                if(nParallelBatches > 1)
                    return currentBatchIndex == 0 ? 1 : 0;
                else
                    assert(false);
                    return currentBatchIndex;
            };
			
		
		
			TransitionFunctionData transFuncData;
			
			transFuncData.mybatchgen = &mybatchgen;
			transFuncData.readStorage = threadOpts.readStorage;
			transFuncData.minhasher = threadOpts.minhasher;
			transFuncData.min_overlap_ratio = goodAlignmentProperties.min_overlap_ratio;
			transFuncData.min_overlap = goodAlignmentProperties.min_overlap;
			transFuncData.estimatedErrorrate = correctionOptions.estimatedErrorrate;
			transFuncData.maxErrorRate = goodAlignmentProperties.maxErrorRate;
			transFuncData.estimatedCoverage = correctionOptions.estimatedCoverage;
			transFuncData.m_coverage = correctionOptions.m_coverage;
			transFuncData.new_columns_to_correct = correctionOptions.new_columns_to_correct;
			transFuncData.correctCandidates = correctionOptions.correctCandidates;
			transFuncData.useQualityScores = correctionOptions.useQualityScores;
			transFuncData.kmerlength = correctionOptions.kmerlength;
			transFuncData.num_ids_per_add_tasks = num_ids_per_add_tasks;
			transFuncData.minimum_candidates_per_batch = minimum_candidates_per_batch;
			transFuncData.maxSequenceLength = fileProperties.maxSequenceLength;
			transFuncData.locksForProcessedFlags = threadOpts.locksForProcessedFlags;
			transFuncData.nLocksForProcessedFlags = threadOpts.nLocksForProcessedFlags;
			transFuncData.readIsCorrectedVector = threadOpts.readIsCorrectedVector;
			transFuncData.write_read_to_stream = [&](const ReadId_t readId, const std::string& sequence){
                //std::cout << readId << " " << sequence << std::endl;
    			auto& stream = outputstream;
    			stream << readId << '\n';
    			stream << sequence << '\n';
    		};
			transFuncData.lock = [&](ReadId_t readId){
				ReadId_t index = readId % transFuncData.nLocksForProcessedFlags;
				transFuncData.locksForProcessedFlags[index].lock();
			};
			transFuncData.unlock = [&](ReadId_t readId){
				ReadId_t index = readId % transFuncData.nLocksForProcessedFlags;
				transFuncData.locksForProcessedFlags[index].unlock();
			};

            int batchIndex = 0; // the main stream we are working on in the current loop iteration

            //int num_finished_batches = 0;
            
            int stacksize = 0;

    		//while(!stopAndAbort && !(num_finished_batches == nParallelBatches && readIds.empty())){
			while(!stopAndAbort && !(batchQueue.empty() && mybatchgen.empty())){
				
				if(stacksize != 0)
						assert(stacksize == 0);

				Batch mainBatch;

				if(!batchQueue.empty()){
						mainBatch = std::move(batchQueue.front());
						batchQueue.pop();

                        //std::cout << "queuebatch : " << int(mainBatch.state) << std::endl;
				}


				AdvanceResult mainBatchAdvanceResult;
				bool firstMainIter = true;
				bool popMain = false;
				
				
				
                while(!(mainBatch.state == BatchState::Finished || mainBatch.state == BatchState::Aborted)){
	
					if(firstMainIter){
						assert(popMain == false);
						push_range("mainBatch"+nameOf(mainBatch.state)+"first", int(mainBatch.state));
						++stacksize;
						popMain = true;
					}else{
						if(mainBatchAdvanceResult.oldState != mainBatchAdvanceResult.newState){
							assert(popMain == false);
							push_range("mainBatch"+nameOf(mainBatchAdvanceResult.newState), int(mainBatchAdvanceResult.newState));
							++stacksize;
							popMain = true;
						}
					}
					
                    mainBatchAdvanceResult = advance_one_step(mainBatch,
                                                            dataArrays[batchIndex],
                                                            streams[batchIndex],
                                                            cudaevents[batchIndex],
                                                            true, //can block
                                                            true, //can launch kernels
															transFuncData);
#if 0
					IsWaitingResult isWaitingResult = isWaiting(mainBatch.state);
                    if(isWaitingResults.isWaiting){
                        /*
                            Prepare next batch while waiting for the mainBatch gpu work to finish
                        */
                        if(nParallelBatches > 1){
							PUSH_RANGE_2("sidework",3);
                            do{
                                const int nextBatchId = nextBatchIndex(batchIndex, nParallelBatches);
                                assert(nextBatchId != batchIndex);
                                Batch sideBatch;

                                if(!batchQueue.empty()){
                                    sideBatch = std::move(batchQueue.front());
                                    batchQueue.pop();

                                    //assert(sideBatch.state != BatchState::Unprepared);
                                }else{
                                    if(mybatchgen.empty()){
                                        break; //no next batch to prepare
                                    }
                                }

                                cudaError_t eventquerystatus = cudaSuccess;
                                cudaEvent_t eventToWaitFor = cudaevents[batchIndex][isWaitingResults.eventIndexToWaitFor];
															 
								AdvanceResult sideBatchAdvanceResult;
								bool firstSideIter = true;
								bool popSide = false;
								
                                //while mainBatch is not ready for next step...
                                //this condition is checked after each step of sideBatch
                                while((eventquerystatus = cudaEventQuery(eventToWaitFor)) == cudaErrorNotReady){
									
									if(firstSideIter){
										assert(popSide == false);
										push_range("sideBatch"+nameOf(sideBatch.state)+"first", int(sideBatch.state));
										++stacksize;
										popSide = true;
									}else{
										if(sideBatchAdvanceResult.oldState != sideBatchAdvanceResult.newState){
											assert(popSide == false);
											push_range("sideBatch"+nameOf(sideBatchAdvanceResult.newState), int(sideBatchAdvanceResult.newState));
											++stacksize;
											popSide = true;
										}
									}

                                    sideBatchAdvanceResult = advance_one_step(sideBatch,
                                                                            dataArrays[nextBatchId],
                                                                            streams[nextBatchId],
                                                                            cudaevents[nextBatchId],
                                                                            false, //must not block
																		    false); //cannot launch kernels
									
									/*if((sideBatchAdvanceResult.noProgressBlocking || sideBatchAdvanceResult.noProgressLaunching) && batchQueue.size() < 2){
										//the current side batch cannot make progress. switch it with a second side batch
										int queuesize = batchQueue.size();
										
										batchQueue.push(std::move(sideBatch));
										
										if(queuesize == 0){
											sideBatch = Batch{};
										}else{
											sideBatch = std::move(batchQueue.front());
											batchQueue.pop();
										}
									}*/
									
									if(sideBatchAdvanceResult.oldState != sideBatchAdvanceResult.newState){
										pop_range("side inner");
										popSide = false;
										--stacksize;
									}
									
									firstSideIter = false;
                                }
                                
                                if(popSide){
									pop_range("side outer");
									popSide = false;
									--stacksize;
								}

                                assert(eventquerystatus == cudaSuccess);

                                //if(sideBatch.state != BatchState::Unprepared){
								batchQueue.push(std::move(sideBatch));
                                //}
                            }while(0);
							POP_RANGE_2;
                        }
                    }
#endif
#if 0
                    if(mainBatch.state == BatchState::FinishedCorrectionWork){
						/*
						Launch alignment kernels of side batch
						*/
						if(nParallelBatches > 1){
							PUSH_RANGE_2("sidework",3);
							const int nextBatchId = nextBatchIndex(batchIndex, nParallelBatches);
							assert(nextBatchId != batchIndex);
							Batch sideBatch;

							if(!batchQueue.empty()){
								sideBatch = std::move(batchQueue.front());
								batchQueue.pop();

								assert(sideBatch.state != BatchState::Unprepared);
								
								if(sideBatch.state == BatchState::DataOnGPU){
									push_range("sideBatch"+nameOf(sideBatch.state), int(sideBatch.state));
									++stacksize;
									
									advance_one_step(sideBatch,
													dataArrays[nextBatchId],
													streams[nextBatchId],
													cudaevents[nextBatchId],
													false, //must not block
													true); //can launch kernels
									
									pop_range("side");
									--stacksize;
								}
								
								batchQueue.push(std::move(sideBatch));
							}
							
							POP_RANGE_2;
						}
                    }
#endif                    
                    
                    if((mainBatchAdvanceResult.oldState != mainBatchAdvanceResult.newState)){
						pop_range("main inner");
						popMain = false;
						--stacksize;
					}

					firstMainIter = false;
                }
                
                if(popMain){
					pop_range("main outer");
					popMain = false;
					--stacksize;
				}
				
				assert(mainBatch.state == BatchState::Finished || mainBatch.state == BatchState::Aborted); 
#if 0
                assert(mainBatch.state == BatchState::WriteResults || mainBatch.state == BatchState::Aborted); 
				
				if(mainBatch.state == BatchState::WriteResults){

					//write result to file
					push_range("mainBatch"+nameOf(mainBatch.state), int(mainBatch.state));
					++stacksize;
					
					PUSH_RANGE_2("write_results", 8);
					for(std::size_t subject_index = 0; subject_index < mainBatch.tasks.size(); ++subject_index){

						const auto& task = mainBatch.tasks[subject_index];

						//std::cout << "finished readId " << task.readId << std::endl;

						if(task.corrected){
							//std::cout << task.readId << " " << task.corrected_subject << std::endl;
							write_read(task.readId, task.corrected_subject);
							lock(task.readId);
							(*threadOpts.readIsCorrectedVector)[task.readId] = 1;
							unlock(task.readId);
						}

						for(std::size_t corrected_candidate_index = 0; corrected_candidate_index < task.corrected_candidates.size(); ++corrected_candidate_index){

							ReadId_t candidateId = task.corrected_candidates_read_ids[corrected_candidate_index];
							const std::string& corrected_candidate = task.corrected_candidates[corrected_candidate_index];

							bool savingIsOk = false;
							if((*threadOpts.readIsCorrectedVector)[candidateId] == 0){
								lock(candidateId);
								if((*threadOpts.readIsCorrectedVector)[candidateId]== 0) {
									(*threadOpts.readIsCorrectedVector)[candidateId] = 1; // we will process this read
									savingIsOk = true;
									nCorrectedCandidates++;
								}
								unlock(candidateId);
							}
							if (savingIsOk) {
								write_read(candidateId, corrected_candidate);
							}
						}
					}
					POP_RANGE_2;
					pop_range("main");
					--stacksize;
					
					mainBatch.state = BatchState::Finished;
				
				}
#endif
				if(nParallelBatches > 1){
					batchIndex = nextBatchIndex(batchIndex, nParallelBatches);
				}
				
				nProcessedReads = mybatchgen.currentId - mybatchgen.firstId;




#if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_MEMCOPY && 0
				//DEBUGGING
				cudaMemcpyAsync(dataArrays[batchIndex].msa_data_host,
								dataArrays[batchIndex].msa_data_device,
								dataArrays[batchIndex].msa_data_usable_size,
								D2H,
								streams[batchIndex][0]); CUERR;
				cudaStreamSynchronize(streams[batchIndex][0]); CUERR;

				//DEBUGGING
				cudaMemcpyAsync(dataArrays[batchIndex].alignment_result_data_host,
								dataArrays[batchIndex].alignment_result_data_device,
								dataArrays[batchIndex].alignment_result_data_usable_size,
								D2H,
								streams[batchIndex][0]); CUERR;
				cudaStreamSynchronize(streams[batchIndex][0]); CUERR;

				//DEBUGGING
				cudaMemcpyAsync(dataArrays[batchIndex].subject_indices_data_host,
								dataArrays[batchIndex].subject_indices_data_device,
								dataArrays[batchIndex].subject_indices_data_usable_size,
								D2H,
								streams[batchIndex][0]); CUERR;
				cudaStreamSynchronize(streams[batchIndex][0]); CUERR;
#endif

/*std::cout << "h_is_high_quality_subject" << std::endl;
for(int i = 0; i< dataArrays[batchIndex].n_subjects; i++){
    std::cout << dataArrays[batchIndex].h_is_high_quality_subject[i] << "\t";
}
std::cout << std::endl;*/

#if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_PRINT_ARRAYS && 0
				//DEBUGGING
				std::cout << "alignment scores" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_queries * 2; i++){
					std::cout << dataArrays[batchIndex].h_alignment_scores[i] << "\t";
				}
				std::cout << std::endl;
				//DEBUGGING
				std::cout << "alignment overlaps" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_queries * 2; i++){
					std::cout << dataArrays[batchIndex].h_alignment_overlaps[i] << "\t";
				}
				std::cout << std::endl;
				//DEBUGGING
				std::cout << "alignment shifts" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_queries * 2; i++){
					std::cout << dataArrays[batchIndex].h_alignment_shifts[i] << "\t";
				}
				std::cout << std::endl;
				//DEBUGGING
				std::cout << "alignment nOps" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_queries * 2; i++){
					std::cout << dataArrays[batchIndex].h_alignment_nOps[i] << "\t";
				}
				std::cout << std::endl;
				//DEBUGGING
				std::cout << "alignment isvalid" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_queries * 2; i++){
					std::cout << dataArrays[batchIndex].h_alignment_isValid[i] << "\t";
				}
				std::cout << std::endl;
				//DEBUGGING
				std::cout << "alignment flags" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_queries * 2; i++){
					std::cout << int(dataArrays[batchIndex].h_alignment_best_alignment_flags[i]) << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_candidates_per_subject_prefixsum" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_subjects +1; i++){
					std::cout << dataArrays[batchIndex].h_candidates_per_subject_prefixsum[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_num_indices" << std::endl;
				for(int i = 0; i< 1; i++){
					std::cout << dataArrays[batchIndex].h_num_indices[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_indices" << std::endl;
				for(int i = 0; i< *dataArrays[batchIndex].h_num_indices; i++){
					std::cout << dataArrays[batchIndex].h_indices[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_indices_per_subject" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_subjects; i++){
					std::cout << dataArrays[batchIndex].h_indices_per_subject[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_indices_per_subject_prefixsum" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_subjects; i++){
					std::cout << dataArrays[batchIndex].h_indices_per_subject_prefixsum[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_high_quality_subject_indices" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_subjects; i++){
					std::cout << dataArrays[batchIndex].h_high_quality_subject_indices[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_is_high_quality_subject" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_subjects; i++){
					std::cout << dataArrays[batchIndex].h_is_high_quality_subject[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_num_high_quality_subject_indices" << std::endl;
				for(int i = 0; i< 1; i++){
					std::cout << dataArrays[batchIndex].h_num_high_quality_subject_indices[i] << "\t";
				}
				std::cout << std::endl;

                //DEBUGGING
				std::cout << "h_num_corrected_candidates" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_subjects; i++){
					std::cout << dataArrays[batchIndex].h_num_corrected_candidates[i] << "\t";
				}
				std::cout << std::endl;

                //DEBUGGING
                std::cout << "h_subject_is_corrected" << std::endl;
				for(int i = 0; i< dataArrays[batchIndex].n_subjects; i++){
					std::cout << dataArrays[batchIndex].h_subject_is_corrected[i] << "\t";
				}
				std::cout << std::endl;

                //DEBUGGING
				std::cout << "h_indices_of_corrected_candidates" << std::endl;
				for(int i = 0; i< *dataArrays[batchIndex].h_num_indices; i++){
					std::cout << dataArrays[batchIndex].h_indices_of_corrected_candidates[i] << "\t";
				}
				std::cout << std::endl;

#if 0
				{
					auto& arrays = dataArrays[batchIndex];
					for(int row = 0; row < *arrays[batchIndex].h_num_indices+1 && row < 50; ++row){
						for(int col = 0; col < arrays.msa_pitch; col++){
							char c = arrays.h_multiple_sequence_alignments[row * arrays.msa_pitch + col];
							std::cout << (c == '\0' ? '0' : c);
						}
						std::cout << std::endl;
					}
				}
#endif

#endif


#if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_PRINT_MSA && 0

				//DEBUGGING
				for(std::size_t subject_index = 0; subject_index < correctionTasks[batchIndex].size(); ++subject_index){
					auto& task = correctionTasks[batchIndex][subject_index];
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
					for(int row = 0; row < msa_rows; row++){
						for(int col = 0; col < columnsToCheck; col++){
							//multiple_sequence_alignment[row * msa_row_pitch + globalIndex]
							char c = my_multiple_sequence_alignment[row * arrays.msa_pitch + col];
                            assert(c != 'F');
							std::cout << (c == '\0' ? '0' : c);
							if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
								std::cout << " ";
						}
						if(row > 0){
							const int queryIndex = indices_for_this_subject[row-1];
							const int shift = arrays.h_alignment_shifts[queryIndex];

							std::cout << " shift " << shift;
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;

					std::cout << "Consensus: "<< std::endl;
					for(int col = 0; col < columnsToCheck; col++){
						char c = my_consensus[col];
						std::cout << (c == '\0' ? '0' : c);
						if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
								std::cout << " ";
					}
					std::cout << std::endl;

					std::cout << "MSA weights:" << std::endl;
					for(int row = 0; row < msa_rows; row++){
						for(int col = 0; col < columnsToCheck; col++){
							float f = my_multiple_sequence_alignment_weight[row * msa_weights_pitch_floats + col];
							std::cout << f << " ";
							if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
								std::cout << " ";
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;

					std::cout << "Support: "<< std::endl;
					for(int col = 0; col < columnsToCheck; col++){
						std::cout << my_support[col] << " ";
					}
					std::cout << std::endl;

					std::cout << "Coverage: "<< std::endl;
					for(int col = 0; col < columnsToCheck; col++){
						std::cout << my_coverage[col] << " ";
					}
					std::cout << std::endl;

					std::cout << "Orig weights: "<< std::endl;
					for(int col = 0; col < columnsToCheck; col++){
						std::cout << my_orig_weights[col] << " ";
					}
					std::cout << std::endl;

					std::cout << "Orig coverage: "<< std::endl;
					for(int col = 0; col < columnsToCheck; col++){
						std::cout << my_orig_coverage[col] << " ";
					}
					std::cout << std::endl;


				}
#endif




//#ifdef CARE_GPU_DEBUG
				//stopAndAbort = true; //remove
//#endif


    		} // end batch processing

            outputstream.flush();
            featurestream.flush();
			
			std::cout << "new gpu thread finished" << std::endl;

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

            for(auto& array : dataArrays){
                array.reset();
            }

            for(auto& streamarray : streams){
                for(auto& stream : streamarray)
                    cudaStreamDestroy(stream); CUERR;
            }

            for(auto& eventarray : cudaevents){
                for(auto& event : eventarray)
                    cudaEventDestroy(event); CUERR;
            }
    	}
    };

#endif

}
}

#endif
