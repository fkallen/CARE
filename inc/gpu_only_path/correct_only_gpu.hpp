#ifndef CARE_CORRECT_ONLY_GPU_HPP
#define CARE_CORRECT_ONLY_GPU_HPP

#define USE_NVTX2


#if defined USE_NVTX2 && defined __NVCC__
#include <nvToolsExt.h>

const uint32_t colors_[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff, 0xdeadbeef };
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
    BatchGenerator(std::uint64_t firstId, std::uint64_t lastIdExcl, std::uint64_t batchsize)
            : batchsize(batchsize), firstId(firstId), lastIdExcl(lastIdExcl), currentId(firstId){
                if(batchsize == 0) throw std::runtime_error("BatchGenerator: invalid batch size");
                if(firstId >= lastIdExcl) throw std::runtime_error("BatchGenerator: firstId >= lastIdExcl");
            }
    BatchGenerator(std::uint64_t totalNumberOfReads, std::uint64_t batchsize_, int threadId, int nThreads){
        if(threadId < 0) throw std::runtime_error("BatchGenerator: invalid threadId");
        if(nThreads < 0) throw std::runtime_error("BatchGenerator: invalid nThreads");

    	std::uint64_t chunksize = totalNumberOfReads / nThreads;
    	int leftover = totalNumberOfReads % nThreads;

    	if(threadId < leftover){
    		chunksize++;
    		firstId = threadId == 0 ? 0 : threadId * chunksize;
    		lastIdExcl = firstId + chunksize;
    	}else{
    		firstId = leftover * (chunksize+1) + (threadId - leftover) * chunksize;;
    		lastIdExcl = firstId + chunksize;
    	}


        currentId = firstId;
        batchsize = batchsize_;
        //std::cout << "thread " << threadId << " firstId " << firstId << " lastIdExcl " << lastIdExcl << " batchsize " << batchsize << std::endl;
    };

    std::vector<ReadId_t> getNextReadIds(){
        std::vector<ReadId_t> result;
    	while(result.size() < batchsize && currentId < lastIdExcl){
    		result.push_back(currentId);
    		currentId++;
    	}
        return result;
    }

    std::uint64_t batchsize;
    std::uint64_t firstId;
    std::uint64_t lastIdExcl;
    std::uint64_t currentId;
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

		enum class BatchState{
			Unprepared,
			SubjectsPresent,
			CandidatesPresent,
			DataOnGPU,
			RunningAlignmentWork,
            TransferingQualityScores,
			RunningCorrectionWork,
			Finished,
		};

		struct Batch{
			std::vector<CorrectionTask_t> tasks;
			int maxSubjectLength = 0;
			int maxQueryLength = 0;
			BatchState state = BatchState::Unprepared;

			int copiedTasks = 0; // used if state == CandidatesPresent
			int copiedCandidates = 0; // used if state == CandidatesPresent
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

        BatchState advance_one_step(Batch& sideBatch,
                                const std::vector<ReadId_t>& readIds,
                                DataArrays<Sequence_t>& dataArrays,
                                std::array<cudaStream_t, nStreamsPerBatch>& streams,
                                std::array<cudaEvent_t, nStreamsPerBatch>& events,
                                bool canBlock){

            auto accessor = [] __device__ (const char* data, int length, int index){
                return Sequence_t::get(data, length, index);
            };

            auto nucleotide_accessor = [] __device__ (const char* data, int length, int index){
                return Sequence_t::get_as_nucleotide(data, length, index);
            };

            auto make_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
                return Sequence_t::make_reverse_complement_inplace(sequence, sequencelength);
            };

            auto make_unpacked_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
                return care::SequenceString::make_reverse_complement_inplace(sequence, sequencelength);
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
                                            goodAlignmentProperties.min_overlap_ratio,
                                            goodAlignmentProperties.min_overlap,
                                            correctionOptions.estimatedErrorrate * 4.0);
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

            BatchState oldState = sideBatch.state;

            switch(sideBatch.state){
                case BatchState::Finished:{
                    //do nothing;
                    break;
                }

                case BatchState::Unprepared:{
                    PUSH_RANGE_2("get_subjects_next", 0);

                    get_subjects(sideBatch, readIds);

                    POP_RANGE_2;

                    sideBatch.state = BatchState::SubjectsPresent;

                    break;
                }

                case BatchState::SubjectsPresent:{
                    PUSH_RANGE_2("get_candidates_next", 1);

                    get_candidates(sideBatch);

                    POP_RANGE_2;

                    std::remove_if(sideBatch.tasks.begin(),
                                    sideBatch.tasks.end(),
                                    [](const auto& t){return !t.active;});

                    int nTotalCandidates = std::accumulate(sideBatch.tasks.begin(),
                                                            sideBatch.tasks.end(),
                                                            int(0),
                                                            [](const auto& l, const auto& r){
                                                                return l + int(r.candidate_read_ids.size());
                                                            });

                    if(nTotalCandidates == 0){
                        if(!readIds.empty()){
                            //reset sideBatch state and start over with the next chunk of read ids
                            sideBatch.state = BatchState::Unprepared;
                        }
                    }else{
                        //allocate data arrays
                        PUSH_RANGE_2("set_problem_dimensions_next", 7);
                        dataArrays.set_problem_dimensions(int(sideBatch.tasks.size()),
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
                                                            streams[0]); CUERR;

                        max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

                        cub::DeviceSelect::Flagged((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                                                    (bool*)nullptr, (int*)nullptr, (int*)nullptr,
                                                    nTotalCandidates,
                                                    streams[0]); CUERR;

                        max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

                        cub::DeviceScan::ExclusiveSum((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                                                        (int*)nullptr,
                                                        dataArrays.n_subjects,
                                                        streams[0]); CUERR;

                        max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);
                        temp_storage_bytes = max_temp_storage_bytes;

                        dataArrays.set_tmp_storage_size(max_temp_storage_bytes);

                        POP_RANGE_2;

                        PUSH_RANGE_2("zero_gpu_next", 5);
                        dataArrays.zero_gpu(streams[0]);
                        POP_RANGE_2;

                        sideBatch.state = BatchState::CandidatesPresent;
                    }

                    break;
                }

                case BatchState::CandidatesPresent:{
                    dataArrays.h_candidates_per_subject_prefixsum[0] = 0;

                    PUSH_RANGE_2("copy_to_buffer_next", 2);

                    //copy one task
                    if(sideBatch.copiedTasks < int(sideBatch.tasks.size())){
                        const auto& task = sideBatch.tasks[sideBatch.copiedTasks];
                        auto& arrays = dataArrays;

                        //fill subject
                        std::memcpy(arrays.h_subject_sequences_data + sideBatch.copiedTasks * arrays.encoded_sequence_pitch,
                                    task.subject_sequence->begin(),
                                    task.subject_sequence->getNumBytes());
                        arrays.h_subject_sequences_lengths[sideBatch.copiedTasks] = task.subject_sequence->length();
                        sideBatch.maxSubjectLength = std::max(int(task.subject_sequence->length()),
                                                                        sideBatch.maxSubjectLength);

                        //fill candidates
                        for(const Sequence_t* candidate_sequence : task.candidate_sequences){

                            std::memcpy(arrays.h_candidate_sequences_data
                                            + sideBatch.copiedCandidates * arrays.encoded_sequence_pitch,
                                        candidate_sequence->begin(),
                                        candidate_sequence->getNumBytes());

                            arrays.h_candidate_sequences_lengths[sideBatch.copiedCandidates] = candidate_sequence->length();
                            sideBatch.maxQueryLength = std::max(int(candidate_sequence->length()),
                                                                        sideBatch.maxQueryLength);

                            ++sideBatch.copiedCandidates;
                        }

                        //make prefix sum
                        arrays.h_candidates_per_subject_prefixsum[sideBatch.copiedTasks+1]
                                        = arrays.h_candidates_per_subject_prefixsum[sideBatch.copiedTasks]
                                            + int(task.candidate_read_ids.size());
                    }

                    ++sideBatch.copiedTasks;

                    POP_RANGE_2;

                    //if sidebatch is fully copied, transfer to gpu
                    if(sideBatch.copiedTasks == int(sideBatch.tasks.size())){
                        cudaMemcpyAsync(dataArrays.alignment_transfer_data_device,
                                        dataArrays.alignment_transfer_data_host,
                                        dataArrays.alignment_transfer_data_usable_size,
                                        H2D,
                                        streams[0]); CUERR;

                        sideBatch.copiedTasks = 0;
                        sideBatch.copiedCandidates = 0;

                        sideBatch.state = BatchState::DataOnGPU;
                    }

                    break;
                }
#if 1
                case BatchState::DataOnGPU:{
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
                                            goodAlignmentProperties.min_overlap,
                                            goodAlignmentProperties.maxErrorRate, //correctionOptions.estimatedErrorrate * 4.0,
                                            goodAlignmentProperties.min_overlap_ratio,
                                            accessor,
                                            make_reverse_complement_inplace,
                                            dataArrays.n_queries,
                                            sideBatch.maxSubjectLength,
                                            sideBatch.maxQueryLength,
                                            streams[0]);

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
                                                        goodAlignmentProperties.min_overlap_ratio,
                                                        goodAlignmentProperties.min_overlap,
                                                        goodAlignmentProperties.maxErrorRate, //correctionOptions.estimatedErrorrate * 4.0,
                                                        best_alignment_comp,
                                                        dataArrays.n_queries,
                                                        streams[0]);

                    //Determine indices where d_alignment_best_alignment_flags[i] != BestAlignment_t::None. this selects all good alignments
                    select_alignments_by_flag(dataArrays, streams[0]);

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
                                                        streams[0]); CUERR;

                    cub::DeviceScan::ExclusiveSum(dataArrays.d_temp_storage,
                                                    dataArrays.tmp_storage_allocation_size,
                                                    dataArrays.d_indices_per_subject,
                                                    dataArrays.d_indices_per_subject_prefixsum,
                                                    dataArrays.n_subjects,
                                                    streams[0]); CUERR;

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
                                            correctionOptions.estimatedErrorrate,
                                            correctionOptions.estimatedCoverage * correctionOptions.m_coverage,
                                            streams[0]);

                    //determine indices of remaining alignments
                    select_alignments_by_flag(dataArrays, streams[0]);

                    //update indices_per_subject
                    cub::DeviceHistogram::HistogramRange(dataArrays.d_temp_storage,
                                                        dataArrays.tmp_storage_allocation_size,
                                                        dataArrays.d_indices,
                                                        dataArrays.d_indices_per_subject,
                                                        dataArrays.n_subjects+1,
                                                        dataArrays.d_candidates_per_subject_prefixsum,
                                                        // *dataArrays.h_num_indices,
                                                        dataArrays.n_queries,
                                                        streams[0]); CUERR;

                    //Make indices_per_subject_prefixsum
                    cub::DeviceScan::ExclusiveSum(dataArrays.d_temp_storage,
                                                    dataArrays.tmp_storage_allocation_size,
                                                    dataArrays.d_indices_per_subject,
                                                    dataArrays.d_indices_per_subject_prefixsum,
                                                    dataArrays.n_subjects,
                                                    streams[0]); CUERR;

                    cudaMemcpyAsync(dataArrays.h_num_indices, dataArrays.d_num_indices, sizeof(int), D2H, streams[0]); CUERR;

                    cudaEventRecord(events[0], streams[0]); CUERR;

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
                                    streams[0]);

                    sideBatch.state = BatchState::RunningAlignmentWork;

                    break;
                }

                case BatchState::RunningAlignmentWork: {
                    //Step 10. Copy quality scores to gpu. This overlaps with alignment kernel

                    if(correctionOptions.useQualityScores){
                        PUSH_RANGE_2("get_quality_scores", 5);

                        if(sideBatch.copiedTasks < int(sideBatch.tasks.size())){
                            const auto& task = sideBatch.tasks[sideBatch.copiedTasks];
                            auto& arrays = dataArrays;

                            //fill subject
                            std::memcpy(arrays.h_subject_qualities + sideBatch.copiedTasks * arrays.quality_pitch,
                                        task.subject_quality->c_str(),
                                        task.subject_quality->length());

                            for(const std::string* qual : task.candidate_qualities){
                                std::memcpy(arrays.h_candidate_qualities + sideBatch.copiedCandidates * arrays.quality_pitch,
                                            qual->c_str(),
                                            qual->length());
                                ++sideBatch.copiedCandidates;
                            }
                        }

                        ++sideBatch.copiedTasks;

                        //if sidebatch is fully copied, transfer to gpu
                        if(sideBatch.copiedTasks == int(sideBatch.tasks.size())){
                            cudaMemcpyAsync(dataArrays.qualities_transfer_data_device,
                                            dataArrays.qualities_transfer_data_host,
                                            dataArrays.qualities_transfer_data_usable_size,
                                            H2D,
                                            streams[1]); CUERR;

                            cudaEventRecord(events[1], streams[1]); CUERR;

                            sideBatch.copiedTasks = 0;
                            sideBatch.copiedCandidates = 0;

                            sideBatch.state = BatchState::TransferingQualityScores;
                        }

                        POP_RANGE_2;
                    }else{
                        // indicate state progress in case no quality scores are used
                        sideBatch.state = BatchState::TransferingQualityScores;
                    }

                    break;
                }
#endif

#if 1

                case BatchState::TransferingQualityScores: {

                    cudaError_t querystatus = cudaEventQuery(events[0]); CUERR;
                    if(querystatus == cudaErrorNotReady && !canBlock){
                        //we cannot continue yet and cannot block either. do nothing
                        break;
                    }

                    assert(querystatus == cudaSuccess);

                    querystatus = cudaEventQuery(events[1]); CUERR;
                    if(querystatus == cudaErrorNotReady && !canBlock){
                        //we cannot continue yet and cannot block either. do nothing
                        break;
                    }

                    assert(querystatus == cudaSuccess);

                    PUSH_RANGE_2("wait_for_num_indices_2", 4);
                    cudaEventSynchronize(events[0]); CUERR; // need h_num_indices before continuing
                    POP_RANGE_2;

                    if(*dataArrays.h_num_indices == 0){
                        sideBatch.state == BatchState::Unprepared;
                        break;
                    }

                    //Step 8. Copy d_indices, d_indices_per_subject, d_indices_per_subject_prefixsum to host

                    cudaMemcpyAsync(dataArrays.indices_transfer_data_host,
                                    dataArrays.indices_transfer_data_device,
                                    dataArrays.indices_transfer_data_usable_size,
                                    D2H,
                                    streams[1]); CUERR;

                    cudaEventRecord(events[0], streams[1]); CUERR;

                    //Step 12. Fill multiple sequence alignment
                    cudaEventSynchronize(events[1]); CUERR; // quality transfer to gpu must be finished

                    const double avg_support_threshold = 1.0-1.0*correctionOptions.estimatedErrorrate;
                    const double min_support_threshold = 1.0-3.0*correctionOptions.estimatedErrorrate;
                    // coverage is always >= 1
                    const double min_coverage_threshold = std::max(1.0,
                                                                correctionOptions.m_coverage / 6.0 * correctionOptions.estimatedCoverage);
                    const int new_columns_to_correct = correctionOptions.new_columns_to_correct;
                    const float desiredAlignmentMaxErrorRate = goodAlignmentProperties.maxErrorRate;

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
                                    correctionOptions.useQualityScores,
                                    desiredAlignmentMaxErrorRate,
                                    dataArrays.encoded_sequence_pitch,
                                    dataArrays.quality_pitch,
                                    dataArrays.msa_pitch,
                                    dataArrays.msa_weights_pitch,
                                    nucleotide_accessor,
                                    make_unpacked_reverse_complement_inplace,
                                    streams[0]);

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
                                    3*dataArrays.maximum_sequence_length - 2*goodAlignmentProperties.min_overlap,
                                    streams[0]);

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
                                    correctionOptions.estimatedErrorrate,
                                    avg_support_threshold,
                                    min_support_threshold,
                                    min_coverage_threshold,
                                    correctionOptions.kmerlength,
                                    dataArrays.maximum_sequence_length,
                                    streams[0]);


                    if(correctionOptions.correctCandidates){


                        // find subject ids of subjects with high quality multiple sequence alignment

                        cub::DeviceSelect::Flagged(dataArrays.d_temp_storage,
                                        dataArrays.tmp_storage_allocation_size,
                                        cub::CountingInputIterator<int>(0),
                                        dataArrays.d_is_high_quality_subject,
                                        dataArrays.d_high_quality_subject_indices,
                                        dataArrays.d_num_high_quality_subject_indices,
                                        dataArrays.n_subjects,
                                        streams[0]); CUERR;

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
                                        streams[0]);

                    }

                    //copy correction results to host
                    cudaMemcpyAsync(dataArrays.correction_results_transfer_data_host,
                                    dataArrays.correction_results_transfer_data_device,
                                    dataArrays.correction_results_transfer_data_usable_size,
                                    D2H,
                                    streams[0]); CUERR;

                    cudaEventRecord(events[1], streams[0]); CUERR;

                    sideBatch.state = BatchState::RunningCorrectionWork;

                    break;
                }

                case BatchState::RunningCorrectionWork: {

                    cudaError_t querystatus = cudaEventQuery(events[0]); CUERR;
                    if(querystatus == cudaErrorNotReady && !canBlock){
                        //we cannot continue yet and cannot block either. do nothing
                        break;
                    }

                    assert(querystatus == cudaSuccess);

                    querystatus = cudaEventQuery(events[1]); CUERR;
                    if(querystatus == cudaErrorNotReady && !canBlock){
                        //we cannot continue yet and cannot block either. do nothing
                        break;
                    }

                    assert(querystatus == cudaSuccess);

                    PUSH_RANGE_2("wait_for_results", 6);
					cudaEventSynchronize(events[1]); CUERR; //wait for result transfer to host to finish
                    cudaEventSynchronize(events[0]); CUERR; //wait for index transfer to host to finish
                    POP_RANGE_2;

                    //unpack results
                    PUSH_RANGE_2("unpack_results", 7);
                    for(std::size_t subject_index = 0; subject_index < sideBatch.tasks.size(); ++subject_index){
                        auto& task = sideBatch.tasks[subject_index];
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

                        if(correctionOptions.correctCandidates){
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
                    POP_RANGE_2;

                    // results are present in batch
                    sideBatch.state = BatchState::Finished;

                    break;
                }

#endif
                default:
                    break;
            }

            return oldState;
        }

        void get_subjects(Batch& batch, const std::vector<ReadId_t>& readIds){

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
        }

        void lock(ReadId_t readId){
            ReadId_t index = readId % threadOpts.nLocksForProcessedFlags;
            threadOpts.locksForProcessedFlags[index].lock();
        };

        void unlock(ReadId_t readId){
            ReadId_t index = readId % threadOpts.nLocksForProcessedFlags;
            threadOpts.locksForProcessedFlags[index].unlock();
        };

    	void execute() {
    		isRunning = true;

			assert(threadOpts.canUseGpu);

    		//std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

    		std::ofstream outputstream(threadOpts.outputfile);

            std::ofstream featurestream(threadOpts.outputfile + "_features");
    		auto write_read = [&](const ReadId_t readId, const auto& sequence){
                //std::cout << readId << " " << sequence << std::endl;
    			auto& stream = outputstream;
    			stream << readId << '\n';
    			stream << sequence << '\n';
    		};

            auto accessor = [] __device__ (const char* data, int length, int index){
                return Sequence_t::get(data, length, index);
            };

            auto nucleotide_accessor = [] __device__ (const char* data, int length, int index){
                return Sequence_t::get_as_nucleotide(data, length, index);
            };

            auto make_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
                return Sequence_t::make_reverse_complement_inplace(sequence, sequencelength);
            };

            auto make_unpacked_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
                return care::SequenceString::make_reverse_complement_inplace(sequence, sequencelength);
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
                                            goodAlignmentProperties.min_overlap_ratio,
                                            goodAlignmentProperties.min_overlap,
                                            correctionOptions.estimatedErrorrate * 4.0);
            };

    		constexpr int nParallelBatches = 2;

			cudaSetDevice(threadOpts.deviceId); CUERR;

            std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();

    		std::vector<DataArrays<Sequence_t>> dataArrays;
            //std::array<Batch, nParallelBatches> batches;
            std::array<std::array<cudaStream_t, nStreamsPerBatch>, nParallelBatches> streams;
			std::array<std::array<cudaEvent_t, nStreamsPerBatch>, nParallelBatches> cudaevents;

			std::queue<Batch> batchQueue;

            for(int i = 0; i < nParallelBatches; i++){
                dataArrays.emplace_back(threadOpts.deviceId);

                for(int j = 0; j < nStreamsPerBatch; ++j){
                    cudaStreamCreate(&streams[i][j]); CUERR;
                    cudaEventCreateWithFlags(&cudaevents[i][j], cudaEventDisableTiming); CUERR;
                }

                dataArrays[i].set_problem_dimensions(readIds.size(),
                                                            max_candidates * readIds.size(),
                                                            fileProperties.maxSequenceLength,
                                                            goodAlignmentProperties.min_overlap,
                                                            goodAlignmentProperties.min_overlap_ratio,
                                                            correctionOptions.useQualityScores);

            }

            const double avg_support_threshold = 1.0-1.0*correctionOptions.estimatedErrorrate;
            const double min_support_threshold = 1.0-3.0*correctionOptions.estimatedErrorrate;
			// coverage is always >= 1
            const double min_coverage_threshold = std::max(1.0,
														correctionOptions.m_coverage / 6.0 * correctionOptions.estimatedCoverage);
			const int new_columns_to_correct = correctionOptions.new_columns_to_correct;
            const float desiredAlignmentMaxErrorRate = goodAlignmentProperties.maxErrorRate;
			//std::uint64_t itercounter = 0;




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

            auto nextBatchIndex = [](int currentBatchIndex, int nParallelBatches){
                if(nParallelBatches > 1)
                    return currentBatchIndex == 0 ? 1 : 0;
                else
                    assert(false);
                    return currentBatchIndex;
            };

            int batchIndex = 0; // the main stream we are working on in the current loop iteration

            //int num_finished_batches = 0;

    		//while(!stopAndAbort && !(num_finished_batches == nParallelBatches && readIds.empty())){
			while(!stopAndAbort && !(batchQueue.empty() && readIds.empty())){

				Batch mainBatch;

				if(!batchQueue.empty()){
						mainBatch = std::move(batchQueue.front());
						batchQueue.pop();
				}

                if(mainBatch.state == BatchState::Finished){
                    mainBatch.state = BatchState::Unprepared;
                }

                if(mainBatch.state == BatchState::Unprepared){
                    PUSH_RANGE_2("get_subjects_own", 0);

                    get_subjects(mainBatch, readIds);

                    nProcessedReads += readIds.size();
                    readIds = threadOpts.batchGen->getNextReadIds();

                    POP_RANGE_2;

                    mainBatch.state = BatchState::SubjectsPresent;
                }

                if(mainBatch.state == BatchState::SubjectsPresent){
                    PUSH_RANGE_2("get_candidates_own", 1);

                    get_candidates(mainBatch);

                    POP_RANGE_2;

                    std::remove_if(mainBatch.tasks.begin(),
                                    mainBatch.tasks.end(),
                                    [](const auto& t){return !t.active;});

                    int nTotalCandidates = std::accumulate(mainBatch.tasks.begin(),
                                                            mainBatch.tasks.end(),
                                                            int(0),
                                                            [](const auto& l, const auto& r){
                                                                return l + int(r.candidate_read_ids.size());
                                                            });

                    if(nTotalCandidates == 0){
                        mainBatch.state = BatchState::Unprepared;
                        continue;
                    }

                    //allocate data arrays

                    PUSH_RANGE_2("set_problem_dimensions_firstiter", 7);
                    dataArrays[batchIndex].set_problem_dimensions(int(mainBatch.tasks.size()),
                                                                nTotalCandidates,
                                                                fileProperties.maxSequenceLength,
                                                                goodAlignmentProperties.min_overlap,
                                                                goodAlignmentProperties.min_overlap_ratio,
                                                                correctionOptions.useQualityScores); CUERR;

                    std::size_t temp_storage_bytes = 0;
                    std::size_t max_temp_storage_bytes = 0;
                    cub::DeviceHistogram::HistogramRange((void*)nullptr, temp_storage_bytes,
                                                        (int*)nullptr, (int*)nullptr,
                                                        dataArrays[batchIndex].n_subjects+1,
                                                        (int*)nullptr,
                                                        dataArrays[batchIndex].n_queries,
                                                        streams[batchIndex][0]); CUERR;

                    max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

                    cub::DeviceSelect::Flagged((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                                                (bool*)nullptr, (int*)nullptr, (int*)nullptr,
                                                nTotalCandidates,
                                                streams[batchIndex][0]); CUERR;

                    max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

                    cub::DeviceScan::ExclusiveSum((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                                                    (int*)nullptr,
                                                    dataArrays[batchIndex].n_subjects,
                                                    streams[batchIndex][0]); CUERR;

                    max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);
                    temp_storage_bytes = max_temp_storage_bytes;

                    dataArrays[batchIndex].set_tmp_storage_size(max_temp_storage_bytes);

                    POP_RANGE_2;

                    PUSH_RANGE_2("zero_gpu_firstiter", 5);
    				dataArrays[batchIndex].zero_gpu(streams[batchIndex][0]);
                    POP_RANGE_2;

                    mainBatch.state = BatchState::CandidatesPresent;
                }

                //fill data arrays
                if(mainBatch.state == BatchState::CandidatesPresent){

                    dataArrays[batchIndex].h_candidates_per_subject_prefixsum[0] = 0;


                    PUSH_RANGE_2("copy_to_buffer", 2);

					for(; mainBatch.copiedTasks < int(mainBatch.tasks.size()); ++mainBatch.copiedTasks){
						const auto& task = mainBatch.tasks[mainBatch.copiedTasks];
                        auto& arrays = dataArrays[batchIndex];

                        //fill subject
                        std::memcpy(arrays.h_subject_sequences_data + mainBatch.copiedTasks * arrays.encoded_sequence_pitch,
                                    task.subject_sequence->begin(),
                                    task.subject_sequence->getNumBytes());
                        arrays.h_subject_sequences_lengths[mainBatch.copiedTasks] = task.subject_sequence->length();
                        mainBatch.maxSubjectLength = std::max(int(task.subject_sequence->length()),
																		 mainBatch.maxSubjectLength);

                        //fill candidates
                        for(const Sequence_t* candidate_sequence : task.candidate_sequences){

                            std::memcpy(arrays.h_candidate_sequences_data
											+ mainBatch.copiedCandidates * arrays.encoded_sequence_pitch,
                                        candidate_sequence->begin(),
                                        candidate_sequence->getNumBytes());

                            arrays.h_candidate_sequences_lengths[mainBatch.copiedCandidates] = candidate_sequence->length();
                            mainBatch.maxQueryLength = std::max(int(candidate_sequence->length()),
																		   mainBatch.maxQueryLength);

                            ++mainBatch.copiedCandidates;
                        }

                        //make prefix sum
                        arrays.h_candidates_per_subject_prefixsum[mainBatch.copiedTasks+1]
                            = arrays.h_candidates_per_subject_prefixsum[mainBatch.copiedTasks] + int(task.candidate_read_ids.size());
					}

                    POP_RANGE_2;

                    //data required for alignment is now ready for transfer

                    cudaMemcpyAsync(dataArrays[batchIndex].alignment_transfer_data_device,
                                    dataArrays[batchIndex].alignment_transfer_data_host,
                                    dataArrays[batchIndex].alignment_transfer_data_usable_size,
                                    H2D,
                                    streams[batchIndex][0]); CUERR;

                    mainBatch.copiedTasks = 0;
                    mainBatch.copiedCandidates = 0;

                    mainBatch.state = BatchState::DataOnGPU;
                }

                if(mainBatch.state == BatchState::DataOnGPU){
                    //

                    call_shd_with_revcompl_kernel_async(
                                            dataArrays[batchIndex].d_alignment_scores,
                                            dataArrays[batchIndex].d_alignment_overlaps,
                                            dataArrays[batchIndex].d_alignment_shifts,
                                            dataArrays[batchIndex].d_alignment_nOps,
                                            dataArrays[batchIndex].d_alignment_isValid,
                                            dataArrays[batchIndex].d_subject_sequences_data,
                                            dataArrays[batchIndex].d_candidate_sequences_data,
                                            dataArrays[batchIndex].d_subject_sequences_lengths,
                                            dataArrays[batchIndex].d_candidate_sequences_lengths,
                                            dataArrays[batchIndex].d_candidates_per_subject_prefixsum,
                                            dataArrays[batchIndex].n_subjects,
                                            Sequence_t::getNumBytes(dataArrays[batchIndex].maximum_sequence_length),
                                            dataArrays[batchIndex].encoded_sequence_pitch,
                                            goodAlignmentProperties.min_overlap,
                                            goodAlignmentProperties.maxErrorRate, //correctionOptions.estimatedErrorrate * 4.0,
                                            goodAlignmentProperties.min_overlap_ratio,
                                            accessor,
                                            make_reverse_complement_inplace,
                                            dataArrays[batchIndex].n_queries,
                                            mainBatch.maxSubjectLength,
                                            mainBatch.maxQueryLength,
                                            streams[batchIndex][0]);

                    //Step 5. Compare each forward alignment with the correspoding reverse complement alignment and keep the best, if any.
                    //    If reverse complement is the best, it is copied into the first half, replacing the forward alignment

                    call_cuda_find_best_alignment_kernel_async(
                                                        dataArrays[batchIndex].d_alignment_best_alignment_flags,
                                                        dataArrays[batchIndex].d_alignment_scores,
                                                        dataArrays[batchIndex].d_alignment_overlaps,
                                                        dataArrays[batchIndex].d_alignment_shifts,
                                                        dataArrays[batchIndex].d_alignment_nOps,
                                                        dataArrays[batchIndex].d_alignment_isValid,
                                                        dataArrays[batchIndex].d_subject_sequences_lengths,
                                                        dataArrays[batchIndex].d_candidate_sequences_lengths,
                                                        dataArrays[batchIndex].d_candidates_per_subject_prefixsum,
                                                        dataArrays[batchIndex].n_subjects,
                                                        goodAlignmentProperties.min_overlap_ratio,
                                                        goodAlignmentProperties.min_overlap,
                                                        goodAlignmentProperties.maxErrorRate, //correctionOptions.estimatedErrorrate * 4.0,
                                                        best_alignment_comp,
                                                        dataArrays[batchIndex].n_queries,
                                                        streams[batchIndex][0]);

                    //Determine indices where d_alignment_best_alignment_flags[i] != BestAlignment_t::None. this selects all good alignments
                    select_alignments_by_flag(dataArrays[batchIndex], streams[batchIndex][0]);

                    //Get number of indices per subject by creating histrogram.
                    //The elements d_indices[d_num_indices] to d_indices[n_queries - 1] will be -1.
                    //Thus, they will not be accounted for by the histrogram, since the histrogram bins (d_candidates_per_subject_prefixsum) are >= 0.
                    cub::DeviceHistogram::HistogramRange(dataArrays[batchIndex].d_temp_storage,
                                                        dataArrays[batchIndex].tmp_storage_allocation_size,
                                                        dataArrays[batchIndex].d_indices,
                                                        dataArrays[batchIndex].d_indices_per_subject,
                                                        dataArrays[batchIndex].n_subjects+1,
                                                        dataArrays[batchIndex].d_candidates_per_subject_prefixsum,
                                                        dataArrays[batchIndex].n_queries,
                                                        streams[batchIndex][0]); CUERR;

                    cub::DeviceScan::ExclusiveSum(dataArrays[batchIndex].d_temp_storage,
                                                    dataArrays[batchIndex].tmp_storage_allocation_size,
                                                    dataArrays[batchIndex].d_indices_per_subject,
                                                    dataArrays[batchIndex].d_indices_per_subject_prefixsum,
                                                    dataArrays[batchIndex].n_subjects,
                                                    streams[batchIndex][0]); CUERR;

                    //choose the most appropriate subset of alignments from the good alignments.
                    //This sets d_alignment_best_alignment_flags[i] = BestAlignment_t::None for all non-appropriate alignments
                    call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                                            dataArrays[batchIndex].d_alignment_best_alignment_flags,
                                            dataArrays[batchIndex].d_alignment_overlaps,
                                            dataArrays[batchIndex].d_alignment_nOps,
                                            dataArrays[batchIndex].d_indices,
                                            dataArrays[batchIndex].d_indices_per_subject,
                                            dataArrays[batchIndex].d_indices_per_subject_prefixsum,
                                            dataArrays[batchIndex].n_subjects,
                                            dataArrays[batchIndex].n_queries,
                                            dataArrays[batchIndex].d_num_indices,
                                            correctionOptions.estimatedErrorrate,
                                            correctionOptions.estimatedCoverage * correctionOptions.m_coverage,
                                            streams[batchIndex][0]);

                    //determine indices of remaining alignments
                    select_alignments_by_flag(dataArrays[batchIndex], streams[batchIndex][0]);

                    //update indices_per_subject
                    cub::DeviceHistogram::HistogramRange(dataArrays[batchIndex].d_temp_storage,
                                                        dataArrays[batchIndex].tmp_storage_allocation_size,
                                                        dataArrays[batchIndex].d_indices,
                                                        dataArrays[batchIndex].d_indices_per_subject,
                                                        dataArrays[batchIndex].n_subjects+1,
                                                        dataArrays[batchIndex].d_candidates_per_subject_prefixsum,
                                                        // *dataArrays[batchIndex].h_num_indices,
                                                        dataArrays[batchIndex].n_queries,
                                                        streams[batchIndex][0]); CUERR;

                    //Make indices_per_subject_prefixsum
                    cub::DeviceScan::ExclusiveSum(dataArrays[batchIndex].d_temp_storage,
                                                    dataArrays[batchIndex].tmp_storage_allocation_size,
                                                    dataArrays[batchIndex].d_indices_per_subject,
                                                    dataArrays[batchIndex].d_indices_per_subject_prefixsum,
                                                    dataArrays[batchIndex].n_subjects,
                                                    streams[batchIndex][0]); CUERR;

                    cudaMemcpyAsync(dataArrays[batchIndex].h_num_indices, dataArrays[batchIndex].d_num_indices, sizeof(int), D2H, streams[batchIndex][0]); CUERR;

					cudaEventRecord(cudaevents[batchIndex][0], streams[batchIndex][0]); CUERR;

                    //Determine multiple sequence alignment properties
                    call_msa_init_kernel_async(
                                    dataArrays[batchIndex].d_msa_column_properties,
                                    dataArrays[batchIndex].d_alignment_shifts,
                                    dataArrays[batchIndex].d_alignment_best_alignment_flags,
                                    dataArrays[batchIndex].d_subject_sequences_lengths,
                                    dataArrays[batchIndex].d_candidate_sequences_lengths,
                                    dataArrays[batchIndex].d_indices,
                                    dataArrays[batchIndex].d_indices_per_subject,
                                    dataArrays[batchIndex].d_indices_per_subject_prefixsum,
                                    dataArrays[batchIndex].n_subjects,
                                    dataArrays[batchIndex].n_queries,
                                    streams[batchIndex][0]);

                    mainBatch.state = BatchState::RunningAlignmentWork;
                }

                if(mainBatch.state == BatchState::RunningAlignmentWork){
                    //Step 10. Copy quality scores to gpu. This overlaps with alignment kernel

                    if(correctionOptions.useQualityScores){
                        PUSH_RANGE_2("get_quality_scores", 5);

                        for(; mainBatch.copiedTasks < mainBatch.tasks.size(); ++mainBatch.copiedTasks){
                            const auto& task = mainBatch.tasks[mainBatch.copiedTasks];
                            auto& arrays = dataArrays[batchIndex];

                            std::memcpy(arrays.h_subject_qualities + mainBatch.copiedTasks * arrays.quality_pitch,
                                        task.subject_quality->c_str(),
                                        task.subject_quality->length());

                            for(const std::string* qual : task.candidate_qualities){
                                std::memcpy(arrays.h_candidate_qualities + mainBatch.copiedCandidates * arrays.quality_pitch,
                                            qual->c_str(),
                                            qual->length());
                                ++mainBatch.copiedCandidates;
                            }
                        }

                        cudaMemcpyAsync(dataArrays[batchIndex].qualities_transfer_data_device,
                                        dataArrays[batchIndex].qualities_transfer_data_host,
                                        dataArrays[batchIndex].qualities_transfer_data_usable_size,
                                        H2D,
                                        streams[batchIndex][1]); CUERR;

                        cudaEventRecord(cudaevents[batchIndex][1], streams[batchIndex][1]); CUERR;

                        mainBatch.copiedTasks = 0;
                        mainBatch.copiedCandidates = 0;

                        POP_RANGE_2;

                        mainBatch.state = BatchState::TransferingQualityScores;
                    }else{
                        mainBatch.state = BatchState::TransferingQualityScores;
                    }

                }


				/*
                    Prepare next batch while waiting for the mainBatch gpu work to finish
                */
                if(nParallelBatches > 1){
                    do{
    					const int nextBatchId = nextBatchIndex(batchIndex, nParallelBatches);

                        Batch sideBatch;

    					if(!batchQueue.empty()){
    						sideBatch = std::move(batchQueue.front());
    						batchQueue.pop();

                            assert(sideBatch.state != BatchState::Unprepared);
    					}else{
                            if(readIds.empty()){
                                break; //no next batch to prepare
                            }
                        }

    					cudaError_t eventquerystatus = cudaSuccess;

    					//while mainBatch is not ready for next step...
    					//this condition is checked after each step of sideBatch
    					while((eventquerystatus = cudaEventQuery(cudaevents[batchIndex][0])) == cudaErrorNotReady){

                            BatchState oldState = advance_one_step(sideBatch, readIds,
                                                                dataArrays[nextBatchId],
                                                                streams[nextBatchId],
                                                                cudaevents[nextBatchId],
                                                                false); //must not block

                            if(oldState == BatchState::Unprepared){
                                nProcessedReads += readIds.size();
                                readIds = threadOpts.batchGen->getNextReadIds();
                            }

    					}

    					assert(eventquerystatus == cudaSuccess);

    					if(sideBatch.state != BatchState::Unprepared){
                            batchQueue.push(std::move(sideBatch));
                        }
                    }while(0);
                }



                if(mainBatch.state == BatchState::TransferingQualityScores){

                    PUSH_RANGE_2("wait_for_num_indices_2", 4);
					cudaEventSynchronize(cudaevents[batchIndex][0]); CUERR; // need h_num_indices before continuing
                    POP_RANGE_2;

                    if(*dataArrays[batchIndex].h_num_indices == 0){
                        mainBatch.state == BatchState::Unprepared;
                        continue;
                    }

                    //Step 8. Copy d_indices, d_indices_per_subject, d_indices_per_subject_prefixsum to host

                    cudaMemcpyAsync(dataArrays[batchIndex].indices_transfer_data_host,
                                    dataArrays[batchIndex].indices_transfer_data_device,
                                    dataArrays[batchIndex].indices_transfer_data_usable_size,
                                    D2H,
                                    streams[batchIndex][1]); CUERR;

                    cudaEventRecord(cudaevents[batchIndex][0], streams[batchIndex][1]); CUERR;

                    //Step 12. Fill multiple sequence alignment
                    cudaEventSynchronize(cudaevents[batchIndex][1]); CUERR; // quality transfer to gpu must be finished

                    call_msa_add_sequences_kernel_async(
                                    dataArrays[batchIndex].d_multiple_sequence_alignments,
                                    dataArrays[batchIndex].d_multiple_sequence_alignment_weights,
                                    dataArrays[batchIndex].d_alignment_shifts,
                                    dataArrays[batchIndex].d_alignment_best_alignment_flags,
                                    dataArrays[batchIndex].d_subject_sequences_data,
                                    dataArrays[batchIndex].d_candidate_sequences_data,
                                    dataArrays[batchIndex].d_subject_sequences_lengths,
                                    dataArrays[batchIndex].d_candidate_sequences_lengths,
                                    dataArrays[batchIndex].d_subject_qualities,
                                    dataArrays[batchIndex].d_candidate_qualities,
                                    dataArrays[batchIndex].d_alignment_overlaps,
                                    dataArrays[batchIndex].d_alignment_nOps,
                                    dataArrays[batchIndex].d_msa_column_properties,
                                    dataArrays[batchIndex].d_candidates_per_subject_prefixsum,
                                    dataArrays[batchIndex].d_indices,
                                    dataArrays[batchIndex].d_indices_per_subject,
                                    dataArrays[batchIndex].d_indices_per_subject_prefixsum,
                                    dataArrays[batchIndex].n_subjects,
                                    dataArrays[batchIndex].n_queries,
                                    *dataArrays[batchIndex].h_num_indices,
                                    correctionOptions.useQualityScores,
                                    desiredAlignmentMaxErrorRate,
                                    dataArrays[batchIndex].encoded_sequence_pitch,
                                    dataArrays[batchIndex].quality_pitch,
                                    dataArrays[batchIndex].msa_pitch,
                                    dataArrays[batchIndex].msa_weights_pitch,
                                    nucleotide_accessor,
                                    make_unpacked_reverse_complement_inplace,
                                    streams[batchIndex][0]);

                    //Step 13. Determine consensus in multiple sequence alignment

                    call_msa_find_consensus_kernel_async(
    								dataArrays[batchIndex].d_consensus,
    								dataArrays[batchIndex].d_support,
    								dataArrays[batchIndex].d_coverage,
    								dataArrays[batchIndex].d_origWeights,
    								dataArrays[batchIndex].d_origCoverages,
    								dataArrays[batchIndex].d_multiple_sequence_alignments,
    								dataArrays[batchIndex].d_multiple_sequence_alignment_weights,
    								dataArrays[batchIndex].d_msa_column_properties,
    								dataArrays[batchIndex].d_candidates_per_subject_prefixsum,
    								dataArrays[batchIndex].d_indices_per_subject,
    								dataArrays[batchIndex].d_indices_per_subject_prefixsum,
    								dataArrays[batchIndex].n_subjects,
    								dataArrays[batchIndex].n_queries,
                                    *dataArrays[batchIndex].h_num_indices,
    								dataArrays[batchIndex].msa_pitch,
    								dataArrays[batchIndex].msa_weights_pitch,
    								3*dataArrays[batchIndex].maximum_sequence_length - 2*goodAlignmentProperties.min_overlap,
    								streams[batchIndex][0]);

                    // Step 14. Correction

                    // correct subjects
                    call_msa_correct_subject_kernel_async(
    								dataArrays[batchIndex].d_consensus,
    								dataArrays[batchIndex].d_support,
    								dataArrays[batchIndex].d_coverage,
    								dataArrays[batchIndex].d_origCoverages,
    								dataArrays[batchIndex].d_multiple_sequence_alignments,
    								dataArrays[batchIndex].d_msa_column_properties,
    								dataArrays[batchIndex].d_indices_per_subject_prefixsum,
    								dataArrays[batchIndex].d_is_high_quality_subject,
    								dataArrays[batchIndex].d_corrected_subjects,
    								dataArrays[batchIndex].d_subject_is_corrected,
    								dataArrays[batchIndex].n_subjects,
    								dataArrays[batchIndex].n_queries,
                                    *dataArrays[batchIndex].h_num_indices,
    								dataArrays[batchIndex].sequence_pitch,
    								dataArrays[batchIndex].msa_pitch,
    								dataArrays[batchIndex].msa_weights_pitch,
    								correctionOptions.estimatedErrorrate,
    								avg_support_threshold,
    								min_support_threshold,
    								min_coverage_threshold,
    								correctionOptions.kmerlength,
    								dataArrays[batchIndex].maximum_sequence_length,
    								streams[batchIndex][0]);


                    if(correctionOptions.correctCandidates){


                        // find subject ids of subjects with high quality multiple sequence alignment

                        cub::DeviceSelect::Flagged(dataArrays[batchIndex].d_temp_storage,
        								dataArrays[batchIndex].tmp_storage_allocation_size,
        								cub::CountingInputIterator<int>(0),
        								dataArrays[batchIndex].d_is_high_quality_subject,
        								dataArrays[batchIndex].d_high_quality_subject_indices,
        								dataArrays[batchIndex].d_num_high_quality_subject_indices,
        								dataArrays[batchIndex].n_subjects,
        								streams[batchIndex][0]); CUERR;

        				// correct candidates
        				call_msa_correct_candidates_kernel_async(
        								dataArrays[batchIndex].d_consensus,
        								dataArrays[batchIndex].d_support,
        								dataArrays[batchIndex].d_coverage,
        								dataArrays[batchIndex].d_origCoverages,
        								dataArrays[batchIndex].d_multiple_sequence_alignments,
        								dataArrays[batchIndex].d_msa_column_properties,
        								dataArrays[batchIndex].d_indices,
        								dataArrays[batchIndex].d_indices_per_subject,
        								dataArrays[batchIndex].d_indices_per_subject_prefixsum,
        								dataArrays[batchIndex].d_high_quality_subject_indices,
        								dataArrays[batchIndex].d_num_high_quality_subject_indices,
        								dataArrays[batchIndex].d_alignment_shifts,
        								dataArrays[batchIndex].d_alignment_best_alignment_flags,
        								dataArrays[batchIndex].d_candidate_sequences_lengths,
        								dataArrays[batchIndex].d_num_corrected_candidates,
        								dataArrays[batchIndex].d_corrected_candidates,
        								dataArrays[batchIndex].d_indices_of_corrected_candidates,
        								dataArrays[batchIndex].n_subjects,
        								dataArrays[batchIndex].n_queries,
        								//dataArrays[batchIndex].n_indices,
                                        *dataArrays[batchIndex].h_num_indices,
        								dataArrays[batchIndex].sequence_pitch,
        								dataArrays[batchIndex].msa_pitch,
        								dataArrays[batchIndex].msa_weights_pitch,
        								min_support_threshold,
        								min_coverage_threshold,
        								new_columns_to_correct,
        								make_unpacked_reverse_complement_inplace,
        								dataArrays[batchIndex].maximum_sequence_length,
        								streams[batchIndex][0]);

                    }

    				//copy correction results to host
    				cudaMemcpyAsync(dataArrays[batchIndex].correction_results_transfer_data_host,
    								dataArrays[batchIndex].correction_results_transfer_data_device,
    								dataArrays[batchIndex].correction_results_transfer_data_usable_size,
    								D2H,
    								streams[batchIndex][0]); CUERR;

					cudaEventRecord(cudaevents[batchIndex][1], streams[batchIndex][0]); CUERR;

                    mainBatch.state = BatchState::RunningCorrectionWork;
                }




                /*
                    Prepare subjects and candidates of next batch while waiting for the previous gpu work to finish
                */
                if(nParallelBatches > 1){
                    do{
                        const int nextBatchId = nextBatchIndex(batchIndex, nParallelBatches);

                        Batch sideBatch;

                        if(!batchQueue.empty()){
                            sideBatch = std::move(batchQueue.front());
                            batchQueue.pop();

                            assert(sideBatch.state != BatchState::Unprepared);
                        }else{
                            if(readIds.empty()){
                                break; //no next batch to prepare
                            }
                        }

                        cudaError_t eventquerystatus = cudaSuccess;

                        //while mainBatch is not ready for next step...
                        //this condition is checked after each step of sideBatch
                        while((eventquerystatus = cudaEventQuery(cudaevents[batchIndex][1])) == cudaErrorNotReady){

                            BatchState oldState = advance_one_step(sideBatch, readIds,
                                                                dataArrays[nextBatchId],
                                                                streams[nextBatchId],
                                                                cudaevents[nextBatchId],
                                                                false); //must not block

                            if(oldState == BatchState::Unprepared){
                                nProcessedReads += readIds.size();
                                readIds = threadOpts.batchGen->getNextReadIds();
                            }

                        }

                        assert(eventquerystatus == cudaSuccess);

                        if(sideBatch.state != BatchState::Unprepared){
                            batchQueue.push(std::move(sideBatch));
                        }
                    }while(0);
                }


                if(mainBatch.state == BatchState::RunningCorrectionWork){
                    PUSH_RANGE_2("wait_for_results", 6);
					cudaEventSynchronize(cudaevents[batchIndex][1]); CUERR; //wait for result transfer to host to finish
                    cudaEventSynchronize(cudaevents[batchIndex][0]); CUERR; //wait for index transfer to host to finish
                    POP_RANGE_2;

                    //unpack results
                    PUSH_RANGE_2("unpack_results", 7);
                    for(std::size_t subject_index = 0; subject_index < mainBatch.tasks.size(); ++subject_index){
                        auto& task = mainBatch.tasks[subject_index];
                        auto& arrays = dataArrays[batchIndex];

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

                        if(correctionOptions.correctCandidates){
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
                    POP_RANGE_2;

                    mainBatch.state = BatchState::Finished;
                }

                if(mainBatch.state == BatchState::Finished){

                    //write result to file

                    for(std::size_t subject_index = 0; subject_index < mainBatch.tasks.size(); ++subject_index){
                        const auto& task = mainBatch.tasks[subject_index];

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

                    mainBatch.state = BatchState::Unprepared;
                }

				if(nParallelBatches > 1){
					batchIndex = nextBatchIndex(batchIndex, nParallelBatches);
				}





#if 0
                /*
                    Prepare subjects and candidates of next batch while waiting for the previous gpu work to finish
                */
                {
                    correctionTasks[nextBatchIndex(batchIndex, nParallelBatches)].clear();

                    // Step 1 and 2: Get subject sequence and its candidate read ids from hash map
                    // Get candidate sequences from read storage

                    PUSH_RANGE_2("get_subjects_next", 0);

                    get_subjects(correctionTasks[nextBatchIndex(batchIndex, nParallelBatches)], readIds);

                    nProcessedReads += readIds.size();
                    readIds = threadOpts.batchGen->getNextReadIds();

                    POP_RANGE_2;

                    PUSH_RANGE_2("get_candidates_next", 1);

                    get_candidates(correctionTasks[nextBatchIndex(batchIndex, nParallelBatches)]);

                    POP_RANGE_2;
                }
#endif



#if 0
                cudaMemcpyAsync(dataArrays[batchIndex].indices_transfer_data_host,
                                dataArrays[batchIndex].indices_transfer_data_device,
                                dataArrays[batchIndex].indices_transfer_data_usable_size,
                                D2H,
                                streams[batchIndex][0]); CUERR;
                cudaStreamSynchronize(streams[batchIndex][0]); CUERR; //remove

                std::cout << "indices" << std::endl;
                for(int i = 0; i < dataArrays[batchIndex].n_queries; i++){
                    std::cout << dataArrays[batchIndex].h_indices[i] << " ";
                }
                std::cout << std::endl;

                std::cout << "indices_per_subject" << std::endl;
                for(int i = 0; i < dataArrays[batchIndex].n_subjects; i++){
                    std::cout << dataArrays[batchIndex].h_indices_per_subject[i] << " ";
                }
                std::cout << std::endl;

                std::cout << "candidates_per_subject_prefixsum" << std::endl;
                for(int i = 0; i < dataArrays[batchIndex].n_subjects + 1; i++){
                    std::cout << dataArrays[batchIndex].h_candidates_per_subject_prefixsum[i] << " ";
                }
                std::cout << std::endl;

#endif





                /*
                    Continue preparation of next batch while waiting for correction results
                */
#if 0
                {
                    std::remove_if(correctionTasks[nextBatchIndex(batchIndex, nParallelBatches)].begin(),
                                    correctionTasks[nextBatchIndex(batchIndex, nParallelBatches)].end(),
                                    [](const auto& t){return !t.active;});

                    int nTotalCandidates = std::accumulate(correctionTasks[nextBatchIndex(batchIndex, nParallelBatches)].begin(),
                                                            correctionTasks[nextBatchIndex(batchIndex, nParallelBatches)].end(),
                                                            int(0),
                                                            [](const auto& l, const auto& r){
                                                                return l + int(r.candidate_read_ids.size());
                                                            });

                    if(nTotalCandidates == 0){
                        firstIter = true; // no batch prepared for next iter
                        continue;
                    }


                    //Step 3. Copy subject sequences, subject sequence lengths, candidate sequences,
                    //candidate sequence lengths, candidates_per_subject_prefixsum to GPU

                    //allocate data arrays

                    PUSH_RANGE_2("set_problem_dimensions_firstiter", 7);
                    dataArrays[nextBatchIndex(batchIndex, nParallelBatches)].set_problem_dimensions(int(correctionTasks[nextBatchIndex(batchIndex, nParallelBatches)].size()),
                                                                nTotalCandidates,
                                                                fileProperties.maxSequenceLength,
                                                                goodAlignmentProperties.min_overlap,
                                                                goodAlignmentProperties.min_overlap_ratio,
                                                                correctionOptions.useQualityScores); CUERR;

                    std::size_t temp_storage_bytes = 0;
                    std::size_t max_temp_storage_bytes = 0;
                    cub::DeviceHistogram::HistogramRange((void*)nullptr, temp_storage_bytes,
                                                        (int*)nullptr, (int*)nullptr,
                                                        dataArrays[nextBatchIndex(batchIndex, nParallelBatches)].n_subjects+1,
                                                        (int*)nullptr,
                                                        dataArrays[nextBatchIndex(batchIndex, nParallelBatches)].n_queries,
                                                        streams[nextBatchIndex(batchIndex, nParallelBatches)]); CUERR;

                    max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

                    cub::DeviceSelect::Flagged((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                                                (bool*)nullptr, (int*)nullptr, (int*)nullptr,
                                                nTotalCandidates,
                                                streams[nextBatchIndex(batchIndex, nParallelBatches)]); CUERR;

                    max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

                    cub::DeviceScan::ExclusiveSum((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                                                    (int*)nullptr,
                                                    dataArrays[nextBatchIndex(batchIndex, nParallelBatches)].n_subjects,
                                                    streams[nextBatchIndex(batchIndex, nParallelBatches)]); CUERR;

                    max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);
                    temp_storage_bytes = max_temp_storage_bytes;

                    dataArrays[nextBatchIndex(batchIndex, nParallelBatches)].set_tmp_storage_size(max_temp_storage_bytes);

                    POP_RANGE_2;

                    //dataArrays[batchIndex].set_n_indices(nTotalCandidates); CUERR;
                    //PUSH_RANGE_2("zero_cpu", 4);
                    //dataArrays[batchIndex].zero_cpu();
                    //POP_RANGE_2;
                    PUSH_RANGE_2("zero_gpu_firstiter", 5);
                    dataArrays[nextBatchIndex(batchIndex, nParallelBatches)].zero_gpu(streams[nextBatchIndex(batchIndex, nParallelBatches)]);
                    POP_RANGE_2;
                }
#endif




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
