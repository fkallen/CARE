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

    	void execute() {
    		isRunning = true;

			assert(threadOpts.canUseGpu);

    		std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

    		std::ofstream outputstream(threadOpts.outputfile);

            std::ofstream featurestream(threadOpts.outputfile + "_features");
    		auto write_read = [&](const ReadId_t readId, const auto& sequence){
                //std::cout << readId << " " << sequence << std::endl;
    			auto& stream = outputstream;
    			stream << readId << '\n';
    			stream << sequence << '\n';
    		};

    		auto lock = [&](ReadId_t readId){
    			ReadId_t index = readId % threadOpts.nLocksForProcessedFlags;
    			threadOpts.locksForProcessedFlags[index].lock();
    		};

    		auto unlock = [&](ReadId_t readId){
    			ReadId_t index = readId % threadOpts.nLocksForProcessedFlags;
    			threadOpts.locksForProcessedFlags[index].unlock();
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

    		constexpr int nStreams = 1;
			cudaSetDevice(threadOpts.deviceId);
            std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();

            using Batch = std::vector<CorrectionTask_t>;

    		std::vector<DataArrays<Sequence_t>> dataArrays;
            std::array<Batch, nStreams> correctionTasks;
            std::array<cudaStream_t, nStreams> streams;


            //determine maximum temp storage size for cub


            for(int i = 0; i < nStreams; i++){
                dataArrays.emplace_back(threadOpts.deviceId);
                cudaStreamCreate(&streams[i]); CUERR;
                dataArrays[i].set_problem_dimensions(readIds.size(),
                                                            max_candidates * readIds.size(),
                                                            fileProperties.maxSequenceLength,
                                                            goodAlignmentProperties.min_overlap,
                                                            goodAlignmentProperties.min_overlap_ratio,
                                                            correctionOptions.useQualityScores); CUERR;
            }





            const auto& readStorage = threadOpts.readStorage;
            const auto& minhasher = threadOpts.minhasher;

            const double avg_support_threshold = 1.0-1.0*correctionOptions.estimatedErrorrate;
            const double min_support_threshold = 1.0-3.0*correctionOptions.estimatedErrorrate;
			// coverage is always >= 1
            const double min_coverage_threshold = std::max(1.0,
														correctionOptions.m_coverage / 6.0 * correctionOptions.estimatedCoverage);
			const int new_columns_to_correct = correctionOptions.new_columns_to_correct;
            const float desiredAlignmentMaxErrorRate = goodAlignmentProperties.maxErrorRate;
			//std::uint64_t itercounter = 0;

            auto get_subjects = [&](Batch& batch, const auto& readIds){
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

                        batch.emplace_back(id, sequenceptr, qualityptr);
                    }
                }
            };

            auto get_candidates = [&](Batch& batch){
                for(auto& task : batch){
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
            };

            auto nextStreamIndex = [](int currentStreamIndex){
                    return currentStreamIndex == 0 ? 1 : 0;
            };

            int streamIndex = 0;

    		while(!stopAndAbort && !readIds.empty()){

				//std::cout << "iter " << itercounter << std::endl;
				//++itercounter;

                correctionTasks[streamIndex].clear();

                // Step 1 and 2: Get subject sequence and its candidate read ids from hash map
                // Get candidate sequences from read storage

                PUSH_RANGE_2("get_subjects", 0);

                get_subjects(correctionTasks[streamIndex], readIds);

                nProcessedReads += readIds.size();
                readIds = threadOpts.batchGen->getNextReadIds();

                POP_RANGE_2;

                PUSH_RANGE_2("get_candidates", 1);

                get_candidates(correctionTasks[streamIndex]);

                POP_RANGE_2;

                std::remove_if(correctionTasks[streamIndex].begin(),
                                correctionTasks[streamIndex].end(),
                                [](const auto& t){return !t.active;});

                int nTotalCandidates = std::accumulate(correctionTasks[streamIndex].begin(),
                                                        correctionTasks[streamIndex].end(),
                                                        int(0),
                                                        [](const auto& l, const auto& r){
                                                            return l + int(r.candidate_read_ids.size());
                                                        });

                if(nTotalCandidates == 0){
                    continue;
                }


                //Step 3. Copy subject sequences, subject sequence lengths, candidate sequences,
                //candidate sequence lengths, candidates_per_subject_prefixsum to GPU

                //allocate data arrays

                PUSH_RANGE_2("set_problem_dimensions", 7);
                dataArrays[streamIndex].set_problem_dimensions(int(correctionTasks[streamIndex].size()),
                                                            nTotalCandidates,
                                                            fileProperties.maxSequenceLength,
                                                            goodAlignmentProperties.min_overlap,
                                                            goodAlignmentProperties.min_overlap_ratio,
                                                            correctionOptions.useQualityScores); CUERR;

                std::size_t temp_storage_bytes = 0;
                std::size_t max_temp_storage_bytes = 0;
                cub::DeviceHistogram::HistogramRange((void*)nullptr, temp_storage_bytes,
                                                    (int*)nullptr, (int*)nullptr,
                                                    dataArrays[streamIndex].n_subjects+1,
                                                    (int*)nullptr,
                                                    dataArrays[streamIndex].n_queries,
                                                    streams[streamIndex]); CUERR;

                max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

                cub::DeviceSelect::Flagged((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                                            (bool*)nullptr, (int*)nullptr, (int*)nullptr,
                                            nTotalCandidates,
                                            streams[streamIndex]); CUERR;

                max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

                cub::DeviceScan::ExclusiveSum((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                                                (int*)nullptr,
                                                dataArrays[streamIndex].n_subjects,
                                                streams[streamIndex]); CUERR;

                max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);
                temp_storage_bytes = max_temp_storage_bytes;

                dataArrays[streamIndex].set_tmp_storage_size(max_temp_storage_bytes);

                POP_RANGE_2;

                //dataArrays[streamIndex].set_n_indices(nTotalCandidates); CUERR;
//PUSH_RANGE_2("zero_cpu", 4);
				//dataArrays[streamIndex].zero_cpu();
//POP_RANGE_2;
PUSH_RANGE_2("zero_gpu", 5);
				dataArrays[streamIndex].zero_gpu(streams[streamIndex]);
POP_RANGE_2;

                //fill data arrays
                dataArrays[streamIndex].h_candidates_per_subject_prefixsum[0] = 0;

                int maxSubjectLength = 0;
                int maxQueryLength = 0;

                PUSH_RANGE_2("copy_to_buffer", 2);

                for(std::size_t i = 0, count = 0; i < correctionTasks[streamIndex].size(); ++i){
                    const auto& task = correctionTasks[streamIndex][i];
                    auto& arrays = dataArrays[streamIndex];

                    //fill subject
                    std::memcpy(arrays.h_subject_sequences_data + i * arrays.encoded_sequence_pitch,
                                task.subject_sequence->begin(),
                                task.subject_sequence->getNumBytes());
                    arrays.h_subject_sequences_lengths[i] = task.subject_sequence->length();
                    maxSubjectLength = std::max(int(task.subject_sequence->length()), maxSubjectLength);

                    //fill candidates
                    for(const Sequence_t* candidate_sequence : task.candidate_sequences){

                        std::memcpy(arrays.h_candidate_sequences_data + count * arrays.encoded_sequence_pitch,
                                    candidate_sequence->begin(),
                                    candidate_sequence->getNumBytes());

                        arrays.h_candidate_sequences_lengths[count] = candidate_sequence->length();
                        maxQueryLength = std::max(int(candidate_sequence->length()), maxQueryLength);

                        ++count;
                    }

                    //make prefix sum
                    arrays.h_candidates_per_subject_prefixsum[i+1]
                        = arrays.h_candidates_per_subject_prefixsum[i] + int(task.candidate_read_ids.size());
                }

                POP_RANGE_2;

                //data required for alignment is now ready for transfer

                cudaMemcpyAsync(dataArrays[streamIndex].alignment_transfer_data_device,
                                dataArrays[streamIndex].alignment_transfer_data_host,
                                dataArrays[streamIndex].alignment_transfer_data_usable_size,
                                H2D,
                                streams[streamIndex]); CUERR;

#ifdef CARE_GPU_DEBUG
                cudaStreamSynchronize(streams[streamIndex]); CUERR;
#endif

                //Step 4. Perform Alignment. Produces 2*M alignments, M alignments for forward sequences, M alignments for reverse complement sequences


                call_shd_with_revcompl_kernel_async(
                                        dataArrays[streamIndex].d_alignment_scores,
                                        dataArrays[streamIndex].d_alignment_overlaps,
                                        dataArrays[streamIndex].d_alignment_shifts,
                                        dataArrays[streamIndex].d_alignment_nOps,
                                        dataArrays[streamIndex].d_alignment_isValid,
                                        dataArrays[streamIndex].d_subject_sequences_data,
                                        dataArrays[streamIndex].d_candidate_sequences_data,
                                        dataArrays[streamIndex].d_subject_sequences_lengths,
                                        dataArrays[streamIndex].d_candidate_sequences_lengths,
                                        dataArrays[streamIndex].d_candidates_per_subject_prefixsum,
                                        dataArrays[streamIndex].n_subjects,
                                        Sequence_t::getNumBytes(dataArrays[streamIndex].maximum_sequence_length),
                                        dataArrays[streamIndex].encoded_sequence_pitch,
                                        goodAlignmentProperties.min_overlap,
                                        goodAlignmentProperties.maxErrorRate, //correctionOptions.estimatedErrorrate * 4.0,
                                        goodAlignmentProperties.min_overlap_ratio,
                                        accessor,
                                        make_reverse_complement_inplace,
                                        dataArrays[streamIndex].n_queries,
                                        maxSubjectLength,
                                        maxQueryLength,
                                        streams[streamIndex]);

#ifdef CARE_GPU_DEBUG
                cudaStreamSynchronize(streams[streamIndex]); CUERR;
#endif

#if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_MEMCOPY
                                        cudaMemcpyAsync(dataArrays[streamIndex].alignment_result_data_host,
                        								dataArrays[streamIndex].alignment_result_data_device,
                        								dataArrays[streamIndex].alignment_result_data_usable_size,
                        								D2H,
                        								streams[streamIndex]); CUERR;
                        				cudaStreamSynchronize(streams[streamIndex]); CUERR;
#endif
                //Step 5. Compare each forward alignment with the correspoding reverse complement alignment and keep the best, if any.
                //    If reverse complement is the best, it is copied into the first half, replacing the forward alignment

                call_cuda_find_best_alignment_kernel_async(
                                                    dataArrays[streamIndex].d_alignment_best_alignment_flags,
                                                    dataArrays[streamIndex].d_alignment_scores,
                                                    dataArrays[streamIndex].d_alignment_overlaps,
                                                    dataArrays[streamIndex].d_alignment_shifts,
                                                    dataArrays[streamIndex].d_alignment_nOps,
                                                    dataArrays[streamIndex].d_alignment_isValid,
                                                    dataArrays[streamIndex].d_subject_sequences_lengths,
                                                    dataArrays[streamIndex].d_candidate_sequences_lengths,
                                                    dataArrays[streamIndex].d_candidates_per_subject_prefixsum,
                                                    dataArrays[streamIndex].n_subjects,
													goodAlignmentProperties.min_overlap_ratio,
													goodAlignmentProperties.min_overlap,
													goodAlignmentProperties.maxErrorRate, //correctionOptions.estimatedErrorrate * 4.0,
                                                    best_alignment_comp,
                                                    dataArrays[streamIndex].n_queries,
                                                    streams[streamIndex]);

#ifdef CARE_GPU_DEBUG
                cudaStreamSynchronize(streams[streamIndex]); CUERR;
#endif

#if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_MEMCOPY
                                        cudaMemcpyAsync(dataArrays[streamIndex].alignment_result_data_host,
                        								dataArrays[streamIndex].alignment_result_data_device,
                        								dataArrays[streamIndex].alignment_result_data_usable_size,
                        								D2H,
                        								streams[streamIndex]); CUERR;
                        				cudaStreamSynchronize(streams[streamIndex]); CUERR;
#endif

                //Step 6. Determine alignments which should be used for correction

                /*
                    Writes indices of candidates with alignmentflag != None to dataArrays[streamIndex].d_indices
                    Writes number of writen indices to dataArrays[streamIndex].d_num_indices
                    Elements after d_indices[d_num_indices - 1] are set to -1;
                */
                auto select_alignments_by_flag = [&](){
                    dataArrays[streamIndex].fill_d_indices(-1, streams[streamIndex]);

                    auto select_alignment_op = [] __device__ (const BestAlignment_t& flag){
                        return flag != BestAlignment_t::None;
                    };

                    cub::TransformInputIterator<bool,decltype(select_alignment_op), BestAlignment_t*>
                                d_isGoodAlignment(dataArrays[streamIndex].d_alignment_best_alignment_flags,
                                                  select_alignment_op);

                    cub::DeviceSelect::Flagged(dataArrays[streamIndex].d_temp_storage,
                                                dataArrays[streamIndex].tmp_storage_allocation_size,
                                                cub::CountingInputIterator<int>(0),
                                                d_isGoodAlignment,
                                                dataArrays[streamIndex].d_indices,
                                                dataArrays[streamIndex].d_num_indices,
                                                nTotalCandidates,
                                                streams[streamIndex]); CUERR;
                };

                //Determine indices i < M where d_alignment_best_alignment_flags[i] != BestAlignment_t::None. this selects all good alignments
                select_alignments_by_flag();

                //Step 10. Copy quality scores to gpu. This overlaps with alignment kernel
                PUSH_RANGE_2("get_quality_scores", 5);
                if(correctionOptions.useQualityScores){

                    for(std::size_t subject_index = 0, count = 0; subject_index < correctionTasks[streamIndex].size(); ++subject_index){
                        const auto& task = correctionTasks[streamIndex][subject_index];
                        auto& arrays = dataArrays[streamIndex];

                        std::memcpy(arrays.h_subject_qualities + subject_index * arrays.quality_pitch,
                                    task.subject_quality->c_str(),
                                    task.subject_quality->length());

                        for(const std::string* qual : task.candidate_qualities){
                            std::memcpy(arrays.h_candidate_qualities + count * arrays.quality_pitch,
                                        qual->c_str(),
                                        qual->length());
                            ++count;
                        }
                    }

                    cudaMemcpyAsync(dataArrays[streamIndex].qualities_transfer_data_device,
                                    dataArrays[streamIndex].qualities_transfer_data_host,
                                    dataArrays[streamIndex].qualities_transfer_data_usable_size,
                                    H2D,
                                    streams[streamIndex]); CUERR;

                }
                POP_RANGE_2;

        #ifdef CARE_GPU_DEBUG
                        cudaStreamSynchronize(streams[streamIndex]); CUERR;
        #endif

                /*cudaMemcpyAsync(dataArrays[streamIndex].h_num_indices, dataArrays[streamIndex].d_num_indices, sizeof(int), D2H, streams[streamIndex]); CUERR;
                PUSH_RANGE_2("wait_for_num_indices_1", 3);
                cudaStreamSynchronize(streams[streamIndex]); CUERR; // need h_num_indices before continuing
                POP_RANGE_2;*/

        #if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_MEMCOPY
                        cudaMemcpyAsync(dataArrays[streamIndex].indices_transfer_data_host,
                                        dataArrays[streamIndex].indices_transfer_data_device,
                                        dataArrays[streamIndex].indices_transfer_data_usable_size,
                                        D2H,
                                        streams[streamIndex]); CUERR;
                        cudaStreamSynchronize(streams[streamIndex]); CUERR; //remove
        #endif

                //Step 7. Determine number of indices per subject and the corresponding prefix sum

                //Get number of indices per subject by creating histrogram.
                //The elements d_indices[d_num_indices] to d_indices[n_queries - 1] will be -1.
                //Thus, they will not be accounted for by the histrogram, since the histrogram bins (d_candidates_per_subject_prefixsum) are >= 0.
                cub::DeviceHistogram::HistogramRange(dataArrays[streamIndex].d_temp_storage,
                                                    dataArrays[streamIndex].tmp_storage_allocation_size,
                                                    dataArrays[streamIndex].d_indices,
                                                    dataArrays[streamIndex].d_indices_per_subject,
                                                    dataArrays[streamIndex].n_subjects+1,
                                                    dataArrays[streamIndex].d_candidates_per_subject_prefixsum,
                                                    // *dataArrays[streamIndex].h_num_indices,
                                                    dataArrays[streamIndex].n_queries,
                                                    streams[streamIndex]); CUERR;


        #ifdef CARE_GPU_DEBUG
                        cudaStreamSynchronize(streams[streamIndex]); CUERR;
        #endif

                cub::DeviceScan::ExclusiveSum(dataArrays[streamIndex].d_temp_storage,
                                                dataArrays[streamIndex].tmp_storage_allocation_size,
                                                dataArrays[streamIndex].d_indices_per_subject,
                                                dataArrays[streamIndex].d_indices_per_subject_prefixsum,
                                                dataArrays[streamIndex].n_subjects,
                                                streams[streamIndex]); CUERR;

        #ifdef CARE_GPU_DEBUG
                        cudaStreamSynchronize(streams[streamIndex]); CUERR;
        #endif

                //choose the most appropriate subset of alignments from the good alignments.
                //This sets d_alignment_best_alignment_flags[i] = BestAlignment_t::None for all non-appropriate alignments
                call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                                        dataArrays[streamIndex].d_alignment_best_alignment_flags,
                                        dataArrays[streamIndex].d_alignment_overlaps,
                                        dataArrays[streamIndex].d_alignment_nOps,
                                        dataArrays[streamIndex].d_indices,
                                        dataArrays[streamIndex].d_indices_per_subject,
                                        dataArrays[streamIndex].d_indices_per_subject_prefixsum,
                                        dataArrays[streamIndex].n_subjects,
                                        dataArrays[streamIndex].n_queries,
                                        dataArrays[streamIndex].d_num_indices,
                                        correctionOptions.estimatedErrorrate,
                                        correctionOptions.estimatedCoverage * correctionOptions.m_coverage,
                                        streams[streamIndex]);

#ifdef CARE_GPU_DEBUG
                cudaStreamSynchronize(streams[streamIndex]); CUERR;
#endif

                //determine indices of remaining alignments
                select_alignments_by_flag();

                //update indices_per_subject
                cub::DeviceHistogram::HistogramRange(dataArrays[streamIndex].d_temp_storage,
                                                    dataArrays[streamIndex].tmp_storage_allocation_size,
                                                    dataArrays[streamIndex].d_indices,
                                                    dataArrays[streamIndex].d_indices_per_subject,
                                                    dataArrays[streamIndex].n_subjects+1,
                                                    dataArrays[streamIndex].d_candidates_per_subject_prefixsum,
                                                    // *dataArrays[streamIndex].h_num_indices,
                                                    dataArrays[streamIndex].n_queries,
                                                    streams[streamIndex]); CUERR;

#ifdef CARE_GPU_DEBUG
                cudaStreamSynchronize(streams[streamIndex]); CUERR;
#endif

                //Make indices_per_subject_prefixsum
                cub::DeviceScan::ExclusiveSum(dataArrays[streamIndex].d_temp_storage,
                                                dataArrays[streamIndex].tmp_storage_allocation_size,
                                                dataArrays[streamIndex].d_indices_per_subject,
                                                dataArrays[streamIndex].d_indices_per_subject_prefixsum,
                                                dataArrays[streamIndex].n_subjects,
                                                streams[streamIndex]); CUERR;

#ifdef CARE_GPU_DEBUG
                cudaStreamSynchronize(streams[streamIndex]); CUERR;
#endif

                // Step 11. Determine multiple sequence alignment properties

                call_msa_init_kernel_async(
                                dataArrays[streamIndex].d_msa_column_properties,
                                dataArrays[streamIndex].d_alignment_shifts,
                                dataArrays[streamIndex].d_alignment_best_alignment_flags,
                                dataArrays[streamIndex].d_subject_sequences_lengths,
                                dataArrays[streamIndex].d_candidate_sequences_lengths,
                                dataArrays[streamIndex].d_indices,
                                dataArrays[streamIndex].d_indices_per_subject,
                                dataArrays[streamIndex].d_indices_per_subject_prefixsum,
                                dataArrays[streamIndex].n_subjects,
                                dataArrays[streamIndex].n_queries,
                                streams[streamIndex]);

#ifdef CARE_GPU_DEBUG
                cudaStreamSynchronize(streams[streamIndex]); CUERR; //remove
#endif

                cudaMemcpyAsync(dataArrays[streamIndex].h_num_indices, dataArrays[streamIndex].d_num_indices, sizeof(int), D2H, streams[streamIndex]); CUERR;
                PUSH_RANGE_2("wait_for_num_indices_2", 4);
                cudaStreamSynchronize(streams[streamIndex]); CUERR; // need h_num_indices before continuing
                POP_RANGE_2;

        #if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_MEMCOPY
                        cudaMemcpyAsync(dataArrays[streamIndex].indices_transfer_data_host,
                                        dataArrays[streamIndex].indices_transfer_data_device,
                                        dataArrays[streamIndex].indices_transfer_data_usable_size,
                                        D2H,
                                        streams[streamIndex]); CUERR;
                        cudaStreamSynchronize(streams[streamIndex]); CUERR; //remove
        #endif

                if(*dataArrays[streamIndex].h_num_indices == 0){
                    continue;
                }

#if 0
                cudaMemcpyAsync(dataArrays[streamIndex].indices_transfer_data_host,
                                dataArrays[streamIndex].indices_transfer_data_device,
                                dataArrays[streamIndex].indices_transfer_data_usable_size,
                                D2H,
                                streams[streamIndex]); CUERR;
                cudaStreamSynchronize(streams[streamIndex]); CUERR; //remove

                std::cout << "indices" << std::endl;
                for(int i = 0; i < dataArrays[streamIndex].n_queries; i++){
                    std::cout << dataArrays[streamIndex].h_indices[i] << " ";
                }
                std::cout << std::endl;

                std::cout << "indices_per_subject" << std::endl;
                for(int i = 0; i < dataArrays[streamIndex].n_subjects; i++){
                    std::cout << dataArrays[streamIndex].h_indices_per_subject[i] << " ";
                }
                std::cout << std::endl;

                std::cout << "candidates_per_subject_prefixsum" << std::endl;
                for(int i = 0; i < dataArrays[streamIndex].n_subjects + 1; i++){
                    std::cout << dataArrays[streamIndex].h_candidates_per_subject_prefixsum[i] << " ";
                }
                std::cout << std::endl;

#endif


#ifdef CARE_GPU_DEBUG
                cudaStreamSynchronize(streams[streamIndex]); CUERR;
#endif


                //Step 8. Copy d_indices, d_indices_per_subject, d_indices_per_subject_prefixsum to host

                cudaMemcpyAsync(dataArrays[streamIndex].indices_transfer_data_host,
                                dataArrays[streamIndex].indices_transfer_data_device,
                                dataArrays[streamIndex].indices_transfer_data_usable_size,
                                D2H,
                                streams[streamIndex]); CUERR;

#ifdef CARE_GPU_DEBUG
                cudaStreamSynchronize(streams[streamIndex]); CUERR;
#endif


                //Step 12. Fill multiple sequence alignment

                call_msa_add_sequences_kernel_async(
                                dataArrays[streamIndex].d_multiple_sequence_alignments,
                                dataArrays[streamIndex].d_multiple_sequence_alignment_weights,
                                dataArrays[streamIndex].d_alignment_shifts,
                                dataArrays[streamIndex].d_alignment_best_alignment_flags,
                                dataArrays[streamIndex].d_subject_sequences_data,
                                dataArrays[streamIndex].d_candidate_sequences_data,
                                dataArrays[streamIndex].d_subject_sequences_lengths,
                                dataArrays[streamIndex].d_candidate_sequences_lengths,
                                dataArrays[streamIndex].d_subject_qualities,
                                dataArrays[streamIndex].d_candidate_qualities,
                                dataArrays[streamIndex].d_alignment_overlaps,
                                dataArrays[streamIndex].d_alignment_nOps,
                                dataArrays[streamIndex].d_msa_column_properties,
                                dataArrays[streamIndex].d_candidates_per_subject_prefixsum,
                                dataArrays[streamIndex].d_indices,
                                dataArrays[streamIndex].d_indices_per_subject,
                                dataArrays[streamIndex].d_indices_per_subject_prefixsum,
                                dataArrays[streamIndex].n_subjects,
                                dataArrays[streamIndex].n_queries,
                                *dataArrays[streamIndex].h_num_indices,
                                correctionOptions.useQualityScores,
                                desiredAlignmentMaxErrorRate,
                                dataArrays[streamIndex].encoded_sequence_pitch,
                                dataArrays[streamIndex].quality_pitch,
                                dataArrays[streamIndex].msa_pitch,
                                dataArrays[streamIndex].msa_weights_pitch,
                                nucleotide_accessor,
                                make_unpacked_reverse_complement_inplace,
                                streams[streamIndex]);

#ifdef CARE_GPU_DEBUG
				cudaStreamSynchronize(streams[streamIndex]); CUERR; //remove
#endif

                //Step 13. Determine consensus in multiple sequence alignment

                call_msa_find_consensus_kernel_async(
								dataArrays[streamIndex].d_consensus,
								dataArrays[streamIndex].d_support,
								dataArrays[streamIndex].d_coverage,
								dataArrays[streamIndex].d_origWeights,
								dataArrays[streamIndex].d_origCoverages,
								dataArrays[streamIndex].d_multiple_sequence_alignments,
								dataArrays[streamIndex].d_multiple_sequence_alignment_weights,
								dataArrays[streamIndex].d_msa_column_properties,
								dataArrays[streamIndex].d_candidates_per_subject_prefixsum,
								dataArrays[streamIndex].d_indices_per_subject,
								dataArrays[streamIndex].d_indices_per_subject_prefixsum,
								dataArrays[streamIndex].n_subjects,
								dataArrays[streamIndex].n_queries,
                                *dataArrays[streamIndex].h_num_indices,
								dataArrays[streamIndex].msa_pitch,
								dataArrays[streamIndex].msa_weights_pitch,
								3*dataArrays[streamIndex].maximum_sequence_length - 2*goodAlignmentProperties.min_overlap,
								streams[streamIndex]);

#ifdef CARE_GPU_DEBUG
				cudaStreamSynchronize(streams[streamIndex]); CUERR; //remove
#endif

                // Step 14. Correction

                // correct subjects
                call_msa_correct_subject_kernel_async(
								dataArrays[streamIndex].d_consensus,
								dataArrays[streamIndex].d_support,
								dataArrays[streamIndex].d_coverage,
								dataArrays[streamIndex].d_origCoverages,
								dataArrays[streamIndex].d_multiple_sequence_alignments,
								dataArrays[streamIndex].d_msa_column_properties,
								dataArrays[streamIndex].d_indices_per_subject_prefixsum,
								dataArrays[streamIndex].d_is_high_quality_subject,
								dataArrays[streamIndex].d_corrected_subjects,
								dataArrays[streamIndex].d_subject_is_corrected,
								dataArrays[streamIndex].n_subjects,
								dataArrays[streamIndex].n_queries,
                                *dataArrays[streamIndex].h_num_indices,
								dataArrays[streamIndex].sequence_pitch,
								dataArrays[streamIndex].msa_pitch,
								dataArrays[streamIndex].msa_weights_pitch,
								correctionOptions.estimatedErrorrate,
								avg_support_threshold,
								min_support_threshold,
								min_coverage_threshold,
								correctionOptions.kmerlength,
								dataArrays[streamIndex].maximum_sequence_length,
								streams[streamIndex]);

#ifdef CARE_GPU_DEBUG
				cudaStreamSynchronize(streams[streamIndex]); CUERR; //remove
#endif

                if(correctionOptions.correctCandidates){


                    // find subject ids of subjects with high quality multiple sequence alignment

                    cub::DeviceSelect::Flagged(dataArrays[streamIndex].d_temp_storage,
    								dataArrays[streamIndex].tmp_storage_allocation_size,
    								cub::CountingInputIterator<int>(0),
    								dataArrays[streamIndex].d_is_high_quality_subject,
    								dataArrays[streamIndex].d_high_quality_subject_indices,
    								dataArrays[streamIndex].d_num_high_quality_subject_indices,
    								dataArrays[streamIndex].n_subjects,
    								streams[streamIndex]); CUERR;
#ifdef CARE_GPU_DEBUG
				cudaStreamSynchronize(streams[streamIndex]); CUERR; //remove
#endif

    				// correct candidates
    				call_msa_correct_candidates_kernel_async(
    								dataArrays[streamIndex].d_consensus,
    								dataArrays[streamIndex].d_support,
    								dataArrays[streamIndex].d_coverage,
    								dataArrays[streamIndex].d_origCoverages,
    								dataArrays[streamIndex].d_multiple_sequence_alignments,
    								dataArrays[streamIndex].d_msa_column_properties,
    								dataArrays[streamIndex].d_indices,
    								dataArrays[streamIndex].d_indices_per_subject,
    								dataArrays[streamIndex].d_indices_per_subject_prefixsum,
    								dataArrays[streamIndex].d_high_quality_subject_indices,
    								dataArrays[streamIndex].d_num_high_quality_subject_indices,
    								dataArrays[streamIndex].d_alignment_shifts,
    								dataArrays[streamIndex].d_alignment_best_alignment_flags,
    								dataArrays[streamIndex].d_candidate_sequences_lengths,
    								dataArrays[streamIndex].d_num_corrected_candidates,
    								dataArrays[streamIndex].d_corrected_candidates,
    								dataArrays[streamIndex].d_indices_of_corrected_candidates,
    								dataArrays[streamIndex].n_subjects,
    								dataArrays[streamIndex].n_queries,
    								//dataArrays[streamIndex].n_indices,
                                    *dataArrays[streamIndex].h_num_indices,
    								dataArrays[streamIndex].sequence_pitch,
    								dataArrays[streamIndex].msa_pitch,
    								dataArrays[streamIndex].msa_weights_pitch,
    								min_support_threshold,
    								min_coverage_threshold,
    								new_columns_to_correct,
    								make_unpacked_reverse_complement_inplace,
    								dataArrays[streamIndex].maximum_sequence_length,
    								streams[streamIndex]);
#ifdef CARE_GPU_DEBUG
				cudaStreamSynchronize(streams[streamIndex]); CUERR; //remove
#endif

                }

				//copy correction results to host
				cudaMemcpyAsync(dataArrays[streamIndex].correction_results_transfer_data_host,
								dataArrays[streamIndex].correction_results_transfer_data_device,
								dataArrays[streamIndex].correction_results_transfer_data_usable_size,
								D2H,
								streams[streamIndex]); CUERR;
                PUSH_RANGE_2("wait_for_results", 6);
				cudaStreamSynchronize(streams[streamIndex]); CUERR; //remove
                POP_RANGE_2;


#if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_MEMCOPY
				//DEBUGGING
				cudaMemcpyAsync(dataArrays[streamIndex].msa_data_host,
								dataArrays[streamIndex].msa_data_device,
								dataArrays[streamIndex].msa_data_usable_size,
								D2H,
								streams[streamIndex]); CUERR;
				cudaStreamSynchronize(streams[streamIndex]); CUERR;

				//DEBUGGING
				cudaMemcpyAsync(dataArrays[streamIndex].alignment_result_data_host,
								dataArrays[streamIndex].alignment_result_data_device,
								dataArrays[streamIndex].alignment_result_data_usable_size,
								D2H,
								streams[streamIndex]); CUERR;
				cudaStreamSynchronize(streams[streamIndex]); CUERR;

				//DEBUGGING
				cudaMemcpyAsync(dataArrays[streamIndex].subject_indices_data_host,
								dataArrays[streamIndex].subject_indices_data_device,
								dataArrays[streamIndex].subject_indices_data_usable_size,
								D2H,
								streams[streamIndex]); CUERR;
				cudaStreamSynchronize(streams[streamIndex]); CUERR;
#endif

/*std::cout << "h_is_high_quality_subject" << std::endl;
for(int i = 0; i< dataArrays[streamIndex].n_subjects; i++){
    std::cout << dataArrays[streamIndex].h_is_high_quality_subject[i] << "\t";
}
std::cout << std::endl;*/

#if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_PRINT_ARRAYS
				//DEBUGGING
				std::cout << "alignment scores" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_queries * 2; i++){
					std::cout << dataArrays[streamIndex].h_alignment_scores[i] << "\t";
				}
				std::cout << std::endl;
				//DEBUGGING
				std::cout << "alignment overlaps" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_queries * 2; i++){
					std::cout << dataArrays[streamIndex].h_alignment_overlaps[i] << "\t";
				}
				std::cout << std::endl;
				//DEBUGGING
				std::cout << "alignment shifts" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_queries * 2; i++){
					std::cout << dataArrays[streamIndex].h_alignment_shifts[i] << "\t";
				}
				std::cout << std::endl;
				//DEBUGGING
				std::cout << "alignment nOps" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_queries * 2; i++){
					std::cout << dataArrays[streamIndex].h_alignment_nOps[i] << "\t";
				}
				std::cout << std::endl;
				//DEBUGGING
				std::cout << "alignment isvalid" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_queries * 2; i++){
					std::cout << dataArrays[streamIndex].h_alignment_isValid[i] << "\t";
				}
				std::cout << std::endl;
				//DEBUGGING
				std::cout << "alignment flags" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_queries * 2; i++){
					std::cout << int(dataArrays[streamIndex].h_alignment_best_alignment_flags[i]) << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_candidates_per_subject_prefixsum" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_subjects +1; i++){
					std::cout << dataArrays[streamIndex].h_candidates_per_subject_prefixsum[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_num_indices" << std::endl;
				for(int i = 0; i< 1; i++){
					std::cout << dataArrays[streamIndex].h_num_indices[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_indices" << std::endl;
				for(int i = 0; i< *dataArrays[streamIndex].h_num_indices; i++){
					std::cout << dataArrays[streamIndex].h_indices[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_indices_per_subject" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_subjects; i++){
					std::cout << dataArrays[streamIndex].h_indices_per_subject[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_indices_per_subject_prefixsum" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_subjects; i++){
					std::cout << dataArrays[streamIndex].h_indices_per_subject_prefixsum[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_high_quality_subject_indices" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_subjects; i++){
					std::cout << dataArrays[streamIndex].h_high_quality_subject_indices[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_is_high_quality_subject" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_subjects; i++){
					std::cout << dataArrays[streamIndex].h_is_high_quality_subject[i] << "\t";
				}
				std::cout << std::endl;

				//DEBUGGING
				std::cout << "h_num_high_quality_subject_indices" << std::endl;
				for(int i = 0; i< 1; i++){
					std::cout << dataArrays[streamIndex].h_num_high_quality_subject_indices[i] << "\t";
				}
				std::cout << std::endl;

                //DEBUGGING
				std::cout << "h_num_corrected_candidates" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_subjects; i++){
					std::cout << dataArrays[streamIndex].h_num_corrected_candidates[i] << "\t";
				}
				std::cout << std::endl;

                //DEBUGGING
                std::cout << "h_subject_is_corrected" << std::endl;
				for(int i = 0; i< dataArrays[streamIndex].n_subjects; i++){
					std::cout << dataArrays[streamIndex].h_subject_is_corrected[i] << "\t";
				}
				std::cout << std::endl;

                //DEBUGGING
				std::cout << "h_indices_of_corrected_candidates" << std::endl;
				for(int i = 0; i< *dataArrays[streamIndex].h_num_indices; i++){
					std::cout << dataArrays[streamIndex].h_indices_of_corrected_candidates[i] << "\t";
				}
				std::cout << std::endl;

#if 0
				{
					auto& arrays = dataArrays[streamIndex];
					for(int row = 0; row < *arrays[streamIndex].h_num_indices+1 && row < 50; ++row){
						for(int col = 0; col < arrays.msa_pitch; col++){
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
				for(std::size_t subject_index = 0; subject_index < correctionTasks[streamIndex].size(); ++subject_index){
					auto& task = correctionTasks[streamIndex][subject_index];
					auto& arrays = dataArrays[streamIndex];

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

                //unpack results
                PUSH_RANGE_2("unpack_results", 7);
				for(std::size_t subject_index = 0; subject_index < correctionTasks[streamIndex].size(); ++subject_index){
                    auto& task = correctionTasks[streamIndex][subject_index];
                    auto& arrays = dataArrays[streamIndex];

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


                //write result to file

                for(std::size_t subject_index = 0; subject_index < correctionTasks[streamIndex].size(); ++subject_index){
					const auto& task = correctionTasks[streamIndex][subject_index];

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

            for(auto& stream : streams){
                cudaStreamDestroy(stream); CUERR;
            }
    	}
    };

#endif

}
}

#endif
