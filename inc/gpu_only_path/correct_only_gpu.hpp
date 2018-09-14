#ifndef CARE_CORRECT_ONLY_GPU_HPP
#define CARE_CORRECT_ONLY_GPU_HPP

#include "../shifted_hamming_distance.hpp"
#include "../hpc_helpers.cuh"
#include "dataarrays.hpp"

#include <vector>
#include <string>
#include <mutex>
#include <cstdint>

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif

namespace care{
namespace gpu{


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



        6. Determine indices i < M where d_alignment_best_alignment_flags[i] != BestAlignment_t::None

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
    		 class readStorage_t>
    struct ErrorCorrectionThreadOnlyGPU{
    using Minhasher_t = minhasher_t;
    	using ReadStorage_t = readStorage_t;
    	using Sequence_t = typename ReadStorage_t::Sequence_t;
    	using ReadId_t = typename ReadStorage_t::ReadId_t;
        using CorrectionTask_t = CorrectionTask<ReadId_t, Sequence_t>;

    	struct CorrectionThreadOptions{
    		int threadId;
    		int deviceId;

    		std::string outputfile;
    		BatchGenerator<ReadId_t>* batchGen;
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

        DetermineGoodAlignmentStats goodAlignmentStats;
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
            if(isRunning) throw std::runtime_error("ErrorCorrectionThreadCombined::run: Is already running.");
            isRunning = true;
            thread = std::move(std::thread(&ErrorCorrectionThreadCombined::execute, this));
        }

        void join(){
            thread.join();
            isRunning = false;
        }

    private:

    	void execute() {
    		isRunning = true;

    		std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

    		std::ofstream outputstream(threadOpts.outputfile);

            std::ofstream featurestream(threadOpts.outputfile + "_features");

    		auto write_read = [&](const ReadId_t readId, const auto& sequence){
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

            // device lambdas for sequence functions
            auto getNumBytes = [] __device__ (int length){
                return Sequence_t::getNumBytes(length);
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
                return SequenceString::make_reverse_complement_inplace(sequence, sequencelength);
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
                                            min_overlap_ratio,
                                            min_overlap,
                                            maxErrorRate);
            };

            std::uint64_t cpuAlignments = 0;
            std::uint64_t gpuAlignments = 0;
            //std::uint64_t savedAlignments = 0;
            //std::uint64_t performedAlignments = 0;

    		constexpr int nStreams = 1;

    		std::vector<DataArrays<Sequence_t>> dataArrays;
            std::array<std::vector<CorrectionTask_t>, nStreams> correctionTasks;
            std::array<cudaStream_t, nStreams> streams;

            for(int i = 0; i < nStreams; i++){
                dataArrays.emplace_back(threadOpts.deviceId);
                cudaStreamCreate(&streams[i]); CUERR;
            }

    		std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();

            const auto& readStorage = threadOpts->readStorage;
            const auto& minhasher = threadOpts->minhasher;

            const double avg_support_threshold = 1.0-1.0*correctionOptions.estimatedErrorrate;
            const double min_support_threshold = 1.0-3.0*correctionOptions.estimatedErrorrate;
            const double min_coverage_threshold = correctionOptions.m_coverage / 6.0 * correctionOptions.estimatedCoverage;

    		while(!stopAndAbort && !readIds.empty()){

                int streamIndex = 0;

                correctionTasks[streamIndex].clear();

                // Step 1 and 2: Get subject sequence and its candidate read ids from hash map
                // Get candidate sequences from read storage

                // Get subject sequence
                for(std::size_t i = 0; i < readIds.size(); i++){
                    ReadId_t id = readIds[i];

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

                        if(correctionOptions.canUseQualityScores)
                            qualityptr = readStorage->fetchQuality_ptr(id);

                        correctionTasks[streamIndex].emplace_back(id, sequenceptr, qualityptr);

                        nProcessedQueries++;
                    }
                }

                // Get candidates

                for(auto& task : correctionTasks[streamIndex]){
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
                            if(correctionOptions.canUseQualityScores)
                                task.candidate_qualities.emplace_back(readStorage->fetchQuality_ptr(candidate_read_id));
                        }
                    }
                }

                std::remove_if(correctionTasks[streamIndex].begin(),
                                correctionTasks[streamIndex].end(),
                                [](const auto& t){return !t.active;});

                int nTotalCandidates = std::accumulate(correctionTasks[streamIndex].begin(),
                                                        correctionTasks[streamIndex].end(),
                                                        int(0),
                                                        [](const auto& l, const auto& r){
                                                            return int(l.candidate_read_ids.size()) + int(r.candidate_read_ids.size())
                                                        });


                //Step 3. Copy subject sequences, subject sequence lengths, candidate sequences,
                //candidate sequence lengths, candidates_per_subject_prefixsum to GPU

                //allocate data arrays

                dataArrays[streamIndex].set_problem_dimensions(int(correctionTasks[streamIndex]).size(),
                                                            nTotalCandidates,
                                                            fileProperties.maxSequenceLength,
                                                            goodAlignmentProperties.min_overlap,
                                                            goodAlignmentProperties.min_overlap_ratio);
                dataArrays[streamIndex].set_n_indices(nTotalCandidates);

                //fill data arrays
                dataArrays[streamIndex].h_candidates_per_subject_prefixsum[0] = 0;

                int maxSubjectLength = 0;
                int maxQueryLength = 0;

                for(std::size_t i = 0, count = 0; i < correctionTasks[streamIndex].size(); ++i){
                    const auto& task = correctionTasks[streamIndex][i];
                    auto& arrays = dataArrays[streamIndex];

                    //fill subject
                    std::memcpy(arrays.h_subject_sequences_data + i * arrays.encoded_sequence_pitch,
                                subject_sequence->begin(),
                                subject_sequence->getNumBytes());
                    arrays.h_subject_sequences_lengths[i] = candidate_sequence->length();
                    maxSubjectLength = std::max(int(subject_sequence->length()), maxSubjectLength);

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

                //data required for alignment is now ready for transfer

                cudaMemcpyAsync(arrays.alignment_transfer_data_device,
                                arrays.alignment_transfer_data_host,
                                arrays.alignment_transfer_data_size,
                                H2D,
                                streams[streamIndex]); CUERR;

                //Step 4. Perform Alignment. Produces 2*M alignments, M alignments for forward sequences, M alignments for reverse complement sequences


                call_shd_with_revcompl_kernel_async(
                                        arrays.d_alignment_scores,
                                        arrays.d_alignment_overlaps,
                                        arrays.d_alignment_shifts,
                                        arrays.d_alignment_nOps,
                                        arrays.d_alignment_isValid,
                                        arrays.d_subject_sequences_data,
                                        arrays.d_candidate_sequences_data,
                                        arrays.d_subject_sequences_lengths,
                                        arrays.d_candidate_sequences_lengths,
                                        arrays.d_candidates_per_subject_prefixsum,
                                        arrays.n_subjects,
                                        Sequence_t::getNumBytes(arrays.maximum_sequence_length),
                                        arrays.encoded_sequence_pitch,
                                        goodAlignmentProperties.min_overlap,
                                        goodAlignmentProperties.maxErrorRate,
                                        goodAlignmentProperties.min_overlap_ratio,
                                        accessor,
                                        make_reverse_complement_inplace,
                                        arrays.n_queries,
                                        maxSubjectLength,
                                        maxQueryLength,
                                        streams[streamIndex]);

                //Step 5. Compare each forward alignment with the correspoding reverse complement alignment and keep the best, if any.
                //    If reverse complement is the best, it is copied into the first half, replacing the forward alignment

                call_cuda_find_best_alignment_kernel_async(
                                                    arrays.d_alignment_best_alignment_flags,
                                                    arrays.d_alignment_scores
                                                    arrays.d_alignment_overlaps,
                                                    arrays.d_alignment_shifts
                                                    arrays.d_alignment_nOps,
                                                    arrays.d_alignment_isValid,
                                                    arrays.d_subject_sequences_lengths,
                                                    arrays.d_candidate_sequences_lengths,
                                                    arrays.d_candidates_per_subject_prefixsum,
                                                    arrays.n_subjects,
                                                    best_alignment_comp,
                                                    arrays.n_queries,
                                                    streams[streamIndex]);

                //Step 6. Determine indices i < M where d_alignment_best_alignment_flags[i] != BestAlignment_t::None

                auto select_alignment_op = [] __device__ (const BestAlignment_t& flag){
                    return flag != BestAlignment_t::None;
                };

                cub::TransformInputIterator<bool,decltype(select_alignment_op), BestAlignment_t*>
                                d_isGoodAlignment(arrays.d_alignment_best_alignment_flags,
                                                  select_alignment_op);


                std::size_t temp_storage_bytes = 0;
                cub::DeviceSelect::Flagged(nullptr,
                                            temp_storage_bytes,
                                            cub::CountingInputIterator<int>(0),
                                            arrays.d_isGoodAlignment,
                                            arrays.d_indices,
                                            arrays.d_num_indices,
                                            nTotalCandidates,
                                            streams[streamIndex]); CUERR;

                dataArrays[streamIndex].set_tmp_storage_size(temp_storage_bytes);

                cub::DeviceSelect::Flagged(arrays.d_temp_storage,
                                            temp_storage_bytes,
                                            cub::CountingInputIterator<int>(0),
                                            arrays.d_isGoodAlignment,
                                            arrays.d_indices,
                                            arrays.d_num_indices,
                                            nTotalCandidates,
                                            streams[streamIndex]); CUERR;

                cudaMemcpyAsync(arrays.h_num_indices, arrays.d_num_indices, sizeof(int), D2H, streams[streamIndex]); CUERR;
                cudaStreamSynchronize(streams[streamIndex]); CUERR; // need h_num_indices before continuing

                dataArrays[streamIndex].set_n_indices(*arrays.h_num_indices);

                dataArrays[streamIndex].allocateCorrectionData();

                //Step 7. Determine number of indices per subject and the corresponding prefix sum

                //Get number of indices per subject by creating histrogram
                cub::DeviceHistogram::HistogramRange(nullptr,
                                                    temp_storage_bytes,
                                                    arrays.d_indices,
                                                    arrays.d_indices_per_subject,
                                                    arrays.n_subjects+1,
                                                    arrays.d_candidates_per_subject_prefixsum,
                                                    arrays.n_indices,
                                                    streams[streamIndex]); CUERR;

                dataArrays[streamIndex].set_tmp_storage_size(temp_storage_bytes);

                cub::DeviceHistogram::HistogramRange(arrays.d_temp_storage,
                                                    temp_storage_bytes,
                                                    arrays.d_indices,
                                                    arrays.d_indices_per_subject,
                                                    arrays.n_subjects+1,
                                                    arrays.d_candidates_per_subject_prefixsum,
                                                    arrays.n_indices,
                                                    streams[streamIndex]); CUERR;

                //Make prefix sum
                cub::DeviceScan::ExclusiveSum(nullptr,
                                                temp_storage_bytes,
                                                arrays.d_indices_per_subject,
                                                arrays.d_indices_per_subject_prefixsum,
                                                arrays.n_subjects,
                                                streams[streamIndex]); CUERR;

                dataArrays[streamIndex].set_tmp_storage_size(temp_storage_bytes);

                cub::DeviceScan::ExclusiveSum(arrays.d_temp_storage,
                                                temp_storage_bytes,
                                                arrays.d_indices_per_subject,
                                                arrays.d_indices_per_subject_prefixsum,
                                                arrays.n_subjects,
                                                streams[streamIndex]); CUERR;

                //Step 8. Copy d_indices, d_indices_per_subject, d_indices_per_subject_prefixsum to host

                cudaMemcpyAsync(arrays.indices_transfer_data_host,
                                arrays.indices_transfer_data_device,
                                arrays.total_indices_transfer_data_size,
                                D2H,
                                streams[streamIndex]); CUERR;

                // Step 9. Allocate quality score data and correction data

                arrays.allocateCorrectionData();

                cudaStreamSynchronize(streams[streamIndex]); CUERR; // need indices on host before continuing

                //Step 10. Copy quality scores of candidates referenced by h_indices to gpu

                /*
                    assume task 0 has candidates q0, q1, q2
                    assume task 1 has candidates w0, w1, w2
                    h_candidates_per_subject_prefixsum = [0, 3, 6]

                    assume selected candidates q0, q2, w2
                    h_indices = [0, 2, 5]
                    h_indices_per_subject = [2, 1]
                    h_indices_per_subject_prefixsum = [0, 2]
                    ----------------------------

                    task 0:
                    num_indices_for_this_task = 2
                    indices_for_this_task = h_indices + 0
                    num_candidates_before_this_task = 0
                    copy candidates 0-0 = q0, 2-0 = q2

                    task 1:
                    num_indices_for_this_task = 1
                    indices_for_this_task = h_indices + 2
                    num_candidates_before_this_task = 2
                    copy candidates 5-3 = w2
                */

                for(std::size_t subject_index = 0, count = 0; subject_index < correctionTasks[streamIndex].size(); ++subject_index){
                    const auto& task = correctionTasks[streamIndex][subject_index];
                    auto& arrays = dataArrays[streamIndex];

                    const int num_indices_for_this_task = arrays.h_indices_per_subject[subject_index];
                    const int* const indices_for_this_task = arrays.h_indices + arrays.h_indices_per_subject_prefixsum[subject_index];
                    const int num_candidates_before_this_task = arrays.h_candidates_per_subject_prefixsum[subject_index];

                    std::memcpy(arrays.h_subject_qualities + subject_index * quality_pitch,
                                task.subject_quality->begin(),
                                task.subject_quality->length());

                    for(int j = 0; j < num_indices_for_this_task; ++j, ++count){
                        //calculate task local index
                        const int candidate_index = indices_for_this_task[j] - num_candidates_before_this_task;

                        std::memcpy(arrays.h_candidate_qualities + count * quality_pitch,
                                    task.candidate_qualities[candidate_index]->begin(),
                                    task.candidate_qualities[candidate_index]->length());
                    }
                }

                cudaMemcpyAsync(arrays.qualities_transfer_data_device,
                                arrays.qualities_transfer_data_host,
                                arrays.qualities_transfer_data_size,
                                H2D,
                                streams[streamIndex]); CUERR;

                // Step 11. Determine multiple sequence alignment properties

                call_msa_init_kernel_async(
                                arrays.d_msa_column_properties,
                                arrays.d_alignment_shifts,
                                arrays.d_alignment_best_alignment_flags,
                                arrays.d_subject_sequences_lengths,
                                arrays.d_candidate_sequences_lengths,
                                arrays.d_indices,
                                arrays.d_indices_per_subject,
                                arrays.d_indices_per_subject_prefixsum,
                                arrays.n_subjects,
                                arrays.n_queries,
                                streams[streamIndex]);


                //Step 12. Fill multiple sequence alignment

                call_msa_add_sequences_kernel_async(
                                arrays.d_multiple_sequence_alignments,
                                arrays.d_multiple_sequence_alignment_weights,
                                arrays.d_alignment_shifts,
                                arrays.d_alignment_best_alignment_flags,
                                arrays.d_subject_sequences_data,
                                arrays.d_candidate_sequences_data,
                                arrays.d_subject_sequences_lengths,
                                arrays.d_candidate_sequences_lengths,
                                arrays.d_subject_qualities,
                                arrays.d_candidate_qualities,
                                arrays.d_msa_column_properties,
                                arrays.d_candidates_per_subject_prefixsum,
                                arrays.d_indices,
                                arrays.d_indices_per_subject,
                                arrays.d_indices_per_subject_prefixsum,
                                arrays.n_subjects,
                                arrays.n_queries,
                                arrays.n_indices,
                                correctionOptions.useQualityScores,
                                arrays.encoded_sequence_pitch,
                                arrays.quality_pitch
                                arrays.msa_row_pitch,
                                arrays.msa_weights_row_pitch,
                                nucleotide_accessor,
                                streams[streamIndex]);

                //Step 13. Determine consensus in multiple sequence alignment

                call_msa_find_consensus_kernel_async(
                                        arrays.d_consensus,
                                        arrays.d_support,
                                        arrays.d_coverage,
                                        arrays.d_origWeights,
                                        arrays.d_origCoverages,
                                        arrays.d_multiple_sequence_alignments,
                                        arrays.d_multiple_sequence_alignment_weights,
                                        arrays.d_msa_column_properties,
                                        arrays.d_candidates_per_subject_prefixsum,
                                        arrays.d_indices_per_subject,
                                        arrays.d_indices_per_subject_prefixsum,
                                        arrays.n_subjects,
                                        arrays.n_queries,
                                        arrays.n_indices,
                                        arrays.msa_pitch,
                                        arrays.msa_weights_pitch,
                                        3*arrays.maximum_sequence_length - 2*goodAlignmentProperties.min_overlap,
                                        streams[streamIndex]);

                // Step 14. Correction

                // correct subjects
                call_msa_correct_subject_kernel_async(
                                        arrays.d_consensus,
                                        arrays.d_support,
                                        arrays.d_coverage,
                                        arrays.d_origCoverages,
                                        arrays.d_multiple_sequence_alignments,
                                        arrays.d_msa_column_properties,
                                        arrays.d_indices_per_subject_prefixsum,
                                        arrays.d_is_high_quality_subject,
                                        arrays.d_corrected_subjects,
                                        arrays.n_subjects,
                                        arrays.n_queries,
                                        arrays.n_indices,
                                        arrays.sequence_pitch,
                                        arrays.msa_pitch,
                                        arrays.msa_weights_pitch,
                                        correctionOptions.estimatedErrorrate,
                                        avg_support_threshold,
                                        min_support_threshold,
                                        min_coverage_threshold,
                                        correctionOptions.kmerlength,
                                        arrays.maximum_sequence_length,
                                        streams[streamIndex]);

                // find subject ids of subjects with high quality multiple sequence alignment
                cub::DeviceSelect::Flagged(nullptr,
                                            temp_storage_bytes,
                                            cub::CountingInputIterator<int>(0),
                                            arrays.d_high_quality_subject_indices,
                                            arrays.d_num_high_quality_subject_indices,
                                            arrays.n_subjects,
                                            streams[streamIndex]); CUERR;

                dataArrays[streamIndex].set_tmp_storage_size(temp_storage_bytes);

                cub::DeviceSelect::Flagged(arrays.d_temp_storage,
                                            temp_storage_bytes,
                                            cub::CountingInputIterator<int>(0),
                                            arrays.d_high_quality_subject_indices,
                                            arrays.d_num_high_quality_subject_indices,
                                            arrays.n_subjects,
                                            streams[streamIndex]); CUERR;



				nProcessedReads += readIds.size();
				readIds = threadOpts.batchGen->getNextReadIds();





				tpc = std::chrono::system_clock::now();
				std::pair<PileupCorrectionResult, TaskTimings> res =
											correct(pileupImage,
												b,
												goodAlignmentProperties.maxErrorRate,
												correctionOptions.estimatedErrorrate,
												correctionOptions.estimatedCoverage,
												correctionOptions.correctCandidates,
												correctionOptions.new_columns_to_correct,
                                                correctionOptions.classicMode);

				tpd = std::chrono::system_clock::now();
				readcorrectionTimeTotal += tpd - tpc;

                detailedCorrectionTimings += res.second;

				/*
					features
				*/

				/*if(correctionOptions.extractFeatures){
					std::vector<MSAFeature> MSAFeatures =  extractFeatures(pileupImage, b.fwdSequenceString,
													threadOpts.minhasher->minparams.k, 0.0,
													correctionOptions.estimatedCoverage);

					if(MSAFeatures.size() > 0){
						for(const auto& msafeature : MSAFeatures){
							featurestream << b.readId << '\t' << msafeature.position << '\n';
							featurestream << msafeature << '\n';
						}

					}
				}*/
				auto& correctionResult = res.first;

				avgsupportfail += correctionResult.stats.failedAvgSupport;
				minsupportfail += correctionResult.stats.failedMinSupport;
				mincoveragefail += correctionResult.stats.failedMinCoverage;
				verygoodalignment += correctionResult.stats.isHQ;

				if(correctionResult.isCorrected){
					write_read(b.readId, correctionResult.correctedSequence);
					lock(b.readId);
					(*threadOpts.readIsCorrectedVector)[b.readId] = 1;
					unlock(b.readId);
				}

				for(const auto& correctedCandidate : correctionResult.correctedCandidates){
					const int count = 1;//b.candidateCounts[correctedCandidate.index];
					for(int f = 0; f < count; f++){
						//ReadId_t candidateId = b.candidateIds[b.candidateCountsPrefixSum[correctedCandidate.index] + f];
                    ReadId_t candidateId = b.candidateIds[correctedCandidate.index];
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
							//if (b.bestIsForward[correctedCandidate.index])
                        if(b.bestAlignmentFlags[correctedCandidate.index] == BestAlignment_t::Forward)
								write_read(candidateId, correctedCandidate.sequence);
							else {
								//correctedCandidate.sequence is reverse complement, make reverse complement again
								//const std::string fwd = SequenceGeneral(correctedCandidate.sequence, false).reverseComplement().toString();
                            const std::string fwd = SequenceString(correctedCandidate.sequence).reverseComplement().toString();
								write_read(candidateId, fwd);
							}
						}
					}
				}

    			// update local progress
    			//nProcessedReads += readIds.size();

    			//readIds = threadOpts.batchGen->getNextReadIds();

    		} // end batch processing

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

    	#if 1
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
