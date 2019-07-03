#if 0

#include <gpu/correct_gpu.hpp>
#include <gpu/readstorage.hpp>
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

//#include <thrust/inner_product.h>
//#include <thrust/iterator/counting_iterator.h>


//#define REARRANGE_INDICES
//#define USE_MSA_MINIMIZATION

//#define DO_PROFILE


namespace care{
namespace gpu{



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


    // Caching allocator for device memory
    cub::CachingDeviceAllocator cubCachingAllocator(
                8,                                                  ///< Geometric growth factor for bin-sizes
                3,                                                  ///< Minimum bin (default is bin_growth ^ 1)
                cub::CachingDeviceAllocator::INVALID_BIN,           ///< Maximum bin (default is no max bin)
                cub::CachingDeviceAllocator::INVALID_SIZE,          ///< Maximum aggregate cached bytes per device (default is no limit)
                true,                                               ///< Whether or not to skip a call to \p FreeAllCached() when the destructor is called (default is to deallocate)
                false);                                             ///< Whether or not to print (de)allocation events to stdout (default is no stderr output)


    void build_msa_async(gpu::MSAColumnProperties* d_msa_column_properties,
                    const int* d_alignment_shifts,
                    const gpu::BestAlignment_t* d_alignment_best_alignment_flags,
                    const int* d_alignment_overlaps,
                    const int* d_alignment_nOps,
                    const char* d_subject_sequences_data,
                    const char* d_candidate_sequences_data,
                    const int* d_subject_sequences_lengths,
                    const int* d_candidate_sequences_lengths,
                    const char* d_subject_qualities,
                    const char* d_candidate_qualities,
                    const int* d_candidates_per_subject_prefixsum,
                    const int* d_indices,
                    const int* d_indices_per_subject,
                    const int* d_indices_per_subject_prefixsum,
                    int* d_counts,
                    float* d_weights,
                    int* d_coverage,
                    float* d_support,
                    char* d_consensus,
                    float* d_origWeights,
                    int* d_origCoverages,
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
                d_msa_column_properties,
                d_alignment_shifts,
                d_alignment_best_alignment_flags,
                d_subject_sequences_lengths,
                d_candidate_sequences_lengths,
                d_indices,
                d_indices_per_subject,
                d_indices_per_subject_prefixsum,
                n_subjects,
                n_queries,
                stream,
                kernelLaunchHandle);
#if 1
        call_msa_add_sequences_kernel_implicit_async(
                    d_counts,
                    d_weights,
                    d_coverage,
                    d_alignment_shifts,
                    d_alignment_best_alignment_flags,
                    d_alignment_overlaps,
                    d_alignment_nOps,
                    d_subject_sequences_data,
                    d_candidate_sequences_data,
                    d_subject_sequences_lengths,
                    d_candidate_sequences_lengths,
                    d_subject_qualities,
                    d_candidate_qualities,
                    d_msa_column_properties,
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
                    d_counts,
                    d_weights,
                    d_coverage,
                    d_alignment_shifts,
                    d_alignment_best_alignment_flags,
                    d_alignment_overlaps,
                    d_alignment_nOps,
                    d_subject_sequences_data,
                    d_candidate_sequences_data,
                    d_subject_sequences_lengths,
                    d_candidate_sequences_lengths,
                    d_subject_qualities,
                    d_candidate_qualities,
                    d_msa_column_properties,
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
                    d_counts,
                    d_weights,
                    d_consensus,
                    d_support,
                    d_coverage,
                    d_origWeights,
                    d_origCoverages,
                    d_subject_sequences_data,
                    d_indices_per_subject,
                    d_msa_column_properties,
                    n_subjects,
                    encoded_sequence_pitch,
                    msa_pitch,
                    msa_weights_pitch,
                    stream,
                    kernelLaunchHandle);
    };





void gatherSubjectQualitiesFromHost(DataArrays& dataArrays, const gpu::ContiguousReadStorage& gpuReadStorage, int begin, int end){
    const std::size_t maxsubjectqualitychars = dataArrays.n_subjects * dataArrays.quality_pitch;

    constexpr int prefetch_distance = 4;

    #pragma omp parallel for
    for(int subjectIndex = begin; subjectIndex < end; subjectIndex++){

        if(subjectIndex + prefetch_distance < dataArrays.n_subjects) {
            const read_number next_subject_read_id = dataArrays.h_subject_read_ids[subjectIndex + prefetch_distance];
            const char* nextqualityptr = gpuReadStorage.fetchQuality_ptr(next_subject_read_id);
            __builtin_prefetch(nextqualityptr, 0, 0);
        }

        const read_number readId = dataArrays.h_subject_read_ids[subjectIndex];
        const char* qualityptr = gpuReadStorage.fetchQuality_ptr(readId);
        const int sequencelength = gpuReadStorage.fetchSequenceLength(readId);

        assert(subjectIndex * dataArrays.quality_pitch + sequencelength <= maxsubjectqualitychars);

        std::memcpy(dataArrays.h_subject_qualities + subjectIndex * dataArrays.quality_pitch,
                    qualityptr,
                    sequencelength);
    }
}

void gatherCandidateQualitiesFromHost(DataArrays& dataArrays, const gpu::ContiguousReadStorage& gpuReadStorage, int begin, int end){
    const std::size_t maxcandidatequalitychars = dataArrays.n_queries * dataArrays.quality_pitch;

    constexpr int prefetch_distance = 4;

    #pragma omp parallel for
    for(int indexiter = begin; indexiter < end; indexiter++){

        const int candidateIndex = dataArrays.h_indices[indexiter];

        if(indexiter + prefetch_distance < dataArrays.h_num_indices[0]) {
            const int nextIndex = dataArrays.h_indices[indexiter + prefetch_distance];
            const read_number next_candidate_read_id = dataArrays.h_candidate_read_ids[nextIndex];
            const char* nextqualityptr = gpuReadStorage.fetchQuality_ptr(next_candidate_read_id);
            __builtin_prefetch(nextqualityptr, 0, 0);
        }

        const read_number candidate_read_id = dataArrays.h_candidate_read_ids[candidateIndex];
        const char* qualityptr = gpuReadStorage.fetchQuality_ptr(candidate_read_id);
        const int sequencelength = gpuReadStorage.fetchSequenceLength(candidate_read_id);

        assert(indexiter * dataArrays.quality_pitch + sequencelength <= maxcandidatequalitychars);

        std::memcpy(dataArrays.h_candidate_qualities + indexiter * dataArrays.quality_pitch,
                    qualityptr,
                    sequencelength);
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
                  cpu::ContiguousReadStorage& cpuReadStorage,
                  std::uint64_t maxCandidatesPerRead,
                  std::vector<char>& readIsCorrectedVector,
                  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
                  std::size_t nLocksForProcessedFlags){

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

    std::cerr << "Simpleimpl\n";


    assert(runtimeOptions.canUseGpu);
    assert(runtimeOptions.max_candidates > 0);
    assert(runtimeOptions.deviceIds.size() > 0);

    std::chrono::time_point<std::chrono::system_clock> timepoint_begin = std::chrono::system_clock::now();
    std::chrono::duration<double> runtime = std::chrono::seconds(0);

    const auto& deviceIds = runtimeOptions.deviceIds;

    std::vector<std::string> tmpfiles{fileOptions.outputfile + "_tmp"};

    std::unique_ptr<SequenceFileReader> reader = makeSequenceReader(fileOptions.inputfile, fileOptions.format);

    std::ofstream outputstream;
    std::unique_ptr<SequenceFileWriter> writer;

    //if candidate correction is not enabled, it is possible to write directly into the result file
    if(!correctionOptions.correctCandidates){
        //writer = std::move(makeSequenceWriter(fileOptions.outputfile, FileFormat::FASTQGZ));
        outputstream = std::move(std::ofstream(fileOptions.outputfile));
        if(!outputstream){
            throw std::runtime_error("Could not open output file " + fileOptions.outputfile);
        }
    }else{
        outputstream = std::move(std::ofstream(tmpfiles[0]));
        if(!outputstream){
            throw std::runtime_error("Could not open output file " + tmpfiles[0]);
        }
    }


    std::ofstream featurestream;
    //if(correctionOptions.extractFeatures){
        featurestream = std::move(std::ofstream(tmpfiles[0] + "_features"));
        if(!featurestream && correctionOptions.extractFeatures){
            throw std::runtime_error("Could not open output feature file");
        }
    //}

    std::mutex outputstreamlock;

    Read readInFile;

    auto writeProcessedTaskToStream = [&](const auto& correctionTask, const Read& readFromFile){
        std::lock_guard<std::mutex> lg(outputstreamlock);

        if(correctionTask.subject_string != readFromFile.sequence){
            assert(correctionTask.subject_string.size() == readFromFile.sequence.size());
            for(size_t k = 0; k < readFromFile.sequence.size(); k++){
                char fc = readFromFile.sequence[k];
                fc = std::toupper(fc);
                if(fc != 'N' && correctionTask.subject_string[k] != fc){
                    std::cerr << "file sequence " << readFromFile.sequence << ", task sequence " << correctionTask.subject_string << "\n";
                    break;
                }
            }

        }

        if(correctionTask.corrected){
            outputstream << readFromFile.header << '\n' << correctionTask.corrected_subject << '\n';
            if(fileOptions.format == FileFormat::FASTQ)
                outputstream << '+' << '\n' << readFromFile.quality << '\n';
            //writer->writeRead(readFromFile.header, correctionTask.corrected_subject, readFromFile.quality);
        }else{
            outputstream << readFromFile.header << '\n' << readFromFile.sequence << '\n';
            if(fileOptions.format == FileFormat::FASTQ)
                outputstream << '+' << '\n' << readFromFile.quality << '\n';
            //writer->writeRead(readFromFile.header, readFromFile.sequence, readFromFile.quality);
        }
    };


    auto write_read_to_stream = [&](const read_number readId, const std::string& sequence){
                             //std::cout << readId << " " << sequence << std::endl;
                             auto& stream = outputstream;
#if 1
                             stream << readId << ' ' << sequence << '\n';
#else
                             stream << readId << '\n';
                             stream << sequence << '\n';
#endif
                         };
    auto lock = [&](read_number readId){
        read_number index = readId % nLocksForProcessedFlags;
        locksForProcessedFlags[index].lock();
    };
    auto unlock = [&](read_number readId){
        read_number index = readId % nLocksForProcessedFlags;
        locksForProcessedFlags[index].unlock();
    };

    auto identity = [](const auto& i){return i;};

    ForestClassifier fc;
    if(correctionOptions.correctionType == CorrectionType::Forest){
       fc = std::move(ForestClassifier{fileOptions.forestfilename});
    }

    NN_Correction_Classifier_Base nnClassifierBase;
    NN_Correction_Classifier nnClassifier;
    if(correctionOptions.correctionType == CorrectionType::Convnet){
        nnClassifierBase = std::move(NN_Correction_Classifier_Base{"./nn_sources", fileOptions.nnmodelfilename});
        nnClassifier = std::move(NN_Correction_Classifier{&nnClassifierBase});
    }


    cudaSetDevice(deviceIds[0]); CUERR;

    gpu::init_weights(deviceIds);

    gpu::ContiguousReadStorage gpuReadStorage(&cpuReadStorage, deviceIds);

    gpuReadStorage.initGPUData();

    std::cout << "Sequence Type: " << gpuReadStorage.getNameOfSequenceType() << std::endl;
    std::cout << "Quality Type: " << gpuReadStorage.getNameOfQualityType() << std::endl;

    auto readStorageGpuData = gpuReadStorage.getGPUData(deviceIds[0]);


    std::vector<DataArrays > dataArraysVector;

    std::array<std::array<cudaStream_t, nStreamsPerBatch>, nParallelBatches> streamsVector;
    std::array<std::array<cudaEvent_t, nEventsPerBatch>, nParallelBatches> eventsVector;

    for(int i = 0; i < nParallelBatches; i++) {
        dataArraysVector.emplace_back(deviceIds[0]);

        for(int j = 0; j < nStreamsPerBatch; ++j) {
            cudaStreamCreate(&streamsVector[i][j]); CUERR;
        }

        for(int j = 0; j < nEventsPerBatch; ++j) {
            cudaEventCreateWithFlags(&eventsVector[i][j], cudaEventDisableTiming); CUERR;
        }
    }

    auto kernelLaunchHandle = make_kernel_launch_handle(deviceIds[0]);

    int oldNumOMPThreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        oldNumOMPThreads = omp_get_num_threads();
    }

    omp_set_num_threads(runtimeOptions.nCorrectorThreads);

    size_t batchsize = correctionOptions.batchsize; // 1000
    size_t numbatches = SDIV(sequenceFileProperties.nReads, batchsize);


    constexpr int numProc = 1;
    constexpr size_t candidatehashingblocksize = 128;

    struct ProcData{
        std::vector<CorrectionTask> tasks;
        std::vector<Read> readsFromFile;

        size_t initialNumberOfCandidates = 0;
        read_number batchbegin = 0;
        read_number batchend = 0;
        size_t candidateHashingIter = 0;

        void reset(){
            tasks.clear();
            readsFromFile.clear();
            initialNumberOfCandidates = 0;
            batchbegin = 0;
            batchend = 0;
            candidateHashingIter = 0;
        }
    };

    std::vector<ProcData> procDataVector(numProc);


#ifdef DO_PROFILE
    cudaProfilerStart();
#endif

    //int currentProcId = 1;
    int currentProcId = 0;

    for(size_t batchNum = 0; batchNum < numbatches; batchNum++){

        //reset the ProcData which was used in the previous iteration
        procDataVector[currentProcId].reset();

        //currentProcId = 1 - currentProcId;
        //int nextProcId = 1 - currentProcId;
        currentProcId = 0;
        int nextProcId = 0;

        runtime = std::chrono::system_clock::now() - timepoint_begin;

#ifndef DO_PROFILE
        if(runtimeOptions.showProgress){
            printf("Processed %10lu of %10lu reads (Runtime: %03d:%02d:%02d)\r",
                    batchNum * batchsize, sequenceFileProperties.nReads,
                    int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
                    int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count()) % 60,
                    int(runtime.count()) % 60);
        }
#endif

#ifdef DO_PROFILE
        if(batchNum > 20){
            break;
        }
#endif

        push_range("total", 6);

        read_number batchbegin = batchNum * batchsize;
        read_number batchend = std::min(sequenceFileProperties.nReads, (batchNum + 1) * batchsize);
        procDataVector[currentProcId].batchbegin = batchbegin;
        procDataVector[currentProcId].batchend = batchend;

        size_t nextBatchNum = (batchNum + 1);
        if(numProc > 1){

            read_number nextBatchbegin = nextBatchNum * batchsize;
            read_number nextBatchend = std::min(sequenceFileProperties.nReads, (nextBatchNum + 1) * batchsize);
            procDataVector[nextProcId].batchbegin = nextBatchbegin;
            procDataVector[nextProcId].batchend = nextBatchend;
        }

        if(!correctionOptions.correctCandidates){
            procDataVector[currentProcId].readsFromFile.resize(batchend - batchbegin);
            for(read_number k = batchbegin; k < batchend; k++){
                bool ok = reader->getNextRead(&procDataVector[currentProcId].readsFromFile[k - batchbegin]);
                assert(ok);
                size_t readIndex = reader->getReadnum() - 1;
                //std::cout << k << " " << readIndex << std::endl;
                assert(k == readIndex);
            }
        }

        auto& dataArrays = dataArraysVector[0];
        auto& streams = streamsVector[0];
        auto& events = eventsVector[0];

        //query minhasher to get candidate read ids of each read



        auto candidatehashing = [&](ProcData& procData,
                                    read_number batchbegin, read_number batchend,
                                    size_t minhashbegin, size_t minhashend){

            push_range("minhashing", 0);

            size_t initialReadsInBatch = batchend - batchbegin;
            procData.tasks.resize(initialReadsInBatch);

            size_t readsToProcess = minhashend - minhashbegin;

            #pragma omp parallel for
            //for(size_t k = 0; k < initialReadsInBatch; k++){
            for(size_t k = 0; k < readsToProcess; k++){
                //auto& task = procData.tasks[k];
                auto& task = procData.tasks[minhashbegin + k];

                const read_number readId = batchbegin + minhashbegin + k;
                // const read_number readId = batchbegin + k;
                task = CorrectionTask(readId);

                bool ok = false;
                if (readIsCorrectedVector[readId] == 0) {
                    ok = true;
                }

                if(ok){
                    const char* sequenceptr = gpuReadStorage.fetchSequenceData_ptr(readId);
                    const int sequencelength = gpuReadStorage.fetchSequenceLength(readId);

                    task.subject_string.resize(sequencelength);
                    decode2BitHiLoSequence(&task.subject_string[0], (const unsigned int*)sequenceptr, sequencelength, identity);
                    task.candidate_read_ids = minhasher.getCandidates(task.subject_string, correctionOptions.hits_per_candidate, runtimeOptions.max_candidates);

                    auto readIdPos = std::lower_bound(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.readId);

                    if(readIdPos != task.candidate_read_ids.end() && *readIdPos == task.readId) {
                        task.candidate_read_ids.erase(readIdPos);
                    }

                    std::size_t myNumCandidates = task.candidate_read_ids.size();

                    assert(myNumCandidates <= std::size_t(runtimeOptions.max_candidates));

                    if(myNumCandidates == 0) {
                        task.active = false;
                    }
                }else{
                    task.active = false;
                }
            }

            procData.initialNumberOfCandidates = std::accumulate(procData.tasks.begin(),
                                                                    procData.tasks.end(),
                                                                    size_t(0),
                                                                    [](const size_t acc, const auto& task){
                                                                        return acc + task.candidate_read_ids.size();
                                                                    });

            pop_range();
        };

        auto write_results_to_file = [&](auto&& tasks, size_t numActiveTasks, auto&& readsFromFile){

            if(!correctionOptions.correctCandidates){
                auto lambda = [&, tasks = std::move(tasks), readsFromFile = std::move(readsFromFile)]() mutable{
                    push_range("write_results_to_file", 4);

                    std::sort(tasks.begin(), tasks.end(), [](const auto& l, const auto&r){
                        return l.readId < r.readId;
                    });
                    for(std::size_t subject_index = 0; subject_index < tasks.size(); ++subject_index) {
                        const auto& task = tasks[subject_index];
                        const auto& readFromFile = readsFromFile[subject_index];
                        writeProcessedTaskToStream(task, readFromFile);
                    }

                    pop_range();
                };

                //std::thread(lambda).detach();
                lambda();


            }else{
                push_range("write_results_to_file", 4);

                for(std::size_t subject_index = 0; subject_index < numActiveTasks; ++subject_index) {

                    const auto& task = tasks[subject_index];
                    //std::cout << task.readId << "result" << std::endl;

                    //std::cout << "finished readId " << task.readId << std::endl;

                    if(task.corrected/* && task.corrected_subject != task.subject_string*/) {
                        write_read_to_stream(task.readId, task.corrected_subject);
                    }else{
                        //mark read as not corrected
                        if(readIsCorrectedVector[task.readId] == 1) {
                            lock(task.readId);
                            if(readIsCorrectedVector[task.readId] == 1) {
                                readIsCorrectedVector[task.readId] = 0;
                            }
                            unlock(task.readId);
                        }
                    }

                    for(std::size_t corrected_candidate_index = 0; corrected_candidate_index < task.corrected_candidates.size(); ++corrected_candidate_index) {

                        read_number candidateId = task.corrected_candidates_read_ids[corrected_candidate_index];
                        const std::string& corrected_candidate = task.corrected_candidates[corrected_candidate_index];

                        //const char* sequenceptr = gpuReadStorage.fetchSequenceData_ptr(candidateId);
                        //const int sequencelength = gpuReadStorage.fetchSequenceLength(candidateId);
                        //const std::string original_candidate = Sequence_t::Impl_t::toString((const std::uint8_t*)sequenceptr, sequencelength);

                        //if(corrected_candidate == original_candidate){
                            bool savingIsOk = false;
                            if(readIsCorrectedVector[candidateId] == 0) {
                                lock(candidateId);
                                if(readIsCorrectedVector[candidateId]== 0) {
                                    readIsCorrectedVector[candidateId] = 1;         // we will process this read
                                    savingIsOk = true;
                                    //nCorrectedCandidates++;
                                }
                                unlock(candidateId);
                            }
                            if (savingIsOk) {
                                write_read_to_stream(candidateId, corrected_candidate);
                            }
                        //}


                    }
                }

                pop_range();
            }
        };

        auto workOnNextBatchIfPossibleWhileEventNotReady = [&](cudaEvent_t event, ProcData& procData){

            size_t numReadsInBatch = size_t(procData.batchend-procData.batchbegin);
            const size_t candidatehashingmaxiters = SDIV(numReadsInBatch, candidatehashingblocksize);

            cudaError_t status = cudaEventQuery(event); CUERR;

            while(status == cudaErrorNotReady && procData.candidateHashingIter < candidatehashingmaxiters){
                size_t begin = procData.candidateHashingIter * candidatehashingblocksize;
                size_t end = std::min((procData.candidateHashingIter+1) * candidatehashingblocksize, numReadsInBatch);

                candidatehashing(procData, procData.batchbegin, procData.batchend, begin, end);

                procData.candidateHashingIter++;
                status = cudaEventQuery(event); CUERR;
            }
        };


        //if(batchNum == 0){
        //    candidatehashing(procDataVector[currentProcId], batchbegin, batchend, 0, batchend-batchbegin);
        //}

        {
            size_t numReadsInBatch = size_t(procDataVector[currentProcId].batchend-procDataVector[currentProcId].batchbegin);
            const size_t iters = SDIV(numReadsInBatch, candidatehashingblocksize);

            for(; procDataVector[currentProcId].candidateHashingIter < iters; procDataVector[currentProcId].candidateHashingIter++){
                size_t begin = procDataVector[currentProcId].candidateHashingIter * candidatehashingblocksize;
                size_t end = std::min((procDataVector[currentProcId].candidateHashingIter+1) * candidatehashingblocksize, numReadsInBatch);

                candidatehashing(procDataVector[currentProcId], procDataVector[currentProcId].batchbegin, procDataVector[currentProcId].batchend, begin, end);
            }
        }
        //candidatehashing(procDataVector[currentProcId], batchbegin, batchend);

        std::vector<CorrectionTask>& tasks = procDataVector[currentProcId].tasks;
        size_t initialNumberOfCandidates = procDataVector[currentProcId].initialNumberOfCandidates;

        if(initialNumberOfCandidates == 0){
            write_results_to_file(procDataVector[currentProcId].tasks,
                                    procDataVector[currentProcId].tasks.size(),
                                    procDataVector[currentProcId].readsFromFile);
            continue;
        }

        //remove inactive tasks
        //partition needs to be stable such that
        /*std::vector<int> tmpindicesvector(tasks.size());
        std::iota(tmpindicesvector.begin(), tmpindicesvector.end(), 0);
        auto tmpindicesvectoractiveend = std::partition(tmpindicesvector.begin(),
                                                        tmpindicesvector.end(),
                                                        [&](int i){return tasks[i].active;});*/

        //const size_t numActiveTasks = std::distance(tmpindicesvector.begin(), tmpindicesvectoractiveend);
        auto activeTasksEnd = std::partition(tasks.begin(),
                                            tasks.end(),
                                            [](const auto& task){return task.active;});
        const size_t numActiveTasks = std::distance(tasks.begin(), activeTasksEnd);
        /*std::vector<Read> tmpReadsFromFile(tasks.size());
        for(int index : tmpindicesvector)*/




        push_range("memory_reservation", 1);

        //set up storage arrays
        dataArrays.set_problem_dimensions(int(numActiveTasks),
                    initialNumberOfCandidates,
                    sequenceFileProperties.maxSequenceLength,
                    sizeof(unsigned int) * getEncodedNumInts2BitHiLo(sequenceFileProperties.maxSequenceLength),
                    goodAlignmentProperties.min_overlap,
                    goodAlignmentProperties.min_overlap_ratio,
                    correctionOptions.useQualityScores); CUERR;

        //set up temp arrays

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
                    initialNumberOfCandidates,
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
                                                initialNumberOfCandidates,
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

        pop_range();


        push_range("gather_sequence_data", 2);

        //calculate prefix sum of candidates. copy prefixsum and read ids to gpu
        std::transform(tasks.begin(), activeTasksEnd,
                        dataArrays.h_candidates_per_subject.get(),
                        [&](const auto& task){
                            return task.candidate_read_ids.size();
                        });

        dataArrays.h_candidates_per_subject_prefixsum[0] = 0;
        std::partial_sum(dataArrays.h_candidates_per_subject.get(),
                        dataArrays.h_candidates_per_subject.get() + numActiveTasks,
                        dataArrays.h_candidates_per_subject_prefixsum.get()+1);

        std::transform(tasks.begin(), activeTasksEnd,
                        dataArrays.h_subject_read_ids.get(),
                        [&](const auto& task){
                            return task.readId;
                        });

        for(size_t i = 0; i < numActiveTasks; i++){
            const auto& task = tasks[i];

            const int offset = dataArrays.h_candidates_per_subject_prefixsum[i];
            std::copy(task.candidate_read_ids.begin(),
                        task.candidate_read_ids.end(),
                        dataArrays.h_candidate_read_ids.get() + offset);
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

        if(readStorageGpuData.isValidSequenceData()) {
            //directly gather reads from gpu data


            gpuReadStorage.copyGpuLengthsToGpuBufferAsync(dataArrays.d_subject_sequences_lengths,
                                                                         dataArrays.d_subject_read_ids,
                                                                         dataArrays.n_subjects,
                                                                         deviceIds[0], streams[primary_stream_index]);

            gpuReadStorage.copyGpuLengthsToGpuBufferAsync(dataArrays.d_candidate_sequences_lengths,
                                                                         dataArrays.d_candidate_read_ids,
                                                                         dataArrays.n_queries,
                                                                         deviceIds[0], streams[primary_stream_index]);

            gpuReadStorage.copyGpuSequenceDataToGpuBufferAsync(dataArrays.d_subject_sequences_data,
                                                                         dataArrays.encoded_sequence_pitch,
                                                                         dataArrays.d_subject_read_ids,
                                                                         dataArrays.n_subjects,
                                                                         deviceIds[0], streams[primary_stream_index]);

            gpuReadStorage.copyGpuSequenceDataToGpuBufferAsync(dataArrays.d_candidate_sequences_data,
                                                                         dataArrays.encoded_sequence_pitch,
                                                                         dataArrays.d_candidate_read_ids,
                                                                         dataArrays.n_queries,
                                                                         deviceIds[0], streams[primary_stream_index]);

            assert(dataArrays.encoded_sequence_pitch % sizeof(int) == 0);

            call_transpose_kernel((int*)dataArrays.d_subject_sequences_data_transposed.get(),
                                (const int*)dataArrays.d_subject_sequences_data.get(),
                                dataArrays.n_subjects,
                                getEncodedNumInts2BitHiLo(sequenceFileProperties.maxSequenceLength),
                                dataArrays.encoded_sequence_pitch / sizeof(int),
                                streams[primary_stream_index]);

            call_transpose_kernel((int*)dataArrays.d_candidate_sequences_data_transposed.get(),
                                (const int*)dataArrays.d_candidate_sequences_data.get(),
                                dataArrays.n_queries,
                                getEncodedNumInts2BitHiLo(sequenceFileProperties.maxSequenceLength),
                                dataArrays.encoded_sequence_pitch / sizeof(int),
                                streams[primary_stream_index]);

            cudaEventRecord(events[alignment_data_transfer_h2d_finished_event_index], streams[primary_stream_index]); CUERR;

            //copy the gathers GPU sequence data to pinned host memory

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

        }else{

            constexpr int prefetch_distance = 4;
            const std::size_t subjectsequencedatabytes = dataArrays.d_subject_sequences_data.sizeInBytes();
            const std::size_t candidatesequencedatabytes = dataArrays.d_candidate_sequences_data.sizeInBytes();

            //copy subjects to pinned host buffers

            #pragma omp parallel for
            for(int subjectIndex = 0; subjectIndex < dataArrays.n_subjects; subjectIndex++){
                if(subjectIndex + prefetch_distance < dataArrays.n_subjects) {
                    const read_number next_subject_read_id = dataArrays.h_subject_read_ids[subjectIndex + prefetch_distance];
                    const char* nextsequenceptr = gpuReadStorage.fetchSequenceData_ptr(next_subject_read_id);
                    __builtin_prefetch(nextsequenceptr, 0, 0);
                }

                const read_number readId = dataArrays.h_subject_read_ids[subjectIndex];
                const char* sequenceptr = gpuReadStorage.fetchSequenceData_ptr(readId);
                const int sequencelength = gpuReadStorage.fetchSequenceLength(readId);

                assert(subjectIndex * dataArrays.encoded_sequence_pitch + sizeof(unsigned int) * getEncodedNumInts2BitHiLo(sequencelength) <= subjectsequencedatabytes);

                std::memcpy(dataArrays.h_subject_sequences_data + subjectIndex * dataArrays.encoded_sequence_pitch,
                            sequenceptr,
                            sizeof(unsigned int) * getEncodedNumInts2BitHiLo(sequencelength));

                //copy subject length
                dataArrays.h_subject_sequences_lengths[subjectIndex] = sequencelength;
            }

            //copy candidates to pinned host buffers

            #pragma omp parallel for
            for(int candidateIndex = 0; candidateIndex < dataArrays.n_queries; candidateIndex++){

                if(candidateIndex + prefetch_distance < dataArrays.n_queries) {
                    const read_number next_candidate_read_id = dataArrays.h_candidate_read_ids[candidateIndex + prefetch_distance];
                    const char* nextsequenceptr = gpuReadStorage.fetchSequenceData_ptr(next_candidate_read_id);
                    __builtin_prefetch(nextsequenceptr, 0, 0);
                }

                const read_number candidate_read_id = dataArrays.h_candidate_read_ids[candidateIndex];
                const char* sequenceptr = gpuReadStorage.fetchSequenceData_ptr(candidate_read_id);
                const int sequencelength = gpuReadStorage.fetchSequenceLength(candidate_read_id);

                assert(candidateIndex * dataArrays.encoded_sequence_pitch + sizeof(unsigned int) * getEncodedNumInts2BitHiLo(sequencelength) <= candidatesequencedatabytes);

                std::memcpy(dataArrays.h_candidate_sequences_data
                            + candidateIndex * dataArrays.encoded_sequence_pitch,
                            sequenceptr,
                            sizeof(unsigned int) * getEncodedNumInts2BitHiLo(sequencelength));

                dataArrays.h_candidate_sequences_lengths[candidateIndex] = sequencelength;
            }

            // copy sequence data in pinned buffers to GPU

            cudaMemcpyAsync(dataArrays.d_subject_sequences_data,
                            dataArrays.h_subject_sequences_data,
                            dataArrays.h_subject_sequences_data.sizeInBytes(),
                            H2D,
                            streams[primary_stream_index]); CUERR;

            cudaMemcpyAsync(dataArrays.d_candidate_sequences_data,
                            dataArrays.h_candidate_sequences_data,
                            dataArrays.h_candidate_sequences_data.sizeInBytes(),
                            H2D,
                            streams[primary_stream_index]); CUERR;

            cudaMemcpyAsync(dataArrays.d_subject_sequences_lengths,
                            dataArrays.h_subject_sequences_lengths,
                            dataArrays.h_subject_sequences_lengths.sizeInBytes(),
                            H2D,
                            streams[primary_stream_index]); CUERR;

            cudaMemcpyAsync(dataArrays.d_candidate_sequences_lengths,
                            dataArrays.h_candidate_sequences_lengths,
                            dataArrays.h_candidate_sequences_lengths.sizeInBytes(),
                            H2D,
                            streams[primary_stream_index]); CUERR;

            call_transpose_kernel((int*)dataArrays.d_subject_sequences_data_transposed.get(),
                                (const int*)dataArrays.d_subject_sequences_data.get(),
                                dataArrays.n_subjects,
                                getEncodedNumInts2BitHiLo(sequenceFileProperties.maxSequenceLength),
                                dataArrays.encoded_sequence_pitch / sizeof(int),
                                streams[primary_stream_index]);

            call_transpose_kernel((int*)dataArrays.d_candidate_sequences_data_transposed.get(),
                                (const int*)dataArrays.d_candidate_sequences_data.get(),
                                dataArrays.n_queries,
                                getEncodedNumInts2BitHiLo(sequenceFileProperties.maxSequenceLength),
                                dataArrays.encoded_sequence_pitch / sizeof(int),
                                streams[primary_stream_index]);

            cudaEventRecord(events[alignment_data_transfer_h2d_finished_event_index], streams[primary_stream_index]); CUERR;
        }

        pop_range();

        push_range("start_alignments", 3);


        // Calculate alignments

        call_cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel_async(
                    dataArrays.d_alignment_scores,
                    dataArrays.d_alignment_overlaps,
                    dataArrays.d_alignment_shifts,
                    dataArrays.d_alignment_nOps,
                    dataArrays.d_alignment_isValid,
                    dataArrays.d_subject_sequences_data,
                    //dataArrays.d_candidate_sequences_data,
                    dataArrays.d_candidate_sequences_data_transposed,
                    dataArrays.d_subject_sequences_lengths,
                    dataArrays.d_candidate_sequences_lengths,
                    dataArrays.d_candidates_per_subject_prefixsum,
                    dataArrays.h_candidates_per_subject,
                    dataArrays.d_candidates_per_subject,
                    dataArrays.n_subjects,
                    dataArrays.n_queries,
                    dataArrays.encoded_sequence_pitch,
                    sizeof(unsigned int) * getEncodedNumInts2BitHiLo(dataArrays.maximum_sequence_length),
                    goodAlignmentProperties.min_overlap,
                    goodAlignmentProperties.maxErrorRate,
                    goodAlignmentProperties.min_overlap_ratio,
                    //batch.maxSubjectLength,
                    streams[primary_stream_index],
                    kernelLaunchHandle);

        // Compare each forward alignment with the correspoding reverse complement alignment and keep the best, if any.
        //    If reverse complement is the best, it is copied into the first half, replacing the forward alignment

        call_cuda_find_best_alignment_kernel_async_exp(
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
                    dataArrays.n_queries,
                    goodAlignmentProperties.min_overlap_ratio,
                    goodAlignmentProperties.min_overlap,
                    correctionOptions.estimatedErrorrate,
                    streams[primary_stream_index],
                    kernelLaunchHandle);

        //choose the most appropriate subset of alignments from the good alignments.
        //This sets d_alignment_best_alignment_flags[i] = BestAlignment_t::None for all non-appropriate alignments
        call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                    dataArrays.d_alignment_best_alignment_flags,
                    dataArrays.d_alignment_overlaps,
                    dataArrays.d_alignment_nOps,
                    dataArrays.d_candidates_per_subject_prefixsum,
                    dataArrays.n_subjects,
                    dataArrays.n_queries,
                    correctionOptions.estimatedErrorrate,
                    correctionOptions.estimatedCoverage * correctionOptions.m_coverage,
                    streams[primary_stream_index],
                    kernelLaunchHandle);

        //make index list to indicate candidates which passed the filtering step

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

        cudaEventRecord(events[indices_calculated_event_index], streams[primary_stream_index]); CUERR;

        cudaStreamWaitEvent(streams[secondary_stream_index], events[indices_calculated_event_index], 0); CUERR;

        cudaMemcpyAsync(dataArrays.h_num_indices,
                        dataArrays.d_num_indices,
                        sizeof(int),
                        D2H,
                        streams[primary_stream_index]); CUERR;

        cudaEventRecord(events[num_indices_transfered_event_index], streams[primary_stream_index]); CUERR;

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


        pop_range();


        push_range("wait_for_h_num_indices", 4);

        // wait for transfer of h_num_indices. find candidates of next batch in the meantime
        if(nextBatchNum < numbatches && numProc > 1){
            if(numProc > 1){
                workOnNextBatchIfPossibleWhileEventNotReady(events[num_indices_transfered_event_index], procDataVector[nextProcId]);

                cudaEventSynchronize(events[num_indices_transfered_event_index]); CUERR;
            }
        }else{
            cudaEventSynchronize(events[num_indices_transfered_event_index]); CUERR;
        }

        pop_range();

        if(dataArrays.h_num_indices[0] == 0){
            write_results_to_file(procDataVector[currentProcId].tasks, numActiveTasks, procDataVector[currentProcId].readsFromFile);
            continue;
        }

        push_range("wait_for_indices", 5);

        if(nextBatchNum < numbatches && numProc > 1){
            if(numProc > 1){
                workOnNextBatchIfPossibleWhileEventNotReady(events[indices_transfer_finished_event_index], procDataVector[nextProcId]);

                cudaEventSynchronize(events[num_indices_transfered_event_index]); CUERR;
            }
        }else{
            cudaEventSynchronize(events[indices_transfer_finished_event_index]); CUERR;
        }

        pop_range();

        push_range("gather_quality_data", 6);

        //gather quality scores, if required
        if(correctionOptions.useQualityScores) {

            if(readStorageGpuData.isValidQualityData()) {

                //gather subject quality scores
                gpuReadStorage.copyGpuQualityDataToGpuBufferAsync(dataArrays.d_subject_qualities,
                                                                  dataArrays.quality_pitch,
                                                                  dataArrays.d_subject_read_ids,
                                                                  dataArrays.n_subjects,
                                                                  deviceIds[0],
                                                                  streams[primary_stream_index]);

                // gather candidate read ids of active candidates into temp array
                read_number* d_tmp_read_ids = nullptr;
                cubCachingAllocator.DeviceAllocate((void**)&d_tmp_read_ids, dataArrays.n_queries * sizeof(read_number), streams[primary_stream_index]); CUERR;

                call_compact_kernel_async(d_tmp_read_ids,
                                            dataArrays.d_candidate_read_ids.get(),
                                            dataArrays.d_indices,
                                            dataArrays.h_num_indices[0],
                                            streams[primary_stream_index]);

                //gather candidate quality scores
                gpuReadStorage.copyGpuQualityDataToGpuBufferAsync(dataArrays.d_candidate_qualities,
                                                                  dataArrays.quality_pitch,
                                                                  d_tmp_read_ids,
                                                                  dataArrays.h_num_indices[0],
                                                                  deviceIds[0],
                                                                  streams[primary_stream_index]);

                cubCachingAllocator.DeviceFree(d_tmp_read_ids); CUERR;

                cudaEventRecord(events[quality_transfer_finished_event_index], streams[primary_stream_index]); CUERR;

                cudaStreamWaitEvent(streams[secondary_stream_index], events[quality_transfer_finished_event_index], 0); CUERR;

                //copy gathered quality scores to pinned host arrays
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

                //transpose the quality scores

                call_transpose_kernel(dataArrays.d_candidate_qualities_tmp.get(),
                                      dataArrays.d_candidate_qualities.get(),
                                      dataArrays.h_num_indices[0],
                                      dataArrays.maximum_sequence_length,
                                      dataArrays.quality_pitch,
                                      streams[primary_stream_index]);

                std::swap(dataArrays.d_candidate_qualities_tmp, dataArrays.d_candidate_qualities_transposed);


            }else{

                //copy subject qualities to pinned host buffers

                gatherSubjectQualitiesFromHost(dataArrays, gpuReadStorage, 0, dataArrays.n_subjects);

                cudaEventSynchronize(events[num_indices_transfered_event_index]); CUERR;

                cudaEventSynchronize(events[indices_transfer_finished_event_index]); CUERR;

                //copy candidate qualities to pinned host buffers

                gatherCandidateQualitiesFromHost(dataArrays, gpuReadStorage, 0, dataArrays.h_num_indices[0]);

                // copy quality data in pinned buffers to GPU

                cudaMemcpyAsync(dataArrays.d_subject_qualities,
                                dataArrays.h_subject_qualities,
                                dataArrays.h_subject_qualities.sizeInBytes(),
                                H2D,
                                streams[primary_stream_index]);

                cudaMemcpyAsync(dataArrays.d_candidate_qualities,
                                dataArrays.h_candidate_qualities,
                                dataArrays.h_candidate_qualities.sizeInBytes(),
                                H2D,
                                streams[primary_stream_index]);

                //transpose the quality scores

                call_transpose_kernel(dataArrays.d_candidate_qualities_tmp.get(),
                                      dataArrays.d_candidate_qualities.get(),
                                      dataArrays.h_num_indices[0],
                                      dataArrays.maximum_sequence_length,
                                      dataArrays.quality_pitch,
                                      streams[primary_stream_index]);

                std::swap(dataArrays.d_candidate_qualities_tmp, dataArrays.d_candidate_qualities_transposed);

                cudaEventRecord(events[quality_transfer_finished_event_index], streams[primary_stream_index]); CUERR;

            }

        }

        pop_range();

        cudaEventSynchronize(events[num_indices_transfered_event_index]); CUERR;

        if(dataArrays.h_num_indices[0] == 0){
            write_results_to_file(procDataVector[currentProcId].tasks, numActiveTasks, procDataVector[currentProcId].readsFromFile);
            continue;
        }

        const float desiredAlignmentMaxErrorRate = goodAlignmentProperties.maxErrorRate;

        push_range("build_and_minimize_msa", 0);

        build_msa_async(dataArrays.d_msa_column_properties,
                        dataArrays.d_alignment_shifts,
                        dataArrays.d_alignment_best_alignment_flags,
                        dataArrays.d_alignment_overlaps,
                        dataArrays.d_alignment_nOps,
                        dataArrays.d_subject_sequences_data,
                        dataArrays.d_candidate_sequences_data,
                        dataArrays.d_subject_sequences_lengths,
                        dataArrays.d_candidate_sequences_lengths,
                        dataArrays.d_subject_qualities,
                        dataArrays.d_candidate_qualities,
                        //dataArrays.d_candidate_qualities_transposed,
                        dataArrays.d_candidates_per_subject_prefixsum,
                        dataArrays.d_indices,
                        dataArrays.d_indices_per_subject,
                        dataArrays.d_indices_per_subject_prefixsum,
                        dataArrays.d_counts,
                        dataArrays.d_weights,
                        dataArrays.d_coverage,
                        dataArrays.d_support,
                        dataArrays.d_consensus,
                        dataArrays.d_origWeights,
                        dataArrays.d_origCoverages,
                        dataArrays.n_subjects,
                        dataArrays.n_queries,
                        dataArrays.h_num_indices,
                        dataArrays.d_num_indices,
                        1.0f,
                        correctionOptions.useQualityScores,
                        desiredAlignmentMaxErrorRate,
                        dataArrays.maximum_sequence_length,
                        sizeof(unsigned int) * getEncodedNumInts2BitHiLo(dataArrays.maximum_sequence_length),
                        dataArrays.encoded_sequence_pitch,
                        dataArrays.quality_pitch,
                        dataArrays.msa_pitch,
                        dataArrays.msa_weights_pitch,
                        streams[primary_stream_index],
                        kernelLaunchHandle);

        //At this point the msa is built
        cudaEventRecord(events[msa_build_finished_event_index], streams[primary_stream_index]); CUERR;

        cudaEventSynchronize(events[num_indices_transfered_event_index]); CUERR;

        if(dataArrays.h_num_indices[0] == 0){
            write_results_to_file(procDataVector[currentProcId].tasks, numActiveTasks, procDataVector[currentProcId].readsFromFile);
            continue;
        }

    #ifdef USE_MSA_MINIMIZATION

        constexpr int max_num_minimizations = 5;

        int previousNumIndices = 0;

        if(max_num_minimizations > 0){
            for(int numMinimizations = 0; numMinimizations < max_num_minimizations; numMinimizations++){

                cudaEventSynchronize(events[num_indices_transfered_event_index]); CUERR;

                if(numMinimizations > 0 && previousNumIndices == dataArrays.h_num_indices[0]){
                    break;
                }

                if(dataArrays.h_num_indices[0] == 0){
                    break;
                }

                const int currentNumIndices = dataArrays.h_num_indices[0];

                bool* d_shouldBeKept;


                cubCachingAllocator.DeviceAllocate((void**)&d_shouldBeKept, sizeof(bool) * dataArrays.n_queries, streams[primary_stream_index]); CUERR;
                call_fill_kernel_async(d_shouldBeKept, dataArrays.n_queries, true, streams[primary_stream_index]);

                //select candidates which are to be removed
                call_msa_findCandidatesOfDifferentRegion_kernel_async(
                            d_shouldBeKept,
                            dataArrays.d_subject_sequences_data,
                            dataArrays.d_candidate_sequences_data,
                            dataArrays.d_subject_sequences_lengths,
                            dataArrays.d_candidate_sequences_lengths,
                            dataArrays.d_candidates_per_subject_prefixsum,
                            dataArrays.d_alignment_shifts,
                            dataArrays.d_alignment_best_alignment_flags,
                            dataArrays.n_subjects,
                            dataArrays.n_queries,
                            sizeof(unsigned int) * getEncodedNumInts2BitHiLo(dataArrays.maximum_sequence_length),
                            dataArrays.encoded_sequence_pitch,
                            dataArrays.d_consensus,
                            dataArrays.d_counts,
                            dataArrays.d_weights,
                            dataArrays.d_msa_column_properties,
                            dataArrays.msa_pitch,
                            dataArrays.msa_weights_pitch,
                            dataArrays.d_indices,
                            dataArrays.d_indices_per_subject,
                            dataArrays.d_indices_per_subject_prefixsum,
                            correctionOptions.estimatedCoverage,
                            streams[primary_stream_index],
                            kernelLaunchHandle,
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
                if(correctionOptions.useQualityScores){
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

                    std::swap(dataArrays.d_candidate_qualities, dataArrays.d_candidate_qualities_tmp);

                    call_transpose_kernel(dataArrays.d_candidate_qualities_tmp.get(),
                                          dataArrays.d_candidate_qualities.get(),
                                          dataArrays.h_num_indices[0],
                                          dataArrays.maximum_sequence_length,
                                          dataArrays.quality_pitch,
                                          streams[primary_stream_index]);

                    std::swap(dataArrays.d_candidate_qualities_tmp, dataArrays.d_candidate_qualities_transposed);
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
                }

                const float desiredAlignmentMaxErrorRate = goodAlignmentProperties.maxErrorRate;

                build_msa_async(dataArrays.d_msa_column_properties,
                                dataArrays.d_alignment_shifts,
                                dataArrays.d_alignment_best_alignment_flags,
                                dataArrays.d_alignment_overlaps,
                                dataArrays.d_alignment_nOps,
                                dataArrays.d_subject_sequences_data,
                                dataArrays.d_candidate_sequences_data,
                                dataArrays.d_subject_sequences_lengths,
                                dataArrays.d_candidate_sequences_lengths,
                                dataArrays.d_subject_qualities,
                                dataArrays.d_candidate_qualities,
                                //dataArrays.d_candidate_qualities_transposed,
                                dataArrays.d_candidates_per_subject_prefixsum,
                                dataArrays.d_indices,
                                d_indices_per_subject_tmp,
                                dataArrays.d_indices_per_subject_prefixsum,
                                dataArrays.d_counts,
                                dataArrays.d_weights,
                                dataArrays.d_coverage,
                                dataArrays.d_support,
                                dataArrays.d_consensus,
                                dataArrays.d_origWeights,
                                dataArrays.d_origCoverages,
                                dataArrays.n_subjects,
                                dataArrays.n_queries,
                                dataArrays.h_num_indices,
                                dataArrays.d_num_indices,
                                0.05f, //
                                correctionOptions.useQualityScores,
                                desiredAlignmentMaxErrorRate,
                                dataArrays.maximum_sequence_length,
                                sizeof(unsigned int) * getEncodedNumInts2BitHiLo(dataArrays.maximum_sequence_length),
                                dataArrays.encoded_sequence_pitch,
                                dataArrays.quality_pitch,
                                dataArrays.msa_pitch,
                                dataArrays.msa_weights_pitch,
                                streams[primary_stream_index],
                                kernelLaunchHandle);

                cudaMemcpyAsync(dataArrays.h_num_indices, dataArrays.d_num_indices, sizeof(int), D2H, streams[primary_stream_index]);  CUERR;

                cudaEventRecord(events[num_indices_transfered_event_index], streams[primary_stream_index]); CUERR;

                cubCachingAllocator.DeviceFree(d_shouldBeKept); CUERR;
                cubCachingAllocator.DeviceFree(d_newIndices); CUERR;
                cubCachingAllocator.DeviceFree(d_indices_per_subject_tmp); CUERR;
                cubCachingAllocator.DeviceFree(d_shouldBeKept_positions); CUERR;

                previousNumIndices = currentNumIndices;

            }

            if(dataArrays.h_num_indices[0] == 0){
                write_results_to_file(procDataVector[currentProcId].tasks, numActiveTasks, procDataVector[currentProcId].readsFromFile);
                continue; //discard batch
            }

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

    #endif

        cudaEventRecord(events[msa_build_finished_event_index], streams[primary_stream_index]); CUERR;

        pop_range();

        if(correctionOptions.extractFeatures || correctionOptions.correctionType != CorrectionType::Classic) {

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

        if(correctionOptions.extractFeatures){
            //BatchState::WriteFeatures;
            cudaEventSynchronize(events[msadata_transfer_finished_event_index]); CUERR;

            for(std::size_t subject_index = 0; subject_index < numActiveTasks; ++subject_index) {

                const auto& task = tasks[subject_index];
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
                            correctionOptions.kmerlength, 0.0f,
                            correctionOptions.estimatedCoverage);
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
                                            correctionOptions.estimatedCoverage);

    #endif
                for(const auto& msafeature : MSAFeatures) {
                    featurestream << task.readId << '\t' << msafeature.position << '\t' << msafeature.consensus << '\n';
                    featurestream << msafeature << '\n';
                }
            }

            continue; //batch finished.
        }else{

            push_range("correct", 1);

            switch(correctionOptions.correctionType){
            default:
            case CorrectionType::Classic:
                {
                    //BatchState::StartClassicCorrection;

                    const float avg_support_threshold = 1.0f-1.0f*correctionOptions.estimatedErrorrate;
                    const float min_support_threshold = 1.0f-3.0f*correctionOptions.estimatedErrorrate;
                    // coverage is always >= 1
                    const float min_coverage_threshold = std::max(1.0f,
                                correctionOptions.m_coverage / 6.0f * correctionOptions.estimatedCoverage);
                    const int new_columns_to_correct = correctionOptions.new_columns_to_correct;

                    // correct subjects

                    call_msa_correct_subject_implicit_kernel_async(
                                dataArrays.d_consensus,
                                dataArrays.d_support,
                                dataArrays.d_coverage,
                                dataArrays.d_origCoverages,
                                dataArrays.d_msa_column_properties,
                                dataArrays.d_subject_sequences_data,
                                dataArrays.d_is_high_quality_subject,
                                dataArrays.d_corrected_subjects,
                                dataArrays.d_subject_is_corrected,
                                dataArrays.n_subjects,
                                dataArrays.encoded_sequence_pitch,
                                dataArrays.sequence_pitch,
                                dataArrays.msa_pitch,
                                dataArrays.msa_weights_pitch,
                                correctionOptions.estimatedErrorrate,
                                avg_support_threshold,
                                min_support_threshold,
                                min_coverage_threshold,
                                correctionOptions.kmerlength,
                                dataArrays.maximum_sequence_length,
                                streams[primary_stream_index],
                                kernelLaunchHandle);

                    if(correctionOptions.correctCandidates) {


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
                                dataArrays.d_consensus,
                                dataArrays.d_support,
                                dataArrays.d_coverage,
                                dataArrays.d_origCoverages,
                                dataArrays.d_msa_column_properties,
                                dataArrays.d_candidate_sequences_lengths,
                                dataArrays.d_indices,
                                dataArrays.d_indices_per_subject,
                                dataArrays.d_indices_per_subject_prefixsum,
                                dataArrays.d_high_quality_subject_indices,
                                dataArrays.d_num_high_quality_subject_indices,
                                dataArrays.d_alignment_shifts,
                                dataArrays.d_alignment_best_alignment_flags,
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
                                streams[primary_stream_index],
                                kernelLaunchHandle);
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

                    if(correctionOptions.correctCandidates){
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


                    if(nextBatchNum < numbatches && numProc > 1){
                        if(numProc > 1){
                            workOnNextBatchIfPossibleWhileEventNotReady(events[correction_finished_event_index], procDataVector[nextProcId]);
                            workOnNextBatchIfPossibleWhileEventNotReady(events[indices_transfer_finished_event_index], procDataVector[nextProcId]);
                            workOnNextBatchIfPossibleWhileEventNotReady(events[result_transfer_finished_event_index], procDataVector[nextProcId]);

                            cudaEventSynchronize(events[correction_finished_event_index]); CUERR;
                            cudaEventSynchronize(events[indices_transfer_finished_event_index]); CUERR;
                            cudaEventSynchronize(events[result_transfer_finished_event_index]); CUERR;
                        }
                    }else{
                        cudaEventSynchronize(events[correction_finished_event_index]); CUERR;
                        cudaEventSynchronize(events[indices_transfer_finished_event_index]); CUERR;
                        cudaEventSynchronize(events[result_transfer_finished_event_index]); CUERR;
                    }

                    #pragma omp parallel for
                    for(std::size_t subject_index = 0; subject_index < numActiveTasks; ++subject_index) {
                        auto& task = tasks[subject_index];
                        const char* const my_corrected_subject_data = dataArrays.h_corrected_subjects + subject_index * dataArrays.sequence_pitch;
                        task.corrected = dataArrays.h_subject_is_corrected[subject_index];
                        //if(task.readId == 207){
                        //    std::cerr << "\n\ncorrected: " << task.corrected << "\n";
                        //}
                        if(task.corrected) {
                            const int subject_length = dataArrays.h_subject_sequences_lengths[subject_index];//task.subject_string.length();//dataArrays.h_subject_sequences_lengths[subject_index];
                            task.corrected_subject = std::move(std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length});

                            //if(task.readId == 207){
                            //    std::cerr << "\n\ncorrected sequence: " << task.corrected_subject << "\n";
                            //}
                        }
                    }

                    if(correctionOptions.correctCandidates) {

                        #pragma omp parallel for schedule(dynamic, 4)
                        for(std::size_t subject_index = 0; subject_index < numActiveTasks; ++subject_index) {
                            auto& task = tasks[subject_index];
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
                                const int candidate_length = dataArrays.h_candidate_sequences_lengths[global_candidate_index];//gpuReadStorage.fetchSequenceLength(candidate_read_id);

                                const char* const candidate_data = my_corrected_candidates_data + i * dataArrays.sequence_pitch;

                                task.corrected_candidates_read_ids[i] = candidate_read_id;
                                task.corrected_candidates[i] = std::move(std::string{candidate_data, candidate_data + candidate_length});

                                //task.corrected_candidates_read_ids.emplace_back(candidate_read_id);
                                //task.corrected_candidates.emplace_back(std::move(std::string{candidate_data, candidate_data + candidate_length}));
                            }
                        }
                    }

                }

                break;
            case CorrectionType::Forest:
                {
                    //BatchState::StartForestCorrection;

                    cudaEventSynchronize(events[msadata_transfer_finished_event_index]);

                    std::vector<MSAFeature> MSAFeatures;
                    std::vector<int> MSAFeaturesPerSubject(numActiveTasks);
                    std::vector<int> MSAFeaturesPerSubjectPrefixSum(numActiveTasks+1);

                    for(std::size_t subject_index = 0; subject_index < numActiveTasks; ++subject_index) {
                        auto& task = tasks[subject_index];

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
                                                            correctionOptions.kmerlength, 0.0f,
                                                            correctionOptions.estimatedCoverage);

                        MSAFeatures.insert(MSAFeatures.end(), tmpfeatures.begin(), tmpfeatures.end());
                        MSAFeaturesPerSubject[subject_index] = tmpfeatures.size();

                    }

                    MSAFeaturesPerSubjectPrefixSum[0] = 0;
                    std::partial_sum(MSAFeaturesPerSubject.begin(), MSAFeaturesPerSubject.end(), MSAFeaturesPerSubjectPrefixSum.begin()+1);

                    constexpr float maxgini = 0.05f;
                    constexpr float forest_correction_fraction = 0.5f;

                    #pragma omp parallel for schedule(dynamic,2)
                    for(std::size_t subject_index = 0; subject_index < numActiveTasks; ++subject_index) {
                        auto& task = tasks[subject_index];
                        task.corrected_subject = task.subject_string;

                        const int offset = MSAFeaturesPerSubjectPrefixSum[subject_index];
                        const int end_index = offset + MSAFeaturesPerSubject[subject_index];

                        const char* const consensus = &dataArrays.h_consensus[subject_index * dataArrays.msa_pitch];
                        const auto& columnProperties = dataArrays.h_msa_column_properties[subject_index];

                        for(int index = offset; index < end_index; index++){
                            const auto& msafeature = MSAFeatures[index];

                            const bool doCorrect = fc.shouldCorrect(msafeature.position_support,
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

                }

                break;
            case CorrectionType::Convnet:
                {
                    //BatchState::StartConvnetCorrection;

                    cudaEventSynchronize(events[msadata_transfer_finished_event_index]); CUERR;

                    std::vector<MSAFeature3> MSAFeatures;
                    std::vector<int> MSAFeaturesPerSubject(numActiveTasks);
                    std::vector<int> MSAFeaturesPerSubjectPrefixSum(numActiveTasks+1);
                    MSAFeaturesPerSubjectPrefixSum[0] = 0;

                    for(std::size_t subject_index = 0; subject_index < numActiveTasks; ++subject_index) {

                        auto& task = tasks[subject_index];

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
                                                    correctionOptions.estimatedCoverage);
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
                                                    correctionOptions.useQualityScores,
                                                    dataArrays.h_consensus + subject_index * dataArrays.msa_pitch,
                                                    dataArrays.h_support + subject_index * msa_weights_pitch_floats,
                                                    dataArrays.h_coverage + subject_index * msa_weights_pitch_floats,
                                                    dataArrays.h_origCoverages + subject_index * msa_weights_pitch_floats,
                                                    columnProperties.subjectColumnsBegin_incl,
                                                    columnProperties.subjectColumnsEnd_excl,
                                                    task.subject_string,
                                                    correctionOptions.estimatedCoverage,
                                                    true,
                                                    dataArrays.msa_pitch,
                                                    msa_weights_pitch_floats);
            #endif
                        MSAFeatures.insert(MSAFeatures.end(), tmpfeatures.begin(), tmpfeatures.end());
                        MSAFeaturesPerSubject[subject_index] = tmpfeatures.size();
                    }

                    std::partial_sum(MSAFeaturesPerSubject.begin(), MSAFeaturesPerSubject.end(),MSAFeaturesPerSubjectPrefixSum.begin()+1);

                    std::vector<float> predictions = nnClassifier.infer(MSAFeatures);
                    assert(predictions.size() == MSAFeatures.size());

                    for(std::size_t subject_index = 0; subject_index < numActiveTasks; ++subject_index) {
                        auto& task = tasks[subject_index];
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

                }

                break;
            }

            pop_range();


            //write result to file

            write_results_to_file(procDataVector[currentProcId].tasks, numActiveTasks, procDataVector[currentProcId].readsFromFile);

        }

        pop_range();
    }

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


    for(auto& array : dataArraysVector) {
        array.reset();
    }

    for(auto& streamarray : streamsVector) {
        for(auto& stream : streamarray)
            cudaStreamDestroy(stream); CUERR;
    }

    for(auto& eventarray : eventsVector) {
        for(auto& event : eventarray)
            cudaEventDestroy(event); CUERR;
    }

    omp_set_num_threads(oldNumOMPThreads);


    size_t occupiedMemory = minhasher.numBytes() + cpuReadStorage.size();

    minhasher.destroy();
    cpuReadStorage.destroy();
    gpuReadStorage.destroy();

    #ifndef DO_PROFILE

    //if candidate correction is enabled, only the read id and corrected sequence of corrected reads is written to outputfile
    //outputfile needs to be sorted by read id
    //then, the corrected reads from the output file have to be merged with the original input file to get headers, uncorrected reads, and quality scores
    if(correctionOptions.correctCandidates){

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
