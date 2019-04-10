#include <gpu/gpu_correction_thread.hpp>

#include <gpu/nvvptimelinemarkers.hpp>

#include <hpc_helpers.cuh>
#include <options.hpp>
#include <tasktiming.hpp>
#include <sequence.hpp>
#include <featureextractor.hpp>
#include <forestclassifier.hpp>
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
#include <queue>
#include <unordered_map>

#ifdef __NVCC__
#include <cub/cub.cuh>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#endif

//#define CARE_GPU_DEBUG
//#define CARE_GPU_DEBUG_MEMCOPY
//#define CARE_GPU_DEBUG_PRINT_ARRAYS
//#define CARE_GPU_DEBUG_PRINT_MSA

//MSA_IMPLICIT does not work yet
//#define MSA_IMPLICIT

//#define USE_WAIT_FLAGS

#define shd_tilesize 32


constexpr int nParallelBatches = 1;
constexpr int sideBatchStepsPerWaitIter = 1;

namespace care{
namespace gpu{

    bool ErrorCorrectionThreadOnlyGPU::Batch::isWaiting() const{
        #ifdef USE_WAIT_FLAGS
            return 0 != waitCounts[activeWaitIndex].load();
        #else
            return cudaEventQuery((*events)[activeWaitIndex]) == cudaErrorNotReady;
        #endif
	}

    void ErrorCorrectionThreadOnlyGPU::Batch::addWaitSignal(ErrorCorrectionThreadOnlyGPU::BatchState state, cudaStream_t stream){
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
        mycase(BatchState::CopyQualities)
        mycase(BatchState::BuildMSA)
        mycase(BatchState::StartClassicCorrection)
        mycase(BatchState::StartForestCorrection)
        mycase(BatchState::UnpackClassicResults)
        mycase(BatchState::WriteResults)
        mycase(BatchState::WriteFeatures)
        mycase(BatchState::Finished)
        mycase(BatchState::Aborted)
        default: assert(false);
        }

        #undef mycase
        #undef handlethis

        /*auto dataptr = std::make_unique<WaitCallbackData>(this, wait_index);

        auto waitsuccessfunc = [](void* d){
            const WaitCallbackData* const data = static_cast<const WaitCallbackData*>(d);
            Batch* const b = data->b;
            b->waitCounts[data->index]--;
        };

        cudaLaunchHostFunc(stream, waitsuccessfunc, (void*)dataptr.get()); CUERR;
        callbackDataList.emplace_back(dataptr);*/

        /*if(wait_index == wait_before_copyqualites_index){
            auto waitsuccessfunc = [](void* batch){
                Batch* b = static_cast<Batch*>(batch);
                b->waitCounts[wait_before_copyqualites_index]--;
            };

            cudaLaunchHostFunc(stream, waitsuccessfunc, (void*)this); CUERR;
        }else if(wait_index == wait_before_unpackclassicresults_index){
            auto waitsuccessfunc = [](void* batch){
                Batch* b = static_cast<Batch*>(batch);
                b->waitCounts[wait_before_unpackclassicresults_index]--;
            };

            cudaLaunchHostFunc(stream, waitsuccessfunc, (void*)this); CUERR;
        }else if(wait_index == wait_before_startforestcorrection_index){
            auto waitsuccessfunc = [](void* batch){
                Batch* b = static_cast<Batch*>(batch);
                b->waitCounts[wait_before_startforestcorrection_index]--;
            };

            cudaLaunchHostFunc(stream, waitsuccessfunc, (void*)this); CUERR;
        }else{
            assert(false); //every case should be handled above
        }*/


    }

	void ErrorCorrectionThreadOnlyGPU::Batch::reset(){
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
	}

    void ErrorCorrectionThreadOnlyGPU::Batch::waitUntilAllCallbacksFinished() const{
        assert(std::any_of(waitCounts.begin(), waitCounts.end(), [](const auto& i){return i >= 0;}));

        while(std::any_of(waitCounts.begin(), waitCounts.end(), [](const auto& i){return i > 0;})){
            ;
        }
    }

	void ErrorCorrectionThreadOnlyGPU::run(){
		if(isRunning) throw std::runtime_error("ErrorCorrectionThreadOnlyGPU::run: Is already running.");
		isRunning = true;
		thread = std::move(std::thread(&ErrorCorrectionThreadOnlyGPU::execute, this));
	}

	void ErrorCorrectionThreadOnlyGPU::join(){
		thread.join();
		isRunning = false;
	}

	std::string ErrorCorrectionThreadOnlyGPU::nameOf(const BatchState& state) const {
		switch(state) {
		case BatchState::Unprepared: return "Unprepared";
		case BatchState::CopyReads: return "CopyReads";
		case BatchState::StartAlignment: return "StartAlignment";
		case BatchState::CopyQualities: return "CopyQualities";
		case BatchState::BuildMSA: return "BuildMSA";
		case BatchState::StartClassicCorrection: return "StartClassicCorrection";
		case BatchState::StartForestCorrection: return "StartForestCorrection";
		case BatchState::UnpackClassicResults: return "UnpackClassicResults";
		case BatchState::WriteResults: return "WriteResults";
		case BatchState::WriteFeatures: return "WriteFeatures";
		case BatchState::Finished: return "Finished";
		case BatchState::Aborted: return "Aborted";
		default: assert(false); return "None";
		}
	}

	void ErrorCorrectionThreadOnlyGPU::makeTransitionFunctionTable(){
		transitionFunctionTable[BatchState::Unprepared] = state_unprepared_func2;
		transitionFunctionTable[BatchState::CopyReads] = state_copyreads_func2;
		transitionFunctionTable[BatchState::StartAlignment] = state_startalignment_func;
		transitionFunctionTable[BatchState::CopyQualities] = state_copyqualities_func;
		transitionFunctionTable[BatchState::BuildMSA] = state_buildmsa_func;
		transitionFunctionTable[BatchState::StartClassicCorrection] = state_startclassiccorrection_func;
		transitionFunctionTable[BatchState::StartForestCorrection] = state_startforestcorrection_func;
		transitionFunctionTable[BatchState::UnpackClassicResults] = state_unpackclassicresults_func;
		transitionFunctionTable[BatchState::WriteResults] = state_writeresults_func;
		transitionFunctionTable[BatchState::WriteFeatures] = state_writefeatures_func;
		transitionFunctionTable[BatchState::Finished] = state_finished_func;
		transitionFunctionTable[BatchState::Aborted] = state_aborted_func;
	}

	ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_unprepared_func(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::Unprepared);
		assert((batch.initialNumberOfCandidates == 0 && batch.tasks.empty()) || batch.initialNumberOfCandidates > 0);

		const int hits_per_candidate = transFuncData.correctionOptions.hits_per_candidate;

		DataArrays& dataArrays = *batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		//std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		std::vector<read_number>* readIdBuffer = transFuncData.readIdBuffer;

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

			read_number id = readIdBuffer->back();
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
                            Sequence_t::getNumBytes(transFuncData.maxSequenceLength),
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

                cub::DeviceScan::InclusiveSum((void*)nullptr, temp_storage_bytes, (int*)nullptr,
							(int*)nullptr,
							dataArrays.n_subjects,
							streams[primary_stream_index]); CUERR;

				max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

				temp_storage_bytes = max_temp_storage_bytes;

				//dataArrays.set_tmp_storage_size(max_temp_storage_bytes);
				dataArrays.zero_gpu(streams[primary_stream_index]);

				auto roundToNextMultiple = [](int num, int factor){
					return SDIV(num, factor) * factor;
				};

				const int minOverlapForMaxSeqLength = std::max(1,
														std::max(transFuncData.min_overlap,
																	int(transFuncData.maxSequenceLength * transFuncData.min_overlap_ratio)));
				const int msa_max_column_count = (3*transFuncData.maxSequenceLength - 2*minOverlapForMaxSeqLength);

				batch.batchDataDevice.cubTemp.resize(max_temp_storage_bytes);
				batch.batchDataDevice.resize(int(batch.tasks.size()),
											batch.initialNumberOfCandidates,
											roundToNextMultiple(transFuncData.maxSequenceLength, 4),
											roundToNextMultiple(Sequence_t::getNumBytes(transFuncData.maxSequenceLength), 4),
											transFuncData.maxSequenceLength,
											Sequence_t::getNumBytes(transFuncData.maxSequenceLength), msa_max_column_count);

				batch.batchDataHost.resize(int(batch.tasks.size()),
											batch.initialNumberOfCandidates,
											roundToNextMultiple(transFuncData.maxSequenceLength, 4),
											roundToNextMultiple(Sequence_t::getNumBytes(transFuncData.maxSequenceLength), 4),
											transFuncData.maxSequenceLength,
											Sequence_t::getNumBytes(transFuncData.maxSequenceLength), msa_max_column_count);

				batch.initialNumberOfCandidates = 0;

				return BatchState::CopyReads;
			}
		}
	}




    ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_unprepared_func2(ErrorCorrectionThreadOnlyGPU::Batch& batch,
                bool canBlock,
                bool canLaunchKernel,
                bool isPausable,
                const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

        assert(batch.state == BatchState::Unprepared);
        assert((batch.initialNumberOfCandidates == 0 && batch.tasks.empty()) || batch.initialNumberOfCandidates > 0);

        const int hits_per_candidate = transFuncData.correctionOptions.hits_per_candidate;

        DataArrays& dataArrays = *batch.dataArrays;
        std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
        //std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

        std::vector<read_number>* readIdBuffer = transFuncData.readIdBuffer;
        std::vector<CorrectionTask_t>* tmptasksBuffer = transFuncData.tmptasksBuffer;

        auto erase_from_range = [](auto begin, auto end, auto position_to_erase){
                        auto copybegin = position_to_erase;
                        std::advance(copybegin, 1);
                        return std::copy(copybegin, end, position_to_erase);
                    };

        dataArrays.allocCandidateIds(transFuncData.minimum_candidates_per_batch + transFuncData.max_candidates);

        const auto* gpuReadStorage = transFuncData.gpuReadStorage;
        const auto& minhasher = transFuncData.minhasher;


        constexpr int num_simultaneous_tasks = 64;

        std::vector<CorrectionTask_t> tmptasks;
        //std::vector<bool> tmpokflags(num_simultaneous_tasks);

        while(batch.initialNumberOfCandidates < transFuncData.minimum_candidates_per_batch
              && !(transFuncData.readIdGenerator->empty() && readIdBuffer->empty() && tmptasksBuffer->empty())) {



            if(tmptasksBuffer->empty()){

                if(readIdBuffer->empty())
                    *readIdBuffer = transFuncData.readIdGenerator->next_n(1000);

                if(readIdBuffer->empty())
                    continue;

                const int readIdsInBuffer = readIdBuffer->size();
                const int max_tmp_tasks = std::min(readIdsInBuffer, num_simultaneous_tasks);

                tmptasks.resize(max_tmp_tasks);

                #pragma omp parallel for num_threads(4)
                for(int tmptaskindex = 0; tmptaskindex < max_tmp_tasks; tmptaskindex++){
                    auto& task = tmptasks[tmptaskindex];

                    const read_number readId = (*readIdBuffer)[readIdsInBuffer - 1 - tmptaskindex];
                    task = CorrectionTask_t(readId);

                    bool ok = false;
                    if ((*transFuncData.readIsCorrectedVector)[readId] == 0) {
                        ok = true;
                    }

                    if(ok){
                        const char* sequenceptr = gpuReadStorage->fetchSequenceData_ptr(readId);
                        const int sequencelength = gpuReadStorage->fetchSequenceLength(readId);

                        task.subject_string = Sequence_t::Impl_t::toString((const std::uint8_t*)sequenceptr, sequencelength);
                        task.candidate_read_ids = minhasher->getCandidates(task.subject_string, hits_per_candidate, transFuncData.max_candidates);

                        task.candidate_read_ids_begin = &(task.candidate_read_ids[0]);
                        task.candidate_read_ids_end = &(task.candidate_read_ids[task.candidate_read_ids.size()]);

                        auto readIdPos = std::lower_bound(task.candidate_read_ids_begin, task.candidate_read_ids_end, task.readId);

                        if(readIdPos != task.candidate_read_ids_end && *readIdPos == task.readId) {

                            task.candidate_read_ids_end = erase_from_range(task.candidate_read_ids_begin, task.candidate_read_ids_end, readIdPos);
                            task.candidate_read_ids.resize(std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end));
                        }

                        std::size_t myNumCandidates = std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end);

                        assert(myNumCandidates <= std::size_t(transFuncData.max_candidates));

                        if(myNumCandidates == 0) {
                            task.active = false;
                        }
                    }else{
                        task.active = false;
                    }
                }

                for(int tmptaskindex = 0; tmptaskindex < max_tmp_tasks; tmptaskindex++){
                    readIdBuffer->pop_back();
                }

                std::swap(*tmptasksBuffer, tmptasks);

                //only perform one iteration if pausable
                if(isPausable)
                    break;
            }

            while(batch.initialNumberOfCandidates < transFuncData.minimum_candidates_per_batch
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
                        const size_t myNumCandidates = std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end);

                        batch.tasks.emplace_back(task);
                        batch.initialNumberOfCandidates += int(myNumCandidates);
                    }
                }

                tmptasksBuffer->pop_back();
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
                            Sequence_t::getNumBytes(transFuncData.maxSequenceLength),
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

                cub::DeviceScan::InclusiveSum((void*)nullptr, temp_storage_bytes, (int*)nullptr,
                            (int*)nullptr,
                            dataArrays.n_subjects,
                            streams[primary_stream_index]); CUERR;

                max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

                temp_storage_bytes = max_temp_storage_bytes;

                //dataArrays.set_tmp_storage_size(max_temp_storage_bytes);
                dataArrays.zero_gpu(streams[primary_stream_index]);

                auto roundToNextMultiple = [](int num, int factor){
                    return SDIV(num, factor) * factor;
                };

                const int minOverlapForMaxSeqLength = std::max(1,
                                                        std::max(transFuncData.min_overlap,
                                                                    int(transFuncData.maxSequenceLength * transFuncData.min_overlap_ratio)));
                const int msa_max_column_count = (3*transFuncData.maxSequenceLength - 2*minOverlapForMaxSeqLength);

                batch.batchDataDevice.cubTemp.resize(max_temp_storage_bytes);
                batch.batchDataDevice.resize(int(batch.tasks.size()),
                                            batch.initialNumberOfCandidates,
                                            roundToNextMultiple(transFuncData.maxSequenceLength, 4),
                                            roundToNextMultiple(Sequence_t::getNumBytes(transFuncData.maxSequenceLength), 4),
                                            transFuncData.maxSequenceLength,
                                            Sequence_t::getNumBytes(transFuncData.maxSequenceLength), msa_max_column_count);

                batch.batchDataHost.resize(int(batch.tasks.size()),
                                            batch.initialNumberOfCandidates,
                                            roundToNextMultiple(transFuncData.maxSequenceLength, 4),
                                            roundToNextMultiple(Sequence_t::getNumBytes(transFuncData.maxSequenceLength), 4),
                                            transFuncData.maxSequenceLength,
                                            Sequence_t::getNumBytes(transFuncData.maxSequenceLength), msa_max_column_count);

                batch.initialNumberOfCandidates = 0;

                return BatchState::CopyReads;
            }
        }
    }
















	ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_copyreads_func(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::CopyReads);
		assert(batch.copiedTasks <= int(batch.tasks.size()));

		DataArrays& dataArrays = *batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		dataArrays.h_candidates_per_subject_prefixsum[0] = 0;
        dataArrays.h_tiles_per_subject_prefixsum[0] = 0;
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

			std::vector<read_number> indexList(batch.numsortedCandidateIds);
			std::iota(indexList.begin(), indexList.end(), read_number(0));
			std::sort(indexList.begin(), indexList.end(),
						[&](auto index1, auto index2){
						return batch.allReadIdsOfTasks[index1] < batch.allReadIdsOfTasks[index2];
					});

			std::vector<char> readsSortedById(dataArrays.encoded_sequence_pitch * batch.numsortedCandidateIds, 0);

			constexpr std::size_t prefetch_distance = 4;

			for(int i = 0; i < batch.numsortedCandidateIds && i < prefetch_distance; ++i) {
				const read_number next_candidate_read_id = batch.allReadIdsOfTasks[indexList[i]];
				const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
				__builtin_prefetch(nextsequenceptr, 0, 0);
			}

			const std::size_t candidatesequencedatabytes = dataArrays.memQueries / sizeof(char);

			for(int i = 0; i < batch.numsortedCandidateIds; ++i) {
				if(i + prefetch_distance < batch.numsortedCandidateIds) {
					const read_number next_candidate_read_id = batch.allReadIdsOfTasks[indexList[i + prefetch_distance]];
					const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
					__builtin_prefetch(nextsequenceptr, 0, 0);
				}

				const read_number candidate_read_id = batch.allReadIdsOfTasks[indexList[i]];
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

            arrays.h_subject_read_ids[batch.copiedTasks] = task.readId;

			if(transFuncData.readStorageGpuData.isValidSequenceData()) {
				//copy read Ids

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

				const std::size_t h_candidate_read_ids_size = arrays.memCandidateIds / sizeof(read_number);

				assert(std::size_t(batch.copiedCandidates) + std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end) <= h_candidate_read_ids_size);

				std::copy(task.candidate_read_ids_begin, task.candidate_read_ids_end, arrays.h_candidate_read_ids + batch.copiedCandidates);

				batch.copiedCandidates += std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end);

				//update prefix sum
                const int numcandidates = int(std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end));
				arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks+1]
				        = arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks]
				          + numcandidates;
                arrays.h_tiles_per_subject_prefixsum[batch.copiedTasks+1]
                        = arrays.h_tiles_per_subject_prefixsum[batch.copiedTasks]
                            + SDIV(numcandidates, shd_tilesize);
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
					const read_number next_candidate_read_id = task.candidate_read_ids[i];
					const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
					__builtin_prefetch(nextsequenceptr, 0, 0);
				}

				const std::size_t candidatesequencedatabytes = arrays.memQueries / sizeof(char);

                #pragma omp parallel for num_threads(4)
				for(std::size_t i = 0; i < task.candidate_read_ids.size(); ++i) {
					if(i + prefetch_distance < task.candidate_read_ids.size()) {
						const read_number next_candidate_read_id = task.candidate_read_ids[i + prefetch_distance];
						const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
						__builtin_prefetch(nextsequenceptr, 0, 0);
					}

					const read_number candidate_read_id = task.candidate_read_ids[i];
					const char* sequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(candidate_read_id);
					const int sequencelength = transFuncData.gpuReadStorage->fetchSequenceLength(candidate_read_id);

					assert(batch.copiedCandidates * arrays.encoded_sequence_pitch + Sequence_t::getNumBytes(sequencelength) <= candidatesequencedatabytes);

                    const int global_i = batch.copiedCandidates + i;

					std::memcpy(arrays.h_candidate_sequences_data
								+ global_i * arrays.encoded_sequence_pitch,
								sequenceptr,
								Sequence_t::getNumBytes(sequencelength));

					arrays.h_candidate_sequences_lengths[global_i] = sequencelength;
				}

                batch.copiedCandidates += task.candidate_read_ids.size();
#else

				constexpr std::size_t prefetch_distance = 4;

				for(std::size_t i = 0; i < task.candidate_read_ids.size() && i < prefetch_distance; ++i) {
					const read_number next_candidate_read_id = task.candidate_read_ids[i];
					const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
					__builtin_prefetch(nextsequenceptr, 0, 0);
				}

				const std::size_t candidatesequencedatabytes = arrays.memQueries / sizeof(char);

				for(std::size_t i = 0; i < task.candidate_read_ids.size(); ++i) {
					if(i + prefetch_distance < task.candidate_read_ids.size()) {
						const read_number next_candidate_read_id = task.candidate_read_ids[i + prefetch_distance];
						const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
						__builtin_prefetch(nextsequenceptr, 0, 0);
					}

					const read_number candidate_read_id = task.candidate_read_ids[i];
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
                const int numcandidates = int(std::distance(task.candidate_read_ids_begin, task.candidate_read_ids_end));
				arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks+1]
				        = arrays.h_candidates_per_subject_prefixsum[batch.copiedTasks]
                            + numcandidates;

                arrays.h_tiles_per_subject_prefixsum[batch.copiedTasks+1]
                        = arrays.h_tiles_per_subject_prefixsum[batch.copiedTasks]
                            + SDIV(numcandidates, shd_tilesize);
#endif
			}

			++batch.copiedTasks;

			if(isPausable)
				break;
		}

		//if batch is fully copied, transfer to gpu
		if(batch.copiedTasks == int(batch.tasks.size())) {
			assert(batch.copiedTasks == int(batch.tasks.size()));

            /*thrust::counting_iterator<int> countiterbegin(0);

            const int nUniqueSequences = std::inner_product(
                                                    countiterbegin,
                                                    countiterbegin + batch.copiedCandidates - 1,
                                                    countiterbegin + 1,
                                                    int(1),
                                                    thrust::plus<int>(),
                                                    [&](auto l, auto r){
                                                        return 0 != std::memcmp(dataArrays.h_candidate_sequences_data + l * dataArrays.encoded_sequence_pitch,
                                                                                dataArrays.h_candidate_sequences_data + r * dataArrays.encoded_sequence_pitch,
                                                                                dataArrays.encoded_sequence_pitch);
                                                    });

            std::cout << "Batch candidates: " << batch.copiedCandidates << ", unique: " << nUniqueSequences << std::endl;*/



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
                cudaMemcpyAsync(dataArrays.d_tiles_per_subject_prefixsum,
							dataArrays.h_tiles_per_subject_prefixsum,
							dataArrays.memTilesPrefixSum,
							H2D,
							streams[primary_stream_index]); CUERR;

                transFuncData.gpuReadStorage->copyGpuLengthsToGpuBufferAsync(dataArrays.d_subject_sequences_lengths,
                                                                             dataArrays.d_subject_read_ids,
                                                                             dataArrays.n_subjects,
                                                                             transFuncData.threadOpts.deviceId, streams[primary_stream_index]);

                transFuncData.gpuReadStorage->copyGpuLengthsToGpuBufferAsync(dataArrays.d_candidate_sequences_lengths,
                                                                             dataArrays.d_candidate_read_ids,
                                                                             dataArrays.n_queries,
                                                                             transFuncData.threadOpts.deviceId, streams[primary_stream_index]);

                transFuncData.gpuReadStorage->copyGpuSequenceDataToGpuBufferAsync(dataArrays.d_subject_sequences_data,
                                                                             dataArrays.encoded_sequence_pitch,
                                                                             dataArrays.d_subject_read_ids,
                                                                             dataArrays.n_subjects,
                                                                             transFuncData.threadOpts.deviceId, streams[primary_stream_index]);

                transFuncData.gpuReadStorage->copyGpuSequenceDataToGpuBufferAsync(dataArrays.d_candidate_sequences_data,
                                                                             dataArrays.encoded_sequence_pitch,
                                                                             dataArrays.d_candidate_read_ids,
                                                                             dataArrays.n_queries,
                                                                             transFuncData.threadOpts.deviceId, streams[primary_stream_index]);

			}else{

				/*int memcmpresult = std::memcmp(batch.collectedCandidateReads.data(),
				                                dataArrays.h_candidate_sequences_data,
				                                dataArrays.encoded_sequence_pitch * batch.numsortedCandidateIds);
				   if(memcmpresult == 0){
				    std::cerr << "ok\n";
				   }else{
				    std::cerr << "not ok\n";
				   }*/
                
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



#if 1
    ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_copyreads_func2(ErrorCorrectionThreadOnlyGPU::Batch& batch,
                bool canBlock,
                bool canLaunchKernel,
                bool isPausable,
                const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

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
                dataArrays.h_candidates_per_subject_prefixsum[i+1] = dataArrays.h_candidates_per_subject_prefixsum[i] + num;
            }

            dataArrays.h_tiles_per_subject_prefixsum[0] = 0;
            for(size_t i = 0; i < batch.tasks.size(); i++){
                const size_t num = batch.tasks[i].candidate_read_ids.size();
                dataArrays.h_tiles_per_subject_prefixsum[i+1] = dataArrays.h_tiles_per_subject_prefixsum[i] + SDIV(num, shd_tilesize);
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
                            
            cudaMemcpyAsync(dataArrays.d_tiles_per_subject_prefixsum,
                            dataArrays.h_tiles_per_subject_prefixsum,
                            dataArrays.memTilesPrefixSum,
                            H2D,
                            streams[primary_stream_index]); CUERR;

            batch.handledReadIds = true;
        }

        if(transFuncData.readStorageGpuData.isValidSequenceData()) {

            

            transFuncData.gpuReadStorage->copyGpuLengthsToGpuBufferAsync(dataArrays.d_subject_sequences_lengths,
                                                                         dataArrays.d_subject_read_ids,
                                                                         dataArrays.n_subjects,
                                                                         transFuncData.threadOpts.deviceId, streams[primary_stream_index]);

            transFuncData.gpuReadStorage->copyGpuLengthsToGpuBufferAsync(dataArrays.d_candidate_sequences_lengths,
                                                                         dataArrays.d_candidate_read_ids,
                                                                         dataArrays.n_queries,
                                                                         transFuncData.threadOpts.deviceId, streams[primary_stream_index]);

            transFuncData.gpuReadStorage->copyGpuSequenceDataToGpuBufferAsync(dataArrays.d_subject_sequences_data,
                                                                         dataArrays.encoded_sequence_pitch,
                                                                         dataArrays.d_subject_read_ids,
                                                                         dataArrays.n_subjects,
                                                                         transFuncData.threadOpts.deviceId, streams[primary_stream_index]);

            transFuncData.gpuReadStorage->copyGpuSequenceDataToGpuBufferAsync(dataArrays.d_candidate_sequences_data,
                                                                         dataArrays.encoded_sequence_pitch,
                                                                         dataArrays.d_candidate_read_ids,
                                                                         dataArrays.n_queries,
                                                                         transFuncData.threadOpts.deviceId, streams[primary_stream_index]);

        }else{
            constexpr int subjectschunksize = 1000;
            constexpr int candidateschunksize = 1000;
            constexpr int prefetch_distance = 4;

            const int firstSubjectIndex = batch.copiedSubjects;
            const int lastSubjectIndexExcl = dataArrays.n_subjects;
            const int subjectChunks = SDIV((lastSubjectIndexExcl - firstSubjectIndex), subjectschunksize);

            const std::size_t subjectsequencedatabytes = dataArrays.memSubjects / sizeof(char);
            const std::size_t candidatesequencedatabytes = dataArrays.memQueries / sizeof(char);

            for(int chunkId = 0; chunkId < subjectChunks; chunkId++){
                const int chunkoffset = chunkId * subjectschunksize;
                const int loop_begin = firstSubjectIndex + chunkoffset;
                const int loop_end_excl = std::min(loop_begin + subjectschunksize, lastSubjectIndexExcl);

                //std::cout << batch.id << " subject chunk [" << loop_begin << ", " << loop_end_excl << "]" << std::endl;

                #pragma omp parallel for num_threads(4)
                for(int subjectIndex = loop_begin; subjectIndex < loop_end_excl; subjectIndex++){

                    if(subjectIndex + prefetch_distance < loop_end_excl) {
                        const read_number next_subject_read_id = dataArrays.h_subject_read_ids[subjectIndex + prefetch_distance];
                        const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_subject_read_id);
                        __builtin_prefetch(nextsequenceptr, 0, 0);
                    }

                    const read_number readId = dataArrays.h_subject_read_ids[subjectIndex];
                    const char* sequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(readId);
                    const int sequencelength = transFuncData.gpuReadStorage->fetchSequenceLength(readId);

                    assert(subjectIndex * dataArrays.encoded_sequence_pitch + Sequence_t::getNumBytes(sequencelength) <= subjectsequencedatabytes);

                    std::memcpy(dataArrays.h_subject_sequences_data + subjectIndex * dataArrays.encoded_sequence_pitch,
                                sequenceptr,
                                Sequence_t::getNumBytes(sequencelength));

                    //copy subject length
                    dataArrays.h_subject_sequences_lengths[subjectIndex] = sequencelength;
                }

                batch.copiedSubjects = loop_end_excl;

                if(isPausable){
                    return BatchState::CopyReads;
                }
            }

            const int firstCandidateIndex = batch.copiedCandidates;
            const int lastCandidateIndexExcl = dataArrays.n_queries;
            const int candidateChunks = SDIV((lastCandidateIndexExcl - firstCandidateIndex), candidateschunksize);

            for(int chunkId = 0; chunkId < candidateChunks; chunkId++){
                const int chunkoffset = chunkId * candidateschunksize;
                const int loop_begin = firstCandidateIndex + chunkoffset;
                const int loop_end_excl = std::min(loop_begin + candidateschunksize, lastCandidateIndexExcl);

                #pragma omp parallel for num_threads(4)
                for(int candidateIndex = loop_begin; candidateIndex < loop_end_excl; candidateIndex++){

                    if(candidateIndex + prefetch_distance < loop_end_excl) {
                        const read_number next_candidate_read_id = dataArrays.h_candidate_read_ids[candidateIndex + prefetch_distance];
                        const char* nextsequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(next_candidate_read_id);
                        __builtin_prefetch(nextsequenceptr, 0, 0);
                    }

                    const read_number candidate_read_id = dataArrays.h_candidate_read_ids[candidateIndex];
                    const char* sequenceptr = transFuncData.gpuReadStorage->fetchSequenceData_ptr(candidate_read_id);
                    const int sequencelength = transFuncData.gpuReadStorage->fetchSequenceLength(candidate_read_id);

                    assert(candidateIndex * dataArrays.encoded_sequence_pitch + Sequence_t::getNumBytes(sequencelength) <= candidatesequencedatabytes);

                    std::memcpy(dataArrays.h_candidate_sequences_data
                                + candidateIndex * dataArrays.encoded_sequence_pitch,
                                sequenceptr,
                                Sequence_t::getNumBytes(sequencelength));

                    dataArrays.h_candidate_sequences_lengths[candidateIndex] = sequencelength;
                }

                batch.copiedCandidates = loop_end_excl;

                if(isPausable && (loop_end_excl != lastCandidateIndexExcl)){
                    return BatchState::CopyReads;
                }
            }

            cudaMemcpyAsync(dataArrays.alignment_transfer_data_device,
                        dataArrays.alignment_transfer_data_host,
                        dataArrays.alignment_transfer_data_usable_size,
                        H2D,
                        streams[primary_stream_index]); CUERR;
        }

        cudaEventRecord(events[alignment_data_transfer_h2d_finished_event_index], streams[primary_stream_index]); CUERR;

        batch.copiedTasks = 0;
        batch.copiedCandidates = 0;
        batch.copiedSubjects = 0;
        batch.handledReadIds = false;

        return BatchState::StartAlignment;
    }
#endif



	ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_startalignment_func(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::StartAlignment);

		DataArrays& dataArrays = *batch.dataArrays;
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

							 /*cub::DeviceSelect::Flagged(dataArray.d_temp_storage,
										 dataArray.tmp_storage_allocation_size,
										 cub::CountingInputIterator<int>(0),
										 d_isGoodAlignment,
										 dataArray.d_indices,
										 dataArray.d_num_indices,
										 //nTotalCandidates,
										 dataArray.n_queries,
										 stream); CUERR;*/

                             cub::DeviceSelect::Flagged(batch.batchDataDevice.cubTemp.get(),
                                         batch.batchDataDevice.cubTemp.sizeRef(),
                                         cub::CountingInputIterator<int>(0),
                                         d_isGoodAlignment,
                                         dataArray.d_indices,
                                         dataArray.d_num_indices,
                                         //nTotalCandidates,
                                         dataArray.n_queries,
                                         stream); CUERR;
						 };

        call_cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel_async(
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
                    dataArrays.h_tiles_per_subject_prefixsum,
                    dataArrays.d_tiles_per_subject_prefixsum,
                    shd_tilesize,
                    dataArrays.n_subjects,
                    dataArrays.n_queries,
                    dataArrays.encoded_sequence_pitch,
                    Sequence_t::getNumBytes(dataArrays.maximum_sequence_length),
                    transFuncData.min_overlap,
                    transFuncData.maxErrorRate,
                    transFuncData.min_overlap_ratio,
                    //batch.maxSubjectLength,
                    streams[primary_stream_index],
                    batch.kernelLaunchHandle);


		//Step 5. Compare each forward alignment with the correspoding reverse complement alignment and keep the best, if any.
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
                    transFuncData.min_overlap_ratio,
                    transFuncData.min_overlap,
                    transFuncData.estimatedErrorrate,
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
					dataArrays.d_candidates_per_subject_prefixsum,
					dataArrays.n_subjects,
					dataArrays.n_queries,
					transFuncData.estimatedErrorrate,
					transFuncData.estimatedCoverage * transFuncData.m_coverage,
					streams[primary_stream_index],
					batch.kernelLaunchHandle);

		//determine indices of remaining alignments
		select_alignments_by_flag(dataArrays, streams[primary_stream_index]);

        cudaEventRecord(events[num_indices_calculated_event_index], streams[primary_stream_index]); CUERR;
        cudaStreamWaitEvent(streams[secondary_stream_index], events[num_indices_calculated_event_index], 0); CUERR;

        cudaMemcpyAsync(dataArrays.h_num_indices,
                        dataArrays.d_num_indices,
                        sizeof(int),
                        D2H,
                        streams[secondary_stream_index]); CUERR;
#ifdef USE_WAIT_FLAGS
        batch.addWaitSignal(BatchState::BuildMSA, streams[secondary_stream_index]);
#endif
        cudaEventRecord(events[num_indices_transfered_event_index], streams[secondary_stream_index]); CUERR;

		//update indices_per_subject
/*
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
*/


        cub::DeviceHistogram::HistogramRange(batch.batchDataDevice.cubTemp.get(),
                    batch.batchDataDevice.cubTemp.sizeRef(),
                    dataArrays.d_indices,
                    dataArrays.d_indices_per_subject,
                    dataArrays.n_subjects+1,
                    dataArrays.d_candidates_per_subject_prefixsum,
                    // *dataArrays.h_num_indices,
                    dataArrays.n_queries,
                    streams[primary_stream_index]); CUERR;

        //Make indices_per_subject_prefixsum
        cub::DeviceScan::ExclusiveSum(batch.batchDataDevice.cubTemp.get(),
                    batch.batchDataDevice.cubTemp.sizeRef(),
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

#ifdef USE_WAIT_FLAGS
        batch.addWaitSignal(BatchState::CopyQualities, streams[secondary_stream_index]);
        batch.addWaitSignal(BatchState::UnpackClassicResults, streams[secondary_stream_index]);
#endif

		if(transFuncData.correctionOptions.useQualityScores) {
			/*if(transFuncData.readStorageGpuData.isValidQualityData()) {
				return BatchState::BuildMSA;
			}else{*/
                return BatchState::CopyQualities;
			//}
		}else{
			return BatchState::BuildMSA;
		}
	}

	ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_copyqualities_func(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

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
        if(*dataArrays.h_num_indices == 0){
            return BatchState::WriteResults;
        }

		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;

		const auto* gpuReadStorage = transFuncData.gpuReadStorage;

		if(transFuncData.correctionOptions.useQualityScores) {

			//if(transFuncData.useGpuReadStorage && transFuncData.gpuReadStorage->type == GPUReadStorageType::SequencesAndQualities){
			if(transFuncData.readStorageGpuData.isValidQualityData()) {
#if 1
                const char* const rs_quality_data = transFuncData.readStorageGpuData.d_quality_data;
                const size_t readstorage_quality_pitch = std::size_t(gpuReadStorage->getQualityPitch());
                const size_t qualitypitch = dataArrays.quality_pitch;

                char* const subject_qualities = dataArrays.d_subject_qualities;
                char* const candidate_qualities = dataArrays.d_candidate_qualities;
                const read_number* const subject_read_ids = dataArrays.d_subject_read_ids;
                const read_number* const candidate_read_ids = dataArrays.d_candidate_read_ids;
                const int* const subject_sequences_lengths = dataArrays.d_subject_sequences_lengths;
                const int* const candidate_sequences_lengths = dataArrays.d_candidate_sequences_lengths;
                const int* const indices = dataArrays.d_indices;

                const int nSubjects = dataArrays.n_subjects;
                const int num_indices = *dataArrays.h_num_indices;

                dim3 grid(std::min(num_indices, (1<<16)),1,1);
                dim3 block(64,1,1);
                cudaStream_t stream = streams[primary_stream_index];
#if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_PRINT_ARRAYS               
                bool debug = false;
                if(batch.tasks[0].readId == 436){
                    debug = true;
                }
#endif
                generic_kernel<<<grid, block,0, stream>>>([=] __device__ (){


                    for(int index = blockIdx.x; index < nSubjects; index += gridDim.x){
                        const read_number readId = subject_read_ids[index];
                        const int length = subject_sequences_lengths[index];
                        for(int k = threadIdx.x; k < length; k += blockDim.x){
                            subject_qualities[index * qualitypitch + k]
                            = rs_quality_data[size_t(readId) * readstorage_quality_pitch + k];
                        }
                    }

                    for(int i = blockIdx.x; i < num_indices; i += gridDim.x){
                        const int index = indices[i];
                        const read_number readId = candidate_read_ids[index];
                        const int length = candidate_sequences_lengths[index];
#if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_PRINT_ARRAYS
                        if(debug && threadIdx.x == 0){
                            printf("in kernel: %d %d %d %d %d\n", i, num_indices, index, readId, length);
                        }
#endif
                        for(int k = threadIdx.x; k < length; k += blockDim.x){
                            candidate_qualities[i * qualitypitch + k]
                            = rs_quality_data[size_t(readId) * readstorage_quality_pitch + k];
                        }
                    }
                });
#else
                gpuReadStorage->copyGpuQualityDataToGpuBufferAsync(dataArrays.d_subject_qualities,
                                                                   dataArrays.quality_pitch,
                                                                   dataArrays.d_subject_read_ids,
                                                                   dataArrays.n_subjects,
                                                                   transFuncData.threadOpts.deviceId, streams[primary_stream_index]);

                gpuReadStorage->copyGpuQualityDataToGpuBufferAsync(dataArrays.d_candidate_qualities,
                                                                   dataArrays.quality_pitch,
                                                                   dataArrays.d_candidate_read_ids,
                                                                   *dataArrays.h_num_indices,
                                                                   transFuncData.threadOpts.deviceId, streams[primary_stream_index]);
#endif
                assert(cudaSuccess == cudaEventQuery(events[quality_transfer_finished_event_index])); CUERR;

                cudaEventRecord(events[quality_transfer_finished_event_index], streams[primary_stream_index]); CUERR;

                return BatchState::BuildMSA;
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

                    #pragma omp parallel for num_threads(2)
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
                        const int global_i = batch.copiedCandidates + i;
						assert(global_i * arrays.quality_pitch + sequencelength <= maxcandidatequalitychars);
						std::memcpy(arrays.h_candidate_qualities + global_i * arrays.quality_pitch,
									candidate_quality,
									sequencelength);


					}

                    batch.copiedCandidates += my_num_indices;

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


    ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_copyqualities_func2(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

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
        if(*dataArrays.h_num_indices == 0){
            return BatchState::WriteResults;
        }


		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;

		const auto* gpuReadStorage = transFuncData.gpuReadStorage;

		if(transFuncData.correctionOptions.useQualityScores) {

			//if(transFuncData.useGpuReadStorage && transFuncData.gpuReadStorage->type == GPUReadStorageType::SequencesAndQualities){
			if(transFuncData.readStorageGpuData.isValidQualityData()) {
#if 0
                const char* const rs_quality_data = transFuncData.readStorageGpuData.d_quality_data;
                const size_t readstorage_quality_pitch = std::size_t(gpuReadStorage->getQualityPitch());
                const size_t qualitypitch = dataArrays.quality_pitch;

                char* const subject_qualities = dataArrays.d_subject_qualities;
                char* const candidate_qualities = dataArrays.d_candidate_qualities;
                const read_number* const subject_read_ids = dataArrays.d_subject_read_ids;
                const read_number* const candidate_read_ids = dataArrays.d_candidate_read_ids;
                const int* const subject_sequences_lengths = dataArrays.d_subject_sequences_lengths;
                const int* const candidate_sequences_lengths = dataArrays.d_candidate_sequences_lengths;
                const int* const indices = dataArrays.d_indices;

                const int nSubjects = dataArrays.n_subjects;
                const int num_indices = *dataArrays.h_num_indices;

                dim3 grid(std::min(num_indices, (1<<16)),1,1);
                dim3 block(64,1,1);
                cudaStream_t stream = streams[primary_stream_index];

                generic_kernel<<<grid, block,0, stream>>>([=] __device__ (){


                    for(int index = blockIdx.x; index < nSubjects; index += gridDim.x){
                        const read_number readId = subject_read_ids[index];
                        const int length = subject_sequences_lengths[index];
                        for(int k = threadIdx.x; k < length; k += blockDim.x){
                            subject_qualities[index * qualitypitch + k]
                            = rs_quality_data[size_t(readId) * readstorage_quality_pitch + k];
                        }
                    }

                    for(int i = blockIdx.x; i < num_indices; i += gridDim.x){
                        const int index = indices[i];
                        const read_number readId = candidate_read_ids[index];
                        const int length = candidate_sequences_lengths[index];
                        for(int k = threadIdx.x; k < length; k += blockDim.x){
                            candidate_qualities[i * qualitypitch + k]
                            = rs_quality_data[size_t(readId) * readstorage_quality_pitch + k];
                        }
                    }
                });
#else
                gpuReadStorage->copyGpuQualityDataToGpuBufferAsync(dataArrays.d_subject_qualities,
                                                                   dataArrays.quality_pitch,
                                                                   dataArrays.d_subject_read_ids,
                                                                   dataArrays.n_subjects,
                                                                   transFuncData.threadOpts.deviceId, streams[primary_stream_index]);

                gpuReadStorage->copyGpuQualityDataToGpuBufferAsync(dataArrays.d_candidate_qualities,
                                                                   dataArrays.quality_pitch,
                                                                   dataArrays.d_candidate_read_ids,
                                                                   *dataArrays.h_num_indices,
                                                                   transFuncData.threadOpts.deviceId, streams[primary_stream_index]);
#endif
                assert(cudaSuccess == cudaEventQuery(events[quality_transfer_finished_event_index])); CUERR;

                cudaEventRecord(events[quality_transfer_finished_event_index], streams[primary_stream_index]); CUERR;

                return BatchState::BuildMSA;
			}else{

                constexpr int subjectschunksize = 1000;
                constexpr int candidateschunksize = 1000;
                constexpr int prefetch_distance = 4;

                const int firstSubjectIndex = batch.copiedSubjects;
                const int lastSubjectIndexExcl = dataArrays.n_subjects;
                const int subjectChunks = SDIV((lastSubjectIndexExcl - firstSubjectIndex), subjectschunksize);

                for(int chunkId = 0; chunkId < subjectChunks; chunkId++){
                    const int chunkoffset = chunkId * subjectschunksize;
                    const int loop_begin = firstSubjectIndex + chunkoffset;
                    const int loop_end_excl = std::min(loop_begin + subjectschunksize, lastSubjectIndexExcl);

                    //std::cout << batch.id << " subject chunk [" << loop_begin << ", " << loop_end_excl << "]" << std::endl;

                    #pragma omp parallel for num_threads(4)
                    for(int subjectIndex = loop_begin; subjectIndex < loop_end_excl; subjectIndex++){

                        if(subjectIndex + prefetch_distance < loop_end_excl) {
                            const read_number next_subject_read_id = dataArrays.h_subject_read_ids[subjectIndex + prefetch_distance];
                            const char* nextqualityptr = transFuncData.gpuReadStorage->fetchQuality_ptr(next_subject_read_id);
                            __builtin_prefetch(nextqualityptr, 0, 0);
                        }

                        const read_number readId = dataArrays.h_subject_read_ids[subjectIndex];
                        const char* qualityptr = gpuReadStorage->fetchQuality_ptr(readId);
                        const int sequencelength = transFuncData.gpuReadStorage->fetchSequenceLength(readId);

    					//assert(subjectIndex * dataArrays.quality_pitch + sequencelength <= maxsubjectqualitychars);
    					std::memcpy(dataArrays.h_subject_qualities + subjectIndex * dataArrays.quality_pitch,
    								qualityptr,
    								sequencelength);
                    }

                    batch.copiedSubjects = loop_end_excl;

                    if(isPausable){
                        return expectedState;
                    }
                }

                const int firstCandidateIndex = batch.copiedCandidates;
                const int lastCandidateIndexExcl = *dataArrays.h_num_indices;
                const int candidateChunks = SDIV((lastCandidateIndexExcl - firstCandidateIndex), candidateschunksize);

                for(int chunkId = 0; chunkId < candidateChunks; chunkId++){
                    const int chunkoffset = chunkId * candidateschunksize;
                    const int loop_begin = firstCandidateIndex + chunkoffset;
                    const int loop_end_excl = std::min(loop_begin + candidateschunksize, lastCandidateIndexExcl);

                    #pragma omp parallel for num_threads(4)
                    for(int index = loop_begin; index < loop_end_excl; index++){
                        const int candidateIndex = dataArrays.h_indices[index];

                        if(index + prefetch_distance < loop_end_excl) {
                            const int nextIndex = dataArrays.h_indices[index + prefetch_distance];
                            const read_number next_candidate_read_id = dataArrays.h_candidate_read_ids[nextIndex];
                            const char* nextqualityptr = transFuncData.gpuReadStorage->fetchQuality_ptr(next_candidate_read_id);
                            __builtin_prefetch(nextqualityptr, 0, 0);
                        }

                        const read_number candidate_read_id = dataArrays.h_candidate_read_ids[candidateIndex];
                        const char* qualityptr = gpuReadStorage->fetchQuality_ptr(candidate_read_id);
                        const int sequencelength = transFuncData.gpuReadStorage->fetchSequenceLength(candidate_read_id);

    					//assert(candidateIndex * arrays.quality_pitch + sequencelength <= maxcandidatequalitychars);
    					std::memcpy(dataArrays.h_candidate_qualities + candidateIndex * dataArrays.quality_pitch,
    								qualityptr,
    								sequencelength);
                    }

                    batch.copiedCandidates = loop_end_excl;

                    if(isPausable && (loop_end_excl != lastCandidateIndexExcl)){
                        return expectedState;
                    }
                }

                cudaMemcpyAsync(dataArrays.qualities_transfer_data_device,
                            dataArrays.qualities_transfer_data_host,
                            dataArrays.qualities_transfer_data_usable_size,
                            H2D,
                            streams[secondary_stream_index]); CUERR;
            }
            assert(cudaSuccess == cudaEventQuery(events[quality_transfer_finished_event_index])); CUERR;

            cudaEventRecord(events[quality_transfer_finished_event_index], streams[secondary_stream_index]); CUERR;
            batch.copiedTasks = 0;
            batch.copiedCandidates = 0;
            return BatchState::BuildMSA;
        }else{
            return BatchState::BuildMSA;
        }

	}

	ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_buildmsa_func(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

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
        if(*dataArrays.h_num_indices == 0){
            return BatchState::WriteResults;
        }




		if(!canLaunchKernel) {
			return BatchState::BuildMSA;
		}else{

			cudaStreamWaitEvent(streams[primary_stream_index], events[quality_transfer_finished_event_index], 0); CUERR;

			// coverage is always >= 1
			const float min_coverage_threshold = std::max(1.0f,
						transFuncData.m_coverage / 6.0f * transFuncData.estimatedCoverage);

			const float desiredAlignmentMaxErrorRate = transFuncData.maxErrorRate;

			//Determine multiple sequence alignment properties

            call_msa_init_kernel_async_exp(
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
                    streams[primary_stream_index],
                    batch.kernelLaunchHandle);


            //make blocks per subject prefixsum for msa_add_sequences_kernel_implicit

            auto getBlocksPerSubject = [] __device__ (int indices_for_subject){
                return SDIV(indices_for_subject, msa_add_sequences_kernel_implicit_shared_blocksize);
            };
            cub::TransformInputIterator<int,decltype(getBlocksPerSubject), int*>
                d_blocksPerSubject(dataArrays.d_indices_per_subject,
                              getBlocksPerSubject);


            /*
            cub::DeviceScan::InclusiveSum(dataArrays.d_temp_storage,
    					dataArrays.tmp_storage_allocation_size,
    					d_blocksPerSubject,
    					dataArrays.d_tiles_per_subject_prefixsum+1,
    					dataArrays.n_subjects,
    					streams[primary_stream_index]); CUERR;
*/

            cub::DeviceScan::InclusiveSum(batch.batchDataDevice.cubTemp.get(),
                        batch.batchDataDevice.cubTemp.sizeRef(),
                        d_blocksPerSubject,
                        dataArrays.d_tiles_per_subject_prefixsum+1,
                        dataArrays.n_subjects,
                        streams[primary_stream_index]); CUERR;


            call_set_kernel_async(dataArrays.d_tiles_per_subject_prefixsum,
                                    0,
                                    0,
                                    streams[primary_stream_index]);

#ifndef MSA_IMPLICIT

            call_msa_add_sequences_kernel_exp_async(
                    dataArrays.d_multiple_sequence_alignments,
                    dataArrays.d_multiple_sequence_alignment_weights,
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
                    dataArrays.d_msa_column_properties,
                    dataArrays.d_candidates_per_subject_prefixsum,
                    dataArrays.d_indices,
                    dataArrays.d_indices_per_subject,
                    dataArrays.d_indices_per_subject_prefixsum,
                    dataArrays.n_subjects,
                    dataArrays.n_queries,
                    dataArrays.h_num_indices,
                    dataArrays.d_num_indices,
                    transFuncData.correctionOptions.useQualityScores,
                    desiredAlignmentMaxErrorRate,
                    dataArrays.maximum_sequence_length,
                    Sequence_t::getNumBytes(dataArrays.maximum_sequence_length),
                    dataArrays.encoded_sequence_pitch,
                    dataArrays.quality_pitch,
                    dataArrays.msa_pitch,
                    dataArrays.msa_weights_pitch,
                    streams[primary_stream_index],
                    batch.kernelLaunchHandle);

#else

            call_msa_add_sequences_kernel_implicit_async(
                        dataArrays.d_counts,
                        dataArrays.d_weights,
                        dataArrays.d_coverage,
                        dataArrays.d_origWeights,
                        dataArrays.d_origCoverages,
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
                        dataArrays.d_msa_column_properties,
                        dataArrays.d_candidates_per_subject_prefixsum,
                        dataArrays.d_indices,
                        dataArrays.d_indices_per_subject,
                        dataArrays.d_indices_per_subject_prefixsum,
                        dataArrays.d_tiles_per_subject_prefixsum,
                        dataArrays.n_subjects,
                        dataArrays.n_queries,
                        dataArrays.h_num_indices,
                        dataArrays.d_num_indices,
                        transFuncData.correctionOptions.useQualityScores,
                        desiredAlignmentMaxErrorRate,
                        dataArrays.maximum_sequence_length,
                        Sequence_t::getNumBytes(dataArrays.maximum_sequence_length),
                        dataArrays.encoded_sequence_pitch,
                        dataArrays.quality_pitch,
                        dataArrays.msa_pitch,
                        dataArrays.msa_weights_pitch,
                        streams[primary_stream_index],
                        batch.kernelLaunchHandle);

#endif

#ifndef MSA_IMPLICIT

			call_msa_find_consensus_kernel_async(
						dataArrays.d_consensus,
						dataArrays.d_support,
						dataArrays.d_coverage,
						dataArrays.d_origWeights,
						dataArrays.d_origCoverages,
                        dataArrays.d_counts,
                        dataArrays.d_weights,
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
#else

            call_msa_find_consensus_implicit_kernel_async(
                        dataArrays.d_counts,
                        dataArrays.d_weights,
                        dataArrays.d_consensus,
                        dataArrays.d_support,
                        dataArrays.d_coverage,
                        dataArrays.d_origWeights,
                        dataArrays.d_origCoverages,
                        dataArrays.d_subject_sequences_data,
                        dataArrays.d_msa_column_properties,
                        dataArrays.n_subjects,
                        dataArrays.encoded_sequence_pitch,
                        dataArrays.msa_pitch,
                        dataArrays.msa_weights_pitch,
                        streams[primary_stream_index],
                        batch.kernelLaunchHandle);

#endif
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
                cudaMemcpyAsync(dataArrays.h_multiple_sequence_alignment_weights,
							dataArrays.d_multiple_sequence_alignment_weights,
							(dataArrays.n_subjects + dataArrays.n_queries) * dataArrays.msa_weights_pitch,
							D2H,
							streams[secondary_stream_index]); CUERR;
                cudaMemcpyAsync(dataArrays.h_multiple_sequence_alignments,
							dataArrays.d_multiple_sequence_alignments,
							(dataArrays.n_subjects + dataArrays.n_queries) * dataArrays.msa_pitch,
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
#endif

			}

			if(transFuncData.correctionOptions.classicMode) {
				return BatchState::StartClassicCorrection;
			}else{
				return BatchState::StartForestCorrection;
			}
		}
	}

	ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_startclassiccorrection_func(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::StartClassicCorrection);
		assert(transFuncData.correctionOptions.classicMode);

		DataArrays& dataArrays = *batch.dataArrays;
		std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		if(!canLaunchKernel) {
			return BatchState::StartClassicCorrection;
		}else{

			const float avg_support_threshold = 1.0f-1.0f*transFuncData.estimatedErrorrate;
			const float min_support_threshold = 1.0f-3.0f*transFuncData.estimatedErrorrate;
			// coverage is always >= 1
			const float min_coverage_threshold = std::max(1.0f,
						transFuncData.m_coverage / 6.0f * transFuncData.estimatedCoverage);
			const int new_columns_to_correct = transFuncData.new_columns_to_correct;

			// Step 14. Correction
			if(transFuncData.correctionOptions.classicMode) {
				// correct subjects
#ifndef MSA_IMPLICIT
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
#else

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
                            transFuncData.estimatedErrorrate,
                            avg_support_threshold,
                            min_support_threshold,
                            min_coverage_threshold,
                            transFuncData.kmerlength,
                            dataArrays.maximum_sequence_length,
                            streams[primary_stream_index],
                            batch.kernelLaunchHandle);

#endif
				if(transFuncData.correctionOptions.correctCandidates) {


					// find subject ids of subjects with high quality multiple sequence alignment

					/*
                    cub::DeviceSelect::Flagged(dataArrays.d_temp_storage,
								dataArrays.tmp_storage_allocation_size,
								cub::CountingInputIterator<int>(0),
								dataArrays.d_is_high_quality_subject,
								dataArrays.d_high_quality_subject_indices,
								dataArrays.d_num_high_quality_subject_indices,
								dataArrays.n_subjects,
								streams[primary_stream_index]); CUERR;
*/
                    cub::DeviceSelect::Flagged(batch.batchDataDevice.cubTemp.get(),
                                batch.batchDataDevice.cubTemp.sizeRef(),
                                cub::CountingInputIterator<int>(0),
                                dataArrays.d_is_high_quality_subject,
                                dataArrays.d_high_quality_subject_indices,
                                dataArrays.d_num_high_quality_subject_indices,
                                dataArrays.n_subjects,
                                streams[primary_stream_index]); CUERR;

					// correct candidates
                    call_msa_correct_candidates_kernel_async_exp(
                            dataArrays.d_consensus,
                            dataArrays.d_support,
                            dataArrays.d_coverage,
                            dataArrays.d_origCoverages,
                            dataArrays.d_multiple_sequence_alignments,
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

#ifdef USE_WAIT_FLAGS
            batch.addWaitSignal(BatchState::UnpackClassicResults, streams[primary_stream_index]);
#endif

            return BatchState::UnpackClassicResults;
		}
	}

	ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_startforestcorrection_func(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

		assert(!transFuncData.correctionOptions.classicMode);

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

		DataArrays& dataArrays = *batch.dataArrays;

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
						transFuncData.minhasher->minparams.k, 0.0f,
						transFuncData.estimatedCoverage);

			for(const auto& msafeature : MSAFeatures) {
				constexpr float maxgini = 0.05f;
				constexpr float forest_correction_fraction = 0.5f;

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
					task.corrected_subject[msafeature.position] = cons[globalIndex];
				}
			}
        }

		return BatchState::WriteResults;
	}

	ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_unpackclassicresults_func(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

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
        
        cudaMemcpyAsync(dataArrays.qualities_transfer_data_host,
                        dataArrays.qualities_transfer_data_device,
                        dataArrays.qualities_transfer_data_usable_size,
                        D2H,
                        streams[primary_stream_index]); CUERR;
        cudaStreamSynchronize(streams[primary_stream_index]); CUERR;

	    #endif

	    #if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_PRINT_ARRAYS
    if(batch.tasks[0].readId == 436){
        std::cout << "subject read ids" << std::endl;
        for(int i = 0; i< dataArrays.n_subjects; i++) {
            std::cout << dataArrays.h_subject_read_ids[i]<< std::endl;
        }
        
        std::cout << "candidate read ids" << std::endl;
        for(int i = 0; i< dataArrays.n_queries; i++) {
            std::cout << dataArrays.h_candidate_read_ids[i]<< std::endl;
        }
        
        std::cout << "subject quality scores" << std::endl;
        for(int i = 0; i< dataArrays.n_subjects; i++) {
            for(size_t k = 0; k < dataArrays.quality_pitch; k++){
                std::cout << dataArrays.h_subject_qualities[i * dataArrays.quality_pitch + k];
            }
            std::cout << std::endl;
        }
        
        std::cout << "candidate quality scores" << std::endl;
        for(int i = 0; i< *dataArrays.h_num_indices; i++) {
            for(size_t k = 0; k < dataArrays.quality_pitch; k++){
                std::cout << dataArrays.h_candidate_qualities[i * dataArrays.quality_pitch + k];
            }
            std::cout << std::endl;
        }
        
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
    }
	    #endif

	    #if defined CARE_GPU_DEBUG && defined CARE_GPU_DEBUG_PRINT_MSA
#if 0
        for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {
            auto& task = batch.tasks[subject_index];
            auto& arrays = dataArrays;

            const unsigned offset1 = arrays.msa_pitch * (subject_index + arrays.h_indices_per_subject_prefixsum[subject_index]);
            const int* const indices_for_this_subject = arrays.h_indices + arrays.h_indices_per_subject_prefixsum[subject_index];
            const char* const my_multiple_sequence_alignment = arrays.h_multiple_sequence_alignments + offset1;
            const char* const my_consensus = arrays.h_consensus + subject_index * arrays.msa_pitch;
            const int subjectColumnsBegin_incl = arrays.h_msa_column_properties[subject_index].subjectColumnsBegin_incl;
			const int subjectColumnsEnd_excl = arrays.h_msa_column_properties[subject_index].subjectColumnsEnd_excl;
            const int ncolumns = arrays.h_msa_column_properties[subject_index].columnsToCheck;
            const int msa_rows = 1 + arrays.h_indices_per_subject[subject_index];
            bool isHQ = arrays.h_is_high_quality_subject[subject_index];
            bool isCorrected = arrays.h_subject_is_corrected[subject_index];

            const char* const my_corrected_subject_data = arrays.h_corrected_subjects + subject_index * arrays.sequence_pitch;
            const char* const my_corrected_candidates_data = arrays.h_corrected_candidates + arrays.h_indices_per_subject_prefixsum[subject_index] * arrays.sequence_pitch;
            const int* const my_indices_of_corrected_candidates = arrays.h_indices_of_corrected_candidates + arrays.h_indices_per_subject_prefixsum[subject_index];

            auto mismatch = std::mismatch(task.subject_string.begin(), task.subject_string.end(), &my_consensus[subjectColumnsBegin_incl]);
            bool subjectAndConsensusDiffer = !(mismatch.first == task.subject_string.end() || mismatch.second == &my_consensus[subjectColumnsEnd_excl]);



            std::string corrected_subject;
            bool more_candidates_after_correction = false;
            if(isCorrected){
                const int subject_length = task.subject_string.length();
                corrected_subject = std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length};
                const int hits_per_candidate = transFuncData.correctionOptions.hits_per_candidate;
                auto newCandidateList = transFuncData.minhasher->getCandidates(corrected_subject, hits_per_candidate, transFuncData.max_candidates);
                auto readIdPos = std::lower_bound(newCandidateList.begin(), newCandidateList.end(), task.readId);
                if(readIdPos != newCandidateList.end() && *readIdPos == task.readId) {
                    newCandidateList.erase(readIdPos);
                }
                if(newCandidateList.size() > task.candidate_read_ids.size())
                    more_candidates_after_correction = true;
            }



            int comp = -1;
            if(!subjectAndConsensusDiffer && !isCorrected)
                comp = 0;
            else if(!subjectAndConsensusDiffer && isCorrected)
                comp = 1;
            else if(subjectAndConsensusDiffer && !isCorrected)
                comp = 2;
            else if(subjectAndConsensusDiffer && isCorrected)
                comp = 3;

            auto get_shift_of_row = [&](int row){
                if(row == 0) return 0;
                const int queryIndex = indices_for_this_subject[row-1];
                return arrays.h_alignment_shifts[queryIndex];
            };

            std::cout << "ReadId " << task.readId << ": msa rows = " << msa_rows << ", columns = " << ncolumns << ", HQ-MSA: " << (isHQ ? "True" : "False")
                        << ", comp " << comp << ", more cand: " << (more_candidates_after_correction ? "True" : "False")
                        << ", subjectColumnsBegin_incl = " << subjectColumnsBegin_incl << ", subjectColumnsEnd_excl = " << subjectColumnsEnd_excl << '\n';
            if(isCorrected){
                const int subject_length = task.subject_string.length();
                std::string s{my_corrected_subject_data, my_corrected_subject_data + subject_length};
                std::cout << s << '\n';
            }
            print_multiple_sequence_alignment_sorted_by_shift(std::cout, my_multiple_sequence_alignment, msa_rows, ncolumns, arrays.msa_pitch, get_shift_of_row);
            std::cout << '\n';
            print_multiple_sequence_alignment_consensusdiff_sorted_by_shift(std::cout, my_multiple_sequence_alignment, my_consensus,
                                                                            msa_rows, ncolumns, arrays.msa_pitch, get_shift_of_row);
            //print_multiple_sequence_alignment(std::cout, my_multiple_sequence_alignment, msa_rows, ncolumns, arrays.msa_pitch);
            std::cout << '\n';
        }
#else
		//DEBUGGING
		for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {
			auto& task = batch.tasks[subject_index];
			auto& arrays = dataArrays;

            if(task.readId == 436){

			const size_t msa_weights_pitch_floats = arrays.msa_weights_pitch / sizeof(float);

			const unsigned offset1 = arrays.msa_pitch * (subject_index + arrays.h_indices_per_subject_prefixsum[subject_index]);
			const unsigned offset2 = msa_weights_pitch_floats * (subject_index + arrays.h_indices_per_subject_prefixsum[subject_index]);

			const char* const my_multiple_sequence_alignment = arrays.h_multiple_sequence_alignments + offset1;
			const float* const my_multiple_sequence_alignment_weight = arrays.h_multiple_sequence_alignment_weights + offset2;

			char* const my_consensus = arrays.h_consensus + subject_index * arrays.msa_pitch;
			float* const my_support = arrays.h_support + subject_index * msa_weights_pitch_floats;
			int* const my_coverage = arrays.h_coverage + subject_index * msa_weights_pitch_floats;

            const int* my_countsA = arrays.h_counts + 4 * subject_index * msa_weights_pitch_floats + 0 * msa_weights_pitch_floats;
            const int* my_countsC = arrays.h_counts + 4 * subject_index * msa_weights_pitch_floats + 1 * msa_weights_pitch_floats;
            const int* my_countsG = arrays.h_counts + 4 * subject_index * msa_weights_pitch_floats + 2 * msa_weights_pitch_floats;
            const int* my_countsT = arrays.h_counts + 4 * subject_index * msa_weights_pitch_floats + 3 * msa_weights_pitch_floats;

            const float* my_weightsA = arrays.h_weights + 4 * subject_index * msa_weights_pitch_floats + 0 * msa_weights_pitch_floats;
            const float* my_weightsC = arrays.h_weights + 4 * subject_index * msa_weights_pitch_floats + 1 * msa_weights_pitch_floats;
            const float* my_weightsG = arrays.h_weights + 4 * subject_index * msa_weights_pitch_floats + 2 * msa_weights_pitch_floats;
            const float* my_weightsT = arrays.h_weights + 4 * subject_index * msa_weights_pitch_floats + 3 * msa_weights_pitch_floats;

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
			/*std::cout << "MSA:" << std::endl;
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
			std::cout << std::endl;*/

            std::cout << "countsA: "<< std::endl;
			for(int col = 0; col < columnsToCheck; col++) {
				std::cout << my_countsA[col] << " ";
				if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
					std::cout << " ";
			}
			std::cout << std::endl;

            std::cout << "countsC: "<< std::endl;
			for(int col = 0; col < columnsToCheck; col++) {
				std::cout << my_countsC[col] << " ";
				if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
					std::cout << " ";
			}
			std::cout << std::endl;

            std::cout << "countsG: "<< std::endl;
			for(int col = 0; col < columnsToCheck; col++) {
				std::cout << my_countsG[col] << " ";
				if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
					std::cout << " ";
			}
			std::cout << std::endl;

            std::cout << "countsT: "<< std::endl;
			for(int col = 0; col < columnsToCheck; col++) {
				std::cout << my_countsT[col] << " ";
				if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
					std::cout << " ";
			}
			std::cout << std::endl;

            std::cout << "weightsA: "<< std::endl;
			for(int col = 0; col < columnsToCheck; col++) {
				std::cout << my_weightsA[col] << " ";
				if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
					std::cout << " ";
			}
			std::cout << std::endl;

            std::cout << "weightsC: "<< std::endl;
			for(int col = 0; col < columnsToCheck; col++) {
				std::cout << my_weightsC[col] << " ";
				if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
					std::cout << " ";
			}
			std::cout << std::endl;

            std::cout << "weightsG: "<< std::endl;
			for(int col = 0; col < columnsToCheck; col++) {
				std::cout << my_weightsG[col] << " ";
				if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
					std::cout << " ";
			}
			std::cout << std::endl;

            std::cout << "weightsT: "<< std::endl;
			for(int col = 0; col < columnsToCheck; col++) {
				std::cout << my_weightsT[col] << " ";
				if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
					std::cout << " ";
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

			/*std::cout << "MSA weights:" << std::endl;
			for(int row = 0; row < msa_rows; row++) {
				for(int col = 0; col < columnsToCheck; col++) {
					float f = my_multiple_sequence_alignment_weight[row * msa_weights_pitch_floats + col];
					std::cout << f << " ";
					if(col == subjectColumnsBegin_incl - 1 || col == subjectColumnsEnd_excl - 1)
						std::cout << " ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;*/

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

            std::exit(0);
            }
		}
#endif
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
				task.corrected_subject = std::move(std::string{my_corrected_subject_data, my_corrected_subject_data + subject_length});
			}

			if(transFuncData.correctCandidates) {
				const int n_corrected_candidates = arrays.h_num_corrected_candidates[subject_index];

				assert((!any_correction_candidates && n_corrected_candidates == 0) || any_correction_candidates);

				for(int i = 0; i < n_corrected_candidates; ++i) {
					const int global_candidate_index = my_indices_of_corrected_candidates[i];
					const int local_candidate_index = global_candidate_index - arrays.h_candidates_per_subject_prefixsum[subject_index];

					//const read_number candidate_read_id = task.candidate_read_ids[local_candidate_index];
					const read_number candidate_read_id = task.candidate_read_ids_begin[local_candidate_index];
					const int candidate_length = transFuncData.gpuReadStorage->fetchSequenceLength(candidate_read_id);
					//const int candidate_length = task.candidate_sequences[local_candidate_index]->length();
					const char* const candidate_data = my_corrected_candidates_data + i * arrays.sequence_pitch;

					//task.corrected_candidates_read_ids.emplace_back(task.candidate_read_ids[local_candidate_index]);
					task.corrected_candidates_read_ids.emplace_back(task.candidate_read_ids_begin[local_candidate_index]);

					task.corrected_candidates.emplace_back(std::move(std::string{candidate_data, candidate_data + candidate_length}));
				}
			}
		}
        //std::exit(0);
		return BatchState::WriteResults;
	}

	ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_writeresults_func(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::WriteResults);

		//DataArrays<Sequence_t, read_number>& dataArrays = *batch.dataArrays;
		//std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		//std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

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

		if(transFuncData.correctionOptions.extractFeatures)
			return BatchState::WriteFeatures;
		else
			return BatchState::Finished;
	}

	ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_writefeatures_func(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::WriteFeatures);

		DataArrays& dataArrays = *batch.dataArrays;
		//std::array<cudaStream_t, nStreamsPerBatch>& streams = *batch.streams;
		std::array<cudaEvent_t, nEventsPerBatch>& events = *batch.events;

		cudaEventSynchronize(events[msadata_transfer_finished_event_index]); CUERR;

		auto& featurestream = *transFuncData.featurestream;

		for(std::size_t subject_index = 0; subject_index < batch.tasks.size(); ++subject_index) {

			const auto& task = batch.tasks[subject_index];
			const auto& columnProperties = dataArrays.h_msa_column_properties[subject_index];

			//std::cout << task.readId << "feature" << std::endl;

			const std::size_t msa_weights_pitch_floats = dataArrays.msa_weights_pitch / sizeof(float);
#if 0
			std::vector<MSAFeature> MSAFeatures = extractFeatures(dataArrays.h_consensus + subject_index * dataArrays.msa_pitch,
						dataArrays.h_support + subject_index * msa_weights_pitch_floats,
						dataArrays.h_coverage + subject_index * msa_weights_pitch_floats,
						dataArrays.h_origCoverages + subject_index * msa_weights_pitch_floats,
						columnProperties.columnsToCheck,
						columnProperties.subjectColumnsBegin_incl,
						columnProperties.subjectColumnsEnd_excl,
						task.subject_string,
						transFuncData.minhasher->minparams.k, 0.0f,
						transFuncData.estimatedCoverage);
#else
            //const size_t msa_weights_pitch_floats = dataarrays.msa_weights_pitch / sizeof(float);
            const unsigned offset1 = dataArrays.msa_pitch * (subject_index +  dataArrays.h_indices_per_subject_prefixsum[subject_index]);
            const unsigned offset2 = msa_weights_pitch_floats * (subject_index +  dataArrays.h_indices_per_subject_prefixsum[subject_index]);

            const char* const my_multiple_sequence_alignment = dataArrays.h_multiple_sequence_alignments + offset1;
            const float* const my_multiple_sequence_alignment_weight = dataArrays.h_multiple_sequence_alignment_weights + offset2;
            const int msa_rows = 1 + dataArrays.h_indices_per_subject[subject_index];

#if 1

            std::vector<MSAFeature3> MSAFeatures = extractFeatures3(
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
                                        transFuncData.estimatedCoverage,
                                        true,
                                        dataArrays.msa_pitch,
                                        msa_weights_pitch_floats);

#else

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
                                        transFuncData.estimatedCoverage);
#endif

#endif
			for(const auto& msafeature : MSAFeatures) {
				featurestream << task.readId << '\t' << msafeature.position << '\n';
				featurestream << msafeature << '\n';
			}
		}

		return BatchState::Finished;
	}

	ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_finished_func(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::Finished);

		assert(false);         //Finished is end node

		return BatchState::Finished;
	}

	ErrorCorrectionThreadOnlyGPU::BatchState ErrorCorrectionThreadOnlyGPU::state_aborted_func(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

		assert(batch.state == BatchState::Aborted);

		assert(false);         //Aborted is end node

		return BatchState::Aborted;
	}


	ErrorCorrectionThreadOnlyGPU::AdvanceResult ErrorCorrectionThreadOnlyGPU::advance_one_step(ErrorCorrectionThreadOnlyGPU::Batch& batch,
				bool canBlock,
				bool canLaunchKernel,
				bool isPausable,
				const ErrorCorrectionThreadOnlyGPU::TransitionFunctionData& transFuncData){

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


	void ErrorCorrectionThreadOnlyGPU::execute() {
		isRunning = true;

		assert(threadOpts.canUseGpu);
		assert(max_candidates > 0);

		makeTransitionFunctionTable();

		//std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

		std::ofstream outputstream(threadOpts.outputfile);
		if(!outputstream)
			throw std::runtime_error("Could not open output file");

		std::ofstream featurestream(threadOpts.outputfile + "_features");
		if(!featurestream)
			throw std::runtime_error("Could not open output feature file");


		cudaSetDevice(threadOpts.deviceId); CUERR;

		//std::vector<read_number> readIds = threadOpts.batchGen->getNextReadIds();

		std::vector<DataArrays > dataArrays;
		//std::array<Batch, nParallelBatches> batches;
		std::array<std::array<cudaStream_t, nStreamsPerBatch>, nParallelBatches> streams;
		std::array<std::array<cudaEvent_t, nEventsPerBatch>, nParallelBatches> cudaevents;

		std::queue<Batch> batchQueue;
		std::queue<DataArrays*> freeDataArraysQueue;
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

		}

		for(auto& array : dataArrays)
			freeDataArraysQueue.push(&array);
		for(auto& streamArray : streams)
			freeStreamsQueue.push(&streamArray);
		for(auto& eventArray : cudaevents)
			freeEventsQueue.push(&eventArray);

		std::vector<read_number> readIdBuffer;
        std::vector<CorrectionTask_t> tmptasksBuffer;

		TransitionFunctionData transFuncData;

		//transFuncData.mybatchgen = &mybatchgen;
        transFuncData.threadOpts = threadOpts;
		transFuncData.readIdGenerator = threadOpts.readIdGenerator;
		transFuncData.readIdBuffer = &readIdBuffer;
        transFuncData.tmptasksBuffer = &tmptasksBuffer;
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

        if(!correctionOptions.classicMode){
           transFuncData.fc = ForestClassifier{fileOptions.forestfilename};
        }


		std::array<Batch, nParallelBatches> batches;
        std::array<Batch*, nParallelBatches> batchPointers;

		for(int i = 0; i < nParallelBatches; ++i) {
            batches[i].id = i;
			batches[i].dataArrays = &dataArrays[i];
			batches[i].streams = &streams[i];
			batches[i].events = &cudaevents[i];
			batches[i].kernelLaunchHandle = make_kernel_launch_handle(threadOpts.deviceId);
            batchPointers[i] = &batches[i];
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

			Batch& mainBatch = *batchPointers[0];

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

			//rotate left to position next batch at index 0
			std::rotate(batchPointers.begin(), batchPointers.begin()+1, batchPointers.end());






//#ifdef CARE_GPU_DEBUG
			//stopAndAbort = true; //remove
//#endif


		} // end batch processing


		outputstream.flush();
		featurestream.flush();

        for(const auto& batch : batches){
            batch.waitUntilAllCallbacksFinished();
        }

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



}
}
