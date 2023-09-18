#ifndef CARE_MULTIGPUCORRECTOR_CUH
#define CARE_MULTIGPUCORRECTOR_CUH



#include <hpc_helpers.cuh>
#include <hpc_helpers/include/nvtx_markers.cuh>

#include <gpu/gpuminhasher.cuh>
//#include <gpu/multigpuminhasher.cuh>

#include <gpu/kernels.hpp>
#include <gpu/gpucorrectorkernels.cuh>
#include <gpu/gpureadstorage.cuh>
#include <gpu/asyncresult.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/cubwrappers.cuh>
#include <gpu/gpumsamanaged.cuh>
#include <gpu/global_cuda_stream_pool.cuh>
#include <gpu/minhashqueryfilter.cuh>
#include <gpu/gpubitarray.cuh>

#include <config.hpp>
#include <util.hpp>
#include <corrector_common.hpp>
#include <threadpool.hpp>

#include <options.hpp>
#include <correctedsequence.hpp>
#include <memorymanagement.hpp>
#include <msa.hpp>
#include <classification.hpp>

#include <forest.hpp>
#include <gpu/forest_gpu.cuh>

#include <algorithm>
#include <array>
#include <map>

#include <cub/cub.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/equal.h>
#include <thrust/logical.h>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/exec_policy.hpp>
#include <gpu/rmm_utilities.cuh>


namespace care{
namespace gpu{

    class MultiGpuErrorCorrectorInput{
    public:
        template<class T>
        using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

        std::vector<int> deviceIds;
        std::vector<CudaEvent> vec_event;
        std::vector<CudaEvent> vec_h_candidate_read_ids_readyEvent;
        std::vector<CudaStream> vec_d2hstream;

        std::vector<PinnedBuffer<int>> vec_h_numAnchors;
        std::vector<PinnedBuffer<int>> vec_h_numCandidates;
        std::vector<PinnedBuffer<read_number>> vec_h_anchorReadIds;
        std::vector<PinnedBuffer<read_number>> vec_h_candidate_read_ids;

        std::vector<rmm::device_uvector<int>> vec_d_numAnchors;
        std::vector<rmm::device_uvector<int>> vec_d_numCandidates;
        std::vector<rmm::device_uvector<read_number>> vec_d_anchorReadIds;
        std::vector<rmm::device_uvector<unsigned int>> vec_d_anchor_sequences_data;
        std::vector<rmm::device_uvector<int>> vec_d_anchor_sequences_lengths;
        std::vector<rmm::device_uvector<read_number>> vec_d_candidate_read_ids;
        std::vector<rmm::device_uvector<unsigned int>> vec_d_candidate_sequences_data;
        std::vector<rmm::device_uvector<int>> vec_d_candidate_sequences_lengths;
        std::vector<rmm::device_uvector<int>> vec_d_candidates_per_anchor;
        std::vector<rmm::device_uvector<int>> vec_d_candidates_per_anchor_prefixsum;

        MultiGpuErrorCorrectorInput(std::vector<int> deviceIds_, const std::vector<cudaStream_t> streams)
        : deviceIds(std::move(deviceIds_))
        {
            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                cudaStream_t stream = streams[g];

                vec_event.emplace_back(cudaEventDisableTiming);
                vec_h_candidate_read_ids_readyEvent.emplace_back(cudaEventDisableTiming);
                vec_d2hstream.emplace_back();
                vec_h_numAnchors.emplace_back(0);
                vec_h_numCandidates.emplace_back(0);
                vec_h_anchorReadIds.emplace_back(0);
                vec_h_candidate_read_ids.emplace_back(0);

                vec_d_numAnchors.emplace_back(0, stream);
                vec_d_numCandidates.emplace_back(0, stream);
                vec_d_anchorReadIds.emplace_back(0, stream);
                vec_d_anchor_sequences_data.emplace_back(0, stream);
                vec_d_anchor_sequences_lengths.emplace_back(0, stream);
                vec_d_candidate_read_ids.emplace_back(0, stream);
                vec_d_candidate_sequences_data.emplace_back(0, stream);
                vec_d_candidate_sequences_lengths.emplace_back(0, stream);
                vec_d_candidates_per_anchor.emplace_back(0, stream);
                vec_d_candidates_per_anchor_prefixsum.emplace_back(0, stream);
            }
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                cudaStream_t stream = streams[g];
                CUDACHECK(cudaStreamSynchronize(stream));
            }
        }

        ~MultiGpuErrorCorrectorInput(){
            //ensure release of memory on the correct device
            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd(deviceIds[g]);
                CUDACHECK(cudaDeviceSynchronize());
                vec_d_numAnchors[g].release();
                vec_d_numCandidates[g].release();
                vec_d_anchorReadIds[g].release();
                vec_d_anchor_sequences_data[g].release();
                vec_d_anchor_sequences_lengths[g].release();
                vec_d_candidate_read_ids[g].release();
                vec_d_candidate_sequences_data[g].release();
                vec_d_candidate_sequences_lengths[g].release();
                vec_d_candidates_per_anchor[g].release();
                vec_d_candidates_per_anchor_prefixsum[g].release();
                CUDACHECK(cudaDeviceSynchronize());
            }
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            auto handleHost = [&](const auto& h){
                info.host += h.sizeInBytes();
            };
            auto handleDevice = [&](const auto& d, int g){
                using ElementType = typename std::remove_reference<decltype(d)>::type::value_type;
                info.device[g] += d.size() * sizeof(ElementType);
            };

            const int numGpus = deviceIds.size();

            for(int g = 0; g < numGpus; g++){

                handleHost(vec_h_numAnchors[g]);
                handleHost(vec_h_numCandidates[g]);
                handleHost(vec_h_anchorReadIds[g]);
                handleHost(vec_h_candidate_read_ids[g]);

                handleDevice(vec_d_numAnchors[g], g);
                handleDevice(vec_d_numCandidates[g], g);
                handleDevice(vec_d_anchorReadIds[g], g);
                handleDevice(vec_d_anchor_sequences_data[g], g);
                handleDevice(vec_d_anchor_sequences_lengths[g], g);
                handleDevice(vec_d_candidate_read_ids[g], g);
                handleDevice(vec_d_candidate_sequences_data[g], g);
                handleDevice(vec_d_candidate_sequences_lengths[g], g);
                handleDevice(vec_d_candidates_per_anchor[g], g);
                handleDevice(vec_d_candidates_per_anchor_prefixsum[g], g);
            }

            return info;
        }  
    };


    class MultiGpuAnchorHasher{
    public:

        MultiGpuAnchorHasher() = default;

        MultiGpuAnchorHasher(
            const GpuReadStorage& gpuReadStorage_,
            const GpuMinhasher& gpuMinhasher_,
            std::vector<int> deviceIds_
        ) : deviceIds(std::move(deviceIds_)),
            gpuReadStorage{&gpuReadStorage_},
            gpuMinhasher{&gpuMinhasher_}
        {
            //assert(gpuMinhasher->hasGpuTables());
            // assert(!gpuReadStorage->hasHostSequences());
            // assert(!gpuReadStorage->hasHostQualities());

            maxCandidatesPerRead = gpuMinhasher->getNumResultsPerMapThreshold() * gpuMinhasher->getNumberOfMaps();
            encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(gpuReadStorage->getSequenceLengthUpperBound());
            qualityPitchInBytes = SDIV(gpuReadStorage->getSequenceLengthUpperBound(), 32) * 32;

            h_tempstorage.resize(128 * deviceIds.size());

            for(auto deviceId : deviceIds){
                cub::SwitchDevice sd{deviceId};
                vec_previousBatchFinishedEvent.emplace_back(cudaEventDisableTiming);
                vec_minhashHandle.push_back(gpuMinhasher->makeMinhasherHandle());
                vec_readstorageHandle.push_back(gpuReadStorage->makeHandle());
            }

        }

        ~MultiGpuAnchorHasher(){
            const int numGpus = deviceIds.size();

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                gpuReadStorage->destroyHandle(vec_readstorageHandle[g]);
                gpuMinhasher->destroyHandle(vec_minhashHandle[g]);
            }
        }

        void makeErrorCorrectorInput(
            const read_number* anchorIds,
            int numIds,
            bool useQualityScores,
            MultiGpuErrorCorrectorInput& ecinput,
            const std::vector<cudaStream_t>& streams
        ){
            assert(ecinput.deviceIds == deviceIds);

            const int numGpus = deviceIds.size();

            //batchsize must be even
            assert(numIds % 2 == 0);

            //distribute anchorIds amongst the gpus
            const int numIdPairs = numIds / 2;
            const int numIdPairsPerGpu = numIdPairs / numGpus;
            std::vector<int> numIdsPerGpu(numGpus, numIdPairsPerGpu * 2);
            const int numLeftoverIdPairs = numIdPairs - numGpus * numIdPairsPerGpu;
            assert(numLeftoverIdPairs < numGpus);
            for(int i = 0; i < numLeftoverIdPairs; i++){
                numIdsPerGpu[i] += 2;
            }

            // std::vector<int> numIdsPerGpu(numGpus, numIds / numGpus);
            // const int numLeftoverIds = numIds % numGpus;
            // for(int i = 0; i < numLeftoverIds; i++){
            //     numIdsPerGpu[i]++;
            // }
            std::vector<int> numIdsPerGpuPS(numGpus, 0);
            for(int g = 1; g < numGpus; g++){
                numIdsPerGpuPS[g] = numIdsPerGpuPS[g-1] + numIdsPerGpu[g-1];
            }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                assert(cudaSuccess == ecinput.vec_event[g].query());
                CUDACHECK(vec_previousBatchFinishedEvent[g].synchronize());
            }

            resizeBuffers(ecinput, numIdsPerGpu, streams);

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                *ecinput.vec_h_numAnchors[g].data() = numIdsPerGpu[g];
                std::copy_n(anchorIds + numIdsPerGpuPS[g], numIdsPerGpu[g], ecinput.vec_h_anchorReadIds[g].data());

                CUDACHECK(cudaMemcpyAsync(
                    ecinput.vec_d_numAnchors[g].data(),
                    ecinput.vec_h_numAnchors[g].data(),
                    sizeof(int),
                    H2D,
                    streams[g]
                ));

                CUDACHECK(cudaMemcpyAsync(
                    ecinput.vec_d_anchorReadIds[g].data(),
                    ecinput.vec_h_anchorReadIds[g].data(),
                    sizeof(read_number) * (*ecinput.vec_h_numAnchors[g].data()),
                    H2D,
                    streams[g]
                ));
            }

            nvtx::push_range("getAnchorReads", 0);
            getAnchorReads(ecinput, useQualityScores, streams);
            nvtx::pop_range();

            nvtx::push_range("getCandidateReadIdsWithMinhashing", 1);
            getCandidateReadIdsWithMinhashing(ecinput, streams);
            nvtx::pop_range();

            getCandidateReads(ecinput, useQualityScores, streams);
    
            // for(int g = 0; g < numGpus; g++){
            //     cub::SwitchDevice sd{deviceIds[g]};
            //     CUDACHECK(ecinput.vec_event[g].record(streams[g]));
            //     CUDACHECK(vec_previousBatchFinishedEvent[g].record(streams[g]));
            // }
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
       
            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                info += gpuMinhasher->getMemoryInfo(vec_minhashHandle[g]);
                info += gpuReadStorage->getMemoryInfo(vec_readstorageHandle[g]);
            }
            return info;
        } 

    public: //private:
        void resizeBuffers(
            MultiGpuErrorCorrectorInput& ecinput, 
            const std::vector<int>& numAnchorsPerGpu, 
            const std::vector<cudaStream_t>& streams
        ){
            assert(numAnchorsPerGpu.size() == ecinput.deviceIds.size());

            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                const int numAnchors = numAnchorsPerGpu[g];
                cudaStream_t stream = streams[g];

                ecinput.vec_h_numAnchors[g].resize(1);
                ecinput.vec_h_numCandidates[g].resize(1);
                ecinput.vec_h_anchorReadIds[g].resize(numAnchors);

                ecinput.vec_d_numAnchors[g].resize(1, stream);
                ecinput.vec_d_numCandidates[g].resize(1, stream);
                ecinput.vec_d_anchorReadIds[g].resize(numAnchors, stream);
                ecinput.vec_d_anchor_sequences_data[g].resize(encodedSequencePitchInInts * numAnchors, stream);
                ecinput.vec_d_anchor_sequences_lengths[g].resize(numAnchors, stream);
                ecinput.vec_d_candidates_per_anchor[g].resize(numAnchors, stream);
                ecinput.vec_d_candidates_per_anchor_prefixsum[g].resize(numAnchors + 1, stream);

            }
        }
        
        void getAnchorReads(
            MultiGpuErrorCorrectorInput& ecinput, 
            bool /*useQualityScores*/, 
            const std::vector<cudaStream_t>& streams
        ){
            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                const int numAnchors = (*ecinput.vec_h_numAnchors[g].data());

                if(numAnchors > 0){

                    gpuReadStorage->gatherContiguousSequences(
                        vec_readstorageHandle[g],
                        ecinput.vec_d_anchor_sequences_data[g].data(),
                        encodedSequencePitchInInts,
                        ecinput.vec_h_anchorReadIds[g][0],
                        numAnchors,
                        streams[g],
                        rmm::mr::get_current_device_resource()
                    );

                    gpuReadStorage->gatherSequenceLengths(
                        vec_readstorageHandle[g],
                        ecinput.vec_d_anchor_sequences_lengths[g].data(),
                        ecinput.vec_d_anchorReadIds[g].data(),
                        numAnchors,
                        streams[g]
                    );
                }
            }
            
        }

        void getCandidateReads(
            MultiGpuErrorCorrectorInput& ecinput, 
            bool /*useQualityScores*/, 
            const std::vector<cudaStream_t>& streams
        ){
            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                const int numCandidates = (*ecinput.vec_h_numCandidates[g].data());
                ecinput.vec_d_candidate_sequences_lengths[g].resize(numCandidates, streams[g]);

                if(numCandidates > 0){

                    gpuReadStorage->gatherSequenceLengths(
                        vec_readstorageHandle[g],
                        ecinput.vec_d_candidate_sequences_lengths[g].data(),
                        ecinput.vec_d_candidate_read_ids[g].data(),
                        numCandidates,
                        streams[g]
                    );
                }
            }

            std::vector<unsigned int*> vec_d_sequence_data(numGpus, nullptr);
            std::vector<read_number*> vec_d_readIds(numGpus, nullptr);
            std::vector<AsyncConstBufferWrapper<read_number>> vec_h_readIdsAsync(numGpus);
            std::vector<int> vec_numSequences(numGpus, 0);
            std::vector<rmm::mr::device_memory_resource*> mrs(numGpus, nullptr);

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                const int numCandidates = (*ecinput.vec_h_numCandidates[g].data());
                ecinput.vec_d_candidate_sequences_data[g].resize(encodedSequencePitchInInts * numCandidates, streams[g]);

                vec_d_sequence_data[g] = ecinput.vec_d_candidate_sequences_data[g].data();
                vec_d_readIds[g] = ecinput.vec_d_candidate_read_ids[g].data();
                vec_numSequences[g] = numCandidates;
                mrs[g] = rmm::mr::get_current_device_resource();

                //h_candidate_read_ids will only be valid if readstorage has host sequences or host qualities
                vec_h_readIdsAsync[g] = makeAsyncConstBufferWrapper(
                    ecinput.vec_h_candidate_read_ids[g].data(),
                    ecinput.vec_h_candidate_read_ids_readyEvent[g]
                );
            }

            
            gpuReadStorage->multi_gatherSequences(
                vec_readstorageHandle,
                vec_d_sequence_data,
                encodedSequencePitchInInts,
                vec_h_readIdsAsync,
                vec_d_readIds,
                vec_numSequences,
                streams,
                deviceIds,
                mrs
            );
            // {
            //     int N = 0;
            //     CUDACHECK(cudaGetDeviceCount(&N));
            //     for(int i = 0; i < N; i++){
            //         cub::SwitchDevice sd(cudaSetDevice(i));
            //         CUDACHECK(cudaDeviceSynchronize());
            //     }
            // }

            // gpuReadStorage->multi_gatherSequences(
            //     vec_readstorageHandle,
            //     vec_d_sequence_data,
            //     encodedSequencePitchInInts,
            //     vec_d_readIds,
            //     vec_numSequences,
            //     streams,
            //     deviceIds,
            //     mrs
            // );
        }

        void getCandidateReadIdsWithMinhashing(
            MultiGpuErrorCorrectorInput& ecinput, 
            const std::vector<cudaStream_t>& streams
        ){

            const int numGpus = deviceIds.size();

            std::vector<int> numAnchorsPerGpu(numGpus);
            for(int g = 0; g < numGpus; g++){
                const int numAnchors = *ecinput.vec_h_numAnchors[g].data();
                numAnchorsPerGpu[g] = numAnchors;
            }

            int* h_pinned_totalNumValuesTmp = h_tempstorage.data();
            std::fill(h_pinned_totalNumValuesTmp, h_pinned_totalNumValuesTmp + numGpus, 0);

            //determine num values
            //const auto* multiGpuMinhasher = dynamic_cast<const care::gpu::MultiGpuMinhasher*>(gpuMinhasher);
            const auto* multiGpuMinhasher = dynamic_cast<const care::gpu::GpuMinhasherWithMultiQuery*>(gpuMinhasher);

            
            if(multiGpuMinhasher != nullptr){
                std::vector<const unsigned int*> vec_d_sequenceData2Bit(numGpus, nullptr);
                std::vector<const int*> vec_d_sequenceLengths(numGpus, nullptr);
                std::vector<int*> vec_d_numValuesPerSequence(numGpus, nullptr);
                std::vector<int*> vec_totalNumValues(numGpus, nullptr);
                std::vector<rmm::mr::device_memory_resource*> mrs(numGpus, nullptr);

                for(int g = 0; g < numGpus; g++){
                    vec_d_sequenceData2Bit[g] = ecinput.vec_d_anchor_sequences_data[g].data();
                    vec_d_sequenceLengths[g] = ecinput.vec_d_anchor_sequences_lengths[g].data();
                    vec_d_numValuesPerSequence[g] = ecinput.vec_d_candidates_per_anchor[g].data();
                    vec_totalNumValues[g] = &h_pinned_totalNumValuesTmp[g];
                    mrs[g] = rmm::mr::get_per_device_resource(rmm::cuda_device_id(deviceIds[g]));
                }

                multiGpuMinhasher->multi_determineNumValues(
                    vec_minhashHandle[0],
                    vec_d_sequenceData2Bit,
                    encodedSequencePitchInInts,
                    vec_d_sequenceLengths,
                    numAnchorsPerGpu,
                    vec_d_numValuesPerSequence,
                    vec_totalNumValues,
                    streams,
                    deviceIds,
                    mrs
                );

                // std::vector<std::vector<int>> vec_h_candidates_per_anchor_2(numGpus);
                // std::cout << "func2: ";
                // for(int g = 0; g < numGpus; g++){
                //     vec_h_candidates_per_anchor_2[g].resize(*ecinput.vec_h_numAnchors[g]);
                //     cub::SwitchDevice sd{deviceIds[g]};
                //     CUDACHECK(cudaMemcpy(
                //         vec_h_candidates_per_anchor_2[g].data(),
                //         vec_d_numValuesPerSequence[g],
                //         sizeof(int) * (*ecinput.vec_h_numAnchors[g]),
                //         D2H
                //     ));
                //     std::cout << *vec_totalNumValues[g] << " ";
                // }
                // std::cout << "\n";

                // multiGpuMinhasher->multi_determineNumValues_1(
                //     vec_minhashHandle,
                //     vec_d_sequenceData2Bit,
                //     encodedSequencePitchInInts,
                //     vec_d_sequenceLengths,
                //     numAnchorsPerGpu,
                //     vec_d_numValuesPerSequence,
                //     vec_totalNumValues,
                //     streams,
                //     deviceIds,
                //     mrs
                // );

                // multiGpuMinhasher->multi_determineNumValues_2(
                //     vec_minhashHandle[0],
                //     vec_d_sequenceData2Bit,
                //     encodedSequencePitchInInts,
                //     vec_d_sequenceLengths,
                //     numAnchorsPerGpu,
                //     vec_d_numValuesPerSequence,
                //     vec_totalNumValues,
                //     streams,
                //     deviceIds,
                //     mrs
                // );

                // std::vector<std::vector<int>> vec_h_candidates_per_anchor_1(numGpus);
                // std::cout << "func1: ";
                // for(int g = 0; g < numGpus; g++){
                //     vec_h_candidates_per_anchor_1[g].resize(*ecinput.vec_h_numAnchors[g]);
                //     cub::SwitchDevice sd{deviceIds[g]};
                //     CUDACHECK(cudaMemcpy(
                //         vec_h_candidates_per_anchor_1[g].data(),
                //         vec_d_numValuesPerSequence[g],
                //         sizeof(int) * (*ecinput.vec_h_numAnchors[g]),
                //         D2H
                //     ));
                //     std::cout << *vec_totalNumValues[g] << " ";
                // }
                // std::cout << "\n";

                // for(int g = 0; g < numGpus; g++){
                //     for(int i = 0; i < *ecinput.vec_h_numAnchors[g]; i++){
                //         if(vec_h_candidates_per_anchor_1[g][i] != vec_h_candidates_per_anchor_2[g][i]){
                //             std::cout << "error g " << g << ", i " << i << ", 1: " << vec_h_candidates_per_anchor_1[g][i] << ", 2: " << vec_h_candidates_per_anchor_2[g][i] << "\n";
                //             assert(false);
                //         }
                //     }
                // }

            }else{
                for(int g = 0; g < numGpus; g++){
                    cub::SwitchDevice sd{deviceIds[g]};

                    const int numAnchors = *ecinput.vec_h_numAnchors[g].data();
                    numAnchorsPerGpu[g] = numAnchors;
                    if(numAnchors > 0){
                        gpuMinhasher->determineNumValues(
                            vec_minhashHandle[g],
                            ecinput.vec_d_anchor_sequences_data[g].data(),
                            encodedSequencePitchInInts,
                            ecinput.vec_d_anchor_sequences_lengths[g].data(),
                            numAnchors,
                            ecinput.vec_d_candidates_per_anchor[g].data(),
                            h_pinned_totalNumValuesTmp[g],
                            streams[g],
                            rmm::mr::get_current_device_resource()
                        );
                    }
                }
            }

            //sync
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                CUDACHECK(cudaStreamSynchronize(streams[g]));
            }

            std::vector<int> totalNumValuesPerGpu(h_pinned_totalNumValuesTmp, h_pinned_totalNumValuesTmp + numGpus);

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                ecinput.vec_d_candidate_read_ids[g].resize(totalNumValuesPerGpu[g], streams[g]);

                if(gpuReadStorage->hasHostSequences() || gpuReadStorage->hasHostQualities()){
                    ecinput.vec_h_candidate_read_ids[g].resize(totalNumValuesPerGpu[g]);                
                }

                //std::cout << totalNumValuesPerGpu[g] << " ";
            }
            //std::cout << "\n";

            for(int g = 0; g < numGpus; g++){
                *ecinput.vec_h_numCandidates[g] = 0;
            }

            if(multiGpuMinhasher != nullptr){
                std::vector<read_number*> vec_d_values(numGpus, nullptr);
                std::vector<int*> vec_d_offsets(numGpus, nullptr);
                std::vector<const int*> vec_d_numValuesPerSequence(numGpus, nullptr);
                std::vector<const int*> vec_totalNumValues(numGpus, nullptr);
                std::vector<rmm::mr::device_memory_resource*> mrs(numGpus, nullptr);

                for(int g = 0; g < numGpus; g++){
                    vec_d_values[g] = ecinput.vec_d_candidate_read_ids[g].data();
                    vec_d_offsets[g] = ecinput.vec_d_candidates_per_anchor_prefixsum[g].data();
                    vec_d_numValuesPerSequence[g] = ecinput.vec_d_candidates_per_anchor[g].data();
                    vec_totalNumValues[g] = &h_pinned_totalNumValuesTmp[g];
                    mrs[g] = rmm::mr::get_per_device_resource(rmm::cuda_device_id(deviceIds[g]));
                }

                // multiGpuMinhasher->multi_retrieveValues_1(
                //     vec_minhashHandle,
                //     numAnchorsPerGpu,
                //     vec_totalNumValues,
                //     vec_d_values,
                //     vec_d_numValuesPerSequence,
                //     vec_d_offsets,
                //     streams,
                //     deviceIds,
                //     mrs
                // );
                multiGpuMinhasher->multi_retrieveValues(
                    vec_minhashHandle[0],
                    numAnchorsPerGpu,
                    vec_totalNumValues,
                    vec_d_values,
                    vec_d_numValuesPerSequence,
                    vec_d_offsets,
                    streams,
                    deviceIds,
                    mrs
                );
                // multiGpuMinhasher->multi_retrieveValues_2(
                //     vec_minhashHandle[0],
                //     numAnchorsPerGpu,
                //     vec_totalNumValues,
                //     vec_d_values,
                //     vec_d_numValuesPerSequence,
                //     vec_d_offsets,
                //     streams,
                //     deviceIds,
                //     mrs
                // );
                
                // std::vector<read_number*> vec_d_values_2(numGpus, nullptr);
                // std::vector<int*> vec_d_offsets_2(numGpus, nullptr);
                // std::vector<const int*> vec_d_numValuesPerSequence_2(numGpus, nullptr);
                // std::vector<const int*> vec_totalNumValues_2(numGpus, nullptr);
                // std::vector<rmm::mr::device_memory_resource*> mrs_2(numGpus, nullptr);

                // std::vector<rmm::device_uvector<read_number>> vec_d_values_2_rmm;
                // std::vector<rmm::device_uvector<int>> vec_d_offsets_2_rmm;
                // for(int g = 0; g < numGpus; g++){
                //     cub::SwitchDevice sd_(deviceIds[g]);
                //     vec_d_values_2_rmm.emplace_back(totalNumValuesPerGpu[g], streams[g]);
                //     vec_d_offsets_2_rmm.emplace_back(numAnchorsPerGpu[g]+1, streams[g]);

                //     vec_d_values_2[g] = vec_d_values_2_rmm.back().data();
                //     vec_d_offsets_2[g] = vec_d_offsets_2_rmm.back().data();
                //     vec_d_numValuesPerSequence_2[g] = ecinput.vec_d_candidates_per_anchor[g].data();
                //     vec_totalNumValues_2[g] = &h_pinned_totalNumValuesTmp[g];
                //     mrs_2[g] = rmm::mr::get_per_device_resource(rmm::cuda_device_id(deviceIds[g]));
                // }

                // multiGpuMinhasher->multi_retrieveValues_2(
                //     vec_minhashHandle[0],
                //     numAnchorsPerGpu,
                //     vec_totalNumValues_2,
                //     vec_d_values_2,
                //     vec_d_numValuesPerSequence_2,
                //     vec_d_offsets_2,
                //     streams,
                //     deviceIds,
                //     mrs_2
                // );

                // for(int g = 0; g < numGpus; g++){
                //     cub::SwitchDevice sd_(deviceIds[g]);
                //     const bool equalOffsets = (thrust::equal(
                //         rmm::exec_policy_nosync(streams[g]),
                //         vec_d_offsets[g],
                //         vec_d_offsets[g] + numAnchorsPerGpu[g]+1,
                //         vec_d_offsets_2[g]
                //     ));
                //     if(!equalOffsets){
                //         std::cout << "g = " << g << ", not equalOffsets\n";
                //         assert(false);
                //     }

                //     std::vector<read_number> oldcands(totalNumValuesPerGpu[g]);
                //     std::vector<read_number> newcands(totalNumValuesPerGpu[g]);
                //     std::vector<int> oldoffsets(numAnchorsPerGpu[g]+1);
                //     CUDACHECK(cudaMemcpy(
                //         oldcands.data(),
                //         vec_d_values[g],
                //         sizeof(read_number) * totalNumValuesPerGpu[g],
                //         D2H
                //     ));
                //     CUDACHECK(cudaMemcpy(
                //         newcands.data(),
                //         vec_d_values_2[g],
                //         sizeof(read_number) * totalNumValuesPerGpu[g],
                //         D2H
                //     ));
                //     CUDACHECK(cudaMemcpy(
                //         oldoffsets.data(),
                //         vec_d_offsets[g],
                //         sizeof(int) * (numAnchorsPerGpu[g]+1),
                //         D2H
                //     ));
                //     for(int a = 0; a < numAnchorsPerGpu[g]; a++){
                //         const int begin = oldoffsets[a];
                //         const int end = oldoffsets[a+1];
                //         const int num = end - begin;

                //         std::sort(oldcands.begin() + begin, oldcands.begin() + end);
                //         std::sort(newcands.begin() + begin, newcands.begin() + end);

                //         for(int c = 0; c < num; c++){
                //             if(oldcands[begin+c] != newcands[begin+c]){
                //                 std::cout << "cands error g " << g << ", a " << a << ", c " << c << "\n";
                //                 std::cout << "old\n";
                //                 for(int x = 0; x < num; x++){
                //                     std::cout << oldcands[begin+x] << " ";
                //                 }
                //                 std::cout << "\n";
                //                 std::cout << "new\n";
                //                 for(int x = 0; x < num; x++){
                //                     std::cout << newcands[begin+x] << " ";
                //                 }
                //                 std::cout << "\n";
                //                 assert(false);
                //             }
                //         }
                //     }
                // }

            }else{
                for(int g = 0; g < numGpus; g++){
                    cub::SwitchDevice sd{deviceIds[g]};

                    const int numAnchors = (*ecinput.vec_h_numAnchors[g]);
                    if(totalNumValuesPerGpu[g] == 0){
                        CUDACHECK(cudaMemsetAsync(ecinput.vec_d_numCandidates[g].data(), 0, sizeof(int), streams[g]));
                        CUDACHECK(cudaMemsetAsync(ecinput.vec_d_candidates_per_anchor[g].data(), 0, sizeof(int) * numAnchors, streams[g]));
                        CUDACHECK(cudaMemsetAsync(ecinput.vec_d_candidates_per_anchor_prefixsum[g].data(), 0, sizeof(int) * (1 + numAnchors), streams[g]));
                    }else{
                        gpuMinhasher->retrieveValues(
                            vec_minhashHandle[g],
                            numAnchors,                
                            totalNumValuesPerGpu[g],
                            ecinput.vec_d_candidate_read_ids[g].data(),
                            ecinput.vec_d_candidates_per_anchor[g].data(),
                            ecinput.vec_d_candidates_per_anchor_prefixsum[g].data(),
                            streams[g],
                            rmm::mr::get_current_device_resource()
                        );
                    }
                }
            }

            std::vector<rmm::device_uvector<read_number>> vec_d_candidate_read_ids2;
            std::vector<rmm::device_uvector<int>> vec_d_candidates_per_anchor2;
            std::vector<rmm::device_uvector<int>> vec_d_candidates_per_anchor_prefixsum2;

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                const int numAnchors = (*ecinput.vec_h_numAnchors[g]);

                vec_d_candidate_read_ids2.emplace_back(totalNumValuesPerGpu[g], streams[g]);
                vec_d_candidates_per_anchor2.emplace_back(numAnchors, streams[g]);
                vec_d_candidates_per_anchor_prefixsum2.emplace_back(1 + numAnchors, streams[g]);
            }

            std::vector<cub::DoubleBuffer<read_number>> vec_d_items;
            std::vector<cub::DoubleBuffer<int>> vec_d_numItemsPerSegment;
            std::vector<cub::DoubleBuffer<int>> vec_d_numItemsPerSegmentPrefixSum;
            std::vector<const read_number*> vec_d_anchorReadIds_ptrs;

            for(int g = 0; g < numGpus; g++){
                vec_d_items.emplace_back(
                    ecinput.vec_d_candidate_read_ids[g].data(), 
                    vec_d_candidate_read_ids2[g].data()
                );
                vec_d_numItemsPerSegment.emplace_back(
                    ecinput.vec_d_candidates_per_anchor[g].data(), 
                    vec_d_candidates_per_anchor2[g].data()
                );
                vec_d_numItemsPerSegmentPrefixSum.emplace_back(
                    ecinput.vec_d_candidates_per_anchor_prefixsum[g].data(), 
                    vec_d_candidates_per_anchor_prefixsum2[g].data()
                );
                vec_d_anchorReadIds_ptrs.push_back(ecinput.vec_d_anchorReadIds[g].data());
            }

            GpuMinhashQueryFilter::keepDistinctAndNotMatching(
                vec_d_anchorReadIds_ptrs,
                vec_d_items,
                vec_d_numItemsPerSegment,
                vec_d_numItemsPerSegmentPrefixSum, //numSegments + 1
                numAnchorsPerGpu,
                totalNumValuesPerGpu,
                streams,
                deviceIds,
                h_tempstorage.data()
            );

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                if(vec_d_items[g].Current() != ecinput.vec_d_candidate_read_ids[g].data()){
                    //std::cerr << "swap d_candidate_read_ids\n";
                    std::swap(ecinput.vec_d_candidate_read_ids[g], vec_d_candidate_read_ids2[g]);
                }
                if(vec_d_numItemsPerSegment[g].Current() != ecinput.vec_d_candidates_per_anchor[g].data()){
                    //std::cerr << "swap d_candidates_per_anchor\n";
                    std::swap(ecinput.vec_d_candidates_per_anchor[g], vec_d_candidates_per_anchor2[g]);
                }
                if(vec_d_numItemsPerSegmentPrefixSum[g].Current() != ecinput.vec_d_candidates_per_anchor_prefixsum[g].data()){
                    //std::cerr << "swap d_candidates_per_anchor_prefixsum\n";
                    std::swap(ecinput.vec_d_candidates_per_anchor_prefixsum[g], vec_d_candidates_per_anchor_prefixsum2[g]);
                }

                gpucorrectorkernels::copyMinhashResultsKernel<<<1, 1, 0, streams[g]>>>(
                    ecinput.vec_d_numCandidates[g].data(),
                    ecinput.vec_h_numCandidates[g].data(),
                    ecinput.vec_d_candidates_per_anchor_prefixsum[g].data(),
                    numAnchorsPerGpu[g]
                ); CUDACHECKASYNC;
            }

            for(int g = 0; g < numGpus; g++){
                //wait for copyMinhashResultsKernel (writes ecinput.vec_h_numCandidates[g])
                cub::SwitchDevice sd{deviceIds[g]};
                CUDACHECK(cudaStreamSynchronize(streams[g]));

                //ensure release of memory on the correct device
                vec_d_candidate_read_ids2[g].release();
                vec_d_candidates_per_anchor2[g].release();
                vec_d_candidates_per_anchor_prefixsum2[g].release();

                if(gpuReadStorage->hasHostSequences() || gpuReadStorage->hasHostQualities()){
                    CUDACHECK(cudaMemcpyAsync(
                        ecinput.vec_h_candidate_read_ids[g].data(),
                        ecinput.vec_d_candidate_read_ids[g].data(),
                        sizeof(read_number) * ecinput.vec_h_numCandidates[g][0],
                        D2H,
                        ecinput.vec_d2hstream[g]
                    ));
                    CUDACHECK(cudaEventRecord(ecinput.vec_h_candidate_read_ids_readyEvent[g], ecinput.vec_d2hstream[g]));
                }
            }



        }
    
        std::vector<int> deviceIds;
        int maxCandidatesPerRead;
        std::size_t encodedSequencePitchInInts;
        std::size_t qualityPitchInBytes;
        std::vector<CudaEvent> vec_previousBatchFinishedEvent;
        const GpuReadStorage* gpuReadStorage;
        const GpuMinhasher* gpuMinhasher;
        std::vector<MinhasherHandle> vec_minhashHandle;
        std::vector<ReadStorageHandle> vec_readstorageHandle;
        helpers::SimpleAllocationPinnedHost<int> h_tempstorage;
    };

#if 1

    class MultiGpuErrorCorrector{

    public:

        template<class T>
        using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

        static constexpr int getNumRefinementIterations() noexcept{
            return 5;
        }

        static constexpr bool useMsaRefinement() noexcept{
            return getNumRefinementIterations() > 0;
        }

        MultiGpuErrorCorrector() = default;

        MultiGpuErrorCorrector(
            const GpuReadStorage& gpuReadStorage_,
            GpuReadCorrectionFlags& correctionFlags_,
            const ProgramOptions& programOptions_,
            int maxAnchorsPerCall,
            std::vector<int> deviceIds_,
            std::vector<const GpuForest*> vec_gpuForestAnchor_,
            std::vector<const GpuForest*> vec_gpuForestCandidate_,
            const std::vector<cudaStream_t>& streams
        ) : 
            maxAnchors{maxAnchorsPerCall},
            correctionFlags{&correctionFlags_},
            gpuReadStorage{&gpuReadStorage_},
            programOptions{&programOptions_},
            deviceIds{std::move(deviceIds_)},
            vec_gpuForestAnchor{std::move(vec_gpuForestAnchor_)},
            vec_gpuForestCandidate{std::move(vec_gpuForestCandidate_)}
            
        {
            std::cout << "MultiGpuErrorCorrector\n";
            const int numGpus = deviceIds.size();


            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                std::array<CudaEvent, 2> events{cudaEventDisableTiming,cudaEventDisableTiming};
                vec_events.push_back(std::move(events));
                vec_previousBatchFinishedEvent.emplace_back(cudaEventDisableTiming);
                vec_inputCandidateDataIsReadyEvent.emplace_back(cudaEventDisableTiming);                
            }

            h_tempstorage.resize(128 * numGpus);

            

            encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(gpuReadStorage->getSequenceLengthUpperBound());
            decodedSequencePitchInBytes = SDIV(gpuReadStorage->getSequenceLengthUpperBound(), 4) * 4;
            qualityPitchInBytes = SDIV(gpuReadStorage->getSequenceLengthUpperBound(), 32) * 32;
            maxNumEditsPerSequence = std::max(1,gpuReadStorage->getSequenceLengthUpperBound() / 7);
            //pad to multiple of 128 bytes
            editsPitchInBytes = SDIV(maxNumEditsPerSequence * sizeof(EncodedCorrectionEdit), 128) * 128;

            const std::size_t min_overlap = std::max(
                1, 
                std::max(
                    programOptions->min_overlap, 
                    int(gpuReadStorage->getSequenceLengthUpperBound() * programOptions->min_overlap_ratio)
                )
            );
            const std::size_t msa_max_column_count = (3*gpuReadStorage->getSequenceLengthUpperBound() - 2*min_overlap);
            //round up to 32 elements
            msaColumnPitchInElements = SDIV(msa_max_column_count, 32) * 32;

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                //vec_extraStream.push_back(streampool::get_current_device_pool()->get_stream());
                vec_extraStream.emplace_back();

                vec_h_num_total_corrected_candidates.emplace_back(0);
                vec_h_num_indices.emplace_back(0);
                vec_h_numSelected.emplace_back(0);
                vec_h_managedmsa_tmp.emplace_back(0);

                vec_d_alignment_overlaps.emplace_back();
                vec_d_alignment_shifts.emplace_back();
                vec_d_alignment_nOps.emplace_back();
                vec_d_alignment_best_alignment_flags.emplace_back();
                vec_d_anchorIndicesOfCandidates.emplace_back();
                vec_d_candidateContainsN.emplace_back();
                vec_d_isPairedCandidate.emplace_back();
                vec_d_indices_of_corrected_candidates.emplace_back();
                vec_d_hqAnchorCorrectionOfCandidateExists.emplace_back();
                vec_d_candidate_read_ids.emplace_back();
                vec_d_candidate_sequences_lengths.emplace_back();
                vec_d_candidate_sequences_data.emplace_back();

                vec_managedgpumsa.emplace_back();
                vec_readstorageHandle.emplace_back(gpuReadStorage->makeHandle());
                vec_d_indicesForGather.emplace_back(0, streams[g]);
                vec_d_anchorContainsN.emplace_back(0, streams[g]);
                vec_d_candidateContainsN_.emplace_back(0, streams[g]);
                vec_d_candidate_sequences_lengths_.emplace_back(0, streams[g]);
                vec_d_candidate_sequences_data_.emplace_back(0, streams[g]);
                vec_d_anchorIndicesOfCandidates_.emplace_back(0, streams[g]);
                vec_d_alignment_overlaps_.emplace_back(0, streams[g]);
                vec_d_alignment_shifts_.emplace_back(0, streams[g]);
                vec_d_alignment_nOps_.emplace_back(0, streams[g]);
                vec_d_alignment_best_alignment_flags_.emplace_back(0, streams[g]); 
                vec_d_indices.emplace_back(0, streams[g]);
                vec_d_indices_per_anchor.emplace_back(0, streams[g]);
                vec_d_indices_per_anchor_prefixsum.emplace_back(0, streams[g]);
                vec_d_num_indices.emplace_back(0, streams[g]);
                vec_d_corrected_anchors.emplace_back(0, streams[g]);
                vec_d_corrected_candidates.emplace_back(0, streams[g]);
                vec_d_num_corrected_candidates_per_anchor.emplace_back(0, streams[g]);
                vec_d_num_corrected_candidates_per_anchor_prefixsum.emplace_back(0, streams[g]);
                vec_d_num_total_corrected_candidates.emplace_back(0, streams[g]);
                vec_d_anchor_is_corrected.emplace_back(0, streams[g]);
                vec_d_is_high_quality_anchor.emplace_back(0, streams[g]);
                vec_d_high_quality_anchor_indices.emplace_back(0, streams[g]);
                vec_d_num_high_quality_anchor_indices.emplace_back(0, streams[g]); 
                vec_d_editsPerCorrectedanchor.emplace_back(0, streams[g]);
                vec_d_numEditsPerCorrectedanchor.emplace_back(0, streams[g]);
                vec_d_editsPerCorrectedCandidate.emplace_back(0, streams[g]);
                vec_d_hqAnchorCorrectionOfCandidateExists_.emplace_back(0, streams[g]);
                vec_d_allCandidateData.emplace_back(0, streams[g]);
                vec_d_numEditsPerCorrectedCandidate.emplace_back(0, streams[g]);
                vec_d_indices_of_corrected_anchors.emplace_back(0, streams[g]);
                vec_d_num_indices_of_corrected_anchors.emplace_back(0, streams[g]);
                vec_d_indices_of_corrected_candidates_.emplace_back(0, streams[g]);
                vec_d_totalNumEdits.emplace_back(0, streams[g]);
                vec_d_isPairedCandidate_.emplace_back(0, streams[g]);
                vec_d_numAnchors.emplace_back(0, streams[g]);
                vec_d_numCandidates.emplace_back(0, streams[g]);
                vec_d_anchorReadIds.emplace_back(0, streams[g]);
                vec_d_anchor_sequences_data.emplace_back(0, streams[g]);
                vec_d_anchor_sequences_lengths.emplace_back(0, streams[g]);
                vec_d_candidate_read_ids_.emplace_back(0, streams[g]);
                vec_d_candidates_per_anchor.emplace_back(0, streams[g]);
                vec_d_candidates_per_anchor_prefixsum.emplace_back(0, streams[g]);
            }

            initFixedSizeBuffers(streams);

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                CUDACHECK(cudaStreamSynchronize(streams[g]));
            }
        }

        ~MultiGpuErrorCorrector(){
            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                gpuReadStorage->destroyHandle(vec_readstorageHandle[g]);

                //ensure release of memory on the correct device

                vec_managedgpumsa[g] = nullptr;

                vec_d_indicesForGather[g].release();
                vec_d_anchorContainsN[g].release();
                vec_d_candidateContainsN_[g].release();
                vec_d_candidate_sequences_lengths_[g].release();
                vec_d_candidate_sequences_data_[g].release();
                vec_d_anchorIndicesOfCandidates_[g].release();
                vec_d_alignment_overlaps_[g].release();
                vec_d_alignment_shifts_[g].release();
                vec_d_alignment_nOps_[g].release();
                vec_d_alignment_best_alignment_flags_[g].release(); 
                vec_d_indices[g].release();
                vec_d_indices_per_anchor[g].release();
                vec_d_indices_per_anchor_prefixsum[g].release();
                vec_d_num_indices[g].release();
                vec_d_corrected_anchors[g].release();
                vec_d_corrected_candidates[g].release();
                vec_d_num_corrected_candidates_per_anchor[g].release();
                vec_d_num_corrected_candidates_per_anchor_prefixsum[g].release();
                vec_d_num_total_corrected_candidates[g].release();
                vec_d_anchor_is_corrected[g].release();
                vec_d_is_high_quality_anchor[g].release();
                vec_d_high_quality_anchor_indices[g].release();
                vec_d_num_high_quality_anchor_indices[g].release(); 
                vec_d_editsPerCorrectedanchor[g].release();
                vec_d_numEditsPerCorrectedanchor[g].release();
                vec_d_editsPerCorrectedCandidate[g].release();
                vec_d_hqAnchorCorrectionOfCandidateExists_[g].release();
                vec_d_allCandidateData[g].release();
                vec_d_numEditsPerCorrectedCandidate[g].release();
                vec_d_indices_of_corrected_anchors[g].release();
                vec_d_num_indices_of_corrected_anchors[g].release();
                vec_d_indices_of_corrected_candidates_[g].release();
                vec_d_totalNumEdits[g].release();
                vec_d_isPairedCandidate_[g].release();
                vec_d_numAnchors[g].release();
                vec_d_numCandidates[g].release();
                vec_d_anchorReadIds[g].release();
                vec_d_anchor_sequences_data[g].release();
                vec_d_anchor_sequences_lengths[g].release();
                vec_d_candidate_read_ids_[g].release();
                vec_d_candidates_per_anchor[g].release();
                vec_d_candidates_per_anchor_prefixsum[g].release();
            }
        }

        void correct(
            MultiGpuErrorCorrectorInput& input, 
            std::vector<GpuErrorCorrectorRawOutput>& outputs, 
            const std::vector<cudaStream_t>& streams
        ){
            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                CUDACHECK(cudaEventSynchronize(vec_previousBatchFinishedEvent[g]));
            }

            currentInput = &input;

            vec_currentNumAnchors.resize(numGpus);
            vec_currentNumCandidates.resize(numGpus);
            vec_currentOutput.resize(numGpus);

            // for(int g = 0; g < numGpus; g++){
            //     std::cout << "gpu " << g << ", numAnchors " << currentInput->vec_h_numAnchors[g][0] << ", numCandidates " << currentInput->vec_h_numCandidates[g][0] << "\n";
            // }

            for(int g = 0; g < numGpus; g++){

                vec_currentNumAnchors[g] = currentInput->vec_h_numAnchors[g][0];
                vec_currentNumCandidates[g] = currentInput->vec_h_numCandidates[g][0];
                assert(vec_currentNumAnchors[g] <= maxAnchors);
                if(gpuReadStorage->isPairedEnd()){
                    if(vec_currentNumAnchors[g] % 2 != 0){
                        std::cout << "vec_currentNumAnchors[" << g << "] = " << vec_currentNumAnchors[g] << "\n";
                    }
                    assert(vec_currentNumAnchors[g] % 2 == 0);
                }

                vec_currentOutput[g] = &outputs[g];
                vec_currentOutput[g]->nothingToDo = false;
                vec_currentOutput[g]->numAnchors = vec_currentNumAnchors[g];
                vec_currentOutput[g]->h_numCorrectedAnchors.resize(1);
                *vec_currentOutput[g]->h_numCorrectedAnchors = 0;
                vec_currentOutput[g]->numCorrectedCandidates = 0;

                if(vec_currentNumCandidates[g] == 0 ||  vec_currentNumAnchors[g] == 0){
                    vec_currentOutput[g]->nothingToDo = true;
                }
            }




            // {
            //     std::vector<int> h_candidates_per_anchor_prefixsum(currentNumAnchors+1);
            //     CUDACHECK(cudaMemcpyAsync(
            //         h_candidates_per_anchor_prefixsum.data(), 
            //         currentInput->d_candidates_per_anchor_prefixsum.data(),
            //         sizeof(int) * (currentNumAnchors+1),
            //         D2H,
            //         stream
            //     ));
            //     CUDACHECK(cudaStreamSynchronize(stream));

            //     const int numgpus = 8;

            //     std::vector<size_t> bucketLimits(1,0);
            //     size_t currentBegin = 0;
            //     const size_t end = currentNumCandidates;
            //     while(currentBegin < end){
            //         const size_t searchBegin = currentBegin + (currentNumCandidates / numgpus);
            //         const auto it = std::upper_bound(
            //             h_candidates_per_anchor_prefixsum.begin(), 
            //             h_candidates_per_anchor_prefixsum.end(), 
            //             searchBegin
            //         );
            //         if(it == h_candidates_per_anchor_prefixsum.end()){
            //             bucketLimits.push_back(currentNumAnchors);
            //             currentBegin = end;
            //         }else{
            //             const size_t dist = std::distance(h_candidates_per_anchor_prefixsum.begin(), it);
            //             bucketLimits.push_back(dist);
            //             currentBegin = h_candidates_per_anchor_prefixsum[dist];
            //         }
            //     }
            //     for(auto x : bucketLimits){
            //         std::cout << x << " ";
            //     }
            //     std::cout << "\n";
            //     for(int i = 0; i < bucketLimits.size() - 1; i++){
            //         const int from = h_candidates_per_anchor_prefixsum[bucketLimits[i]];
            //         const int to = h_candidates_per_anchor_prefixsum[bucketLimits[i+1]];
            //         std::cout << (to - from) << " ";
            //     }
            //     std::cout << "\n";
            // }

            //fixed size memory should already be allocated. However, this will also set the correct working stream for stream-ordered allocations which is important.
            initFixedSizeBuffers(streams); 

            resizeBuffers(vec_currentNumAnchors, vec_currentNumCandidates, streams);

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                const int currentNumAnchors = vec_currentNumAnchors[g];
                const int currentNumCandidates = vec_currentNumCandidates[g];
                cudaStream_t stream = streams[g];

                dim3 block1 = 256;
                dim3 grid1 = std::max(1, SDIV(currentNumCandidates, 256));

                gpucorrectorkernels::copyCorrectionInputDeviceData<<<grid1, block1, 0, stream>>>(
                    vec_d_numAnchors[g].data(),
                    vec_d_numCandidates[g].data(),
                    vec_d_anchorReadIds[g].data(),
                    vec_d_anchor_sequences_data[g].data(),
                    vec_d_anchor_sequences_lengths[g].data(),
                    vec_d_candidate_read_ids[g],
                    vec_d_candidates_per_anchor[g].data(),
                    vec_d_candidates_per_anchor_prefixsum[g].data(),
                    encodedSequencePitchInInts,
                    currentNumAnchors,
                    currentNumCandidates,
                    currentInput->vec_d_anchorReadIds[g].data(),
                    currentInput->vec_d_anchor_sequences_data[g].data(),
                    currentInput->vec_d_anchor_sequences_lengths[g].data(),
                    currentInput->vec_d_candidate_read_ids[g].data(),
                    currentInput->vec_d_candidates_per_anchor[g].data(),
                    currentInput->vec_d_candidates_per_anchor_prefixsum[g].data()
                ); CUDACHECKASYNC;

                CUDACHECK(cudaMemcpyAsync(
                    vec_d_candidate_sequences_data[g],
                    currentInput->vec_d_candidate_sequences_data[g].data(),
                    sizeof(unsigned int) * encodedSequencePitchInInts * currentNumCandidates,
                    D2D,
                    stream
                ));

                CUDACHECK(cudaMemcpyAsync(
                    vec_d_candidate_sequences_lengths[g],
                    currentInput->vec_d_candidate_sequences_lengths[g].data(),
                    sizeof(int) * currentNumCandidates,
                    D2D,
                    stream
                ));

                CUDACHECK(cudaEventRecord(vec_inputCandidateDataIsReadyEvent[g], stream));

                gpucorrectorkernels::setAnchorIndicesOfCandidateskernel
                        <<<std::max(1, currentNumAnchors), 128, 0, stream>>>(
                    vec_d_anchorIndicesOfCandidates[g],
                    vec_d_numAnchors[g].data(),
                    vec_d_candidates_per_anchor[g].data(),
                    vec_d_candidates_per_anchor_prefixsum[g].data()
                ); CUDACHECKASYNC;
            }

            flagPairedCandidates(streams);

            getAmbiguousFlagsOfAnchors(streams);
            getAmbiguousFlagsOfCandidates(streams);


            // nvtx::push_range("getCandidateSequenceData", 3);
            // getCandidateSequenceData(stream); 
            // nvtx::pop_range();

            nvtx::push_range("getCandidateAlignments", 5);
            getCandidateAlignments(streams); 
            nvtx::pop_range();

            nvtx::push_range("buildMultipleSequenceAlignment", 6);
            buildAndRefineMultipleSequenceAlignment(streams);
            nvtx::pop_range();

            nvtx::push_range("correctanchors", 8);
            correctAnchors(streams);
            updateCorrectionFlags(streams);
            nvtx::pop_range();
            
            if(programOptions->correctCandidates) {
                for(int g = 0; g < numGpus; g++){
                    cub::SwitchDevice sd{deviceIds[g]};
                    correctionFlags->isCorrectedAsHQAnchor(
                        vec_d_hqAnchorCorrectionOfCandidateExists[g], 
                        vec_d_candidate_read_ids[g], 
                        vec_currentNumCandidates[g], 
                        streams[g]
                    );
                }

                nvtx::push_range("correctCandidates", 9);
                correctCandidates(streams);
                nvtx::pop_range();
            }
            

            nvtx::push_range("copyAnchorResultsFromDeviceToHost", 3);
            copyAnchorResultsFromDeviceToHost(streams);
            nvtx::pop_range();

            if(programOptions->correctCandidates) { 
                nvtx::push_range("copyCandidateResultsFromDeviceToHost", 4);
                copyCandidateResultsFromDeviceToHost(streams);
                nvtx::pop_range();   
            }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                //ensure release of memory on the correct device
                vec_managedgpumsa[g] = nullptr;
            }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                //after the current work in stream is completed, all results in currentOutput are ready to use.
                CUDACHECK(cudaEventRecord(vec_currentOutput[g]->event, streams[g]));

                CUDACHECK(cudaEventRecord(vec_previousBatchFinishedEvent[g], streams[g]));
            }
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){

                auto handleHost = [&](const auto& h){
                    info.host += h.sizeInBytes();
                };
                auto handleDevice = [&](const auto& d){
                    using ElementType = typename std::remove_reference<decltype(d)>::type::value_type;
                    info.device[deviceIds[g]] += d.size() * sizeof(ElementType);
                };

                info += gpuReadStorage->getMemoryInfo(vec_readstorageHandle[g]);
                if(vec_managedgpumsa[g]){
                    info += vec_managedgpumsa[g]->getMemoryInfo();
                }

                handleHost(vec_h_num_total_corrected_candidates[g]);
                handleHost(vec_h_num_indices[g]);
                handleHost(vec_h_numSelected[g]);
                handleHost(vec_h_managedmsa_tmp[g]);

                handleDevice(vec_d_anchorContainsN[g]);
                handleDevice(vec_d_candidateContainsN_[g]);
                handleDevice(vec_d_candidate_sequences_lengths_[g]);
                handleDevice(vec_d_candidate_sequences_data_[g]);
                handleDevice(vec_d_anchorIndicesOfCandidates_[g]);

                handleDevice(vec_d_allCandidateData[g]);

                handleDevice(vec_d_alignment_overlaps_[g]);
                handleDevice(vec_d_alignment_shifts_[g]);
                handleDevice(vec_d_alignment_nOps_[g]);
                handleDevice(vec_d_alignment_best_alignment_flags_[g]);

                handleDevice(vec_d_indices[g]);
                handleDevice(vec_d_indices_per_anchor[g]);
                handleDevice(vec_d_indices_per_anchor_prefixsum[g]);
                handleDevice(vec_d_num_indices[g]);
                handleDevice(vec_d_corrected_anchors[g]);
                handleDevice(vec_d_corrected_candidates[g]);
                handleDevice(vec_d_num_corrected_candidates_per_anchor[g]);
                handleDevice(vec_d_num_corrected_candidates_per_anchor_prefixsum[g]);
                handleDevice(vec_d_num_total_corrected_candidates[g]);
                handleDevice(vec_d_anchor_is_corrected[g]);
                handleDevice(vec_d_is_high_quality_anchor[g]);
                handleDevice(vec_d_high_quality_anchor_indices[g]);
                handleDevice(vec_d_num_high_quality_anchor_indices[g]);
                handleDevice(vec_d_editsPerCorrectedanchor[g]);
                handleDevice(vec_d_numEditsPerCorrectedanchor[g]);
                handleDevice(vec_d_editsPerCorrectedCandidate[g]);
                handleDevice(vec_d_hqAnchorCorrectionOfCandidateExists_[g]);            
                handleDevice(vec_d_numEditsPerCorrectedCandidate[g]);
                handleDevice(vec_d_indices_of_corrected_anchors[g]);
                handleDevice(vec_d_num_indices_of_corrected_anchors[g]);
                handleDevice(vec_d_indices_of_corrected_candidates_[g]);
                handleDevice(vec_d_numAnchors[g]);
                handleDevice(vec_d_numCandidates[g]);
                handleDevice(vec_d_anchorReadIds[g]);
                handleDevice(vec_d_anchor_sequences_data[g]);
                handleDevice(vec_d_anchor_sequences_lengths[g]);
                handleDevice(vec_d_candidate_read_ids_[g]);
                handleDevice(vec_d_candidates_per_anchor[g]);
                handleDevice(vec_d_candidates_per_anchor_prefixsum[g]);
            }

            return info;
        } 

        void releaseMemory(const std::vector<cudaStream_t>& streams){
            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                auto handleDevice = [&](auto& d){
                    ::destroy(d, streams[g]);
                };

                handleDevice(vec_d_anchorContainsN[g]);
                handleDevice(vec_d_candidateContainsN_[g]);
                handleDevice(vec_d_candidate_sequences_lengths_[g]);
                handleDevice(vec_d_candidate_sequences_data_[g]);
                handleDevice(vec_d_anchorIndicesOfCandidates_[g]);

                handleDevice(vec_d_allCandidateData[g]);

                handleDevice(vec_d_alignment_overlaps_[g]);
                handleDevice(vec_d_alignment_shifts_[g]);
                handleDevice(vec_d_alignment_nOps_[g]);
                handleDevice(vec_d_alignment_best_alignment_flags_[g]);
                handleDevice(vec_d_indices[g]);
                handleDevice(vec_d_indices_per_anchor[g]);
                handleDevice(vec_d_indices_per_anchor_prefixsum[g]);
                handleDevice(vec_d_num_indices[g]);
                handleDevice(vec_d_corrected_anchors[g]);
                handleDevice(vec_d_corrected_candidates[g]);
                handleDevice(vec_d_num_corrected_candidates_per_anchor[g]);
                handleDevice(vec_d_num_corrected_candidates_per_anchor_prefixsum[g]);
                handleDevice(vec_d_num_total_corrected_candidates[g]);
                handleDevice(vec_d_anchor_is_corrected[g]);
                handleDevice(vec_d_is_high_quality_anchor[g]);
                handleDevice(vec_d_high_quality_anchor_indices[g]);
                handleDevice(vec_d_num_high_quality_anchor_indices[g]);
                handleDevice(vec_d_editsPerCorrectedanchor[g]);
                handleDevice(vec_d_numEditsPerCorrectedanchor[g]);
                handleDevice(vec_d_editsPerCorrectedCandidate[g]);
                handleDevice(vec_d_hqAnchorCorrectionOfCandidateExists_[g]);
                handleDevice(vec_d_numEditsPerCorrectedCandidate[g]);
                handleDevice(vec_d_indices_of_corrected_anchors[g]);
                handleDevice(vec_d_num_indices_of_corrected_anchors[g]);
                handleDevice(vec_d_indices_of_corrected_candidates_[g]);
                handleDevice(vec_d_numEditsPerCorrectedanchor[g]);
                handleDevice(vec_d_numAnchors[g]);
                handleDevice(vec_d_numCandidates[g]);
                handleDevice(vec_d_anchorReadIds[g]);
                handleDevice(vec_d_anchor_sequences_data[g]);
                handleDevice(vec_d_anchor_sequences_lengths[g]);
                handleDevice(vec_d_candidate_read_ids_[g]);
            }
        } 

        void releaseCandidateMemory(const std::vector<cudaStream_t>& streams){

            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                auto handleDevice = [&](auto& d){
                    ::destroy(d, streams[g]);
                };

                handleDevice(vec_d_candidateContainsN_[g]);
                handleDevice(vec_d_candidate_sequences_lengths_[g]);
                handleDevice(vec_d_candidate_sequences_data_[g]);
                handleDevice(vec_d_anchorIndicesOfCandidates_[g]);

                handleDevice(vec_d_allCandidateData[g]);

                handleDevice(vec_d_alignment_overlaps_[g]);
                handleDevice(vec_d_alignment_shifts_[g]);
                handleDevice(vec_d_alignment_nOps_[g]);
                handleDevice(vec_d_alignment_best_alignment_flags_[g]);
                handleDevice(vec_d_indices[g]);
                handleDevice(vec_d_corrected_candidates[g]);
                handleDevice(vec_d_editsPerCorrectedCandidate[g]);
                handleDevice(vec_d_hqAnchorCorrectionOfCandidateExists_[g]);
                handleDevice(vec_d_numEditsPerCorrectedCandidate[g]);
                handleDevice(vec_d_indices_of_corrected_candidates_[g]);
                handleDevice(vec_d_candidate_read_ids_[g]);
            }
        } 

        


    public: //private:

        void initFixedSizeBuffers(const std::vector<cudaStream_t>& streams){
            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                cudaStream_t stream = streams[g];

                const std::size_t numEditsAnchors = SDIV(editsPitchInBytes * maxAnchors, sizeof(EncodedCorrectionEdit));          

                vec_h_num_total_corrected_candidates[g].resize(1);
                vec_h_num_indices[g].resize(1);
                vec_h_numSelected[g].resize(1);
                vec_h_managedmsa_tmp[g].resize(1);

                vec_d_anchorContainsN[g].resize(maxAnchors, stream);

                vec_d_indices_per_anchor[g].resize(maxAnchors, stream);
                vec_d_num_indices[g].resize(1, stream);
                vec_d_indices_per_anchor_prefixsum[g].resize(maxAnchors, stream);
                vec_d_corrected_anchors[g].resize(maxAnchors * decodedSequencePitchInBytes, stream);
                vec_d_num_corrected_candidates_per_anchor[g].resize(maxAnchors, stream);
                vec_d_num_corrected_candidates_per_anchor_prefixsum[g].resize(maxAnchors, stream);
                vec_d_num_total_corrected_candidates[g].resize(1, stream);
                vec_d_anchor_is_corrected[g].resize(maxAnchors, stream);
                vec_d_is_high_quality_anchor[g].resize(maxAnchors, stream);
                vec_d_high_quality_anchor_indices[g].resize(maxAnchors, stream);
                vec_d_num_high_quality_anchor_indices[g].resize(1, stream); 
                vec_d_editsPerCorrectedanchor[g].resize(numEditsAnchors, stream);
                vec_d_numEditsPerCorrectedanchor[g].resize(maxAnchors, stream);
                vec_d_indices_of_corrected_anchors[g].resize(maxAnchors, stream);
                vec_d_num_indices_of_corrected_anchors[g].resize(1, stream);

                vec_d_numAnchors[g].resize(1, stream);
                vec_d_numCandidates[g].resize(1, stream);
                vec_d_anchorReadIds[g].resize(maxAnchors, stream);
                vec_d_anchor_sequences_data[g].resize(encodedSequencePitchInInts * maxAnchors, stream);
                vec_d_anchor_sequences_lengths[g].resize(maxAnchors, stream);
                vec_d_candidates_per_anchor[g].resize(maxAnchors, stream);
                vec_d_candidates_per_anchor_prefixsum[g].resize(maxAnchors + 1, stream);
                vec_d_totalNumEdits[g].resize(1, stream);
            }
        }
 
        void resizeBuffers(
            const std::vector<int>& /*vec_numReads*/, 
            const std::vector<int>& vec_numCandidates, 
            const std::vector<cudaStream_t>& streams
        ){  
            //assert(numReads <= maxAnchors);

            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                const int numCandidates = vec_numCandidates[g];
                cudaStream_t stream = streams[g];
                
                vec_d_indices[g].resize(numCandidates + 1, stream);
            }

            #if 0

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                const int numCandidates = vec_numCandidates[g];
                cudaStream_t stream = streams[g];

                vec_d_alignment_overlaps_[g].resize(numCandidates, stream);
                vec_d_alignment_shifts_[g].resize(numCandidates, stream);
                vec_d_alignment_nOps_[g].resize(numCandidates, stream);
                vec_d_alignment_best_alignment_flags_[g].resize(numCandidates, stream);            
                vec_d_anchorIndicesOfCandidates_[g].resize(numCandidates, stream);
                vec_d_candidateContainsN_[g].resize(numCandidates, stream);
                vec_d_isPairedCandidate_[g].resize(numCandidates, stream);
                vec_d_indices_of_corrected_candidates_[g].resize(numCandidates, stream);
                vec_d_hqAnchorCorrectionOfCandidateExists_[g].resize(numCandidates, stream);
                vec_d_candidate_read_ids_[g].resize(numCandidates, stream);
                vec_d_candidate_sequences_lengths_[g].resize(numCandidates, stream);
                vec_d_candidate_sequences_data_[g].resize(numCandidates * encodedSequencePitchInInts, stream);

                d_alignment_overlaps[g] = vec_d_alignment_overlaps_[g].data();
                d_alignment_shifts[g] = vec_d_alignment_shifts_[g].data();
                d_alignment_nOps[g] = vec_d_alignment_nOps_[g].data();
                d_alignment_best_alignment_flags[g] = vec_d_alignment_best_alignment_flags_[g].data();
                d_anchorIndicesOfCandidates[g] = vec_d_anchorIndicesOfCandidates_[g].data();
                d_candidateContainsN[g] = vec_d_candidateContainsN_[g].data();
                d_isPairedCandidate[g] = vec_d_isPairedCandidate_[g].data();
                d_indices_of_corrected_candidates[g] = vec_d_indices_of_corrected_candidates_[g].data();
                d_hqAnchorCorrectionOfCandidateExists[g] = vec_d_hqAnchorCorrectionOfCandidateExists_[g].data();
                d_candidate_read_ids[g] = vec_d_candidate_read_ids_[g].data();
                d_candidate_sequences_lengths[g] = vec_d_candidate_sequences_lengths_[g].data();
                d_candidate_sequences_data[g] = vec_d_candidate_sequences_data_[g].data();
            }

            #else 

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                const int numCandidates = vec_numCandidates[g];
                cudaStream_t stream = streams[g];

                size_t allocation_sizes[12];
                allocation_sizes[0] = sizeof(int) * numCandidates; // d_alignment_overlaps
                allocation_sizes[1] = sizeof(int) * numCandidates; // d_alignment_shifts
                allocation_sizes[2] = sizeof(int) * numCandidates; // d_alignment_nOps
                allocation_sizes[3] = sizeof(AlignmentOrientation) * numCandidates; // d_alignment_best_alignment_flags
                allocation_sizes[4] = sizeof(int) * numCandidates; // d_anchorIndicesOfCandidates
                allocation_sizes[5] = sizeof(bool) * numCandidates; // d_candidateContainsN
                allocation_sizes[6] = sizeof(bool) * numCandidates; // d_isPairedCandidate
                allocation_sizes[7] = sizeof(int) * numCandidates; // d_indices_of_corrected_candidates
                allocation_sizes[8] = sizeof(bool) * numCandidates; // d_hqAnchorCorrectionOfCandidateExists
                allocation_sizes[9] = sizeof(read_number) * numCandidates; // d_candidate_read_ids
                allocation_sizes[10] = sizeof(int) * numCandidates; // d_candidate_sequences_lengths
                allocation_sizes[11] = sizeof(unsigned int) * encodedSequencePitchInInts * numCandidates; // d_candidate_sequences_data

                void* allocations[12]{};

                size_t temp_storage_bytes = 0;

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    temp_storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                resizeUninitialized(vec_d_allCandidateData[g], temp_storage_bytes, stream);

                CUDACHECK(cub::AliasTemporaries(
                    vec_d_allCandidateData[g].data(),
                    temp_storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                vec_d_alignment_overlaps[g] = reinterpret_cast<int*>(allocations[0]);
                vec_d_alignment_shifts[g] = reinterpret_cast<int*>(allocations[1]);
                vec_d_alignment_nOps[g] = reinterpret_cast<int*>(allocations[2]);
                vec_d_alignment_best_alignment_flags[g] = reinterpret_cast<AlignmentOrientation*>(allocations[3]);
                vec_d_anchorIndicesOfCandidates[g] = reinterpret_cast<int*>(allocations[4]);
                vec_d_candidateContainsN[g] = reinterpret_cast<bool*>(allocations[5]);
                vec_d_isPairedCandidate[g] = reinterpret_cast<bool*>(allocations[6]);
                vec_d_indices_of_corrected_candidates[g] = reinterpret_cast<int*>(allocations[7]);
                vec_d_hqAnchorCorrectionOfCandidateExists[g] = reinterpret_cast<bool*>(allocations[8]);
                vec_d_candidate_read_ids[g] = reinterpret_cast<read_number*>(allocations[9]);
                vec_d_candidate_sequences_lengths[g] = reinterpret_cast<int*>(allocations[10]);
                vec_d_candidate_sequences_data[g] = reinterpret_cast<unsigned int*>(allocations[11]);

                #endif
            }            
        }

        void flagPairedCandidates(const std::vector<cudaStream_t>& streams){

            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                const int currentNumAnchors = vec_currentNumAnchors[g];
                const int currentNumCandidates = vec_currentNumCandidates[g];
                cudaStream_t stream = streams[g];

                if(hasAnchorsAndCandidates(g)){

                    if(gpuReadStorage->isPairedEnd()){

                        assert(currentNumAnchors % 2 == 0);
                        assert(currentNumAnchors != 0);

                        helpers::call_fill_kernel_async(vec_d_isPairedCandidate[g], currentNumCandidates, false, stream);

                        dim3 block = 128;
                        dim3 grid = currentNumAnchors / 2;
                        constexpr int staticSmemBytes = 4096;

                        gpucorrectorkernels::flagPairedCandidatesKernel<128,staticSmemBytes>
                        <<<grid, block, 0, stream>>>(
                            currentNumAnchors / 2,
                            vec_d_candidates_per_anchor[g].data(),
                            vec_d_candidates_per_anchor_prefixsum[g].data(),
                            vec_d_candidate_read_ids[g],
                            vec_d_isPairedCandidate[g]
                        ); CUDACHECKASYNC;
                    }else{
                        CUDACHECK(cudaMemsetAsync(
                            vec_d_isPairedCandidate[g],
                            0,
                            sizeof(bool) * currentNumCandidates,
                            stream
                        ));
                    }
                }
            }
        }

        void copyAnchorResultsFromDeviceToHost(const std::vector<cudaStream_t>& streams){
            if(programOptions->correctionType == CorrectionType::Classic){
                copyAnchorResultsFromDeviceToHostClassic(streams);
            }else if(programOptions->correctionType == CorrectionType::Forest){
                copyAnchorResultsFromDeviceToHostForestGpu(streams);
            }else{
                throw std::runtime_error("copyAnchorResultsFromDeviceToHost not implemented for this correctionType");
            }
        }

        void copyAnchorResultsFromDeviceToHostClassic_serialized(const std::vector<cudaStream_t>& streams){
            nvtx::ScopedRange sr("constructSerializedAnchorResults-gpu", 5);

            const int numGpus = deviceIds.size();

            std::vector<std::uint32_t*> vec_d_numBytesPerSerializedAnchor(numGpus);
            std::vector<std::uint32_t*> vec_d_numBytesPerSerializedAnchorPrefixSum(numGpus);
            std::vector<std::uint8_t*> vec_d_serializedAnchorResults(numGpus);
            std::vector<rmm::device_uvector<char>> vec_d_tmp;

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                const std::uint32_t maxSerializedBytesPerAnchor = 
                    sizeof(read_number) 
                    + sizeof(std::uint32_t) 
                    + sizeof(short) 
                    + sizeof(char) * gpuReadStorage->getSequenceLengthUpperBound();

                const std::uint32_t maxResultBytes = maxSerializedBytesPerAnchor * vec_currentNumAnchors[g];

                size_t allocation_sizes[3]{};
                allocation_sizes[0] = sizeof(std::uint32_t) * vec_currentNumAnchors[g]; // d_numBytesPerSerializedAnchor
                allocation_sizes[1] = sizeof(std::uint32_t) * (vec_currentNumAnchors[g]+1); // d_numBytesPerSerializedAnchorPrefixSum
                allocation_sizes[2] = sizeof(uint8_t) * maxResultBytes; // d_serializedAnchorResults
                void* allocations[3]{};
                std::size_t tempbytes = 0;

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    tempbytes,
                    allocations,
                    allocation_sizes
                ));

                rmm::device_uvector<char> d_tmp(tempbytes, streams[g]);

                CUDACHECK(cub::AliasTemporaries(
                    d_tmp.data(),
                    tempbytes,
                    allocations,
                    allocation_sizes
                ));

                vec_d_numBytesPerSerializedAnchor[g] = reinterpret_cast<std::uint32_t*>(allocations[0]);
                vec_d_numBytesPerSerializedAnchorPrefixSum[g] = reinterpret_cast<std::uint32_t*>(allocations[1]);
                vec_d_serializedAnchorResults[g] = reinterpret_cast<std::uint8_t*>(allocations[2]);
                vec_d_tmp.push_back(std::move(d_tmp));

                if(hasAnchorsAndCandidates(g)){
                    CUDACHECK(cudaMemsetAsync(
                        vec_d_numBytesPerSerializedAnchor[g],
                        0,
                        sizeof(std::uint32_t) * vec_currentNumAnchors[g],
                        streams[g]
                    ));

                    //compute bytes per anchor
                    helpers::lambda_kernel<<<SDIV(vec_currentNumAnchors[g], 128), 128, 0, streams[g]>>>(
                        [
                            d_numBytesPerSerializedAnchor = vec_d_numBytesPerSerializedAnchor[g],
                            d_numBytesPerSerializedAnchorPrefixSum = vec_d_numBytesPerSerializedAnchorPrefixSum[g],
                            d_numEditsPerCorrectedanchor = vec_d_numEditsPerCorrectedanchor[g].data(),
                            d_anchor_sequences_lengths = vec_d_anchor_sequences_lengths[g].data(),
                            currentNumAnchors = vec_currentNumAnchors[g],
                            dontUseEditsValue = getDoNotUseEditsValue(),
                            d_num_indices_of_corrected_anchors = vec_d_num_indices_of_corrected_anchors[g].data(),
                            d_indices_of_corrected_anchors = vec_d_indices_of_corrected_anchors[g].data(),
                            maxSerializedBytesPerAnchor
                        ] __device__ (){
                            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                            const int stride = blockDim.x * gridDim.x;
                            if(tid == 0){
                                d_numBytesPerSerializedAnchorPrefixSum[0] = 0;
                            }
                            const int numCorrectedAnchors = *d_num_indices_of_corrected_anchors;
                            for(int outputIndex = tid; outputIndex < numCorrectedAnchors; outputIndex += stride){
                                const int anchorIndex = d_indices_of_corrected_anchors[outputIndex];

                                const int numEdits = d_numEditsPerCorrectedanchor[outputIndex];
                                const bool useEdits = numEdits != dontUseEditsValue;
                                std::uint32_t numBytes = 0;
                                if(useEdits){
                                    numBytes += sizeof(short); //number of edits
                                    numBytes += numEdits * (sizeof(short) + sizeof(char)); //edits
                                }else{
                                    const int sequenceLength = d_anchor_sequences_lengths[anchorIndex];
                                    numBytes += sizeof(short); // sequence length
                                    numBytes += sizeof(char) * sequenceLength;  //sequence
                                }
                                #ifndef NDEBUG
                                //flags use 3 bits, remainings bit can be used for encoding
                                constexpr std::uint32_t maxNumBytes = (std::uint32_t(1) << 29)-1;
                                assert(numBytes <= maxNumBytes);
                                #endif

                                numBytes += sizeof(read_number) + sizeof(std::uint32_t);

                                #ifndef NDEBUG
                                assert(numBytes <= maxSerializedBytesPerAnchor);
                                #endif


                                d_numBytesPerSerializedAnchor[outputIndex] = numBytes;
                            }
                        }
                    ); CUDACHECKASYNC;

                    

                    thrust::inclusive_scan(
                        rmm::exec_policy_nosync(streams[g]),
                        vec_d_numBytesPerSerializedAnchor[g],
                        vec_d_numBytesPerSerializedAnchor[g] + vec_currentNumAnchors[g],
                        vec_d_numBytesPerSerializedAnchorPrefixSum[g] + 1
                    );
                    // helpers::lambda_kernel<<<1,1,0,streams[g]>>>(
                    //     [
                    //         num = vec_currentNumAnchors[g],
                    //         d_numBytesPerSerializedAnchorPrefixSum = vec_d_numBytesPerSerializedAnchorPrefixSum[g]
                    //     ] __device__ (){
                    //         printf("d_numBytesPerSerializedAnchorPrefixSum %u\n", d_numBytesPerSerializedAnchorPrefixSum[num]);
                    //     }
                    // ); CUDACHECKASYNC

                    // std::uint32_t totalRequiredBytesAnchors = 0;
                    // CUDACHECK(cudaMemcpyAsync(
                    //     &totalRequiredBytesAnchors,
                    //     vec_d_numBytesPerSerializedAnchorPrefixSum[g] + vec_currentNumAnchors[g],
                    //     sizeof(std::uint32_t),
                    //     D2H,
                    //     streams[g]
                    // ));
                    // CUDACHECK(cudaStreamSynchronize(streams[g]));
                    // std::cout << "totalRequiredBytesAnchors " << totalRequiredBytesAnchors << "\n";


                    //compute serialized anchors
                    helpers::lambda_kernel<<<std::max(1, SDIV(vec_currentNumAnchors[g], 128)), 128, 0, streams[g]>>>(
                        [
                            d_numBytesPerSerializedAnchor = vec_d_numBytesPerSerializedAnchor[g],
                            d_numBytesPerSerializedAnchorPrefixSum = vec_d_numBytesPerSerializedAnchorPrefixSum[g],
                            d_serializedAnchorResults = vec_d_serializedAnchorResults[g],
                            d_numEditsPerCorrectedanchor = vec_d_numEditsPerCorrectedanchor[g].data(),
                            d_anchor_sequences_lengths = vec_d_anchor_sequences_lengths[g].data(),
                            d_num_indices_of_corrected_anchors = vec_d_num_indices_of_corrected_anchors[g].data(),
                            dontUseEditsValue = getDoNotUseEditsValue(),
                            d_is_high_quality_anchor = vec_d_is_high_quality_anchor[g].data(),
                            d_anchorReadIds = vec_d_anchorReadIds[g].data(),
                            d_corrected_anchors = vec_d_corrected_anchors[g].data(),
                            decodedSequencePitchInBytes = this->decodedSequencePitchInBytes,
                            d_editsPerCorrectedanchor = vec_d_editsPerCorrectedanchor[g].data(),
                            editsPitchInBytes = editsPitchInBytes,
                            d_indices_of_corrected_anchors = vec_d_indices_of_corrected_anchors[g].data()
                        ] __device__ (){
                            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                            const int stride = blockDim.x * gridDim.x;

                            const int numCorrectedAnchors = *d_num_indices_of_corrected_anchors;

                            for(int outputIndex = tid; outputIndex < numCorrectedAnchors; outputIndex += stride){
                                const int anchorIndex = d_indices_of_corrected_anchors[outputIndex];
                                //edit related data is access by outputIndex, other data by anchorIndex

                                const int numEdits = d_numEditsPerCorrectedanchor[outputIndex];
                                const bool useEdits = numEdits != dontUseEditsValue;
                                const int sequenceLength = d_anchor_sequences_lengths[anchorIndex];
                                std::uint32_t numBytes = 0;
                                if(useEdits){
                                    numBytes += sizeof(short); //number of edits
                                    numBytes += numEdits * (sizeof(short) + sizeof(char)); //edits
                                }else{
                                    numBytes += sizeof(short); // sequence length
                                    numBytes += sizeof(char) * sequenceLength;  //sequence
                                }
                                #ifndef NDEBUG
                                //flags use 3 bits, remainings bit can be used for encoding
                                constexpr std::uint32_t maxNumBytes = (std::uint32_t(1) << 29)-1;
                                assert(numBytes <= maxNumBytes);
                                #endif

                                const bool hq = d_is_high_quality_anchor[anchorIndex].hq();
                                const read_number readId = d_anchorReadIds[anchorIndex];

                                std::uint32_t encodedflags = (std::uint32_t(hq) << 31);
                                encodedflags |= (std::uint32_t(useEdits) << 30);
                                encodedflags |= (std::uint32_t(int(TempCorrectedSequenceType::Anchor)) << 29);
                                encodedflags |= numBytes;

                                numBytes += sizeof(read_number) + sizeof(std::uint32_t);

                                std::uint8_t* ptr = d_serializedAnchorResults + d_numBytesPerSerializedAnchorPrefixSum[outputIndex];

                                std::memcpy(ptr, &readId, sizeof(read_number));
                                ptr += sizeof(read_number);
                                std::memcpy(ptr, &encodedflags, sizeof(std::uint32_t));
                                ptr += sizeof(std::uint32_t);

                                if(useEdits){
                                    
                                    const EncodedCorrectionEdit* edits = (const EncodedCorrectionEdit*)(((const char*)d_editsPerCorrectedanchor) + editsPitchInBytes * outputIndex);
                                    short numEditsShort = numEdits;
                                    std::memcpy(ptr, &numEditsShort, sizeof(short));
                                    ptr += sizeof(short);
                                    for(int i = 0; i < numEdits; i++){
                                        const auto& edit = edits[i];
                                        const short p = edit.pos();
                                        std::memcpy(ptr, &p, sizeof(short));
                                        ptr += sizeof(short);
                                    }
                                    for(int i = 0; i < numEdits; i++){
                                        const auto& edit = edits[i];
                                        const char c = edit.base();
                                        std::memcpy(ptr, &c, sizeof(char));
                                        ptr += sizeof(char);
                                    }
                                }else{
                                    short lengthShort = sequenceLength;
                                    std::memcpy(ptr, &lengthShort, sizeof(short));
                                    ptr += sizeof(short);

                                    const char* const sequence = d_corrected_anchors + decodedSequencePitchInBytes * anchorIndex;
                                    std::memcpy(ptr, sequence, sizeof(char) * sequenceLength);
                                    ptr += sizeof(char) * sequenceLength;
                                }

                                // if(! (ptr == d_serializedAnchorResults + d_numBytesPerSerializedAnchorPrefixSum[outputIndex+1]) ){
                                //     printf("A: outputIndex %d, useEdits %d, ser begin %p, ser end %p, expected ser end %p\n", 
                                //         outputIndex,
                                //         useEdits,
                                //         d_serializedAnchorResults + d_numBytesPerSerializedAnchorPrefixSum[outputIndex],
                                //         ptr,
                                //         d_serializedAnchorResults + d_numBytesPerSerializedAnchorPrefixSum[outputIndex+1]
                                //     );
                                // }
                                assert(ptr == d_serializedAnchorResults + d_numBytesPerSerializedAnchorPrefixSum[outputIndex+1]);
                            }
                        }
                    ); CUDACHECKASYNC;
                }
            }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                const std::uint32_t maxSerializedBytesPerAnchor = 
                    sizeof(read_number) 
                    + sizeof(std::uint32_t) 
                    + sizeof(short) 
                    + sizeof(char) * gpuReadStorage->getSequenceLengthUpperBound();

                const std::uint32_t maxResultBytes = maxSerializedBytesPerAnchor * vec_currentNumAnchors[g];

                vec_currentOutput[g]->serializedAnchorResults.resize(maxResultBytes);
                vec_currentOutput[g]->serializedAnchorOffsets.resize(vec_currentNumAnchors[g] + 1);
                vec_currentOutput[g]->h_numCorrectedAnchors.resize(1);
            }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                //copy data to host. since number of output bytes of serialized results is only available 
                // on the device, use a kernel

                if(hasAnchorsAndCandidates(g)){

                    helpers::lambda_kernel<<<480,128,0,streams[g]>>>(
                        [
                            h_numCorrectedAnchors = vec_currentOutput[g]->h_numCorrectedAnchors.data(),
                            d_numCorrectedAnchors = vec_d_num_indices_of_corrected_anchors[g].data(),
                            h_serializedAnchorOffsets = vec_currentOutput[g]->serializedAnchorOffsets.data(),
                            d_numBytesPerSerializedAnchorPrefixSum = vec_d_numBytesPerSerializedAnchorPrefixSum[g],
                            h_serializedAnchorResults = vec_currentOutput[g]->serializedAnchorResults.data(),
                            d_serializedAnchorResults = vec_d_serializedAnchorResults[g],
                            currentNumAnchors = vec_currentNumAnchors[g]
                        ] __device__ (){
                            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                            const int stride = blockDim.x * gridDim.x;

                            const int numCorrectedAnchors = *d_numCorrectedAnchors;

                            if(tid == 0){
                                *h_numCorrectedAnchors = numCorrectedAnchors;
                            }

                            for(int i = tid; i < numCorrectedAnchors + 1; i += stride){
                                h_serializedAnchorOffsets[i] = d_numBytesPerSerializedAnchorPrefixSum[i];
                            }

                            int serializedBytes = d_numBytesPerSerializedAnchorPrefixSum[numCorrectedAnchors];
                            const size_t numIntCopies = serializedBytes / sizeof(int);
                            const char* src = reinterpret_cast<const char*>(d_serializedAnchorResults);
                            char* dst = reinterpret_cast<char*>(h_serializedAnchorResults);
                            for(size_t i = tid; i < numIntCopies; i += stride){
                                reinterpret_cast<int*>(dst)[i] = reinterpret_cast<const int*>(src)[i];
                            }
                            dst += sizeof(int) * numIntCopies;
                            src += sizeof(int) * numIntCopies;
                            serializedBytes -= sizeof(int) * numIntCopies;
                            for(size_t i = tid; i < serializedBytes; i += stride){
                                reinterpret_cast<char*>(dst)[i] = reinterpret_cast<const char*>(src)[i];
                            }

                        }
                    ); CUDACHECKASYNC

                }
            }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                //ensure release of memory on the correct device
                vec_d_tmp[g].release();
            }
        }

        void copyAnchorResultsFromDeviceToHostClassic(const std::vector<cudaStream_t>& streams){
            copyAnchorResultsFromDeviceToHostClassic_serialized(streams);
        }



        void copyAnchorResultsFromDeviceToHostForestGpu(const std::vector<cudaStream_t>& streams){
            copyAnchorResultsFromDeviceToHostClassic(streams);
        }

        void copyCandidateResultsFromDeviceToHost(const std::vector<cudaStream_t>& streams){
            if(programOptions->correctionTypeCands == CorrectionType::Classic){
                copyCandidateResultsFromDeviceToHostClassic(streams);
            }else if(programOptions->correctionTypeCands == CorrectionType::Forest){
                copyCandidateResultsFromDeviceToHostForestGpu(streams);
            }else{
                throw std::runtime_error("copyCandidateResultsFromDeviceToHost not implemented for this correctionTypeCands");
            }
        }

        void copyCandidateResultsFromDeviceToHostClassic_serialized(const std::vector<cudaStream_t>& streams){
            nvtx::ScopedRange sr("constructSerializedCandidateResults-gpu", 5);

            const int numGpus = deviceIds.size();

            std::vector<std::uint32_t*> vec_d_numBytesPerSerializedCandidate(numGpus, nullptr);
            std::vector<std::uint32_t*> vec_d_numBytesPerSerializedCandidatePrefixSum(numGpus, nullptr);
            std::vector<std::uint8_t*> vec_d_serializedCandidateResults(numGpus, nullptr);
            std::vector<rmm::device_uvector<char>> vec_d_tmp;

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                const int numCorrectedCandidates = (*vec_h_num_total_corrected_candidates[g]);

                const std::uint32_t maxSerializedBytesPerCandidate = 
                    sizeof(read_number) 
                    + sizeof(std::uint32_t) 
                    + sizeof(short) 
                    + sizeof(char) * gpuReadStorage->getSequenceLengthUpperBound()
                    + sizeof(short);

                const std::uint32_t maxResultBytes = maxSerializedBytesPerCandidate * numCorrectedCandidates;

                size_t allocation_sizes[3]{};
                allocation_sizes[0] = sizeof(std::uint32_t) * numCorrectedCandidates; // d_numBytesPerSerializedAnchor
                allocation_sizes[1] = sizeof(std::uint32_t) * (numCorrectedCandidates+1); // d_numBytesPerSerializedAnchorPrefixSum
                allocation_sizes[2] = sizeof(uint8_t) * maxResultBytes; // d_serializedAnchorResults
                void* allocations[3]{};
                std::size_t tempbytes = 0;

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    tempbytes,
                    allocations,
                    allocation_sizes
                ));

                rmm::device_uvector<char> d_tmp(tempbytes, streams[g]);

                CUDACHECK(cub::AliasTemporaries(
                    d_tmp.data(),
                    tempbytes,
                    allocations,
                    allocation_sizes
                ));

                vec_d_numBytesPerSerializedCandidate[g] = reinterpret_cast<std::uint32_t*>(allocations[0]);
                vec_d_numBytesPerSerializedCandidatePrefixSum[g] = reinterpret_cast<std::uint32_t*>(allocations[1]);
                vec_d_serializedCandidateResults[g] = reinterpret_cast<std::uint8_t*>(allocations[2]);
                vec_d_tmp.push_back(std::move(d_tmp));

                if(hasAnchorsAndCandidates(g)){
                    //compute bytes per numCorrectedCandidates
                    helpers::lambda_kernel<<<std::max(1, SDIV(numCorrectedCandidates, 128)), 128, 0, streams[g]>>>(
                        [
                            d_numBytesPerSerializedCandidate = vec_d_numBytesPerSerializedCandidate[g],
                            d_numBytesPerSerializedCandidatePrefixSum = vec_d_numBytesPerSerializedCandidatePrefixSum[g],
                            d_numEditsPerCorrectedCandidate = vec_d_numEditsPerCorrectedCandidate[g].data(),
                            d_candidate_sequences_lengths = vec_d_candidate_sequences_lengths[g],
                            numCorrectedCandidates = numCorrectedCandidates,
                            dontUseEditsValue = getDoNotUseEditsValue(),
                            d_indices_of_corrected_candidates = vec_d_indices_of_corrected_candidates[g],
                            maxSerializedBytesPerCandidate
                        ] __device__ (){
                            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                            const int stride = blockDim.x * gridDim.x;
                            if(tid == 0){
                                d_numBytesPerSerializedCandidatePrefixSum[0] = 0;
                            }
                            for(int outputIndex = tid; outputIndex < numCorrectedCandidates; outputIndex += stride){
                                const int candidateIndex = d_indices_of_corrected_candidates[outputIndex];

                                const int numEdits = d_numEditsPerCorrectedCandidate[outputIndex];
                                const bool useEdits = numEdits != dontUseEditsValue;
                                std::uint32_t numBytes = 0;
                                if(useEdits){
                                    numBytes += sizeof(short); //number of edits
                                    numBytes += numEdits * (sizeof(short) + sizeof(char)); //edits
                                }else{
                                    const int sequenceLength = d_candidate_sequences_lengths[candidateIndex];
                                    numBytes += sizeof(short); // sequence length
                                    numBytes += sizeof(char) * sequenceLength;  //sequence
                                }
                                //candidate shift
                                numBytes += sizeof(short);

                                #ifndef NDEBUG
                                //flags use 3 bits, remainings bit can be used for encoding
                                constexpr std::uint32_t maxNumBytes = (std::uint32_t(1) << 29)-1;
                                assert(numBytes <= maxNumBytes);
                                #endif

                                numBytes += sizeof(read_number) + sizeof(std::uint32_t);

                                #ifndef NDEBUG
                                assert(numBytes <= maxSerializedBytesPerCandidate);
                                #endif

                                d_numBytesPerSerializedCandidate[outputIndex] = numBytes;
                            }
                        }
                    ); CUDACHECKASYNC;

                    thrust::inclusive_scan(
                        rmm::exec_policy_nosync(streams[g]),
                        vec_d_numBytesPerSerializedCandidate[g],
                        vec_d_numBytesPerSerializedCandidate[g] + numCorrectedCandidates,
                        vec_d_numBytesPerSerializedCandidatePrefixSum[g] + 1
                    );

                    //compute serialized candidates
                    helpers::lambda_kernel<<<std::max(1, SDIV(numCorrectedCandidates, 128)), 128, 0, streams[g]>>>(
                        [
                            d_numBytesPerSerializedCandidate = vec_d_numBytesPerSerializedCandidate[g],
                            d_numBytesPerSerializedCandidatePrefixSum = vec_d_numBytesPerSerializedCandidatePrefixSum[g],
                            d_serializedCandidateResults = vec_d_serializedCandidateResults[g],
                            d_numEditsPerCorrectedCandidate = vec_d_numEditsPerCorrectedCandidate[g].data(),
                            d_candidate_sequences_lengths = vec_d_candidate_sequences_lengths[g],
                            numCorrectedCandidates = numCorrectedCandidates,
                            dontUseEditsValue = getDoNotUseEditsValue(),
                            d_candidate_read_ids = vec_d_candidate_read_ids[g],
                            d_corrected_candidates = vec_d_corrected_candidates[g].data(),
                            d_alignment_shifts = vec_d_alignment_shifts[g],
                            decodedSequencePitchInBytes = this->decodedSequencePitchInBytes,
                            d_editsPerCorrectedCandidate = vec_d_editsPerCorrectedCandidate[g].data(),
                            editsPitchInBytes = editsPitchInBytes,
                            d_indices_of_corrected_candidates = vec_d_indices_of_corrected_candidates[g]
                        ] __device__ (){
                            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                            const int stride = blockDim.x * gridDim.x;

                            for(int outputIndex = tid; outputIndex < numCorrectedCandidates; outputIndex += stride){
                                const int candidateIndex = d_indices_of_corrected_candidates[outputIndex];

                                const int numEdits = d_numEditsPerCorrectedCandidate[outputIndex];
                                const bool useEdits = numEdits != dontUseEditsValue;
                                const int sequenceLength = d_candidate_sequences_lengths[candidateIndex];
                                std::uint32_t numBytes = 0;
                                if(useEdits){
                                    numBytes += sizeof(short); //number of edits
                                    numBytes += numEdits * (sizeof(short) + sizeof(char)); //edits
                                }else{
                                    numBytes += sizeof(short); // sequence length
                                    numBytes += sizeof(char) * sequenceLength;  //sequence
                                }
                                //candidate shift
                                numBytes += sizeof(short);

                                #ifndef NDEBUG
                                //flags use 3 bits, remainings bit can be used for encoding
                                constexpr std::uint32_t maxNumBytes = (std::uint32_t(1) << 29)-1;
                                assert(numBytes <= maxNumBytes);
                                #endif

                                const bool hq = false;
                                const read_number readId = d_candidate_read_ids[candidateIndex];

                                std::uint32_t encodedflags = (std::uint32_t(hq) << 31);
                                encodedflags |= (std::uint32_t(useEdits) << 30);
                                encodedflags |= (std::uint32_t(int(TempCorrectedSequenceType::Candidate)) << 29);
                                encodedflags |= numBytes;

                                numBytes += sizeof(read_number) + sizeof(std::uint32_t);

                                std::uint8_t* ptr = d_serializedCandidateResults + d_numBytesPerSerializedCandidatePrefixSum[outputIndex];

                                std::memcpy(ptr, &readId, sizeof(read_number));
                                ptr += sizeof(read_number);
                                std::memcpy(ptr, &encodedflags, sizeof(std::uint32_t));
                                ptr += sizeof(std::uint32_t);

                                if(useEdits){
                                    
                                    const EncodedCorrectionEdit* edits = (const EncodedCorrectionEdit*)(((const char*)d_editsPerCorrectedCandidate) + editsPitchInBytes * outputIndex);
                                    short numEditsShort = numEdits;
                                    std::memcpy(ptr, &numEditsShort, sizeof(short));
                                    ptr += sizeof(short);
                                    for(int i = 0; i < numEdits; i++){
                                        const auto& edit = edits[i];
                                        const short p = edit.pos();
                                        std::memcpy(ptr, &p, sizeof(short));
                                        ptr += sizeof(short);
                                    }
                                    for(int i = 0; i < numEdits; i++){
                                        const auto& edit = edits[i];
                                        const char c = edit.base();
                                        std::memcpy(ptr, &c, sizeof(char));
                                        ptr += sizeof(char);
                                    }
                                }else{
                                    short lengthShort = sequenceLength;
                                    std::memcpy(ptr, &lengthShort, sizeof(short));
                                    ptr += sizeof(short);

                                    const char* const sequence = d_corrected_candidates + decodedSequencePitchInBytes * outputIndex;
                                    std::memcpy(ptr, sequence, sizeof(char) * sequenceLength);
                                    ptr += sizeof(char) * sequenceLength;
                                }
                                //candidate shift
                                short shiftShort = d_alignment_shifts[candidateIndex];
                                std::memcpy(ptr, &shiftShort, sizeof(short));
                                ptr += sizeof(short);

                                // if(! (ptr == d_serializedCandidateResults + d_numBytesPerSerializedCandidatePrefixSum[outputIndex+1]) ){
                                //     printf("C: outputIndex %d, ser begin %p, ser end %p, expected ser end %p\n", 
                                //         outputIndex,
                                //         d_serializedCandidateResults + d_numBytesPerSerializedCandidatePrefixSum[outputIndex],
                                //         ptr,
                                //         d_serializedCandidateResults + d_numBytesPerSerializedCandidatePrefixSum[outputIndex+1]
                                //     );
                                // }

                                assert(ptr == d_serializedCandidateResults + d_numBytesPerSerializedCandidatePrefixSum[outputIndex+1]);
                            }
                        }
                    ); CUDACHECKASYNC;
                }
            }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                const int numCorrectedCandidates = (*vec_h_num_total_corrected_candidates[g]);

                const std::uint32_t maxSerializedBytesPerCandidate = 
                    sizeof(read_number) 
                    + sizeof(std::uint32_t) 
                    + sizeof(short) 
                    + sizeof(char) * gpuReadStorage->getSequenceLengthUpperBound()
                    + sizeof(short);

                const std::uint32_t maxResultBytes = maxSerializedBytesPerCandidate * numCorrectedCandidates;

                vec_currentOutput[g]->serializedCandidateResults.resize(maxResultBytes);
                vec_currentOutput[g]->serializedCandidateOffsets.resize(numCorrectedCandidates + 1);
                vec_currentOutput[g]->numCorrectedCandidates = numCorrectedCandidates;
        }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                if(hasAnchorsAndCandidates(g)){

                    const int numCorrectedCandidates = (*vec_h_num_total_corrected_candidates[g]);

                    CUDACHECK(cudaMemcpyAsync(
                        vec_currentOutput[g]->serializedCandidateOffsets.data(),
                        vec_d_numBytesPerSerializedCandidatePrefixSum[g],
                        sizeof(std::uint32_t) * (numCorrectedCandidates+1),
                        D2H,
                        streams[g]
                    ));
                }
            }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                if(hasAnchorsAndCandidates(g)){
                    const int numCorrectedCandidates = (*vec_h_num_total_corrected_candidates[g]);

                    CUDACHECK(cudaStreamSynchronize(streams[g]));
                    std::uint32_t totalSerializedBytes = vec_currentOutput[g]->serializedCandidateOffsets[numCorrectedCandidates];

                    CUDACHECK(cudaMemcpyAsync(
                        vec_currentOutput[g]->serializedCandidateResults.data(),
                        vec_d_serializedCandidateResults[g],
                        totalSerializedBytes,
                        D2H,
                        streams[g]
                    ));
                }
            }

            // for(int g = 0; g < numGpus; g++){
            //     cub::SwitchDevice sd{deviceIds[g]};

            //     if(hasAnchorsAndCandidates(g)){

            //         const int numCorrectedCandidates = (*vec_h_num_total_corrected_candidates[g]);

            //         CUDACHECK(cudaMemcpyAsync(
            //             vec_currentOutput[g]->serializedCandidateOffsets.data(),
            //             vec_d_numBytesPerSerializedCandidatePrefixSum[g],
            //             sizeof(std::uint32_t) * (numCorrectedCandidates+1),
            //             D2H,
            //             streams[g]
            //         ));
            //         CUDACHECK(cudaStreamSynchronize(streams[g]));
            //         std::uint32_t totalSerializedBytes = vec_currentOutput[g]->serializedCandidateOffsets[numCorrectedCandidates];

            //         CUDACHECK(cudaMemcpyAsync(
            //             vec_currentOutput[g]->serializedCandidateResults.data(),
            //             vec_d_serializedCandidateResults[g],
            //             totalSerializedBytes,
            //             D2H,
            //             streams[g]
            //         ));

            //         //copy data to host. since number of output bytes of serialized results is only available 
            //         // on the device, use a kernel

            //         // helpers::lambda_kernel<<<480,128,0,streams[g]>>>(
            //         //     [
            //         //         h_serializedCandidateOffsets = vec_currentOutput[g]->serializedCandidateOffsets.data(),
            //         //         d_numBytesPerSerializedCandidatePrefixSum = vec_d_numBytesPerSerializedCandidatePrefixSum[g],
            //         //         h_serializedCandidateResults = vec_currentOutput[g]->serializedCandidateResults.data(),
            //         //         d_serializedCandidateResults = vec_d_serializedCandidateResults[g],
            //         //         numCorrectedCandidates = numCorrectedCandidates
            //         //     ] __device__ (){
            //         //         const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            //         //         const int stride = blockDim.x * gridDim.x;

            //         //         for(int i = tid; i < numCorrectedCandidates + 1; i += stride){
            //         //             h_serializedCandidateOffsets[i] = d_numBytesPerSerializedCandidatePrefixSum[i];
            //         //         }

            //         //         int serializedBytes = d_numBytesPerSerializedCandidatePrefixSum[numCorrectedCandidates];
            //         //         const size_t numIntCopies = serializedBytes / sizeof(int);
            //         //         const char* src = reinterpret_cast<const char*>(d_serializedCandidateResults);
            //         //         char* dst = reinterpret_cast<char*>(h_serializedCandidateResults);
            //         //         for(size_t i = tid; i < numIntCopies; i += stride){
            //         //             reinterpret_cast<int*>(dst)[i] = reinterpret_cast<const int*>(src)[i];
            //         //         }
            //         //         dst += sizeof(int) * numIntCopies;
            //         //         src += sizeof(int) * numIntCopies;
            //         //         serializedBytes -= sizeof(int) * numIntCopies;
            //         //         for(size_t i = tid; i < serializedBytes; i += stride){
            //         //             reinterpret_cast<char*>(dst)[i] = reinterpret_cast<const char*>(src)[i];
            //         //         }

            //         //     }
            //         // ); CUDACHECKASYNC

            //     }
            // }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                //ensure release of memory on the correct device
                vec_d_tmp[g].release();
            }
        }

        void copyCandidateResultsFromDeviceToHostClassic(const std::vector<cudaStream_t>& streams){
            copyCandidateResultsFromDeviceToHostClassic_serialized(streams);
        }


        void copyCandidateResultsFromDeviceToHostForestGpu(const std::vector<cudaStream_t>& streams){
            copyCandidateResultsFromDeviceToHostClassic(streams);
        }

        void getAmbiguousFlagsOfAnchors(const std::vector<cudaStream_t>& streams){
            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};                

                    gpuReadStorage->areSequencesAmbiguous(
                        vec_readstorageHandle[g],
                        vec_d_anchorContainsN[g].data(),
                        vec_d_anchorReadIds[g].data(),
                        vec_currentNumAnchors[g],
                        streams[g]
                    );
                }
            }
        }

        void getAmbiguousFlagsOfCandidates(const std::vector<cudaStream_t>& streams){
            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};
                    gpuReadStorage->areSequencesAmbiguous(
                        vec_readstorageHandle[g],
                        vec_d_candidateContainsN[g], 
                        vec_d_candidate_read_ids[g], 
                        vec_currentNumCandidates[g],
                        streams[g]
                    ); 
                }
            }
        }

        void getCandidateAlignments(const std::vector<cudaStream_t>& streams){

            const bool removeAmbiguousAnchors = programOptions->excludeAmbiguousReads;
            const bool removeAmbiguousCandidates = programOptions->excludeAmbiguousReads;

            const int numGpus = deviceIds.size();
            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};
    
                    callShiftedHammingDistanceKernel(
                        vec_d_alignment_overlaps[g],
                        vec_d_alignment_shifts[g],
                        vec_d_alignment_nOps[g],
                        vec_d_alignment_best_alignment_flags[g],
                        vec_d_anchor_sequences_data[g].data(),
                        vec_d_candidate_sequences_data[g],
                        vec_d_anchor_sequences_lengths[g].data(),
                        vec_d_candidate_sequences_lengths[g],
                        vec_d_anchorIndicesOfCandidates[g],
                        vec_currentNumAnchors[g],
                        vec_currentNumCandidates[g],
                        vec_d_anchorContainsN[g].data(),
                        removeAmbiguousAnchors,
                        vec_d_candidateContainsN[g],
                        removeAmbiguousCandidates,
                        gpuReadStorage->getSequenceLengthUpperBound(),
                        gpuReadStorage->getSequenceLengthUpperBound(),
                        encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        programOptions->min_overlap,
                        programOptions->maxErrorRate,
                        programOptions->min_overlap_ratio,
                        programOptions->estimatedErrorrate,
                        streams[g]
                    );
                }
            }

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    #if 1
                    if(!gpuReadStorage->isPairedEnd()){
                        //default kernel
                        call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                            vec_d_alignment_best_alignment_flags[g],
                            vec_d_alignment_nOps[g],
                            vec_d_alignment_overlaps[g],
                            vec_d_candidates_per_anchor_prefixsum[g].data(),
                            vec_currentNumAnchors[g],
                            vec_currentNumCandidates[g],
                            programOptions->estimatedErrorrate,
                            programOptions->estimatedCoverage * programOptions->m_coverage,
                            streams[g]
                        );
                    }else{
                        helpers::lambda_kernel<<<SDIV(vec_currentNumCandidates[g], 128), 128, 0, streams[g]>>>(
                            [
                                bestAlignmentFlags = vec_d_alignment_best_alignment_flags[g],
                                nOps = vec_d_alignment_nOps[g],
                                overlaps = vec_d_alignment_overlaps[g],
                                currentNumCandidates = vec_currentNumCandidates[g],
                                d_isPairedCandidate = vec_d_isPairedCandidate[g],
                                pairedFilterThreshold = programOptions->pairedFilterThreshold
                            ] __device__(){
                                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                                const int stride = blockDim.x * gridDim.x;

                                for(int candidate_index = tid; candidate_index < currentNumCandidates; candidate_index += stride){
                                    if(!d_isPairedCandidate[candidate_index]){
                                        if(bestAlignmentFlags[candidate_index] != AlignmentOrientation::None) {

                                            const int alignment_overlap = overlaps[candidate_index];
                                            const int alignment_nops = nOps[candidate_index];

                                            const float mismatchratio = float(alignment_nops) / alignment_overlap;

                                            if(mismatchratio >= pairedFilterThreshold) {
                                                bestAlignmentFlags[candidate_index] = AlignmentOrientation::None;
                                            }
                                        }
                                    }
                                }
                            }
                        );
                    }
                    #else
                        //default kernel
                        call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                            vec_d_alignment_best_alignment_flags[g].data(),
                            vec_d_alignment_nOps[g].data(),
                            vec_d_alignment_overlaps[g].data(),
                            vec_d_candidates_per_anchor_prefixsum[g].data(),
                            vec_d_numAnchors[g].data(),
                            vec_d_numCandidates[g].data(),
                            maxAnchors,
                            vec_currentNumCandidates[g],
                            programOptions->estimatedErrorrate,
                            programOptions->estimatedCoverage * programOptions->m_coverage,
                            streams[g]
                        );
                    #endif
                }
            }

            for(int g = 0; g < numGpus; g++){
                if(hasCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    callSelectIndicesOfGoodCandidatesKernelAsync(
                        vec_d_indices[g].data(),
                        vec_d_indices_per_anchor[g].data(),
                        vec_d_num_indices[g].data(),
                        vec_d_alignment_best_alignment_flags[g],
                        vec_d_candidates_per_anchor[g].data(),
                        vec_d_candidates_per_anchor_prefixsum[g].data(),
                        vec_d_anchorIndicesOfCandidates[g],
                        vec_currentNumAnchors[g],
                        vec_currentNumCandidates[g],
                        streams[g]
                    );
                }else{
                    CUDACHECK(cudaMemsetAsync(
                        vec_d_indices_per_anchor[g].data(),
                        0,
                        sizeof(int) * vec_currentNumAnchors[g],
                        streams[g]
                    ));
                    CUDACHECK(cudaMemsetAsync(
                        vec_d_num_indices[g].data(),
                        0,
                        sizeof(int),
                        streams[g]
                    ));
                }
            }
        }

        void buildAndRefineMultipleSequenceAlignment(const std::vector<cudaStream_t>& streams){
            const int numGpus = deviceIds.size();

            std::vector<size_t> vec_allQualDataBytes(numGpus);
            std::vector<char*> vec_d_allQualData(numGpus);
            std::vector<char*> vec_d_anchor_qual(numGpus);
            std::vector<char*> vec_d_cand_qual(numGpus);

            if(programOptions->useQualityScores){
                for(int g = 0; g < numGpus; g++){
                    cub::SwitchDevice sd{deviceIds[g]};
                    
                    cudaStream_t qualityStream = vec_extraStream[g];

                    CUDACHECK(cudaStreamWaitEvent(qualityStream, vec_inputCandidateDataIsReadyEvent[g], 0));

                    size_t allocation_sizes[2]{};
                    allocation_sizes[0] = vec_currentNumAnchors[g] * qualityPitchInBytes; // d_anchor_qual
                    allocation_sizes[1] = vec_currentNumCandidates[g] * qualityPitchInBytes; // d_cand_qual
                    void* allocations[2]{};

                    CUDACHECK(cub::AliasTemporaries(
                        nullptr,
                        vec_allQualDataBytes[g],
                        allocations,
                        allocation_sizes
                    ));

                    vec_d_allQualData[g] = reinterpret_cast<char*>(
                        rmm::mr::get_current_device_resource()->allocate(vec_allQualDataBytes[g], qualityStream));

                    CUDACHECK(cub::AliasTemporaries(
                        vec_d_allQualData[g],
                        vec_allQualDataBytes[g],
                        allocations,
                        allocation_sizes
                    ));

                    vec_d_anchor_qual[g] = reinterpret_cast<char*>(allocations[0]);
                    vec_d_cand_qual[g] = reinterpret_cast<char*>(allocations[1]);
                }

                for(int g = 0; g < numGpus; g++){
                    if(hasAnchorsAndCandidates(g)){
                        cub::SwitchDevice sd{deviceIds[g]};
                        cudaStream_t qualityStream = vec_extraStream[g];
                        gpuReadStorage->gatherContiguousQualities(
                            vec_readstorageHandle[g],
                            vec_d_anchor_qual[g],
                            qualityPitchInBytes,
                            currentInput->vec_h_anchorReadIds[g][0],
                            vec_currentNumAnchors[g],
                            qualityStream,
                            rmm::mr::get_current_device_resource()
                        );
                    }
                }

                std::vector<rmm::mr::device_memory_resource*> mrs(numGpus, nullptr);
                std::vector<cudaStream_t> extraStreams(numGpus, nullptr);
                std::vector<AsyncConstBufferWrapper<read_number>> vec_h_readIdsAsync(numGpus);

                for(int g = 0; g < numGpus; g++){
                    cub::SwitchDevice sd{deviceIds[g]};
                    mrs[g] = rmm::mr::get_current_device_resource();
                    extraStreams[g] = vec_extraStream[g];
    
                    //h_candidate_read_ids will only be valid if readstorage has host sequences or host qualities
                    vec_h_readIdsAsync[g] = makeAsyncConstBufferWrapper(
                        currentInput->vec_h_candidate_read_ids[g].data(),
                        currentInput->vec_h_candidate_read_ids_readyEvent[g]
                    );
                }
                gpuReadStorage->multi_gatherQualities(
                    vec_readstorageHandle,
                    vec_d_cand_qual,
                    qualityPitchInBytes,
                    vec_h_readIdsAsync,
                    vec_d_candidate_read_ids,
                    vec_currentNumCandidates,
                    extraStreams,
                    deviceIds,
                    mrs
                );

                // gpuReadStorage->multi_gatherQualities(
                //     vec_readstorageHandle,
                //     vec_d_cand_qual,
                //     qualityPitchInBytes,
                //     vec_d_candidate_read_ids,
                //     vec_currentNumCandidates,
                //     extraStreams,
                //     deviceIds,
                //     mrs
                // );

                for(int g = 0; g < numGpus; g++){
                    if(hasAnchors(g) || hasAnchorsAndCandidates(g)){
                        cub::SwitchDevice sd{deviceIds[g]};
                        cudaStream_t qualityStream = vec_extraStream[g];
                    
                        CUDACHECK(cudaEventRecord(vec_events[g][0], qualityStream));
                        CUDACHECK(cudaStreamWaitEvent(streams[g], vec_events[g][0], 0));
                    }
                }
            }

            for(int g = 0; g < numGpus; g++){
                //if(hasAnchors(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    vec_managedgpumsa[g] = std::make_unique<ManagedGPUMultiMSA>(
                        streams[g],
                        rmm::mr::get_current_device_resource(), 
                        vec_h_managedmsa_tmp[g].data()
                    );
                //}
            }


            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};
                    vec_managedgpumsa[g]->construct(
                        vec_d_alignment_overlaps[g],
                        vec_d_alignment_shifts[g],
                        vec_d_alignment_nOps[g],
                        vec_d_alignment_best_alignment_flags[g],
                        vec_d_indices[g].data(),
                        vec_d_indices_per_anchor[g].data(),
                        vec_d_candidates_per_anchor_prefixsum[g].data(),
                        vec_d_anchor_sequences_lengths[g].data(),
                        vec_d_anchor_sequences_data[g].data(),
                        vec_d_anchor_qual[g],
                        vec_currentNumAnchors[g],
                        vec_d_candidate_sequences_lengths[g],
                        vec_d_candidate_sequences_data[g],
                        vec_d_cand_qual[g],
                        vec_d_isPairedCandidate[g],
                        encodedSequencePitchInInts,
                        qualityPitchInBytes,
                        programOptions->useQualityScores,
                        programOptions->maxErrorRate,
                        MSAColumnCount{static_cast<int>(msaColumnPitchInElements)},
                        streams[g]
                    );
                }
            }

            if(useMsaRefinement()){
                for(int g = 0; g < numGpus; g++){
                    if(hasAnchorsAndCandidates(g)){
                        cub::SwitchDevice sd{deviceIds[g]};

                        rmm::device_uvector<int> d_indices_tmp(vec_currentNumCandidates[g]+1, streams[g]);
                        rmm::device_uvector<int> d_indices_per_anchor_tmp(maxAnchors+1, streams[g]);
                        rmm::device_uvector<int> d_num_indices_tmp(1, streams[g]);

                        vec_managedgpumsa[g]->refine(
                            d_indices_tmp.data(),
                            d_indices_per_anchor_tmp.data(),
                            d_num_indices_tmp.data(),
                            vec_d_alignment_overlaps[g],
                            vec_d_alignment_shifts[g],
                            vec_d_alignment_nOps[g],
                            vec_d_alignment_best_alignment_flags[g],
                            vec_d_indices[g].data(),
                            vec_d_indices_per_anchor[g].data(),
                            vec_d_candidates_per_anchor_prefixsum[g].data(),
                            vec_d_anchor_sequences_lengths[g].data(),
                            vec_d_anchor_sequences_data[g].data(),
                            vec_d_anchor_qual[g],
                            vec_currentNumAnchors[g],
                            vec_d_candidate_sequences_lengths[g],
                            vec_d_candidate_sequences_data[g],
                            vec_d_cand_qual[g],
                            vec_d_isPairedCandidate[g],
                            vec_currentNumCandidates[g],
                            encodedSequencePitchInInts,
                            qualityPitchInBytes,
                            programOptions->useQualityScores,
                            programOptions->maxErrorRate,
                            programOptions->estimatedCoverage,
                            getNumRefinementIterations(),
                            streams[g]
                        );

                        std::swap(d_indices_tmp, vec_d_indices[g]);
                        std::swap(d_indices_per_anchor_tmp, vec_d_indices_per_anchor[g]);
                        std::swap(d_num_indices_tmp, vec_d_num_indices[g]);
                    }
                }

            }

            if(programOptions->useQualityScores){
                for(int g = 0; g < numGpus; g++){
                    cub::SwitchDevice sd{deviceIds[g]};
                    rmm::mr::get_current_device_resource()->deallocate(vec_d_allQualData[g], vec_allQualDataBytes[g], streams[g]);
                }
            }
        }


        void correctAnchors(const std::vector<cudaStream_t>& streams){
            if(programOptions->correctionType == CorrectionType::Classic){
                correctAnchorsClassic(streams);
            }else if(programOptions->correctionType == CorrectionType::Forest){
                correctAnchorsForestGpu(streams);
            }else{
                throw std::runtime_error("correctAnchors not implemented for this correctionType");
            }
        }

        void correctAnchorsClassic(const std::vector<cudaStream_t>& streams){
            const int numGpus = deviceIds.size();

            const float avg_support_threshold = 1.0f - 1.0f * programOptions->estimatedErrorrate;
            const float min_support_threshold = 1.0f - 3.0f * programOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                programOptions->m_coverage / 6.0f * programOptions->estimatedCoverage);
            //const float max_coverage_threshold = 0.5 * programOptions->estimatedCoverage;

            // correct anchors
            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    call_msaCorrectAnchorsKernel_async(
                        vec_d_corrected_anchors[g].data(),
                        vec_d_anchor_is_corrected[g].data(),
                        vec_d_is_high_quality_anchor[g].data(),
                        vec_managedgpumsa[g]->multiMSAView(),
                        vec_d_anchor_sequences_data[g].data(),
                        vec_d_indices_per_anchor[g].data(),
                        vec_d_numAnchors[g].data(),
                        maxAnchors,
                        encodedSequencePitchInInts,
                        decodedSequencePitchInBytes,
                        programOptions->estimatedErrorrate,
                        avg_support_threshold,
                        min_support_threshold,
                        min_coverage_threshold,
                        gpuReadStorage->getSequenceLengthUpperBound(),
                        streams[g]
                    );
                }
            }

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, streams[g]>>>(
                        vec_d_indices_of_corrected_anchors[g].data(),
                        vec_d_num_indices_of_corrected_anchors[g].data(),
                        vec_d_anchor_is_corrected[g].data(),
                        vec_d_numAnchors[g].data()
                    ); CUDACHECKASYNC;
                }
            }

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    helpers::call_fill_kernel_async(vec_d_numEditsPerCorrectedanchor[g].data(), vec_currentNumAnchors[g], 0, streams[g]);
                }
            }

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    callConstructSequenceCorrectionResultsKernel(
                        vec_d_editsPerCorrectedanchor[g].data(),
                        vec_d_numEditsPerCorrectedanchor[g].data(),
                        getDoNotUseEditsValue(),
                        vec_d_indices_of_corrected_anchors[g].data(),
                        vec_d_num_indices_of_corrected_anchors[g].data(),
                        vec_d_anchorContainsN[g].data(),
                        vec_d_anchor_sequences_data[g].data(),
                        vec_d_anchor_sequences_lengths[g].data(),
                        vec_d_corrected_anchors[g].data(),
                        vec_currentNumAnchors[g],
                        false,
                        maxNumEditsPerSequence,
                        encodedSequencePitchInInts,
                        decodedSequencePitchInBytes,
                        editsPitchInBytes,      
                        streams[g]
                    );
                }
            }
            
        }

        void correctAnchorsForestGpu(const std::vector<cudaStream_t>& streams){
            const int numGpus = deviceIds.size();
            const float avg_support_threshold = 1.0f - 1.0f * programOptions->estimatedErrorrate;
            const float min_support_threshold = 1.0f - 3.0f * programOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                programOptions->m_coverage / 6.0f * programOptions->estimatedCoverage);
            const float max_coverage_threshold = 0.5 * programOptions->estimatedCoverage;

            std::vector<char*> vec_d_corrected_anchors_ptrs(numGpus, nullptr);
            std::vector<bool*> vec_d_anchor_is_corrected_ptrs(numGpus, nullptr);
            std::vector<AnchorHighQualityFlag*> vec_d_is_high_quality_anchor_ptrs(numGpus, nullptr);
            std::vector<GPUMultiMSA> vec_multiMSA_views(numGpus);
            std::vector<GpuForest::Clf> vec_gpuForest_classifierviews(numGpus);
            std::vector<unsigned int*> vec_d_anchor_sequences_data_ptrs(numGpus, nullptr);
            std::vector<int*> vec_d_indices_per_anchor_ptrs(numGpus, nullptr);

            for(int g = 0; g < numGpus; g++){
                vec_d_corrected_anchors_ptrs[g] = vec_d_corrected_anchors[g].data();
                vec_d_anchor_is_corrected_ptrs[g] = vec_d_anchor_is_corrected[g].data();
                vec_d_is_high_quality_anchor_ptrs[g] = vec_d_is_high_quality_anchor[g].data();
                vec_multiMSA_views[g] = vec_managedgpumsa[g]->multiMSAView();
                vec_gpuForest_classifierviews[g] = vec_gpuForestAnchor[g]->getClf();
                vec_d_anchor_sequences_data_ptrs[g] = vec_d_anchor_sequences_data[g].data();
                vec_d_indices_per_anchor_ptrs[g] = vec_d_indices_per_anchor[g].data();
            }

            // correct anchors
            callMsaCorrectAnchorsWithForestKernel_multiphase(
                vec_d_corrected_anchors_ptrs,
                vec_d_anchor_is_corrected_ptrs,
                vec_d_is_high_quality_anchor_ptrs,
                vec_multiMSA_views,
                vec_gpuForest_classifierviews,
                programOptions->thresholdAnchor,
                vec_d_anchor_sequences_data_ptrs,
                vec_d_indices_per_anchor_ptrs,
                vec_currentNumAnchors,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                gpuReadStorage->getSequenceLengthUpperBound(),
                programOptions->estimatedErrorrate,
                programOptions->estimatedCoverage,
                avg_support_threshold,
                min_support_threshold,
                min_coverage_threshold,
                streams,
                deviceIds,
                reinterpret_cast<int*>(h_tempstorage.data())
            );

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, streams[g]>>>(
                        vec_d_indices_of_corrected_anchors[g].data(),
                        vec_d_num_indices_of_corrected_anchors[g].data(),
                        vec_d_anchor_is_corrected[g].data(),
                        vec_d_numAnchors[g].data()
                    ); CUDACHECKASYNC;
                }
            }

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    helpers::call_fill_kernel_async(vec_d_numEditsPerCorrectedanchor[g].data(), vec_currentNumAnchors[g], 0, streams[g]);
                }
            }

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    callConstructSequenceCorrectionResultsKernel(
                        vec_d_editsPerCorrectedanchor[g].data(),
                        vec_d_numEditsPerCorrectedanchor[g].data(),
                        getDoNotUseEditsValue(),
                        vec_d_indices_of_corrected_anchors[g].data(),
                        vec_d_num_indices_of_corrected_anchors[g].data(),
                        vec_d_anchorContainsN[g].data(),
                        vec_d_anchor_sequences_data[g].data(),
                        vec_d_anchor_sequences_lengths[g].data(),
                        vec_d_corrected_anchors[g].data(),
                        vec_currentNumAnchors[g],
                        false,
                        maxNumEditsPerSequence,
                        encodedSequencePitchInInts,
                        decodedSequencePitchInBytes,
                        editsPitchInBytes,      
                        streams[g]
                    );
                }
            }

        }
        
        void correctCandidates(const std::vector<cudaStream_t>& streams){
            if(programOptions->correctionTypeCands == CorrectionType::Classic){
                correctCandidatesClassic(streams);
            }else if(programOptions->correctionTypeCands == CorrectionType::Forest){
                correctCandidatesForestGpu(streams);
            }else{
                throw std::runtime_error("correctCandidates not implemented for this correctionTypeCands");
            }
        }

        void correctCandidatesClassic(const std::vector<cudaStream_t>& streams){
            const int numGpus = deviceIds.size();            

            const float min_support_threshold = 1.0f-3.0f*programOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                programOptions->m_coverage / 6.0f * programOptions->estimatedCoverage);
            const int new_columns_to_correct = programOptions->new_columns_to_correct;

            std::vector<rmm::device_uvector<bool>> vec_d_candidateCanBeCorrected;
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                vec_d_candidateCanBeCorrected.emplace_back(vec_currentNumCandidates[g], streams[g]);
            }

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    cub::TransformInputIterator<bool, IsHqAnchor, AnchorHighQualityFlag*>
                        d_isHqanchor(vec_d_is_high_quality_anchor[g].data(), IsHqAnchor{});

                    gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, streams[g]>>>(
                        vec_d_high_quality_anchor_indices[g].data(),
                        vec_d_num_high_quality_anchor_indices[g].data(),
                        d_isHqanchor,
                        vec_d_numAnchors[g].data()
                    ); CUDACHECKASYNC;

                    gpucorrectorkernels::initArraysBeforeCandidateCorrectionKernel<<<SDIV(vec_currentNumCandidates[g], 128), 128, 0, streams[g]>>>(
                        vec_currentNumCandidates[g],
                        vec_d_numAnchors[g].data(),
                        vec_d_num_corrected_candidates_per_anchor[g].data(),
                        vec_d_candidateCanBeCorrected[g].data()
                    ); CUDACHECKASYNC;
                }
            }

            #if 1

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};
                    bool* d_excludeFlags = vec_d_hqAnchorCorrectionOfCandidateExists[g];

                    callFlagCandidatesToBeCorrectedWithExcludeFlagsKernel(
                        vec_d_candidateCanBeCorrected[g].data(),
                        vec_d_num_corrected_candidates_per_anchor[g].data(),
                        vec_managedgpumsa[g]->multiMSAView(),
                        d_excludeFlags,
                        vec_d_alignment_shifts[g],
                        vec_d_candidate_sequences_lengths[g],
                        vec_d_anchorIndicesOfCandidates[g],
                        vec_d_is_high_quality_anchor[g].data(),
                        vec_d_candidates_per_anchor_prefixsum[g].data(),
                        vec_d_indices[g].data(),
                        vec_d_indices_per_anchor[g].data(),
                        vec_d_numAnchors[g].data(),
                        vec_d_numCandidates[g].data(),
                        min_support_threshold,
                        min_coverage_threshold,
                        new_columns_to_correct,
                        streams[g]
                    );
                }
            }
            #else
            callFlagCandidatesToBeCorrectedKernel_async(
                d_candidateCanBeCorrected.data(),
                d_num_corrected_candidates_per_anchor.data(),
                managedgpumsa->multiMSAView(),
                d_alignment_shifts.data(),
                d_candidate_sequences_lengths.data(),
                d_anchorIndicesOfCandidates.data(),
                d_is_high_quality_anchor.data(),
                d_candidates_per_anchor_prefixsum.data(),
                d_indices.data(),
                d_indices_per_anchor.data(),
                d_numAnchors.data(),
                d_numCandidates.data(),
                min_support_threshold,
                min_coverage_threshold,
                new_columns_to_correct,
                stream
            );
            #endif

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    CubCallWrapper(rmm::mr::get_current_device_resource()).cubSelectFlagged(
                        cub::CountingInputIterator<int>(0),
                        vec_d_candidateCanBeCorrected[g].data(),
                        vec_d_indices_of_corrected_candidates[g],
                        vec_d_num_total_corrected_candidates[g].data(),
                        vec_currentNumCandidates[g],
                        streams[g]
                    );

                    CUDACHECK(cudaMemcpyAsync(
                        vec_h_num_total_corrected_candidates[g].data(),
                        vec_d_num_total_corrected_candidates[g].data(),
                        sizeof(int),
                        D2H,
                        streams[g]
                    ));
                }else{
                    *vec_h_num_total_corrected_candidates[g] = 0;
                }
            }

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};
                    CUDACHECK(cudaStreamSynchronize(streams[g]));

                    if((*vec_h_num_total_corrected_candidates[g]) > 0){
                        vec_d_corrected_candidates[g].resize(decodedSequencePitchInBytes * (*vec_h_num_total_corrected_candidates[g]), streams[g]);

                        callCorrectCandidatesKernel(
                            vec_d_corrected_candidates[g].data(),            
                            vec_managedgpumsa[g]->multiMSAView(),
                            vec_d_alignment_shifts[g],
                            vec_d_alignment_best_alignment_flags[g],
                            vec_d_candidate_sequences_data[g],
                            vec_d_candidate_sequences_lengths[g],
                            vec_d_candidateContainsN[g],
                            vec_d_indices_of_corrected_candidates[g],
                            vec_d_num_total_corrected_candidates[g].data(),
                            vec_d_anchorIndicesOfCandidates[g],
                            *vec_h_num_total_corrected_candidates[g],
                            encodedSequencePitchInInts,
                            decodedSequencePitchInBytes,
                            gpuReadStorage->getSequenceLengthUpperBound(),
                            streams[g]
                        );   
                    }
                }
            }


            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    if((*vec_h_num_total_corrected_candidates[g]) > 0){
                        vec_d_numEditsPerCorrectedCandidate[g].resize((*vec_h_num_total_corrected_candidates[g]), streams[g]);
                        std::size_t numEditsCandidates = SDIV(editsPitchInBytes * (*vec_h_num_total_corrected_candidates[g]), sizeof(EncodedCorrectionEdit));
                        vec_d_editsPerCorrectedCandidate[g].resize(numEditsCandidates, streams[g]);

                        callConstructSequenceCorrectionResultsKernel(
                            vec_d_editsPerCorrectedCandidate[g].data(),
                            vec_d_numEditsPerCorrectedCandidate[g].data(),
                            getDoNotUseEditsValue(),
                            vec_d_indices_of_corrected_candidates[g],
                            vec_d_num_total_corrected_candidates[g].data(),
                            vec_d_candidateContainsN[g],
                            vec_d_candidate_sequences_data[g],
                            vec_d_candidate_sequences_lengths[g],
                            vec_d_corrected_candidates[g].data(),
                            (*vec_h_num_total_corrected_candidates[g]),
                            true,
                            maxNumEditsPerSequence,
                            encodedSequencePitchInInts,
                            decodedSequencePitchInBytes,
                            editsPitchInBytes,      
                            streams[g]
                        );
                    }
                }
            }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                //ensure release of memory on the correct device
                vec_d_candidateCanBeCorrected[g].release();
            }

        }

        void correctCandidatesForestGpu(const std::vector<cudaStream_t>& streams){

            const int numGpus = deviceIds.size();            

            const float min_support_threshold = 1.0f-3.0f*programOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                programOptions->m_coverage / 6.0f * programOptions->estimatedCoverage);
            const int new_columns_to_correct = programOptions->new_columns_to_correct;

            std::vector<rmm::device_uvector<bool>> vec_d_candidateCanBeCorrected;
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                vec_d_candidateCanBeCorrected.emplace_back(vec_currentNumCandidates[g], streams[g]);
            }

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    cub::TransformInputIterator<bool, IsHqAnchor, AnchorHighQualityFlag*>
                        d_isHqanchor(vec_d_is_high_quality_anchor[g].data(), IsHqAnchor{});

                    gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, streams[g]>>>(
                        vec_d_high_quality_anchor_indices[g].data(),
                        vec_d_num_high_quality_anchor_indices[g].data(),
                        d_isHqanchor,
                        vec_d_numAnchors[g].data()
                    ); CUDACHECKASYNC;

                    gpucorrectorkernels::initArraysBeforeCandidateCorrectionKernel<<<SDIV(vec_currentNumCandidates[g], 128), 128, 0, streams[g]>>>(
                        vec_currentNumCandidates[g],
                        vec_d_numAnchors[g].data(),
                        vec_d_num_corrected_candidates_per_anchor[g].data(),
                        vec_d_candidateCanBeCorrected[g].data()
                    ); CUDACHECKASYNC;
                }
            }

            #if 1

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};
                    bool* d_excludeFlags = vec_d_hqAnchorCorrectionOfCandidateExists[g];

                    callFlagCandidatesToBeCorrectedWithExcludeFlagsKernel(
                        vec_d_candidateCanBeCorrected[g].data(),
                        vec_d_num_corrected_candidates_per_anchor[g].data(),
                        vec_managedgpumsa[g]->multiMSAView(),
                        d_excludeFlags,
                        vec_d_alignment_shifts[g],
                        vec_d_candidate_sequences_lengths[g],
                        vec_d_anchorIndicesOfCandidates[g],
                        vec_d_is_high_quality_anchor[g].data(),
                        vec_d_candidates_per_anchor_prefixsum[g].data(),
                        vec_d_indices[g].data(),
                        vec_d_indices_per_anchor[g].data(),
                        vec_d_numAnchors[g].data(),
                        vec_d_numCandidates[g].data(),
                        min_support_threshold,
                        min_coverage_threshold,
                        new_columns_to_correct,
                        streams[g]
                    );
                }
            }
            #else
            callFlagCandidatesToBeCorrectedKernel_async(
                d_candidateCanBeCorrected.data(),
                d_num_corrected_candidates_per_anchor.data(),
                managedgpumsa->multiMSAView(),
                d_alignment_shifts.data(),
                d_candidate_sequences_lengths.data(),
                d_anchorIndicesOfCandidates.data(),
                d_is_high_quality_anchor.data(),
                d_candidates_per_anchor_prefixsum.data(),
                d_indices.data(),
                d_indices_per_anchor.data(),
                d_numAnchors.data(),
                d_numCandidates.data(),
                min_support_threshold,
                min_coverage_threshold,
                new_columns_to_correct,
                stream
            );
            #endif

            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    CubCallWrapper(rmm::mr::get_current_device_resource()).cubSelectFlagged(
                        cub::CountingInputIterator<int>(0),
                        vec_d_candidateCanBeCorrected[g].data(),
                        vec_d_indices_of_corrected_candidates[g],
                        vec_d_num_total_corrected_candidates[g].data(),
                        vec_currentNumCandidates[g],
                        streams[g]
                    );

                    CUDACHECK(cudaMemcpyAsync(
                        vec_h_num_total_corrected_candidates[g].data(),
                        vec_d_num_total_corrected_candidates[g].data(),
                        sizeof(int),
                        D2H,
                        streams[g]
                    ));
                }else{
                    *vec_h_num_total_corrected_candidates[g] = 0;
                }
            }

            std::vector<char*> vec_d_correctedCandidates_ptrs(numGpus, nullptr);
            std::vector<GPUMultiMSA> vec_multiMSA_views(numGpus);
            std::vector<GpuForest::Clf> vec_gpuForest_classifierviews(numGpus);
            std::vector<int> vec_numCandidatesToProcess(numGpus, 0);

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                CUDACHECK(cudaStreamSynchronize(streams[g]));

                vec_d_corrected_candidates[g].resize(decodedSequencePitchInBytes * (*vec_h_num_total_corrected_candidates[g]), streams[g]);
                vec_d_correctedCandidates_ptrs[g] = vec_d_corrected_candidates[g].data();
                vec_multiMSA_views[g] = vec_managedgpumsa[g]->multiMSAView();
                vec_gpuForest_classifierviews[g] = vec_gpuForestCandidate[g]->getClf();
                vec_numCandidatesToProcess[g] = *vec_h_num_total_corrected_candidates[g];
            }
            callMsaCorrectCandidatesWithForestKernelMultiPhase(
                vec_d_correctedCandidates_ptrs,
                vec_multiMSA_views,
                vec_gpuForest_classifierviews,
                programOptions->thresholdCands,
                programOptions->estimatedCoverage,
                vec_d_alignment_shifts,
                vec_d_alignment_best_alignment_flags,
                vec_d_candidate_sequences_data,
                vec_d_candidate_sequences_lengths,
                vec_d_indices_of_corrected_candidates,
                vec_d_anchorIndicesOfCandidates,
                vec_numCandidatesToProcess,          
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                gpuReadStorage->getSequenceLengthUpperBound(),
                streams,
                deviceIds,
                reinterpret_cast<int*>(h_tempstorage.data())
            );


            for(int g = 0; g < numGpus; g++){
                if(hasAnchorsAndCandidates(g)){
                    cub::SwitchDevice sd{deviceIds[g]};

                    if((*vec_h_num_total_corrected_candidates[g]) > 0){
                        vec_d_numEditsPerCorrectedCandidate[g].resize((*vec_h_num_total_corrected_candidates[g]), streams[g]);
                        std::size_t numEditsCandidates = SDIV(editsPitchInBytes * (*vec_h_num_total_corrected_candidates[g]), sizeof(EncodedCorrectionEdit));
                        vec_d_editsPerCorrectedCandidate[g].resize(numEditsCandidates, streams[g]);

                        callConstructSequenceCorrectionResultsKernel(
                            vec_d_editsPerCorrectedCandidate[g].data(),
                            vec_d_numEditsPerCorrectedCandidate[g].data(),
                            getDoNotUseEditsValue(),
                            vec_d_indices_of_corrected_candidates[g],
                            vec_d_num_total_corrected_candidates[g].data(),
                            vec_d_candidateContainsN[g],
                            vec_d_candidate_sequences_data[g],
                            vec_d_candidate_sequences_lengths[g],
                            vec_d_corrected_candidates[g].data(),
                            (*vec_h_num_total_corrected_candidates[g]),
                            true,
                            maxNumEditsPerSequence,
                            encodedSequencePitchInInts,
                            decodedSequencePitchInBytes,
                            editsPitchInBytes,      
                            streams[g]
                        );
                    }
                }
            }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};

                //ensure release of memory on the correct device
                vec_d_candidateCanBeCorrected[g].release();
            }
            
        }

        static constexpr int getDoNotUseEditsValue() noexcept{
            return -1;
        }

        void updateCorrectionFlags(const std::vector<cudaStream_t>& streams) const{
            const int numGpus = deviceIds.size();
            std::vector<const bool*> vec_d_flags(numGpus, nullptr);
            std::vector<const read_number*> vec_d_readIds(numGpus, nullptr);
            std::vector<cudaStream_t> allocStreams(numGpus, nullptr);

            for(int g = 0; g < numGpus; g++){
                vec_d_flags[g] = reinterpret_cast<const bool*>(vec_d_is_high_quality_anchor[g].data());
                vec_d_readIds[g] = vec_d_anchorReadIds[g].data();
                allocStreams[g] = vec_extraStream[g];
            }

            // correctionFlags->multi_setIsCorrectedAsHQAnchor(
            //     vec_d_flags,             
            //     vec_d_readIds, //d_readIds must be unique
            //     vec_currentNumAnchors, 
            //     streams
            // );

            //works with this sync
            // {
            //     int N = 0;
            //     CUDACHECK(cudaGetDeviceCount(&N));
            //     for(int i = 0; i < N; i++){
            //         cub::SwitchDevice sd(cudaSetDevice(i));
            //         CUDACHECK(cudaDeviceSynchronize());
            //     }
            // }
            correctionFlags->multi_setIsCorrectedAsHQAnchor(
                vec_d_flags,             
                vec_d_readIds, //d_readIds must be unique
                vec_currentNumAnchors, 
                streams,
                allocStreams
            );

            // {
            //     int N = 0;
            //     CUDACHECK(cudaGetDeviceCount(&N));
            //     for(int i = 0; i < N; i++){
            //         cub::SwitchDevice sd(cudaSetDevice(i));
            //         CUDACHECK(cudaDeviceSynchronize());
            //     }
            // }
        }

    private:
        bool hasAnchors(int g) const{
            return (vec_currentNumAnchors[g] > 0);
        }
        bool hasCandidates(int g) const{
            return (vec_currentNumCandidates[g] > 0);
        }
        bool hasAnchorsAndCandidates(int g) const{
            return (hasAnchors(g) && hasCandidates(g));
        }

        std::vector<std::array<CudaEvent, 2>> vec_events;
        //std::vector<cudaStream_t> vec_extraStream;
        std::vector<CudaStream> vec_extraStream;

        std::vector<CudaEvent> vec_previousBatchFinishedEvent;
        std::vector<CudaEvent> vec_inputCandidateDataIsReadyEvent;

        std::size_t msaColumnPitchInElements;
        std::size_t encodedSequencePitchInInts;
        std::size_t decodedSequencePitchInBytes;
        std::size_t qualityPitchInBytes;
        std::size_t editsPitchInBytes;

        int maxAnchors;
        int maxNumEditsPerSequence;
        std::vector<int> vec_currentNumAnchors;
        std::vector<int> vec_currentNumCandidates;

        std::map<int, int> numCandidatesPerReadMap{};

        GpuReadCorrectionFlags* correctionFlags;

        const GpuReadStorage* gpuReadStorage;

        const ProgramOptions* programOptions;

        MultiGpuErrorCorrectorInput* currentInput;
        std::vector<GpuErrorCorrectorRawOutput*> vec_currentOutput;

        std::vector<int> deviceIds{};
        std::vector<const GpuForest*> vec_gpuForestAnchor{};
        std::vector<const GpuForest*> vec_gpuForestCandidate{};

        std::vector<ReadStorageHandle> vec_readstorageHandle;

        std::vector<PinnedBuffer<int>> vec_h_num_total_corrected_candidates;
        std::vector<PinnedBuffer<int>> vec_h_num_indices;
        std::vector<PinnedBuffer<int>> vec_h_numSelected;
        std::vector<PinnedBuffer<int>> vec_h_managedmsa_tmp;

        std::vector<rmm::device_uvector<read_number>> vec_d_indicesForGather;

        std::vector<rmm::device_uvector<bool>> vec_d_anchorContainsN;
        std::vector<rmm::device_uvector<bool>> vec_d_candidateContainsN_;
        std::vector<rmm::device_uvector<int>> vec_d_candidate_sequences_lengths_;
        std::vector<rmm::device_uvector<unsigned int>> vec_d_candidate_sequences_data_;
        std::vector<rmm::device_uvector<int>> vec_d_anchorIndicesOfCandidates_;
        std::vector<rmm::device_uvector<int>> vec_d_alignment_overlaps_;
        std::vector<rmm::device_uvector<int>> vec_d_alignment_shifts_;
        std::vector<rmm::device_uvector<int>> vec_d_alignment_nOps_;
        std::vector<rmm::device_uvector<AlignmentOrientation>> vec_d_alignment_best_alignment_flags_; 
        std::vector<rmm::device_uvector<int>> vec_d_indices;
        std::vector<rmm::device_uvector<int>> vec_d_indices_per_anchor;
        std::vector<rmm::device_uvector<int>> vec_d_indices_per_anchor_prefixsum;
        std::vector<rmm::device_uvector<int>> vec_d_num_indices;
        std::vector<rmm::device_uvector<char>> vec_d_corrected_anchors;
        std::vector<rmm::device_uvector<char>> vec_d_corrected_candidates;
        std::vector<rmm::device_uvector<int>> vec_d_num_corrected_candidates_per_anchor;
        std::vector<rmm::device_uvector<int>> vec_d_num_corrected_candidates_per_anchor_prefixsum;
        std::vector<rmm::device_uvector<int>> vec_d_num_total_corrected_candidates;
        std::vector<rmm::device_uvector<bool>> vec_d_anchor_is_corrected;
        std::vector<rmm::device_uvector<AnchorHighQualityFlag>> vec_d_is_high_quality_anchor;
        std::vector<rmm::device_uvector<int>> vec_d_high_quality_anchor_indices;
        std::vector<rmm::device_uvector<int>> vec_d_num_high_quality_anchor_indices; 
        std::vector<rmm::device_uvector<EncodedCorrectionEdit>> vec_d_editsPerCorrectedanchor;
        std::vector<rmm::device_uvector<int>> vec_d_numEditsPerCorrectedanchor;
        std::vector<rmm::device_uvector<EncodedCorrectionEdit>> vec_d_editsPerCorrectedCandidate;
        std::vector<rmm::device_uvector<bool>> vec_d_hqAnchorCorrectionOfCandidateExists_;

        std::vector<rmm::device_uvector<char>> vec_d_allCandidateData;

        std::vector<int*> vec_d_alignment_overlaps;
        std::vector<int*> vec_d_alignment_shifts;
        std::vector<int*> vec_d_alignment_nOps;
        std::vector<AlignmentOrientation*> vec_d_alignment_best_alignment_flags;
        std::vector<int*> vec_d_anchorIndicesOfCandidates;
        std::vector<bool*> vec_d_candidateContainsN;
        std::vector<bool*> vec_d_isPairedCandidate;
        std::vector<int*> vec_d_indices_of_corrected_candidates;
        std::vector<bool*> vec_d_hqAnchorCorrectionOfCandidateExists;
        std::vector<read_number*> vec_d_candidate_read_ids;
        std::vector<int*> vec_d_candidate_sequences_lengths;
        std::vector<unsigned int*> vec_d_candidate_sequences_data;

        
        std::vector<rmm::device_uvector<int>> vec_d_numEditsPerCorrectedCandidate;
        std::vector<rmm::device_uvector<int>> vec_d_indices_of_corrected_anchors;
        std::vector<rmm::device_uvector<int>> vec_d_num_indices_of_corrected_anchors;
        std::vector<rmm::device_uvector<int>> vec_d_indices_of_corrected_candidates_;
        std::vector<rmm::device_uvector<int>> vec_d_totalNumEdits;
        std::vector<rmm::device_uvector<bool>> vec_d_isPairedCandidate_;

        std::vector<rmm::device_uvector<int>> vec_d_numAnchors;
        std::vector<rmm::device_uvector<int>> vec_d_numCandidates;
        std::vector<rmm::device_uvector<read_number>> vec_d_anchorReadIds;
        std::vector<rmm::device_uvector<unsigned int>> vec_d_anchor_sequences_data;
        std::vector<rmm::device_uvector<int>> vec_d_anchor_sequences_lengths;
        std::vector<rmm::device_uvector<read_number>> vec_d_candidate_read_ids_;
        std::vector<rmm::device_uvector<int>> vec_d_candidates_per_anchor;
        std::vector<rmm::device_uvector<int>> vec_d_candidates_per_anchor_prefixsum; 

        std::vector<std::unique_ptr<ManagedGPUMultiMSA>> vec_managedgpumsa;

        PinnedBuffer<char> h_tempstorage;
    };

#endif

}
}






#endif
