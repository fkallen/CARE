#ifndef CARE_GPUCORRECTOR_CUH
#define CARE_GPUCORRECTOR_CUH


#include <hpc_helpers.cuh>
#include <hpc_helpers/include/nvtx_markers.cuh>

#include <gpu/gpuminhasher.cuh>
#include <gpu/kernels.hpp>
#include <gpu/gpucorrectorkernels.cuh>
#include <gpu/cudagraphhelpers.cuh>
#include <gpu/gpureadstorage.cuh>
#include <gpu/asyncresult.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/cubwrappers.cuh>
#include <gpu/gpumsamanaged.cuh>

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

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <gpu/rmm_utilities.cuh>


namespace care{
namespace gpu{

    struct SequenceFlagMultiplier{
        int pitch{};
        const bool* flags{};

        __host__ __device__
        SequenceFlagMultiplier(const bool* flags_, int pitch_)
            :pitch(pitch_), flags(flags_){

        }

        __host__ __device__
        bool operator()(int i) const{
            return flags[i / pitch];
        }
    };

    template<class Iter>
    struct IteratorMultiplier{
        using value_type = typename std::iterator_traits<Iter>::value_type;

        int factor{};
        Iter data{};

        __host__ __device__
        IteratorMultiplier(Iter data_, int factor_)
            : factor(factor_), data(data_){

        }

        __host__ __device__
        value_type operator()(int i) const{
            return *(data + (i / factor));
        }
    };

    template<class Iter>
    IteratorMultiplier<Iter> make_iterator_multiplier(Iter data, int factor){
        return IteratorMultiplier<Iter>{data, factor};
    }

    struct ReplaceNumberOp{
        int doNotUseEditsValue;
        int decodedSequencePitchInBytes;

        ReplaceNumberOp(int val, int pitch) 
            : doNotUseEditsValue(val), decodedSequencePitchInBytes(pitch)
        {}

        __host__ __device__
        int operator()(const int num) const noexcept{
            return num == doNotUseEditsValue ? decodedSequencePitchInBytes : 0;
        }
    };


    class GpuErrorCorrectorInput{
    public:
        template<class T>
        using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;


        CudaEvent event{cudaEventDisableTiming};

        rmm::mr::device_memory_resource* mr;

        PinnedBuffer<int> h_numAnchors;
        PinnedBuffer<int> h_numCandidates;
        PinnedBuffer<read_number> h_anchorReadIds;
        PinnedBuffer<read_number> h_candidate_read_ids;

        rmm::device_uvector<int> d_numAnchors;
        rmm::device_uvector<int> d_numCandidates;
        rmm::device_uvector<read_number> d_anchorReadIds;
        rmm::device_uvector<unsigned int> d_anchor_sequences_data;
        rmm::device_uvector<int> d_anchor_sequences_lengths;
        rmm::device_uvector<read_number> d_candidate_read_ids;
        rmm::device_uvector<unsigned int> d_candidate_sequences_data;
        rmm::device_uvector<int> d_candidate_sequences_lengths;
        rmm::device_uvector<int> d_candidates_per_anchor;
        rmm::device_uvector<int> d_candidates_per_anchor_prefixsum;

        GpuErrorCorrectorInput()
        : mr(rmm::mr::get_current_device_resource()),
            d_numAnchors(0, cudaStreamPerThread, mr),
            d_numCandidates(0, cudaStreamPerThread, mr),
            d_anchorReadIds(0, cudaStreamPerThread, mr),
            d_anchor_sequences_data(0, cudaStreamPerThread, mr),
            //d_anchor_qualities(0, cudaStreamPerThread, mr),
            d_anchor_sequences_lengths(0, cudaStreamPerThread, mr),
            d_candidate_read_ids(0, cudaStreamPerThread, mr),
            d_candidate_sequences_data(0, cudaStreamPerThread, mr),
            //d_candidate_qualities(0, cudaStreamPerThread, mr),
            d_candidate_sequences_lengths(0, cudaStreamPerThread, mr),
            d_candidates_per_anchor(0, cudaStreamPerThread, mr),
            d_candidates_per_anchor_prefixsum(0, cudaStreamPerThread, mr)
        {
            CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            auto handleHost = [&](const auto& h){
                info.host += h.sizeInBytes();
            };
            auto handleDevice = [&](const auto& d){
                using ElementType = typename std::remove_reference<decltype(d)>::type::value_type;
                info.device[event.getDeviceId()] += d.size() * sizeof(ElementType);
            };

            handleHost(h_numAnchors);
            handleHost(h_numCandidates);
            handleHost(h_anchorReadIds);
            handleHost(h_candidate_read_ids);

            handleDevice(d_numAnchors);
            handleDevice(d_numCandidates);
            handleDevice(d_anchorReadIds);
            handleDevice(d_anchor_sequences_data);
            handleDevice(d_anchor_sequences_lengths);
            handleDevice(d_candidate_read_ids);
            handleDevice(d_candidate_sequences_data);
            handleDevice(d_candidate_sequences_lengths);
            handleDevice(d_candidates_per_anchor);
            handleDevice(d_candidates_per_anchor_prefixsum);

            return info;
        }  
    };

    class GpuErrorCorrectorRawOutput{
    public:
        template<class T>
        using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

        bool nothingToDo;
        int numAnchors;
        int numCandidates;
        int doNotUseEditsValue;
        std::size_t editsPitchInBytes;
        std::size_t decodedSequencePitchInBytes;
        CudaEvent event{cudaEventDisableTiming};
        PinnedBuffer<read_number> h_anchorReadIds;
        PinnedBuffer<read_number> h_candidate_read_ids;
        PinnedBuffer<bool> h_anchor_is_corrected;
        PinnedBuffer<AnchorHighQualityFlag> h_is_high_quality_anchor;
        PinnedBuffer<int> h_num_corrected_candidates_per_anchor;
        PinnedBuffer<int> h_num_corrected_candidates_per_anchor_prefixsum;
        PinnedBuffer<int> h_indices_of_corrected_candidates;

        PinnedBuffer<int> h_candidate_sequences_lengths;
        PinnedBuffer<int> h_numEditsPerCorrectedanchor;
        PinnedBuffer<EncodedCorrectionEdit> h_editsPerCorrectedanchor;
        PinnedBuffer<char> h_corrected_anchors;
        PinnedBuffer<int> h_anchor_sequences_lengths;
        PinnedBuffer<char> h_corrected_candidates;
        PinnedBuffer<int> h_alignment_shifts;
        PinnedBuffer<int> h_numEditsPerCorrectedCandidate;
        PinnedBuffer<EncodedCorrectionEdit> h_editsPerCorrectedCandidate;
        PinnedBuffer<int> h_anchorEditOffsets;
        PinnedBuffer<int> h_correctedAnchorsOffsets;
        PinnedBuffer<int> h_candidateEditOffsets;
        PinnedBuffer<int> h_correctedCandidatesOffsets;

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            auto handleHost = [&](const auto& h){
                info.host += h.sizeInBytes();
            };

            handleHost(h_anchorReadIds);
            handleHost(h_candidate_read_ids);
            handleHost(h_anchor_is_corrected);
            handleHost(h_is_high_quality_anchor);
            handleHost(h_num_corrected_candidates_per_anchor);
            handleHost(h_num_corrected_candidates_per_anchor_prefixsum);
            handleHost(h_indices_of_corrected_candidates);
            handleHost(h_candidate_sequences_lengths);
            handleHost(h_numEditsPerCorrectedanchor);
            handleHost(h_editsPerCorrectedanchor);
            handleHost(h_corrected_anchors);
            handleHost(h_anchor_sequences_lengths);
            handleHost(h_corrected_candidates);
            handleHost(h_alignment_shifts);
            handleHost(h_numEditsPerCorrectedCandidate);
            handleHost(h_editsPerCorrectedCandidate);
            handleHost(h_anchorEditOffsets);
            handleHost(h_correctedAnchorsOffsets);
            handleHost(h_candidateEditOffsets);
            handleHost(h_correctedCandidatesOffsets);

            return info;
        }  
    };



    class GpuAnchorHasher{
    public:

        GpuAnchorHasher() = default;

        GpuAnchorHasher(
            const GpuReadStorage& gpuReadStorage_,
            const GpuMinhasher& gpuMinhasher_,
            ThreadPool* threadPool_,
            rmm::mr::device_memory_resource* mr_
        ) : 
            gpuReadStorage{&gpuReadStorage_},
            gpuMinhasher{&gpuMinhasher_},
            threadPool{threadPool_},
            minhashHandle{gpuMinhasher->makeMinhasherHandle()},
            readstorageHandle{gpuReadStorage->makeHandle()},
            mr{mr_}
        {
            CUDACHECK(cudaGetDevice(&deviceId));            

            maxCandidatesPerRead = gpuMinhasher->getNumResultsPerMapThreshold() * gpuMinhasher->getNumberOfMaps();

            previousBatchFinishedEvent = CudaEvent{};

            encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(gpuReadStorage->getSequenceLengthUpperBound());
            qualityPitchInBytes = SDIV(gpuReadStorage->getSequenceLengthUpperBound(), 32) * 32;
        }

        ~GpuAnchorHasher(){
            // std::cerr << "GpuAnchorHasher::~GpuAnchorHasher(). Memory of minhash handle: ";
            // auto memoryUsage = gpuMinhasher->getMemoryInfo(minhashHandle);
            // std::cerr << memoryUsage.host;
            // for(auto pair : memoryUsage.device){
            //     std::cerr << ", [" << pair.first << "] " << pair.second;
            // }
            // std::cerr << "\n";

            gpuReadStorage->destroyHandle(readstorageHandle);
            gpuMinhasher->destroyHandle(minhashHandle);
        }

        void makeErrorCorrectorInput(
            const read_number* anchorIds,
            int numIds,
            bool useQualityScores,
            GpuErrorCorrectorInput& ecinput,
            cudaStream_t stream
        ){
            cub::SwitchDevice sd{deviceId};

            assert(cudaSuccess == ecinput.event.query());
            CUDACHECK(previousBatchFinishedEvent.synchronize());

            resizeBuffers(ecinput, numIds, stream);
    
            //copy input to pinned memory
            *ecinput.h_numAnchors.data() = numIds;
            std::copy_n(anchorIds, numIds, ecinput.h_anchorReadIds.data());

            CUDACHECK(cudaMemcpyAsync(
                ecinput.d_numAnchors.data(),
                ecinput.h_numAnchors.data(),
                sizeof(int),
                H2D,
                stream
            ));

            DEBUGSTREAMSYNC(stream);

            CUDACHECK(cudaMemcpyAsync(
                ecinput.d_anchorReadIds.data(),
                ecinput.h_anchorReadIds.data(),
                sizeof(read_number) * (*ecinput.h_numAnchors.data()),
                H2D,
                stream
            ));

            DEBUGSTREAMSYNC(stream);

            if(numIds > 0){
                nvtx::push_range("getAnchorReads", 0);
                getAnchorReads(ecinput, useQualityScores, stream);
                nvtx::pop_range();

                DEBUGSTREAMSYNC(stream);

                nvtx::push_range("getCandidateReadIdsWithMinhashing", 1);
                getCandidateReadIdsWithMinhashing(ecinput, stream);
                nvtx::pop_range();

                CUDACHECK(cudaStreamSynchronize(stream));

                getCandidateReads(ecinput, useQualityScores, stream);

                DEBUGSTREAMSYNC(stream);
            }            

            CUDACHECK(ecinput.event.record(stream));
            CUDACHECK(previousBatchFinishedEvent.record(stream));
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
       
            info += gpuMinhasher->getMemoryInfo(minhashHandle);
          
            info += gpuReadStorage->getMemoryInfo(readstorageHandle);
            return info;
        } 

    public: //private:
        void resizeBuffers(GpuErrorCorrectorInput& ecinput, int numAnchors, cudaStream_t stream){
            const std::size_t maxCandidates = maxCandidatesPerRead * numAnchors;
            // large enough to store all minhash results
            ecinput.h_candidate_read_ids.resize(maxCandidates);
            ecinput.d_candidate_read_ids.resize(maxCandidates, stream); 

            ecinput.h_numAnchors.resize(1);
            ecinput.h_numCandidates.resize(1);
            ecinput.h_anchorReadIds.resize(numAnchors);

            ecinput.d_numAnchors.resize(1, stream);
            ecinput.d_numCandidates.resize(1, stream);
            ecinput.d_anchorReadIds.resize(numAnchors, stream);
            ecinput.d_anchor_sequences_data.resize(encodedSequencePitchInInts * numAnchors, stream);
            ecinput.d_anchor_sequences_lengths.resize(numAnchors, stream);
            ecinput.d_candidates_per_anchor.resize(numAnchors, stream);
            ecinput.d_candidates_per_anchor_prefixsum.resize(numAnchors + 1, stream);
        }
        
        void getAnchorReads(GpuErrorCorrectorInput& ecinput, bool /*useQualityScores*/, cudaStream_t stream){
            const int numAnchors = (*ecinput.h_numAnchors.data());

            gpuReadStorage->gatherSequences(
                readstorageHandle,
                ecinput.d_anchor_sequences_data.data(),
                encodedSequencePitchInInts,
                makeAsyncConstBufferWrapper(ecinput.h_anchorReadIds.data()),
                ecinput.d_anchorReadIds.data(),
                numAnchors,
                stream,
                mr
            );

            DEBUGSTREAMSYNC(stream);

            gpuReadStorage->gatherSequenceLengths(
                readstorageHandle,
                ecinput.d_anchor_sequences_lengths.data(),
                ecinput.d_anchorReadIds.data(),
                numAnchors,
                stream
            );

            DEBUGSTREAMSYNC(stream);

            // if(useQualityScores){
            //     ecinput.d_anchor_qualities.resize(qualityPitchInBytes * numAnchors, stream);

            //     gpuReadStorage->gatherQualities(
            //         readstorageHandle,
            //         ecinput.d_anchor_qualities.data(),
            //         qualityPitchInBytes,
            //         makeAsyncConstBufferWrapper(ecinput.h_anchorReadIds.data()),
            //         ecinput.d_anchorReadIds.data(),
            //         numAnchors,
            //         stream,
            //         mr
            //     );

            //     DEBUGSTREAMSYNC(stream);
            // }
        }

        void getCandidateReads(GpuErrorCorrectorInput& ecinput, bool /*useQualityScores*/, cudaStream_t stream){
            const int numCandidates = (*ecinput.h_numCandidates.data());

            ecinput.d_candidate_sequences_data.resize(encodedSequencePitchInInts * numCandidates, stream);

            gpuReadStorage->gatherSequences(
                readstorageHandle,
                ecinput.d_candidate_sequences_data.data(),
                encodedSequencePitchInInts,
                makeAsyncConstBufferWrapper(ecinput.h_candidate_read_ids.data()),
                ecinput.d_candidate_read_ids.data(),
                numCandidates,
                stream,
                mr
            );

            DEBUGSTREAMSYNC(stream);

            ecinput.d_candidate_sequences_lengths.resize(numCandidates, stream);

            gpuReadStorage->gatherSequenceLengths(
                readstorageHandle,
                ecinput.d_candidate_sequences_lengths.data(),
                ecinput.d_candidate_read_ids.data(),
                numCandidates,
                stream
            );

            DEBUGSTREAMSYNC(stream);

            // if(useQualityScores){
            //     ecinput.d_candidate_qualities.resize(qualityPitchInBytes * numCandidates, stream);
                
            //     gpuReadStorage->gatherQualities(
            //         readstorageHandle,
            //         ecinput.d_candidate_qualities.data(),
            //         qualityPitchInBytes,
            //         makeAsyncConstBufferWrapper(ecinput.h_candidate_read_ids.data()),
            //         ecinput.d_candidate_read_ids.data(),
            //         numCandidates,
            //         stream,
            //         mr
            //     );

            //     DEBUGSTREAMSYNC(stream);
            // }
        }

        void getCandidateReadIdsWithMinhashing(GpuErrorCorrectorInput& ecinput, cudaStream_t stream){
            int totalNumValues = 0;

            gpuMinhasher->determineNumValues(
                minhashHandle,
                ecinput.d_anchor_sequences_data.data(),
                encodedSequencePitchInInts,
                ecinput.d_anchor_sequences_lengths.data(),
                (*ecinput.h_numAnchors.data()),
                ecinput.d_candidates_per_anchor.data(),
                totalNumValues,
                stream,
                mr
            );

            CUDACHECK(cudaStreamSynchronize(stream));

            ecinput.d_candidate_read_ids.resize(totalNumValues, stream);
            ecinput.h_candidate_read_ids.resize(totalNumValues);

            if(totalNumValues == 0){
                *ecinput.h_numCandidates = 0;
                CUDACHECK(cudaMemsetAsync(ecinput.d_numCandidates.data(), 0, sizeof(int), stream));
                CUDACHECK(cudaMemsetAsync(ecinput.d_candidates_per_anchor.data(), 0, sizeof(int) * (*ecinput.h_numAnchors), stream));
                CUDACHECK(cudaMemsetAsync(ecinput.d_candidates_per_anchor_prefixsum.data(), 0, sizeof(int) * (1 + (*ecinput.h_numAnchors)), stream));
                return;
            }

            gpuMinhasher->retrieveValues(
                minhashHandle,
                ecinput.d_anchorReadIds.data(),
                (*ecinput.h_numAnchors.data()),                
                totalNumValues,
                ecinput.d_candidate_read_ids.data(),
                ecinput.d_candidates_per_anchor.data(),
                ecinput.d_candidates_per_anchor_prefixsum.data(),
                stream,
                mr
            );

            gpucorrectorkernels::copyMinhashResultsKernel<<<640, 256, 0, stream>>>(
                ecinput.d_numCandidates.data(),
                ecinput.h_numCandidates.data(),
                ecinput.h_candidate_read_ids.data(),
                ecinput.d_candidates_per_anchor_prefixsum.data(),
                ecinput.d_candidate_read_ids.data(),
                *ecinput.h_numAnchors.data()
            ); CUDACHECKASYNC;

            // helpers::lambda_kernel<<<1,1,0,stream>>>(
            //     [
            //         numAnchors = (*ecinput.h_numAnchors.data()),
            //         d_candidate_read_ids = ecinput.d_candidate_read_ids.data(),
            //         d_candidates_per_anchor = ecinput.d_candidates_per_anchor.data(),
            //         d_candidates_per_anchor_prefixsum = ecinput.d_candidates_per_anchor_prefixsum.data(),
            //         d_anchorReadIds = ecinput.d_anchorReadIds.data()
            //     ] __device__ (){
            //         for(int a = 0; a < numAnchors; a++){
            //             if(d_anchorReadIds[a] > 12850 && d_anchorReadIds[a] < 12870){
            //             //if(d_anchorReadIds[a] < 12870){
            //             //if(d_anchorReadIds[a] < 100){
            //                 printf("a = %d %u\n", a, d_anchorReadIds[a]);
            //                 for(int c = 0; c < d_candidates_per_anchor[a]; c++){
            //                     printf("%u ", d_candidate_read_ids[d_candidates_per_anchor_prefixsum[a] + c]);
            //                 }
            //                 printf("\n");
            //             }
            //         }
            //     }
            // );
            // CUDACHECKASYNC;

            // CUDACHECK(cudaStreamSynchronize(stream));


        }
    
        int deviceId;
        int maxCandidatesPerRead;
        std::size_t encodedSequencePitchInInts;
        std::size_t qualityPitchInBytes;
        CudaEvent previousBatchFinishedEvent;
        const GpuReadStorage* gpuReadStorage;
        const GpuMinhasher* gpuMinhasher;
        ThreadPool* threadPool;
        ThreadPool::ParallelForHandle pforHandle;
        MinhasherHandle minhashHandle;
        ReadStorageHandle readstorageHandle;
        rmm::mr::device_memory_resource* mr;
    };


    class OutputConstructor{
    public:

        OutputConstructor() = default;

        OutputConstructor(
            ReadCorrectionFlags& correctionFlags_,
            const ProgramOptions& programOptions_
        ) :
            correctionFlags{&correctionFlags_},
            programOptions{&programOptions_}
        {

        }

        std::vector<int> getAnchorIndicesToProcessAndUpdateCorrectionFlags(const GpuErrorCorrectorRawOutput& currentOutput) const{
            std::vector<int> anchorIndicesToProcess;
            anchorIndicesToProcess.reserve(currentOutput.numAnchors);

            nvtx::push_range("preprocess anchor results",0);

            for(int anchor_index = 0; anchor_index < currentOutput.numAnchors; anchor_index++){
                const read_number readId = currentOutput.h_anchorReadIds[anchor_index];
                const bool isCorrected = currentOutput.h_anchor_is_corrected[anchor_index];
                const bool isHQ = currentOutput.h_is_high_quality_anchor[anchor_index].hq();

                if(isHQ){
                    correctionFlags->setCorrectedAsHqAnchor(readId);
                }

                if(isCorrected){
                    anchorIndicesToProcess.emplace_back(anchor_index);
                }else{
                    correctionFlags->setCouldNotBeCorrectedAsAnchor(readId);
                }

                assert(!(isHQ && !isCorrected));
            }

            nvtx::pop_range();

            return anchorIndicesToProcess;
        }

        std::vector<std::pair<int,int>> getCandidateIndicesToProcess(const GpuErrorCorrectorRawOutput& currentOutput) const{
            std::vector<std::pair<int,int>> candidateIndicesToProcess;

            if(programOptions->correctCandidates){
                candidateIndicesToProcess.reserve(16 * currentOutput.numAnchors);
            }

            if(programOptions->correctCandidates){

                nvtx::push_range("preprocess candidate results",0);

                for(int anchor_index = 0; anchor_index < currentOutput.numAnchors; anchor_index++){

                    const int globalOffset = currentOutput.h_num_corrected_candidates_per_anchor_prefixsum[anchor_index];
                    const int n_corrected_candidates = currentOutput.h_num_corrected_candidates_per_anchor[anchor_index];

                    // const int* const my_indices_of_corrected_candidates = currentOutput.h_indices_of_corrected_candidates
                    //                                     + globalOffset;

                    for(int i = 0; i < n_corrected_candidates; ++i) {
                        //const int global_candidate_index = my_indices_of_corrected_candidates[i];
                        //const read_number candidate_read_id = currentOutput.h_candidate_read_ids[global_candidate_index];
                        const read_number candidate_read_id = currentOutput.h_candidate_read_ids[globalOffset + i];

                        if (!correctionFlags->isCorrectedAsHQAnchor(candidate_read_id)) {
                            candidateIndicesToProcess.emplace_back(std::make_pair(anchor_index, i));
                        }
                    }
                }

                nvtx::pop_range();

            }

            return candidateIndicesToProcess;
        }

        template<class ForLoop>
        CorrectionOutput constructResults(const GpuErrorCorrectorRawOutput& currentOutput, ForLoop loopExecutor) const{
            //assert(cudaSuccess == currentOutput.event.query());

            if(currentOutput.nothingToDo){
                return CorrectionOutput{};
            }

            const std::vector<int> anchorIndicesToProcess = getAnchorIndicesToProcessAndUpdateCorrectionFlags(currentOutput);
            const std::vector<std::pair<int,int>> candidateIndicesToProcess = getCandidateIndicesToProcess(currentOutput);

            const int numCorrectedAnchors = anchorIndicesToProcess.size();
            const int numCorrectedCandidates = candidateIndicesToProcess.size();

            // std::cerr << "numCorrectedAnchors: " << numCorrectedAnchors << 
            //     ", numCorrectedCandidates: " << numCorrectedCandidates << "\n";

            CorrectionOutput correctionOutput;
            correctionOutput.anchorCorrections.resize(numCorrectedAnchors);

            if(programOptions->correctCandidates){
                correctionOutput.candidateCorrections.resize(numCorrectedCandidates);
            }

            auto unpackAnchors = [&](int begin, int end){
                nvtx::push_range("Anchor unpacking " + std::to_string(end - begin), 3);

                //Edits and numEdits are stored compact, only for corrected anchors.
                //they are indexed by positionInVector instead of anchor_index
                            
                for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                    const int anchor_index = anchorIndicesToProcess[positionInVector];

                    auto& tmp = correctionOutput.anchorCorrections[positionInVector];
                    
                    const read_number readId = currentOutput.h_anchorReadIds[anchor_index];

                    tmp.hq = currentOutput.h_is_high_quality_anchor[anchor_index].hq();                    
                    tmp.type = TempCorrectedSequenceType::Anchor;
                    tmp.readId = readId;
                    tmp.edits.clear();
                    
                    const int numEdits = currentOutput.h_numEditsPerCorrectedanchor[positionInVector];
                    if(numEdits != currentOutput.doNotUseEditsValue){
                        const int editOffset = currentOutput.h_anchorEditOffsets[positionInVector];
                        const auto* myedits = currentOutput.h_editsPerCorrectedanchor + editOffset;
                        tmp.edits.insert(tmp.edits.end(), myedits, myedits + numEdits);
                        tmp.useEdits = true;
                    }else{
                        
                        tmp.useEdits = false;

                        const int sequenceOffset = currentOutput.h_correctedAnchorsOffsets[positionInVector];

                        const char* const my_corrected_anchor_data = currentOutput.h_corrected_anchors + sequenceOffset;
                        const int anchor_length = currentOutput.h_anchor_sequences_lengths[anchor_index];
                        tmp.sequence.assign(my_corrected_anchor_data, anchor_length);
                    }

                    // if(tmp.readId == 9273463){
                    //     std::cerr << tmp << "\n";
                    // }
                }

                nvtx::pop_range();
            };

            auto unpackcandidates = [&](int begin, int end){
                nvtx::push_range("candidate unpacking " + std::to_string(end - begin), 3);

                //buffers are stored compact. offsets for each anchor are given by h_num_corrected_candidates_per_anchor_prefixsum
                //Edits, numEdits, h_candidate_read_ids, h_candidate_sequences_lengths, h_alignment_shifts are stored compact, only for corrected candidates.
                //edits are only present for candidates which use edits and have numEdits > 0
                //offsets to the edits of candidates are stored in h_candidateEditOffsets
                

                for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                    

                    //TIMERSTARTCPU(setup);
                    const int anchor_index = candidateIndicesToProcess[positionInVector].first;
                    const int candidateIndex = candidateIndicesToProcess[positionInVector].second;
                    const read_number anchorReadId = currentOutput.h_anchorReadIds[anchor_index];

                    auto& tmp = correctionOutput.candidateCorrections[positionInVector];

                    const size_t offsetForCorrectedCandidateData = currentOutput.h_num_corrected_candidates_per_anchor_prefixsum[anchor_index];

                    const read_number candidate_read_id = currentOutput.h_candidate_read_ids[offsetForCorrectedCandidateData + candidateIndex];
                    const int candidate_shift = currentOutput.h_alignment_shifts[offsetForCorrectedCandidateData + candidateIndex];

                    if(programOptions->new_columns_to_correct < candidate_shift){
                        std::cerr << "readid " << anchorReadId << " candidate readid " << candidate_read_id << " : "
                        << candidate_shift << " " << programOptions->new_columns_to_correct <<"\n";

                        assert(programOptions->new_columns_to_correct >= candidate_shift);
                    }                
                    
                    tmp.type = TempCorrectedSequenceType::Candidate;
                    tmp.shift = candidate_shift;
                    tmp.readId = candidate_read_id;
                    tmp.edits.clear();
                    
                    const int numEdits = currentOutput.h_numEditsPerCorrectedCandidate[offsetForCorrectedCandidateData + candidateIndex];
                    const int editsOffset = currentOutput.h_candidateEditOffsets[offsetForCorrectedCandidateData + candidateIndex];

                    if(numEdits != currentOutput.doNotUseEditsValue){
                        const auto* myEdits = &currentOutput.h_editsPerCorrectedCandidate[editsOffset];
                        tmp.edits.insert(tmp.edits.end(), myEdits, myEdits + numEdits);
                        tmp.useEdits = true;
                    }else{
                        const int correctionOffset = currentOutput.h_correctedCandidatesOffsets[candidateIndex];
                        const int candidate_length = currentOutput.h_candidate_sequences_lengths[candidateIndex];
                        const char* const candidate_data = currentOutput.h_corrected_candidates + correctionOffset * currentOutput.decodedSequencePitchInBytes;
                        tmp.sequence.assign(candidate_data, candidate_length);
                        
                        tmp.useEdits = false;
                    }

                    // if(tmp.readId == 9273463){
                    //     std::cerr << tmp << " with anchorid " << anchorReadId << "\n";
                    // }
                }

                nvtx::pop_range();
            };


            if(!programOptions->correctCandidates){
                loopExecutor(
                    0, 
                    numCorrectedAnchors, 
                    [=](auto begin, auto end, auto /*threadId*/){
                        unpackAnchors(begin, end);
                    }
                );
            }else{
        
  
                loopExecutor(
                    0, 
                    numCorrectedAnchors, 
                    [=](auto begin, auto end, auto /*threadId*/){
                        unpackAnchors(begin, end);
                    }
                );

                loopExecutor(
                    0, 
                    numCorrectedCandidates, 
                    [=](auto begin, auto end, auto /*threadId*/){
                        unpackcandidates(begin, end);
                    }
                );      
            }

            return correctionOutput;
        }

        template<class ForLoop>
        EncodedCorrectionOutput constructEncodedResults(const GpuErrorCorrectorRawOutput& currentOutput, ForLoop loopExecutor) const{
            //assert(cudaSuccess == currentOutput.event.query());

            if(currentOutput.nothingToDo){
                return EncodedCorrectionOutput{};
            }

            const std::vector<int> anchorIndicesToProcess = getAnchorIndicesToProcessAndUpdateCorrectionFlags(currentOutput);
            const std::vector<std::pair<int,int>> candidateIndicesToProcess = getCandidateIndicesToProcess(currentOutput);

            const int numCorrectedAnchors = anchorIndicesToProcess.size();
            const int numCorrectedCandidates = candidateIndicesToProcess.size();

            // std::cerr << "numCorrectedAnchors: " << numCorrectedAnchors << 
            //     ", numCorrectedCandidates: " << numCorrectedCandidates << "\n";

            EncodedCorrectionOutput encodedCorrectionOutput;
            encodedCorrectionOutput.encodedAnchorCorrections.resize(numCorrectedAnchors);

            if(programOptions->correctCandidates){
                encodedCorrectionOutput.encodedCandidateCorrections.resize(numCorrectedCandidates);
            }

            auto unpackAnchors = [&](int begin, int end){
                nvtx::push_range("Anchor unpacking " + std::to_string(end - begin), 3);

                //Edits and numEdits are stored compact, only for corrected anchors.
                //they are indexed by positionInVector instead of anchor_index

                std::vector<CorrectionEdit> edits;
                            
                for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                    const int anchor_index = anchorIndicesToProcess[positionInVector];
                    
                    const read_number readId = currentOutput.h_anchorReadIds[anchor_index];

                    edits.clear();
                    bool useEdits = false;
                    const char* sequence = nullptr;
                    int sequenceLength = 0;
                    
                    const int numEdits = currentOutput.h_numEditsPerCorrectedanchor[positionInVector];
                    if(numEdits != currentOutput.doNotUseEditsValue){
                        const int editOffset = currentOutput.h_anchorEditOffsets[positionInVector];
                        const auto* myEdits = currentOutput.h_editsPerCorrectedanchor + editOffset;
                        edits.insert(edits.end(), myEdits, myEdits + numEdits);
                        useEdits = true;
                    }else{                        
                        const int sequenceOffset = currentOutput.h_correctedAnchorsOffsets[positionInVector];
                        const char* const my_corrected_anchor_data = currentOutput.h_corrected_anchors + sequenceOffset;
                        const int anchor_length = currentOutput.h_anchor_sequences_lengths[anchor_index];
 
                        sequenceLength = anchor_length;
                        sequence = my_corrected_anchor_data;
                    }

                    EncodedTempCorrectedSequence::encodeDataIntoEncodedCorrectedSequence(
                        encodedCorrectionOutput.encodedAnchorCorrections[positionInVector],
                        readId,
                        currentOutput.h_is_high_quality_anchor[anchor_index].hq(),
                        useEdits,
                        TempCorrectedSequenceType::Anchor,
                        0,
                        edits.size(),
                        edits.data(),
                        sequenceLength,
                        sequence
                    );
                }

                nvtx::pop_range();
            };

            auto unpackcandidates = [&](int begin, int end){
                nvtx::push_range("candidate unpacking " + std::to_string(end - begin), 3);

                //buffers are stored compact. offsets for each anchor are given by h_num_corrected_candidates_per_anchor_prefixsum
                //Edits, numEdits, h_candidate_read_ids, h_candidate_sequences_lengths, h_alignment_shifts are stored compact, only for corrected candidates.
                //edits are only present for candidates which use edits and have numEdits > 0
                //offsets to the edits of candidates are stored in h_candidateEditOffsets

                std::vector<CorrectionEdit> edits;          

                for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                    

                    const int anchor_index = candidateIndicesToProcess[positionInVector].first;
                    const int candidateIndex = candidateIndicesToProcess[positionInVector].second;
                    const read_number anchorReadId = currentOutput.h_anchorReadIds[anchor_index];

                    const size_t offsetForCorrectedCandidateData = currentOutput.h_num_corrected_candidates_per_anchor_prefixsum[anchor_index];

                    const read_number candidate_read_id = currentOutput.h_candidate_read_ids[offsetForCorrectedCandidateData + candidateIndex];
                    const int candidate_shift = currentOutput.h_alignment_shifts[offsetForCorrectedCandidateData + candidateIndex];

                    if(programOptions->new_columns_to_correct < candidate_shift){
                        std::cerr << "readid " << anchorReadId << " candidate readid " << candidate_read_id << " : "
                        << candidate_shift << " " << programOptions->new_columns_to_correct <<"\n";

                        assert(programOptions->new_columns_to_correct >= candidate_shift);
                    }

                    edits.clear();
                    
                    const int numEdits = currentOutput.h_numEditsPerCorrectedCandidate[offsetForCorrectedCandidateData + candidateIndex];
                    const int editsOffset = currentOutput.h_candidateEditOffsets[offsetForCorrectedCandidateData + candidateIndex];

                    bool useEdits = false;
                    const char* sequence = nullptr;
                    int sequenceLength = 0;

                    if(numEdits != currentOutput.doNotUseEditsValue){
                        const auto* myEdits = &currentOutput.h_editsPerCorrectedCandidate[editsOffset];
                        edits.insert(edits.end(), myEdits, myEdits + numEdits);
                        useEdits = true;
                    }else{
                        const int correctionOffset = currentOutput.h_correctedCandidatesOffsets[candidateIndex];
                        const int candidate_length = currentOutput.h_candidate_sequences_lengths[candidateIndex];
                        const char* const candidate_data = currentOutput.h_corrected_candidates + correctionOffset * currentOutput.decodedSequencePitchInBytes;

                        sequenceLength = candidate_length;
                        sequence = candidate_data;
                    }

                    EncodedTempCorrectedSequence::encodeDataIntoEncodedCorrectedSequence(
                        encodedCorrectionOutput.encodedCandidateCorrections[positionInVector],
                        candidate_read_id,
                        false,
                        useEdits,
                        TempCorrectedSequenceType::Candidate,
                        candidate_shift,
                        edits.size(),
                        edits.data(),
                        sequenceLength,
                        sequence
                    );
                }

                nvtx::pop_range();
            };


            if(!programOptions->correctCandidates){
                loopExecutor(
                    0, 
                    numCorrectedAnchors, 
                    [=](auto begin, auto end, auto /*threadId*/){
                        unpackAnchors(begin, end);
                    }
                );
            }else{
        
  
                loopExecutor(
                    0, 
                    numCorrectedAnchors, 
                    [=](auto begin, auto end, auto /*threadId*/){
                        unpackAnchors(begin, end);
                    }
                );

                loopExecutor(
                    0, 
                    numCorrectedCandidates, 
                    [=](auto begin, auto end, auto /*threadId*/){
                        unpackcandidates(begin, end);
                    }
                );      
            }

            return encodedCorrectionOutput;
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            return info;
        }
    public: //private:
        ReadCorrectionFlags* correctionFlags;
        const ProgramOptions* programOptions;
    };

    class GpuErrorCorrector{
        static constexpr bool useGraph() noexcept{
            return false;
        }

    public:

        template<class T>
        using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

        static constexpr int getNumRefinementIterations() noexcept{
            return 5;
        }

        static constexpr bool useMsaRefinement() noexcept{
            return getNumRefinementIterations() > 0;
        }

        GpuErrorCorrector() = default;

        GpuErrorCorrector(
            const GpuReadStorage& gpuReadStorage_,
            const ReadCorrectionFlags& correctionFlags_,
            const ProgramOptions& programOptions_,
            int maxAnchorsPerCall,
            rmm::mr::device_memory_resource* mr_,
            ThreadPool* threadPool_,
            const GpuForest* gpuForestAnchor_,
            const GpuForest* gpuForestCandidate_
        ) : 
            maxAnchors{maxAnchorsPerCall},
            maxCandidates{0},
            correctionFlags{&correctionFlags_},
            gpuReadStorage{&gpuReadStorage_},
            programOptions{&programOptions_},
            mr{mr_},
            threadPool{threadPool_},
            gpuForestAnchor{gpuForestAnchor_},
            gpuForestCandidate{gpuForestCandidate_},
            readstorageHandle{gpuReadStorage->makeHandle()},
            d_indicesForGather{0, cudaStreamPerThread, mr},
            //d_candidate_qualities_compact{0, cudaStreamPerThread, mr},
            d_anchorContainsN{0, cudaStreamPerThread, mr},
            d_candidateContainsN{0, cudaStreamPerThread, mr},
            d_candidate_sequences_lengths{0, cudaStreamPerThread, mr},
            d_candidate_sequences_data{0, cudaStreamPerThread, mr},
            //d_transposedCandidateSequencesData{0, cudaStreamPerThread, mr},
            d_anchor_qualities{0, cudaStreamPerThread, mr},
            d_candidate_qualities{0, cudaStreamPerThread, mr},
            d_anchorIndicesOfCandidates{0, cudaStreamPerThread, mr},
            d_alignment_overlaps{0, cudaStreamPerThread, mr},
            d_alignment_shifts{0, cudaStreamPerThread, mr},
            d_alignment_nOps{0, cudaStreamPerThread, mr},
            d_alignment_best_alignment_flags{0, cudaStreamPerThread, mr}, 
            d_indices{0, cudaStreamPerThread, mr},
            d_indices_per_anchor{0, cudaStreamPerThread, mr},
            d_indices_per_anchor_prefixsum{0, cudaStreamPerThread, mr},
            d_num_indices{0, cudaStreamPerThread, mr},
            d_corrected_anchors{0, cudaStreamPerThread, mr},
            d_corrected_candidates{0, cudaStreamPerThread, mr},
            d_num_corrected_candidates_per_anchor{0, cudaStreamPerThread, mr},
            d_num_corrected_candidates_per_anchor_prefixsum{0, cudaStreamPerThread, mr},
            d_num_total_corrected_candidates{0, cudaStreamPerThread, mr},
            d_anchor_is_corrected{0, cudaStreamPerThread, mr},
            d_is_high_quality_anchor{0, cudaStreamPerThread, mr},
            d_high_quality_anchor_indices{0, cudaStreamPerThread, mr},
            d_num_high_quality_anchor_indices{0, cudaStreamPerThread, mr}, 
            d_editsPerCorrectedanchor{0, cudaStreamPerThread, mr},
            d_numEditsPerCorrectedanchor{0, cudaStreamPerThread, mr},
            d_editsPerCorrectedCandidate{0, cudaStreamPerThread, mr},
            d_numEditsPerCorrectedCandidate{0, cudaStreamPerThread, mr},
            d_indices_of_corrected_anchors{0, cudaStreamPerThread, mr},
            d_num_indices_of_corrected_anchors{0, cudaStreamPerThread, mr},
            d_indices_of_corrected_candidates{0, cudaStreamPerThread, mr},
            d_totalNumEdits{0, cudaStreamPerThread, mr},
            d_isPairedCandidate{0, cudaStreamPerThread, mr},
            d_numAnchors{0, cudaStreamPerThread, mr},
            d_numCandidates{0, cudaStreamPerThread, mr},
            d_anchorReadIds{0, cudaStreamPerThread, mr},
            d_anchor_sequences_data{0, cudaStreamPerThread, mr},
            d_anchor_sequences_lengths{0, cudaStreamPerThread, mr},
            d_candidate_read_ids{0, cudaStreamPerThread, mr},
            d_candidates_per_anchor{0, cudaStreamPerThread, mr},
            d_candidates_per_anchor_prefixsum{0, cudaStreamPerThread, mr}
        {
            if(programOptions->correctionType != CorrectionType::Classic){
                assert(gpuForestAnchor != nullptr);
            }

            if(programOptions->correctionTypeCands != CorrectionType::Classic){
                assert(gpuForestCandidate != nullptr);
            }

            CUDACHECK(cudaGetDevice(&deviceId));

            for(auto& event: events){
                event = std::move(CudaEvent{cudaEventDisableTiming});
            }

            previousBatchFinishedEvent = CudaEvent{};

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

            initFixedSizeBuffers(cudaStreamPerThread);

            CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }

        ~GpuErrorCorrector(){
            gpuReadStorage->destroyHandle(readstorageHandle);

            //for(auto pair : numCandidatesPerReadMap){
                //std::cerr << pair.first << " " << pair.second << "\n";
            //}
        }

        void correct(GpuErrorCorrectorInput& input, GpuErrorCorrectorRawOutput& output, cudaStream_t stream){
            CUDACHECK(previousBatchFinishedEvent.synchronize());
            CUDACHECK(cudaStreamSynchronize(stream));

            //assert(cudaSuccess == input.event.query());
            //assert(cudaSuccess == output.event.query());

            currentInput = &input;
            currentOutput = &output;

            assert(*currentInput->h_numAnchors.data() <= maxAnchors);

            currentNumAnchors = *currentInput->h_numAnchors.data();
            currentNumCandidates = *currentInput->h_numCandidates.data();

            if(gpuReadStorage->isPairedEnd()){
                assert(currentNumAnchors % 2 == 0);
            }

            currentOutput->nothingToDo = false;
            currentOutput->numAnchors = currentNumAnchors;
            currentOutput->numCandidates = currentNumCandidates;
            currentOutput->doNotUseEditsValue = getDoNotUseEditsValue();
            currentOutput->editsPitchInBytes = editsPitchInBytes;
            currentOutput->decodedSequencePitchInBytes = decodedSequencePitchInBytes;

            if(currentNumCandidates == 0 || currentNumAnchors == 0){
		        //std::cerr << "currentNumAnchors " << currentNumAnchors << ", currentNumCandidates " << currentNumCandidates << "\n";
                currentOutput->nothingToDo = true;
                return;
            }
            
            cub::SwitchDevice sd{deviceId};

            //fixed size memory should already be allocated. However, this will also set the correct working stream for stream-ordered allocations which is important.
            initFixedSizeBuffers(stream); 

            resizeBuffers(currentNumAnchors, currentNumCandidates, stream);

            gpucorrectorkernels::copyCorrectionInputDeviceData<<<SDIV(currentNumCandidates, 256),256, 0, stream>>>(
                d_numAnchors.data(),
                d_numCandidates.data(),
                d_anchorReadIds.data(),
                d_anchor_sequences_data.data(),
                d_anchor_sequences_lengths.data(),
                d_candidate_read_ids.data(),
                d_candidates_per_anchor.data(),
                d_candidates_per_anchor_prefixsum.data(),
                encodedSequencePitchInInts,
                currentNumAnchors,
                currentNumCandidates,
                currentInput->d_anchorReadIds.data(),
                currentInput->d_anchor_sequences_data.data(),
                currentInput->d_anchor_sequences_lengths.data(),
                currentInput->d_candidate_read_ids.data(),
                currentInput->d_candidates_per_anchor.data(),
                currentInput->d_candidates_per_anchor_prefixsum.data()
            ); CUDACHECKASYNC;

            DEBUGSTREAMSYNC(stream);

            CUDACHECK(cudaMemcpyAsync(
                d_candidate_sequences_data.data(),
                currentInput->d_candidate_sequences_data.data(),
                sizeof(unsigned int) * encodedSequencePitchInInts * currentNumCandidates,
                D2D,
                stream
            ));

            DEBUGSTREAMSYNC(stream);

            CUDACHECK(cudaMemcpyAsync(
                d_candidate_sequences_lengths.data(),
                currentInput->d_candidate_sequences_lengths.data(),
                sizeof(int) * currentNumCandidates,
                D2D,
                stream
            ));

            DEBUGSTREAMSYNC(stream);

            // if(programOptions->useQualityScores){

            //     CUDACHECK(cudaMemcpyAsync(
            //         d_anchor_qualities.data(),
            //         currentInput->d_anchor_qualities.data(),
            //         sizeof(char) * qualityPitchInBytes * currentNumAnchors,
            //         D2D,
            //         stream
            //     ));

            //     DEBUGSTREAMSYNC(stream);

            //     CUDACHECK(cudaMemcpyAsync(
            //         d_candidate_qualities.data(),
            //         currentInput->d_candidate_qualities.data(),
            //         sizeof(char) * qualityPitchInBytes * currentNumCandidates,
            //         D2D,
            //         stream
            //     ));

            //     DEBUGSTREAMSYNC(stream);

            // }

            //after gpu data has been copied to local working set, the gpu data of currentInput can be reused
            CUDACHECK(currentInput->event.record(stream));

            gpucorrectorkernels::setAnchorIndicesOfCandidateskernel
                    <<<currentNumAnchors, 128, 0, stream>>>(
                d_anchorIndicesOfCandidates.data(),
                d_numAnchors.data(),
                d_candidates_per_anchor.data(),
                d_candidates_per_anchor_prefixsum.data()
            ); CUDACHECKASYNC;

            flagPairedCandidates(stream);

            getAmbiguousFlagsOfAnchors(stream);
            getAmbiguousFlagsOfCandidates(stream);


            // nvtx::push_range("getCandidateSequenceData", 3);
            // getCandidateSequenceData(stream); 
            // nvtx::pop_range();


            // if(useGraph()){
            //     //std::cerr << "Launching graph for output " << currentOutput << "\n";
            //     graphMap[currentOutput].execute(stream);
            //     //CUDACHECK(cudaStreamSynchronize(stream));
            // }else{
                execute(stream);
            //}

            managedgpumsa = nullptr;

            nvtx::push_range("copyAnchorResultsFromDeviceToHost", 3);
            copyAnchorResultsFromDeviceToHost(stream);
            nvtx::pop_range();

            if(programOptions->correctCandidates){
                nvtx::push_range("copyCandidateResultsFromDeviceToHost", 4);
                copyCandidateResultsFromDeviceToHost(stream);
                nvtx::pop_range();
            }

            std::copy_n(currentInput->h_anchorReadIds.data(), currentNumAnchors, currentOutput->h_anchorReadIds.data());            
            //std::copy_n(currentInput->h_candidate_read_ids.data(), currentNumCandidates, currentOutput->h_candidate_read_ids.data()); //remove if candidates are compacted


            //after the current work in stream is completed, all results in currentOutput are ready to use.
            CUDACHECK(cudaEventRecord(currentOutput->event, stream));
            CUDACHECK(cudaEventRecord(previousBatchFinishedEvent, stream));
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            auto handleHost = [&](const auto& h){
                info.host += h.sizeInBytes();
            };
            auto handleDevice = [&](const auto& d){
                using ElementType = typename std::remove_reference<decltype(d)>::type::value_type;
                info.device[deviceId] += d.size() * sizeof(ElementType);
            };

            info += gpuReadStorage->getMemoryInfo(readstorageHandle);
            if(managedgpumsa){
                info += managedgpumsa->getMemoryInfo();
            }

            handleHost(h_num_total_corrected_candidates);
            handleHost(h_num_indices);
            handleHost(h_numSelected);
            handleHost(h_numRemainingCandidatesAfterAlignment);
            handleHost(h_managedmsa_tmp);

            handleHost(h_indicesForGather);
            handleHost(h_isPairedCandidate);
            handleHost(h_candidates_per_anchor_prefixsum);
            handleHost(h_indices);
            handleHost(h_flagsCandidates);

            handleDevice(d_anchorContainsN);
            handleDevice(d_candidateContainsN);
            handleDevice(d_candidate_sequences_lengths);
            handleDevice(d_candidate_sequences_data);
            //handleDevice(d_transposedCandidateSequencesData);
            handleDevice(d_anchor_qualities);
            handleDevice(d_candidate_qualities);
            handleDevice(d_anchorIndicesOfCandidates);
            handleDevice(d_alignment_overlaps);
            handleDevice(d_alignment_shifts);
            handleDevice(d_alignment_nOps);
            handleDevice(d_alignment_best_alignment_flags);
            handleDevice(d_indices);
            handleDevice(d_indices_per_anchor);
            handleDevice(d_indices_per_anchor_prefixsum);
            handleDevice(d_num_indices);
            handleDevice(d_corrected_anchors);
            handleDevice(d_corrected_candidates);
            handleDevice(d_num_corrected_candidates_per_anchor);
            handleDevice(d_num_corrected_candidates_per_anchor_prefixsum);
            handleDevice(d_num_total_corrected_candidates);
            handleDevice(d_anchor_is_corrected);
            handleDevice(d_is_high_quality_anchor);
            handleDevice(d_high_quality_anchor_indices);
            handleDevice(d_num_high_quality_anchor_indices);
            handleDevice(d_editsPerCorrectedanchor);
            handleDevice(d_numEditsPerCorrectedanchor);
            handleDevice(d_editsPerCorrectedCandidate);
            handleDevice(d_numEditsPerCorrectedCandidate);
            handleDevice(d_indices_of_corrected_anchors);
            handleDevice(d_num_indices_of_corrected_anchors);
            handleDevice(d_indices_of_corrected_candidates);
            handleDevice(d_numEditsPerCorrectedanchor);
            handleDevice(d_numAnchors);
            handleDevice(d_numCandidates);
            handleDevice(d_anchorReadIds);
            handleDevice(d_anchor_sequences_data);
            handleDevice(d_anchor_sequences_lengths);
            handleDevice(d_candidate_read_ids);
            handleDevice(d_candidates_per_anchor);
            handleDevice(d_candidates_per_anchor_prefixsum);

            return info;
        } 

        void releaseMemory(cudaStream_t stream){
            auto handleDevice = [&](auto& d){
                ::destroy(d, stream);
            };

            handleDevice(d_anchorContainsN);
            handleDevice(d_candidateContainsN);
            handleDevice(d_candidate_sequences_lengths);
            handleDevice(d_candidate_sequences_data);
            handleDevice(d_anchor_qualities);
            handleDevice(d_candidate_qualities);
            handleDevice(d_anchorIndicesOfCandidates);
            handleDevice(d_alignment_overlaps);
            handleDevice(d_alignment_shifts);
            handleDevice(d_alignment_nOps);
            handleDevice(d_alignment_best_alignment_flags);
            handleDevice(d_indices);
            handleDevice(d_indices_per_anchor);
            handleDevice(d_indices_per_anchor_prefixsum);
            handleDevice(d_num_indices);
            handleDevice(d_corrected_anchors);
            handleDevice(d_corrected_candidates);
            handleDevice(d_num_corrected_candidates_per_anchor);
            handleDevice(d_num_corrected_candidates_per_anchor_prefixsum);
            handleDevice(d_num_total_corrected_candidates);
            handleDevice(d_anchor_is_corrected);
            handleDevice(d_is_high_quality_anchor);
            handleDevice(d_high_quality_anchor_indices);
            handleDevice(d_num_high_quality_anchor_indices);
            handleDevice(d_editsPerCorrectedanchor);
            handleDevice(d_numEditsPerCorrectedanchor);
            handleDevice(d_editsPerCorrectedCandidate);
            handleDevice(d_numEditsPerCorrectedCandidate);
            handleDevice(d_indices_of_corrected_anchors);
            handleDevice(d_num_indices_of_corrected_anchors);
            handleDevice(d_indices_of_corrected_candidates);
            handleDevice(d_numEditsPerCorrectedanchor);
            handleDevice(d_numAnchors);
            handleDevice(d_numCandidates);
            handleDevice(d_anchorReadIds);
            handleDevice(d_anchor_sequences_data);
            handleDevice(d_anchor_sequences_lengths);
            handleDevice(d_candidate_read_ids);
            handleDevice(d_candidates_per_anchor);
            handleDevice(d_candidates_per_anchor_prefixsum);
        } 

        void releaseCandidateMemory(cudaStream_t stream){
            auto handleDevice = [&](auto& d){
                ::destroy(d, stream);
            };

            handleDevice(d_candidateContainsN);
            handleDevice(d_candidate_sequences_lengths);
            handleDevice(d_candidate_sequences_data);
            handleDevice(d_candidate_qualities);
            handleDevice(d_anchorIndicesOfCandidates);
            handleDevice(d_alignment_overlaps);
            handleDevice(d_alignment_shifts);
            handleDevice(d_alignment_nOps);
            handleDevice(d_alignment_best_alignment_flags);
            handleDevice(d_indices);
            handleDevice(d_corrected_candidates);
            handleDevice(d_editsPerCorrectedCandidate);
            handleDevice(d_numEditsPerCorrectedCandidate);
            handleDevice(d_indices_of_corrected_candidates);
            handleDevice(d_candidate_read_ids);
            handleDevice(d_candidates_per_anchor);
            handleDevice(d_candidates_per_anchor_prefixsum);
        } 

        


    public: //private:

        void gpuMemsetZero(cudaStream_t stream){
            auto zero = [&](auto& devicebuffer){
                using ElementType = typename std::remove_reference<decltype(devicebuffer)>::type::value_type;
                cudaMemsetAsync(devicebuffer.data(), 0, devicebuffer.size() * sizeof(ElementType), stream);
            };

            zero(d_anchorContainsN);
            zero(d_anchor_qualities);
 
            zero(d_indices_per_anchor);
            zero(d_num_indices);
            zero(d_corrected_anchors);
            zero(d_num_corrected_candidates_per_anchor);
            zero(d_num_corrected_candidates_per_anchor_prefixsum);
            zero(d_num_total_corrected_candidates);
            zero(d_anchor_is_corrected);
            zero(d_is_high_quality_anchor);
            zero(d_high_quality_anchor_indices);
            zero(d_num_high_quality_anchor_indices);
            zero(d_editsPerCorrectedanchor);
            zero(d_numEditsPerCorrectedanchor);
            zero(d_indices_of_corrected_anchors);
            zero(d_num_indices_of_corrected_anchors);
            zero(d_numAnchors);
            zero(d_numCandidates);
            zero(d_anchorReadIds);
            zero(d_anchor_sequences_data);
            zero(d_anchor_sequences_lengths);
            zero(d_candidates_per_anchor);
            zero(d_candidates_per_anchor_prefixsum);
            zero(d_anchorIndicesOfCandidates);
            zero(d_candidateContainsN);
            zero(d_candidate_read_ids);
            zero(d_candidate_sequences_lengths);
            zero(d_candidate_sequences_data);
            //zero(d_transposedCandidateSequencesData);            
            zero(d_candidate_qualities);
            zero(d_alignment_overlaps);
            zero(d_alignment_shifts);
            zero(d_alignment_nOps);
            zero(d_alignment_best_alignment_flags);
            zero(d_indices);
            zero(d_corrected_candidates);
            zero(d_editsPerCorrectedCandidate);
            zero(d_numEditsPerCorrectedCandidate);
            zero(d_indices_of_corrected_candidates);
            
        }

        void initFixedSizeBuffers(cudaStream_t stream){
            const std::size_t numEditsAnchors = SDIV(editsPitchInBytes * maxAnchors, sizeof(EncodedCorrectionEdit));          

            //does not depend on number of candidates
            h_num_total_corrected_candidates.resize(1);
            h_num_indices.resize(1);
            h_numSelected.resize(1);
            h_numRemainingCandidatesAfterAlignment.resize(1);
            h_managedmsa_tmp.resize(1);

            //does not depend on number of candidates
            d_anchorContainsN.resize(maxAnchors, stream);

            if(programOptions->useQualityScores){
                d_anchor_qualities.resize(maxAnchors * qualityPitchInBytes, stream);
            }

            d_indices_per_anchor.resize(maxAnchors, stream);
            d_num_indices.resize(1, stream);
            d_indices_per_anchor_prefixsum.resize(maxAnchors, stream);
            d_corrected_anchors.resize(maxAnchors * decodedSequencePitchInBytes, stream);
            d_num_corrected_candidates_per_anchor.resize(maxAnchors, stream);
            d_num_corrected_candidates_per_anchor_prefixsum.resize(maxAnchors, stream);
            d_num_total_corrected_candidates.resize(1, stream);
            d_anchor_is_corrected.resize(maxAnchors, stream);
            d_is_high_quality_anchor.resize(maxAnchors, stream);
            d_high_quality_anchor_indices.resize(maxAnchors, stream);
            d_num_high_quality_anchor_indices.resize(1, stream); 
            d_editsPerCorrectedanchor.resize(numEditsAnchors, stream);
            d_numEditsPerCorrectedanchor.resize(maxAnchors, stream);
            d_indices_of_corrected_anchors.resize(maxAnchors, stream);
            d_num_indices_of_corrected_anchors.resize(1, stream);

            d_numAnchors.resize(1, stream);
            d_numCandidates.resize(1, stream);
            d_anchorReadIds.resize(maxAnchors, stream);
            d_anchor_sequences_data.resize(encodedSequencePitchInInts * maxAnchors, stream);
            d_anchor_sequences_lengths.resize(maxAnchors, stream);
            d_candidates_per_anchor.resize(maxAnchors, stream);
            h_candidates_per_anchor_prefixsum.resize(maxAnchors + 1);
            d_candidates_per_anchor_prefixsum.resize(maxAnchors + 1, stream);
            d_totalNumEdits.resize(1, stream);
        }
 
        void resizeBuffers(int numReads, int numCandidates, cudaStream_t stream){  
            assert(numReads <= maxAnchors);
            //std::cerr << "numReads: " << numReads << ", numCandidates: " << numCandidates << "\n";

            //bool maxCandidatesDidChange = false;
            constexpr int stepsizeForMaxCandidates = 10000;
            if(numCandidates > maxCandidates){
                //round up numCandidates to next multiple of stepsize
                maxCandidates = SDIV(numCandidates, stepsizeForMaxCandidates) * stepsizeForMaxCandidates;
                //maxCandidatesDidChange = true;

                if(useGraph()){
                    //reallocation will occure. invalidate all graphs and recapture them.
                    for(auto& pair : graphMap){
                        pair.second.valid = false;
                    }
                }
            }

            //std::size_t numEditsCandidates = SDIV(editsPitchInBytes * maxCandidates, sizeof(EncodedCorrectionEdit));

            const std::size_t numEditsAnchors = SDIV(editsPitchInBytes * maxAnchors, sizeof(EncodedCorrectionEdit));          

            //does not depend on number of candidates
            bool outputBuffersReallocated = false;
            outputBuffersReallocated |= currentOutput->h_anchor_sequences_lengths.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_corrected_anchors.resize(maxAnchors * decodedSequencePitchInBytes);            
            outputBuffersReallocated |= currentOutput->h_anchor_is_corrected.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_is_high_quality_anchor.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_editsPerCorrectedanchor.resize(numEditsAnchors);
            outputBuffersReallocated |= currentOutput->h_numEditsPerCorrectedanchor.resize(maxAnchors);            
            outputBuffersReallocated |= currentOutput->h_num_corrected_candidates_per_anchor.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_num_corrected_candidates_per_anchor_prefixsum.resize(maxAnchors);

            outputBuffersReallocated |= currentOutput->h_anchorReadIds.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_anchorEditOffsets.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_correctedAnchorsOffsets.resize(maxAnchors * decodedSequencePitchInBytes);

            // outputBuffersReallocated |= currentOutput->h_candidate_sequences_lengths.resize(maxCandidates);
            // outputBuffersReallocated |= currentOutput->h_corrected_candidates.resize(maxCandidates * decodedSequencePitchInBytes);
            // outputBuffersReallocated |= currentOutput->h_editsPerCorrectedCandidate.resize(numEditsCandidates);
            // outputBuffersReallocated |= currentOutput->h_numEditsPerCorrectedCandidate.resize(maxCandidates);
            // outputBuffersReallocated |= currentOutput->h_candidateEditOffsets.resize(maxCandidates);
            // outputBuffersReallocated |= currentOutput->h_indices_of_corrected_candidates.resize(maxCandidates);
            // outputBuffersReallocated |= currentOutput->h_alignment_shifts.resize(maxCandidates);
            // outputBuffersReallocated |= currentOutput->h_candidate_read_ids.resize(maxCandidates);
            // outputBuffersReallocated |= currentOutput->h_correctedCandidatesOffsets.resize(maxCandidates * decodedSequencePitchInBytes);
            
            
            d_anchorIndicesOfCandidates.resize(maxCandidates, stream);
            d_candidateContainsN.resize(maxCandidates, stream);
            d_candidate_read_ids.resize(maxCandidates, stream);
            d_candidate_sequences_lengths.resize(maxCandidates, stream);
            d_candidate_sequences_data.resize(maxCandidates * encodedSequencePitchInInts, stream);
            //d_transposedCandidateSequencesData.resize(maxCandidates * encodedSequencePitchInInts, stream);
            d_isPairedCandidate.resize(maxCandidates, stream);
            h_isPairedCandidate.resize(maxCandidates);

            h_flagsCandidates.resize(maxCandidates);
            
            if(programOptions->useQualityScores){
                d_candidate_qualities.resize(maxCandidates * qualityPitchInBytes, stream);

                //d_candidate_qualities_compact.resize(maxCandidates * qualityPitchInBytes, stream);
            }

            h_indicesForGather.resize(maxCandidates);
            d_indicesForGather.resize(maxCandidates, stream);
            
            d_alignment_overlaps.resize(maxCandidates, stream);
            d_alignment_shifts.resize(maxCandidates, stream);
            d_alignment_nOps.resize(maxCandidates, stream);
            d_alignment_best_alignment_flags.resize(maxCandidates, stream);
            d_indices.resize(maxCandidates + 1, stream);
            //d_corrected_candidates.resize(maxCandidates * decodedSequencePitchInBytes, stream);
            //d_editsPerCorrectedCandidate.resize(numEditsCandidates, stream);

            //d_numEditsPerCorrectedCandidate.resize(maxCandidates, stream);
            d_indices_of_corrected_candidates.resize(maxCandidates, stream);

            if(numCandidates > maxCandidates){
                CUDACHECK(cudaStreamSynchronize(stream));

                cudaMemPool_t mempool;
                CUDACHECK(cudaDeviceGetMemPool(&mempool, deviceId));
                CUDACHECK(cudaMemPoolTrimTo(mempool, 0));
            }

            // if(maxCandidatesDidChange){
            //     std::cerr << "maxCandidates changed to " << maxCandidates << "\n";
            // }

            if(useGraph()){
                if(!graphMap[currentOutput].valid){
                    if(outputBuffersReallocated){
                        std::cerr << "outputBuffersReallocated " << currentOutput << "\n";
                    }
                    //std::cerr << "Capture graph for output " << currentOutput << "\n";
                    graphMap[currentOutput].capture(
                        [&](cudaStream_t capstream){
                            execute(capstream);
                        }
                    );
                }
            }
        }

        void flagPairedCandidates(cudaStream_t stream){

            if(gpuReadStorage->isPairedEnd()){

                assert(currentNumAnchors % 2 == 0);
                assert(currentNumAnchors != 0);

                d_isPairedCandidate.resize(currentNumCandidates, stream);

                helpers::call_fill_kernel_async(d_isPairedCandidate.data(), currentNumCandidates, false, stream);                   

                dim3 block = 128;
                dim3 grid = currentNumAnchors / 2;
                constexpr int staticSmemBytes = 4096;

                gpucorrectorkernels::flagPairedCandidatesKernel<128,staticSmemBytes>
                <<<grid, block, 0, stream>>>(
                    currentNumAnchors / 2,
                    d_candidates_per_anchor.data(),
                    d_candidates_per_anchor_prefixsum.data(),
                    d_candidate_read_ids.data(),
                    d_isPairedCandidate.data()
                ); CUDACHECKASYNC;

                #if 0
                    // CUDACHECK(cudaDeviceSynchronize());

                    // std::cerr << "currentNumCandidates = " << currentNumCandidates << "\n";
                    // std::cerr << "isPairedCandidate\n";
                    // for(int i = 0; i < currentNumCandidates; i++){
                    //     std::cerr << d_isPairedCandidate[i] << " ";
                    // }
                    // std::cerr << "\n";
                    // std::cerr << "d_anchorIndicesOfCandidates\n";
                    // for(int i = 0; i < currentNumCandidates; i++){
                    //     std::cerr << d_anchorIndicesOfCandidates[i] << " ";
                    // }
                    // std::cerr << "\n";
                    // std::cerr << "d_candidate_read_ids\n";
                    // for(int i = 0; i < currentNumCandidates; i++){
                    //     std::cerr << d_candidate_read_ids[i] << " ";
                    // }
                    // std::cerr << "\n";


                    //remove candidates which are not paired
                    rmm::device_uvector<read_number> d_candidate_read_ids2(currentNumCandidates, stream, mr);
                    rmm::device_uvector<int> d_anchorIndicesOfCandidates2(currentNumCandidates, stream, mr);

                    CubCallWrapper(mr).cubSelectFlagged(
                        thrust::make_zip_iterator(thrust::make_tuple(
                            d_candidate_read_ids.data(),
                            d_anchorIndicesOfCandidates.data()
                        )),                        
                        thrust::make_transform_iterator(d_isPairedCandidate.data(), thrust::identity<bool>()),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            d_candidate_read_ids2.data(),
                            d_anchorIndicesOfCandidates2.data()
                        )),
                        d_numCandidates.data(),
                        currentNumCandidates,
                        stream
                    );

                    // CUDACHECK(cudaDeviceSynchronize());

                    // std::cerr << "currentNumCandidates2 = " << *d_numCandidates << "\n";

                    // std::cerr << "d_anchorIndicesOfCandidates2\n";
                    // for(int i = 0; i < *d_numCandidates; i++){
                    //     std::cerr << d_anchorIndicesOfCandidates2[i] << " ";
                    // }
                    // std::cerr << "\n";
                    // std::cerr << "d_candidate_read_ids2\n";
                    // for(int i = 0; i < *d_numCandidates; i++){
                    //     std::cerr << d_candidate_read_ids2[i] << " ";
                    // }
                    // std::cerr << "\n";

                    CUDACHECK(cudaMemcpyAsync(
                        h_num_indices.data(),
                        d_numCandidates.data(),
                        sizeof(int),
                        D2H,
                        stream
                    ));
                    CUDACHECK(cudaStreamSynchronize(stream));

                    auto oldNumCandidates = currentNumCandidates;
                    currentNumCandidates = *h_num_indices;

                    CUDACHECK(cudaMemcpyAsync(
                        currentInput->h_candidate_read_ids.data(),
                        d_candidate_read_ids2.data(),
                        sizeof(int) * currentNumCandidates,
                        D2H,
                        stream
                    ));
                    CUDACHECK(cudaEventRecord(events[1], stream));

                    std::swap(d_candidate_read_ids, d_candidate_read_ids2);
                    std::swap(d_anchorIndicesOfCandidates, d_anchorIndicesOfCandidates2);

                    CUDACHECK(cudaMemsetAsync(
                        d_candidates_per_anchor.data(),
                        0,
                        sizeof(int) * currentNumAnchors,
                        stream
                    ));


                    if(currentNumCandidates > 0){

                        rmm::device_uvector<int> d_uniqueAnchorIndices(maxNumAnchors, stream, mr);
                        rmm::device_uvector<int> d_aggregates_out(maxNumAnchors, stream, mr);
                        rmm::device_scalar<int> d_numRuns(stream, mr);

                        CubCallWrapper(mr).cubReduceByKey(
                            d_anchorIndicesOfCandidates.data(), 
                            d_uniqueAnchorIndices.data(), 
                            thrust::make_constant_iterator(1), 
                            d_aggregates_out.data(), 
                            d_num_indices.data(), 
                            cub::Sum(), 
                            currentNumCandidates, 
                            stream
                        );

                        //CUDACHECK(cudaDeviceSynchronize());

                        helpers::lambda_kernel<<<SDIV(currentNumAnchors, 256), 256, 0, stream>>>(
                            [
                                d_uniqueAnchorIndices = d_uniqueAnchorIndices.data(),
                                d_aggregates_out = d_aggregates_out.data(),
                                d_candidates_per_anchor = d_candidates_per_anchor.data(),
                                d_numRuns = d_numRuns.data()
                            ] __device__ (){
                                
                                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                                const int stride = blockDim.x * gridDim.x;

                                for(int i = tid; i < *d_numRuns; i += stride){
                                    d_candidates_per_anchor[d_uniqueAnchorIndices[i]]
                                        = d_aggregates_out[i];
                                }
                            }
                        ); CUDACHECKASYNC;

                        //CUDACHECK(cudaDeviceSynchronize());

                        CubCallWrapper(mr).cubInclusiveSum(
                            d_candidates_per_anchor.data(),
                            d_candidates_per_anchor_prefixsum.data() + 1,
                            currentNumAnchors,
                            stream
                        );

                        // CUDACHECK(cudaDeviceSynchronize());
                        // std::cerr << "d_candidates_per_anchor\n";
                        // for(int i = 0; i < currentNumAnchors; i++){
                        //     std::cerr << d_candidates_per_anchor[i] << " ";
                        // }
                        // std::cerr << "\n";
                        // std::cerr << "d_candidates_per_anchor_prefixsum\n";
                        // for(int i = 0; i < currentNumAnchors + 1; i++){
                        //     std::cerr << d_candidates_per_anchor_prefixsum[i] << " ";
                        // }
                        // std::cerr << "\n";

                        //update host candidate read ids

                        // std::remove_if(
                        //     currentInput->h_candidate_read_ids.begin(),
                        //     currentInput->h_candidate_read_ids.begin() + oldNumCandidates,
                        //     [&](const read_number& id){
                        //         const std::size_t i = std::distance((const read_number*)&(*currentInput->h_candidate_read_ids.begin()), &id);
                        //         return h_isPairedCandidate[i];
                        //     }
                        // );

                    }

                    CUDACHECK(cudaEventSynchronize(events[1])); //wait for currentInput->h_candidateReadIds

                    

                #endif
            }else{
                CUDACHECK(cudaMemsetAsync(
                    d_isPairedCandidate.data(),
                    0,
                    sizeof(bool) * currentNumCandidates,
                    stream
                ));
            }
        }

        void execute(cudaStream_t stream){

            nvtx::push_range("getCandidateAlignments", 5);
            getCandidateAlignments(stream); 
            nvtx::pop_range();

            // if(programOptions->useQualityScores) {
                
            //     nvtx::push_range("getQualities", 4);

            //     getQualities(stream);

            //     nvtx::pop_range();

            // }

            #if 0
            nvtx::push_range("buildMultipleSequenceAlignment", 6);
            buildMultipleSequenceAlignment(stream);
            nvtx::pop_range();

            if(useMsaRefinement()){

                nvtx::push_range("refineMultipleSequenceAlignment", 7);
                refineMultipleSequenceAlignment(stream);
                nvtx::pop_range();

            }
            #else 

            nvtx::push_range("buildMultipleSequenceAlignment", 6);
            buildAndRefineMultipleSequenceAlignment(stream);
            nvtx::pop_range();

            #endif

            nvtx::push_range("correctanchors", 8);
            correctAnchors(stream);
            nvtx::pop_range();

            if(programOptions->correctCandidates) {                        

                nvtx::push_range("correctCandidates", 9);
                correctCandidates(stream);
                nvtx::pop_range();
                
            }
        }

        void copyAnchorResultsFromDeviceToHost(cudaStream_t stream){
            if(programOptions->correctionType == CorrectionType::Classic){
                copyAnchorResultsFromDeviceToHostClassic(stream);
            }else if(programOptions->correctionType == CorrectionType::Forest){
                copyAnchorResultsFromDeviceToHostForestGpu(stream);
            }else{
                throw std::runtime_error("copyAnchorResultsFromDeviceToHost not implemented for this correctionType");
            }
        }

        void copyAnchorResultsFromDeviceToHostClassic(cudaStream_t stream){

            if(int(d_indices.capacity()) < currentNumAnchors + 1){
                CUDACHECK(cudaStreamSynchronize(stream));
                d_indices.resize(currentNumAnchors+1, stream);
            }

            rmm::device_uvector<int> d_editsOffsetsTmp(currentNumAnchors + 1, stream, mr);
            int* const d_totalNumberOfEdits = d_editsOffsetsTmp.data() + currentNumAnchors;

            helpers::call_fill_kernel_async(d_editsOffsetsTmp.data(), 1, 0, stream); CUDACHECKASYNC;

            //num edits per anchor prefixsum
            CubCallWrapper(mr).cubInclusiveSum(
                thrust::make_transform_iterator(
                    d_numEditsPerCorrectedanchor.data(),
                    [doNotUseEditsValue = getDoNotUseEditsValue()] __device__ (const auto& num){ return num == doNotUseEditsValue ? 0 : num;}
                ), 
                d_editsOffsetsTmp.data() + 1, 
                currentNumAnchors,
                stream
            );

            //compact edits
            std::size_t numEditsAnchors = SDIV(editsPitchInBytes * currentNumAnchors, sizeof(EncodedCorrectionEdit));
            rmm::device_uvector<EncodedCorrectionEdit> d_editsPerCorrectedanchor2(numEditsAnchors, stream, mr);

            gpucorrectorkernels::compactEditsKernel<<<SDIV(currentNumAnchors, 128), 128, 0, stream>>>(
                d_editsPerCorrectedanchor.data(),
                d_editsPerCorrectedanchor2.data(),
                d_editsOffsetsTmp.data(),
                d_numAnchors.data(),
                d_numEditsPerCorrectedanchor.data(),
                getDoNotUseEditsValue(),
                editsPitchInBytes
            ); CUDACHECKASYNC;

            //copy compacted edits to host
            helpers::call_copy_n_kernel(
                (const int*)d_editsPerCorrectedanchor2.data(), 
                thrust::make_transform_iterator(d_totalNumberOfEdits, [] __device__ (const int num){return SDIV(num * sizeof(EncodedCorrectionEdit), sizeof(int));}),//d_totalNumberOfEdits, 
                (int*)currentOutput->h_editsPerCorrectedanchor.data(), 
                currentNumAnchors, 
                stream
            ); CUDACHECKASYNC;

            //copy other buffers to host
            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    d_anchor_sequences_lengths.data(), 
                    d_anchor_is_corrected.data(),
                    d_is_high_quality_anchor.data(),
                    d_numEditsPerCorrectedanchor.data(),
                    d_editsOffsetsTmp.data()
                )), 
                currentNumAnchors, 
                thrust::make_zip_iterator(thrust::make_tuple(
                    currentOutput->h_anchor_sequences_lengths.data(), 
                    currentOutput->h_anchor_is_corrected.data(),
                    currentOutput->h_is_high_quality_anchor.data(),
                    currentOutput->h_numEditsPerCorrectedanchor.data(),
                    currentOutput->h_anchorEditOffsets.data()
                )), 
                stream
            ); CUDACHECKASYNC;

            
            //compact corrected anchor sequences with numEdits == getDoNotUseEditsValue
            auto correctedAnchorsPitches = thrust::make_transform_iterator(
                d_numEditsPerCorrectedanchor.data(),
                ReplaceNumberOp(getDoNotUseEditsValue(), decodedSequencePitchInBytes)
            );

            int* const d_correctedAnchorOffsetsTmp = d_editsOffsetsTmp.data();
            int* const d_totalCorrectedSequencesBytes = d_correctedAnchorOffsetsTmp + currentNumAnchors;
            helpers::call_fill_kernel_async(d_correctedAnchorOffsetsTmp, 1, 0, stream); CUDACHECKASYNC;

            CubCallWrapper(mr).cubInclusiveSum(
                correctedAnchorsPitches, 
                d_correctedAnchorOffsetsTmp + 1, 
                currentNumAnchors,
                stream
            );

            helpers::call_copy_n_kernel(
                d_correctedAnchorOffsetsTmp, 
                currentNumAnchors, 
                currentOutput->h_correctedAnchorsOffsets.data(), 
                stream
            );

            rmm::device_uvector<char> d_corrected_anchors2(decodedSequencePitchInBytes * currentNumAnchors, stream, mr);

            gpucorrectorkernels::compactCorrectedSequencesKernel<32><<<SDIV(currentNumAnchors, 128), 128, 0, stream>>>(
                d_corrected_anchors.data(),
                d_corrected_anchors2.data(),
                this->decodedSequencePitchInBytes,
                d_num_indices_of_corrected_anchors.data(),
                d_numEditsPerCorrectedanchor.data(),
                getDoNotUseEditsValue(),
                d_correctedAnchorOffsetsTmp,
                d_indices_of_corrected_anchors.data()
            ); CUDACHECKASYNC;

            assert(decodedSequencePitchInBytes % sizeof(int) == 0);

            //copy compacted anchor corrections to host
            helpers::call_copy_n_kernel(
                (const int*)d_corrected_anchors2.data(),
                thrust::make_transform_iterator(d_totalCorrectedSequencesBytes, [] __device__ (const int num){return SDIV(num, sizeof(int));}),
                (int*)currentOutput->h_corrected_anchors.data(), 
                currentNumAnchors * decodedSequencePitchInBytes / sizeof(int),
                stream
            );            

        }

        void copyAnchorResultsFromDeviceToHostForestGpu(cudaStream_t stream){
            copyAnchorResultsFromDeviceToHostClassic(stream);
        }

        void copyCandidateResultsFromDeviceToHost(cudaStream_t stream){
            if(programOptions->correctionTypeCands == CorrectionType::Classic){
                copyCandidateResultsFromDeviceToHostClassic(stream);
            }else if(programOptions->correctionTypeCands == CorrectionType::Forest){
                copyCandidateResultsFromDeviceToHostForestGpu(stream);
            }else{
                throw std::runtime_error("copyCandidateResultsFromDeviceToHost not implemented for this correctionTypeCands");
            }
        }

        void copyCandidateResultsFromDeviceToHostClassic(cudaStream_t stream){
            // helpers::call_copy_n_kernel(
            //     d_num_total_corrected_candidates.data(),
            //     1,
            //     h_num_total_corrected_candidates.data(),
            //     stream
            // );

            CubCallWrapper(mr).cubExclusiveSum(
                d_num_corrected_candidates_per_anchor.data(), 
                d_num_corrected_candidates_per_anchor_prefixsum.data(), 
                maxAnchors, 
                stream
            );

            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    d_num_corrected_candidates_per_anchor_prefixsum.data(), 
                    d_num_corrected_candidates_per_anchor.data()
                )),
                currentNumAnchors,
                thrust::make_zip_iterator(thrust::make_tuple(
                    currentOutput->h_num_corrected_candidates_per_anchor_prefixsum.data(), 
                    currentOutput->h_num_corrected_candidates_per_anchor.data()
                )),
                stream
            );

            if((*h_num_total_corrected_candidates) > 0){

            


                rmm::device_uvector<int> d_alignment_shifts2((*h_num_total_corrected_candidates), stream, mr);
                rmm::device_uvector<read_number> d_candidate_read_ids2((*h_num_total_corrected_candidates), stream, mr);
                rmm::device_uvector<int> d_candidate_sequences_lengths2((*h_num_total_corrected_candidates), stream, mr);

                helpers::call_compact_kernel_async(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        d_alignment_shifts2.data(), 
                        d_candidate_read_ids2.data(),
                        d_candidate_sequences_lengths2.data()
                    )),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        d_alignment_shifts.data(), 
                        d_candidate_read_ids.data(),
                        d_candidate_sequences_lengths.data()
                    )),
                    d_indices_of_corrected_candidates.data(), 
                    d_num_total_corrected_candidates.data(), 
                    (*h_num_total_corrected_candidates),
                    stream
                );

                //compute edit offsets for compacted edits
                auto inputIter2 = thrust::make_transform_iterator(
                    d_numEditsPerCorrectedCandidate.data(),
                    [doNotUseEditsValue = getDoNotUseEditsValue()] __device__ (const auto& num){ return num == doNotUseEditsValue ? 0 : num;}
                );

                int* const d_editsOffsetsTmp = d_indices.data();
                int* const d_totalNumberOfEdits = d_editsOffsetsTmp + (*h_num_total_corrected_candidates);
                helpers::call_fill_kernel_async(d_editsOffsetsTmp, 1, 0, stream); CUDACHECKASYNC;

                CubCallWrapper(mr).cubInclusiveSum(
                    inputIter2, 
                    d_editsOffsetsTmp + 1, 
                    (*h_num_total_corrected_candidates),
                    stream
                );

                //compact edits
                std::size_t numEditsCandidates = SDIV(editsPitchInBytes * (*h_num_total_corrected_candidates), sizeof(EncodedCorrectionEdit));
                rmm::device_uvector<EncodedCorrectionEdit> d_editsPerCorrectedCandidate2(numEditsCandidates, stream, mr);

                gpucorrectorkernels::compactEditsKernel<<<SDIV((*h_num_total_corrected_candidates), 128), 128, 0, stream>>>(
                    d_editsPerCorrectedCandidate.data(),
                    d_editsPerCorrectedCandidate2.data(),
                    d_editsOffsetsTmp,
                    d_num_total_corrected_candidates.data(),
                    d_numEditsPerCorrectedCandidate.data(),
                    getDoNotUseEditsValue(),
                    editsPitchInBytes
                ); CUDACHECKASYNC;


                //copy compact edits to host
                helpers::call_copy_n_kernel(
                    (const int*)d_editsPerCorrectedCandidate2.data(), 
                    thrust::make_transform_iterator(d_totalNumberOfEdits, [] __device__ (const int num){return SDIV(num * sizeof(EncodedCorrectionEdit), sizeof(int));}),//d_totalNumberOfEdits, 
                    (int*)currentOutput->h_editsPerCorrectedCandidate.data(), 
                    numEditsCandidates, 
                    stream
                );

                helpers::call_copy_n_kernel(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        d_alignment_shifts2.data(), 
                        d_candidate_read_ids2.data(),
                        d_candidate_sequences_lengths2.data(),
                        d_indices_of_corrected_candidates.data(),
                        d_editsOffsetsTmp,
                        d_numEditsPerCorrectedCandidate.data()
                    )), 
                    d_num_total_corrected_candidates.data(), 
                    thrust::make_zip_iterator(thrust::make_tuple(
                        currentOutput->h_alignment_shifts.data(), 
                        currentOutput->h_candidate_read_ids.data(),
                        currentOutput->h_candidate_sequences_lengths.data(),
                        currentOutput->h_indices_of_corrected_candidates.data(),
                        currentOutput->h_candidateEditOffsets.data(),
                        currentOutput->h_numEditsPerCorrectedCandidate.data()
                    )), 
                    (*h_num_total_corrected_candidates),
                    stream
                );

                //compact corrected candidate sequences
                auto correctedCandidatesPitches = thrust::make_transform_iterator(
                    d_numEditsPerCorrectedCandidate.data(),
                    ReplaceNumberOp(getDoNotUseEditsValue(), decodedSequencePitchInBytes)
                );

                int* const d_correctedCandidatesOffsetsTmp = d_indices.data();
                int* const d_totalCorrectedSequencesBytes = d_correctedCandidatesOffsetsTmp + (*h_num_total_corrected_candidates);
                helpers::call_fill_kernel_async(d_correctedCandidatesOffsetsTmp, 1, 0, stream); CUDACHECKASYNC;

                CubCallWrapper(mr).cubInclusiveSum(
                    correctedCandidatesPitches, 
                    d_correctedCandidatesOffsetsTmp + 1, 
                    (*h_num_total_corrected_candidates),
                    stream
                );

                helpers::call_copy_n_kernel(
                    d_correctedCandidatesOffsetsTmp, 
                    d_num_total_corrected_candidates.data(), 
                    currentOutput->h_correctedCandidatesOffsets.data(), 
                    (*h_num_total_corrected_candidates),
                    stream
                );

                rmm::device_uvector<char> d_corrected_candidates2(decodedSequencePitchInBytes * (*h_num_total_corrected_candidates), stream, mr);

                gpucorrectorkernels::compactCorrectedSequencesKernel<32><<<SDIV((*h_num_total_corrected_candidates), 128), 128, 0, stream>>>(
                    d_corrected_candidates.data(),
                    d_corrected_candidates2.data(),
                    this->decodedSequencePitchInBytes,
                    d_num_total_corrected_candidates.data(),
                    d_numEditsPerCorrectedCandidate.data(),
                    getDoNotUseEditsValue(),
                    d_correctedCandidatesOffsetsTmp,
                    thrust::make_counting_iterator(0)
                ); CUDACHECKASYNC;

                assert(decodedSequencePitchInBytes % sizeof(int) == 0);

                helpers::call_copy_n_kernel(
                    (const int*)d_corrected_candidates2.data(), 
                    thrust::make_transform_iterator(d_totalCorrectedSequencesBytes, [] __device__ (const int num){return SDIV(num, sizeof(int));}),
                    (int*)currentOutput->h_corrected_candidates.data(), 
                    (*h_num_total_corrected_candidates) * decodedSequencePitchInBytes / sizeof(int),
                    stream
                );
            }

        }

        void copyCandidateResultsFromDeviceToHostForestGpu(cudaStream_t stream){
            copyCandidateResultsFromDeviceToHostClassic(stream);
        }

        void getAmbiguousFlagsOfAnchors(cudaStream_t stream){

            gpuReadStorage->areSequencesAmbiguous(
                readstorageHandle,
                d_anchorContainsN.data(), 
                d_anchorReadIds.data(), 
                currentNumAnchors,
                stream
            );
        }

        void getAmbiguousFlagsOfCandidates(cudaStream_t stream){
            gpuReadStorage->areSequencesAmbiguous(
                readstorageHandle,
                d_candidateContainsN.data(), 
                d_candidate_read_ids.data(), 
                currentNumCandidates,
                stream
            ); 
        }

        //void getCandidateSequenceData(cudaStream_t stream){

            // gpuReadStorage->gatherSequenceLengths(
            //     readstorageHandle,
            //     d_candidate_sequences_lengths.data(),
            //     d_candidate_read_ids.data(),
            //     currentNumCandidates,            
            //     stream
            // );

            // gpuReadStorage->gatherSequences(
            //     readstorageHandle,
            //     d_candidate_sequences_data.data(),
            //     encodedSequencePitchInInts,
            //     makeAsyncConstBufferWrapper(currentInput->h_candidate_read_ids.data()),
            //     d_candidate_read_ids.data(),
            //     currentNumCandidates,
            //     stream,
            //     mr
            // );

            // helpers::call_transpose_kernel(
            //     d_transposedCandidateSequencesData.data(), 
            //     d_candidate_sequences_data.data(), 
            //     currentNumCandidates, 
            //     encodedSequencePitchInInts, 
            //     encodedSequencePitchInInts, 
            //     stream
            // );
        //}

//         void getQualities(cudaStream_t stream){

//             if(programOptions->useQualityScores) {

// //#define COMPACT_GATHER

// #ifndef COMPACT_GATHER

//                 gpuReadStorage->gatherQualities(
//                     readstorageHandle,
//                     d_anchor_qualities.data(),
//                     qualityPitchInBytes,
//                     makeAsyncConstBufferWrapper(currentInput->h_anchorReadIds.data()),
//                     d_anchorReadIds.data(),
//                     maxAnchors,
//                     stream,
//                     mr
//                 );

//                 gpuReadStorage->gatherQualities(
//                     readstorageHandle,
//                     d_candidate_qualities.data(),
//                     qualityPitchInBytes,
//                     makeAsyncConstBufferWrapper(currentInput->h_candidate_read_ids.data()),
//                     d_candidate_read_ids.data(),
//                     currentNumCandidates,
//                     stream,
//                     mr
//                 );

// #else 

//                 CubCallWrapper(mr).cubExclusiveSum(
//                     d_indices_per_anchor.data(),
//                     d_indices_per_anchor_prefixsum.data(),
//                     maxAnchors,
//                     stream
//                 );
                
//                 //from the list of remaining candidates per anchor, compact the corresponding candidate read ids
//                 helpers::lambda_kernel<<<maxAnchors, 128, 0, stream>>>(
//                     [
//                         h_indicesForGather = h_indicesForGather.data(),
//                         d_indicesForGather = d_indicesForGather.data(),
//                         d_indices = d_indices.data(),
//                         d_indices_per_anchor = d_indices_per_anchor.data(),
//                         d_indices_per_anchor_prefixsum = d_indices_per_anchor_prefixsum.data(),
//                         d_num_indices = d_num_indices.data(),
//                         d_candidates_per_anchor_prefixsum = d_candidates_per_anchor_prefixsum.data(),
//                         d_candidate_read_ids = d_candidate_read_ids.data(),
//                         currentNumAnchors = currentNumAnchors
//                     ] __device__ (){

//                         for(int anchor = blockIdx.x; anchor < currentNumAnchors; anchor += gridDim.x){

//                             const int globalCandidateOffset = d_candidates_per_anchor_prefixsum[anchor];
//                             const int* const myIndices = d_indices + globalCandidateOffset;
//                             const int numIndices = d_indices_per_anchor[anchor];
//                             const int offset = d_indices_per_anchor_prefixsum[anchor];

//                             for(int i = threadIdx.x; i < numIndices; i += blockDim.x){
//                                 const int inputpos = myIndices[i];
//                                 d_indicesForGather[offset + i] = d_candidate_read_ids[globalCandidateOffset + inputpos];
//                                 h_indicesForGather[offset + i] = d_candidate_read_ids[globalCandidateOffset + inputpos];
//                             }
//                         }                   
//                     }
//                 ); CUDACHECKASYNC;

//                 CUDACHECK(cudaEventRecord(events[1], stream));

//                 gpuReadStorage->gatherQualities(
//                     readstorageHandle,
//                     d_anchor_qualities,
//                     qualityPitchInBytes,
//                     currentInput->h_anchorReadIds,
//                     d_anchorReadIds,
//                     maxAnchors,
//                     stream
//                 );

//                 //CUDACHECK(cudaStreamSynchronize(stream)); //wait for h_indicesForGather and h_numRemainingCandidatesAfterAlignment
//                 CUDACHECK(cudaEventSynchronize(events[1]));
//                 const int hNumIndices = *h_numRemainingCandidatesAfterAlignment;

//                 nvtx::push_range("get compact qscores " + std::to_string(hNumIndices) + " " + std::to_string(currentNumCandidates), 6);
//                 gpuReadStorage->gatherQualities(
//                     readstorageHandle,
//                     d_candidate_qualities_compact,
//                     qualityPitchInBytes,
//                     h_indicesForGather.data(),
//                     d_indicesForGather.data(),
//                     currentNumCandidates,
//                     stream
//                 );
//                 nvtx::pop_range();

//                 //scatter compact quality scores to correct positions
//                 helpers::lambda_kernel<<<maxAnchors, 256, 0, stream>>>(
//                     [
//                         d_candidate_qualities_compact = d_candidate_qualities_compact.data(),
//                         d_candidate_qualities = d_candidate_qualities.data(),
//                         d_candidate_sequences_lengths = d_candidate_sequences_lengths.data(),
//                         qualityPitchInBytes = qualityPitchInBytes,
//                         d_indices = d_indices.data(),
//                         d_indices_per_anchor = d_indices_per_anchor.data(),
//                         d_indices_per_anchor_prefixsum = d_indices_per_anchor_prefixsum.data(),
//                         d_num_indices = d_num_indices.data(),
//                         d_candidates_per_anchor_prefixsum = d_candidates_per_anchor_prefixsum.data(),
//                         currentNumAnchors = currentNumAnchors
//                     ] __device__ (){
//                         constexpr int groupsize = 32;
//                         auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

//                         const int groupId = threadIdx.x / groupsize;
//                         const int numgroups = blockDim.x / groupsize;

//                         assert(qualityPitchInBytes % sizeof(int) == 0);

//                         for(int anchor = blockIdx.x; anchor < currentNumAnchors; anchor += gridDim.x){

//                             const int globalCandidateOffset = d_candidates_per_anchor_prefixsum[anchor];
//                             const int* const myIndices = d_indices + globalCandidateOffset;
//                             const int numIndices = d_indices_per_anchor[anchor];
//                             const int offset = d_indices_per_anchor_prefixsum[anchor];

//                             for(int c = groupId; c < numIndices; c += numgroups){
//                                 const int outputpos = globalCandidateOffset + myIndices[c];
//                                 const int inputpos = offset + c;
//                                 const int length = d_candidate_sequences_lengths[outputpos];

//                                 const int iters = SDIV(length, sizeof(int));

//                                 const int* const input = (const int*)(d_candidate_qualities_compact + size_t(inputpos) * qualityPitchInBytes);
//                                 int* const output = (int*)(d_candidate_qualities + size_t(outputpos) * qualityPitchInBytes);

//                                 for(int k = group.thread_rank(); k < iters; k += group.size()){
//                                     output[k] = input[k];
//                                 }
//                             }
//                         }
//                     }
//                 ); CUDACHECKASYNC;

//                 // CUDACHECK(cudaStreamSynchronize(stream)); //wait for candidateQualitiesGatherHandle

//                 // // std::cerr << "gather candidate qual\n";
//                 // gpuReadStorage->gatherQualitiesToGpuBufferAsync(
//                 //     threadPool,
//                 //     candidateQualitiesGatherHandle,
//                 //     d_candidate_qualities,
//                 //     qualityPitchInBytes,
//                 //     currentInput->h_candidate_read_ids.data(),
//                 //     d_candidate_read_ids.data(),
//                 //     currentNumCandidates,
//                 //     deviceId,
//                 //     stream
//                 // );
// #undef COMPACT_GATHER                
// #endif                

//             }
//         }

        void getCandidateAlignments(cudaStream_t stream){


            const bool removeAmbiguousAnchors = programOptions->excludeAmbiguousReads;
            const bool removeAmbiguousCandidates = programOptions->excludeAmbiguousReads;
            
            rmm::device_uvector<bool> d_alignment_isValid(maxCandidates, stream, mr);

            std::size_t bytes = 0;

            call_popcount_shifted_hamming_distance_kernel_async(
                nullptr,
                bytes,
                d_alignment_overlaps.data(),
                d_alignment_shifts.data(),
                d_alignment_nOps.data(),
                d_alignment_isValid.data(),
                d_alignment_best_alignment_flags.data(),
                d_anchor_sequences_data.data(),
                d_candidate_sequences_data.data(),
                d_anchor_sequences_lengths.data(),
                d_candidate_sequences_lengths.data(),
                d_candidates_per_anchor_prefixsum.data(),
                d_candidates_per_anchor.data(),
                d_anchorIndicesOfCandidates.data(),
                d_numAnchors.data(),
                d_numCandidates.data(),
                d_anchorContainsN.data(),
                removeAmbiguousAnchors,
                d_candidateContainsN.data(),
                removeAmbiguousCandidates,
                maxAnchors,
                maxCandidates,
                gpuReadStorage->getSequenceLengthUpperBound(),
                encodedSequencePitchInInts,
                programOptions->min_overlap,
                programOptions->maxErrorRate,
                programOptions->min_overlap_ratio,
                programOptions->estimatedErrorrate,
                stream
            );

            rmm::device_uvector<char> d_temp(bytes, stream, mr);

            call_popcount_shifted_hamming_distance_kernel_async(
                d_temp.data(),
                bytes,
                d_alignment_overlaps.data(),
                d_alignment_shifts.data(),
                d_alignment_nOps.data(),
                d_alignment_isValid.data(),
                d_alignment_best_alignment_flags.data(),
                d_anchor_sequences_data.data(),
                d_candidate_sequences_data.data(),
                d_anchor_sequences_lengths.data(),
                d_candidate_sequences_lengths.data(),
                d_candidates_per_anchor_prefixsum.data(),
                d_candidates_per_anchor.data(),
                d_anchorIndicesOfCandidates.data(),
                d_numAnchors.data(),
                d_numCandidates.data(),
                d_anchorContainsN.data(),
                removeAmbiguousAnchors,
                d_candidateContainsN.data(),
                removeAmbiguousCandidates,
                maxAnchors,
                maxCandidates,
                gpuReadStorage->getSequenceLengthUpperBound(),
                encodedSequencePitchInInts,
                programOptions->min_overlap,
                programOptions->maxErrorRate,
                programOptions->min_overlap_ratio,
                programOptions->estimatedErrorrate,
                stream
            );

            #if 1
            if(!gpuReadStorage->isPairedEnd()){
                //default kernel
                call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                    d_alignment_best_alignment_flags.data(),
                    d_alignment_nOps.data(),
                    d_alignment_overlaps.data(),
                    d_candidates_per_anchor_prefixsum.data(),
                    d_numAnchors.data(),
                    d_numCandidates.data(),
                    maxAnchors,
                    maxCandidates,
                    programOptions->estimatedErrorrate,
                    programOptions->estimatedCoverage * programOptions->m_coverage,
                    stream
                );
            }else{
                helpers::lambda_kernel<<<currentNumAnchors, 128, 0, stream>>>(
                    [
                        bestAlignmentFlags = d_alignment_best_alignment_flags.data(),
                        nOps = d_alignment_nOps.data(),
                        overlaps = d_alignment_overlaps.data(),
                        d_candidates_per_anchor_prefixsum = d_candidates_per_anchor_prefixsum.data(),
                        n_anchors = currentNumAnchors,
                        mismatchratioBaseFactor = programOptions->estimatedErrorrate,
                        goodAlignmentsCountThreshold = programOptions->estimatedCoverage * programOptions->m_coverage,
                        d_isPairedCandidate = d_isPairedCandidate.data(),
                        pairedFilterThreshold = programOptions->pairedFilterThreshold
                    ] __device__(){
                        using BlockReduceInt = cub::BlockReduce<int, 128>;

                        // __shared__ union {
                        //     typename BlockReduceInt::TempStorage intreduce;
                        //     int broadcast[3];
                        // } temp_storage;

                        for(int anchorindex = blockIdx.x; anchorindex < n_anchors; anchorindex += gridDim.x) {

                            const int candidatesForAnchor = d_candidates_per_anchor_prefixsum[anchorindex+1]
                                                            - d_candidates_per_anchor_prefixsum[anchorindex];

                            const int firstIndex = d_candidates_per_anchor_prefixsum[anchorindex];

                            //printf("anchorindex %d\n", anchorindex);

                            //int counts[3]{0,0,0};

                            //if(threadIdx.x == 0){
                            //    printf("my_n_indices %d\n", my_n_indices);
                            //}

                            for(int index = threadIdx.x; index < candidatesForAnchor; index += blockDim.x) {

                                const int candidate_index = firstIndex + index;
                                if(!d_isPairedCandidate[candidate_index]){
                                    if(bestAlignmentFlags[candidate_index] != AlignmentOrientation::None) {

                                        const int alignment_overlap = overlaps[candidate_index];
                                        const int alignment_nops = nOps[candidate_index];

                                        const float mismatchratio = float(alignment_nops) / alignment_overlap;

                                        //if(mismatchratio >= 1 * mismatchratioBaseFactor) {
                                        if(mismatchratio >= pairedFilterThreshold) {
                                            bestAlignmentFlags[candidate_index] = AlignmentOrientation::None;
                                        }else{

                                            // #pragma unroll
                                            // for(int i = 2; i <= 2; i++) {
                                            //     counts[i-2] += (mismatchratio < i * mismatchratioBaseFactor);
                                            // }
                                        }

                                    }
                                }
                            }

                            // //accumulate counts over block
                            //     #pragma unroll
                            // for(int i = 0; i < 3; i++) {
                            //     counts[i] = BlockReduceInt(temp_storage.intreduce).Sum(counts[i]);
                            //     __syncthreads();
                            // }

                            // //broadcast accumulated counts to block
                            // if(threadIdx.x == 0) {
                            //     #pragma unroll
                            //     for(int i = 0; i < 3; i++) {
                            //         temp_storage.broadcast[i] = counts[i];
                            //         //printf("count[%d] = %d\n", i, counts[i]);
                            //     }
                            //     //printf("mismatchratioBaseFactor %f, goodAlignmentsCountThreshold %f\n", mismatchratioBaseFactor, goodAlignmentsCountThreshold);
                            // }

                            // __syncthreads();

                            // #pragma unroll
                            // for(int i = 0; i < 3; i++) {
                            //     counts[i] = temp_storage.broadcast[i];
                            // }

                            // float mismatchratioThreshold = 0;
                            // if (counts[0] >= goodAlignmentsCountThreshold) {
                            //     mismatchratioThreshold = 2 * mismatchratioBaseFactor;
                            // } else if (counts[1] >= goodAlignmentsCountThreshold) {
                            //     mismatchratioThreshold = 3 * mismatchratioBaseFactor;
                            // } else if (counts[2] >= goodAlignmentsCountThreshold) {
                            //     mismatchratioThreshold = 4 * mismatchratioBaseFactor;
                            // } else {
                            //     mismatchratioThreshold = -1.0f;                         //this will invalidate all alignments for anchor
                            //     //mismatchratioThreshold = 4 * mismatchratioBaseFactor; //use alignments from every bin
                            //     //mismatchratioThreshold = 1.1f;
                            // }

                            // // Invalidate all alignments for anchor with mismatchratio >= mismatchratioThreshold which are not paired end
                            // for(int index = threadIdx.x; index < candidatesForAnchor; index += blockDim.x) {
                            //     const int candidate_index = firstIndex + index;

                            //     if(!d_isPairedCandidate[candidate_index]){
                            //         if(bestAlignmentFlags[candidate_index] != AlignmentOrientation::None) {

                            //             const int alignment_overlap = overlaps[candidate_index];
                            //             const int alignment_nops = nOps[candidate_index];

                            //             const float mismatchratio = float(alignment_nops) / alignment_overlap;

                            //             const bool doRemove = mismatchratio >= mismatchratioThreshold;
                            //             if(doRemove){
                            //                 bestAlignmentFlags[candidate_index] = AlignmentOrientation::None;
                            //             }
                            //         }
                            //     }
                            // }
                        }
                    }
                );
            }
            #else
                //default kernel
                call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                    d_alignment_best_alignment_flags.data(),
                    d_alignment_nOps.data(),
                    d_alignment_overlaps.data(),
                    d_candidates_per_anchor_prefixsum.data(),
                    d_numAnchors.data(),
                    d_numCandidates.data(),
                    maxAnchors,
                    maxCandidates,
                    programOptions->estimatedErrorrate,
                    programOptions->estimatedCoverage * programOptions->m_coverage,
                    stream
                );
            #endif

            callSelectIndicesOfGoodCandidatesKernelAsync(
                d_indices.data(),
                d_indices_per_anchor.data(),
                d_num_indices.data(),
                d_alignment_best_alignment_flags.data(),
                d_candidates_per_anchor.data(),
                d_candidates_per_anchor_prefixsum.data(),
                d_anchorIndicesOfCandidates.data(),
                d_numAnchors.data(),
                d_numCandidates.data(),
                maxAnchors,
                maxCandidates,
                stream
            );

            CUDACHECK(cudaMemcpyAsync(
                h_numRemainingCandidatesAfterAlignment.data(),
                d_num_indices.data(),
                sizeof(int),
                D2H,
                stream
            ));

            CUDACHECK(cudaEventRecord(events[1], stream));

        }

        void buildAndRefineMultipleSequenceAlignment(cudaStream_t stream){

            if(programOptions->useQualityScores){

                gpuReadStorage->gatherQualities(
                    readstorageHandle,
                    d_anchor_qualities.data(),
                    qualityPitchInBytes,
                    makeAsyncConstBufferWrapper(currentInput->h_anchorReadIds.data()),
                    d_anchorReadIds.data(),
                    currentNumAnchors,
                    stream,
                    mr
                );

                gpuReadStorage->gatherQualities(
                    readstorageHandle,
                    d_candidate_qualities.data(),
                    qualityPitchInBytes,
                    makeAsyncConstBufferWrapper(currentInput->h_candidate_read_ids.data()),
                    d_candidate_read_ids.data(),
                    currentNumCandidates,
                    stream,
                    mr
                );

            }

            managedgpumsa = std::make_unique<ManagedGPUMultiMSA>(stream, mr, h_managedmsa_tmp.data());

            managedgpumsa->construct(
                d_alignment_overlaps.data(),
                d_alignment_shifts.data(),
                d_alignment_nOps.data(),
                d_alignment_best_alignment_flags.data(),
                d_indices.data(),
                d_indices_per_anchor.data(),
                d_candidates_per_anchor_prefixsum.data(),
                d_anchor_sequences_lengths.data(),
                d_anchor_sequences_data.data(),
                d_anchor_qualities.data(),
                currentNumAnchors,
                d_candidate_sequences_lengths.data(),
                d_candidate_sequences_data.data(),
                d_candidate_qualities.data(),
                d_isPairedCandidate.data(),
                maxCandidates,
                d_numAnchors.data(),
                encodedSequencePitchInInts,
                qualityPitchInBytes,
                programOptions->useQualityScores,
                programOptions->maxErrorRate,
                MSAColumnCount{static_cast<int>(msaColumnPitchInElements)},
                stream
            );

            if(useMsaRefinement()){
                
                rmm::device_uvector<int> d_indices_tmp(maxCandidates+1, stream, mr);
                rmm::device_uvector<int> d_indices_per_anchor_tmp(maxAnchors+1, stream, mr);
                rmm::device_uvector<int> d_num_indices_tmp(1, stream, mr);

                managedgpumsa->refine(
                    d_indices_tmp.data(),
                    d_indices_per_anchor_tmp.data(),
                    d_num_indices_tmp.data(),
                    d_alignment_overlaps.data(),
                    d_alignment_shifts.data(),
                    d_alignment_nOps.data(),
                    d_alignment_best_alignment_flags.data(),
                    d_indices.data(),
                    d_indices_per_anchor.data(),
                    d_candidates_per_anchor_prefixsum.data(),
                    d_anchor_sequences_lengths.data(),
                    d_anchor_sequences_data.data(),
                    d_anchor_qualities.data(),
                    currentNumAnchors,
                    d_candidate_sequences_lengths.data(),
                    d_candidate_sequences_data.data(),
                    d_candidate_qualities.data(),
                    d_isPairedCandidate.data(),
                    maxCandidates,
                    d_numAnchors.data(),
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    programOptions->useQualityScores,
                    programOptions->maxErrorRate,
                    programOptions->estimatedCoverage,
                    getNumRefinementIterations(),
                    stream,
                    d_anchorReadIds.data()
                );

                std::swap(d_indices_tmp, d_indices);
                std::swap(d_indices_per_anchor_tmp, d_indices_per_anchor);
                std::swap(d_num_indices_tmp, d_num_indices);

            }

            //CUDACHECK(cudaStreamSynchronize(stream)); //debug
        }


        void correctAnchors(cudaStream_t stream){
            if(programOptions->correctionType == CorrectionType::Classic){
                correctAnchorsClassic(stream);
            }else if(programOptions->correctionType == CorrectionType::Forest){
                correctAnchorsForestGpu(stream);
            }else{
                throw std::runtime_error("correctAnchors not implemented for this correctionType");
            }
        }

        void correctAnchorsClassic(cudaStream_t stream){

            const float avg_support_threshold = 1.0f - 1.0f * programOptions->estimatedErrorrate;
            const float min_support_threshold = 1.0f - 3.0f * programOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                programOptions->m_coverage / 6.0f * programOptions->estimatedCoverage);
            //const float max_coverage_threshold = 0.5 * programOptions->estimatedCoverage;

            // correct anchors

            call_msaCorrectAnchorsKernel_async(
                d_corrected_anchors.data(),
                d_anchor_is_corrected.data(),
                d_is_high_quality_anchor.data(),
                managedgpumsa->multiMSAView(),
                d_anchor_sequences_data.data(),
                d_indices_per_anchor.data(),
                d_numAnchors.data(),
                maxAnchors,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                programOptions->estimatedErrorrate,
                avg_support_threshold,
                min_support_threshold,
                min_coverage_threshold,
                gpuReadStorage->getSequenceLengthUpperBound(),
                stream
            );

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_indices_of_corrected_anchors.data(),
                d_num_indices_of_corrected_anchors.data(),
                d_anchor_is_corrected.data(),
                d_numAnchors.data()
            ); CUDACHECKASYNC;

            helpers::call_fill_kernel_async(d_numEditsPerCorrectedanchor.data(), currentNumAnchors, 0, stream);

            callConstructSequenceCorrectionResultsKernel(
                d_editsPerCorrectedanchor.data(),
                d_numEditsPerCorrectedanchor.data(),
                getDoNotUseEditsValue(),
                d_indices_of_corrected_anchors.data(),
                d_num_indices_of_corrected_anchors.data(),
                d_anchorContainsN.data(),
                d_anchor_sequences_data.data(),
                d_anchor_sequences_lengths.data(),
                d_corrected_anchors.data(),
                currentNumAnchors,
                false,
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,      
                stream
            );
            
        }

        void correctAnchorsForestGpu(cudaStream_t stream){

            const float avg_support_threshold = 1.0f - 1.0f * programOptions->estimatedErrorrate;
            const float min_support_threshold = 1.0f - 3.0f * programOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                programOptions->m_coverage / 6.0f * programOptions->estimatedCoverage);
            const float max_coverage_threshold = 0.5 * programOptions->estimatedCoverage;

            // correct anchors

            callMsaCorrectAnchorsWithForestKernel(
                d_corrected_anchors.data(),
                d_anchor_is_corrected.data(),
                d_is_high_quality_anchor.data(),
                managedgpumsa->multiMSAView(),
                *gpuForestAnchor,
                programOptions->thresholdAnchor,
                d_anchor_sequences_data.data(),
                d_indices_per_anchor.data(),
                currentNumAnchors,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                gpuReadStorage->getSequenceLengthUpperBound(),
                programOptions->estimatedErrorrate,
                programOptions->estimatedCoverage,
                avg_support_threshold,
                min_support_threshold,
                min_coverage_threshold,
                stream
            );

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_indices_of_corrected_anchors.data(),
                d_num_indices_of_corrected_anchors.data(),
                d_anchor_is_corrected.data(),
                d_numAnchors.data()
            ); CUDACHECKASYNC;

            helpers::call_fill_kernel_async(d_numEditsPerCorrectedanchor.data(), currentNumAnchors, 0, stream);

            callConstructSequenceCorrectionResultsKernel(
                d_editsPerCorrectedanchor.data(),
                d_numEditsPerCorrectedanchor.data(),
                getDoNotUseEditsValue(),
                d_indices_of_corrected_anchors.data(),
                d_num_indices_of_corrected_anchors.data(),
                d_anchorContainsN.data(),
                d_anchor_sequences_data.data(),
                d_anchor_sequences_lengths.data(),
                d_corrected_anchors.data(),
                currentNumAnchors,
                false,
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,      
                stream
            );

        }
        
        void correctCandidates(cudaStream_t stream){
            if(programOptions->correctionTypeCands == CorrectionType::Classic){
                correctCandidatesClassic(stream);
            }else if(programOptions->correctionTypeCands == CorrectionType::Forest){
                correctCandidatesForestGpu(stream);
            }else{
                throw std::runtime_error("correctCandidates not implemented for this correctionTypeCands");
            }
        }

        void correctCandidatesClassic(cudaStream_t stream){

            const float min_support_threshold = 1.0f-3.0f*programOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                programOptions->m_coverage / 6.0f * programOptions->estimatedCoverage);
            const int new_columns_to_correct = programOptions->new_columns_to_correct;

            rmm::device_uvector<bool> d_candidateCanBeCorrected(maxCandidates, stream, mr);

            cub::TransformInputIterator<bool, IsHqAnchor, AnchorHighQualityFlag*>
                d_isHqanchor(d_is_high_quality_anchor.data(), IsHqAnchor{});

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_high_quality_anchor_indices.data(),
                d_num_high_quality_anchor_indices.data(),
                d_isHqanchor,
                d_numAnchors.data()
            ); CUDACHECKASYNC;

            gpucorrectorkernels::initArraysBeforeCandidateCorrectionKernel<<<SDIV(maxCandidates, 128), 128, 0, stream>>>(
                maxCandidates,
                d_numAnchors.data(),
                d_num_corrected_candidates_per_anchor.data(),
                d_candidateCanBeCorrected.data()
            ); CUDACHECKASYNC;

            #if 0
                rmm::device_uvector<bool> d_flagsCandidates(currentNumCandidates, stream, mr);
                bool* d_excludeFlags = d_flagsCandidates.data();
                bool* h_excludeFlags = h_flagsCandidates.data();

                //corrections of candidates for which a high quality anchor correction exists will not be used
                //-> don't compute them
                for(int i = 0; i < currentNumCandidates; i++){
                    const read_number candidateReadId = currentInput->h_candidate_read_ids[i];
                    h_excludeFlags[i] = correctionFlags->isCorrectedAsHQAnchor(candidateReadId);
                }

                helpers::call_copy_n_kernel(
                    (const int*)h_excludeFlags,
                    SDIV(currentNumCandidates, sizeof(int)),
                    (int*)d_excludeFlags,
                    stream
                );

                callFlagCandidatesToBeCorrectedWithExcludeFlagsKernel(
                    d_candidateCanBeCorrected,
                    d_num_corrected_candidates_per_anchor.data(),
                    managedgpumsa->multiMSAView(),
                    d_excludeFlags,
                    d_alignment_shifts.data(),
                    d_candidate_sequences_lengths.data(),
                    d_anchorIndicesOfCandidates.data(),
                    d_is_high_quality_anchor.data(),
                    d_candidates_per_anchor_prefixsum,
                    d_indices.data(),
                    d_indices_per_anchor.data(),
                    d_numAnchors,
                    d_numCandidates,
                    min_support_threshold,
                    min_coverage_threshold,
                    new_columns_to_correct,
                    stream
                );
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

            CubCallWrapper(mr).cubSelectFlagged(
                cub::CountingInputIterator<int>(0),
                d_candidateCanBeCorrected.data(),
                d_indices_of_corrected_candidates.data(),
                d_num_total_corrected_candidates.data(),
                maxCandidates,
                stream
            );

            //int h_num_total_corrected_candidates = 0;
            CUDACHECK(cudaMemcpyAsync(
                h_num_total_corrected_candidates.data(),
                d_num_total_corrected_candidates.data(),
                sizeof(int),
                D2H,
                stream
            ));
            CUDACHECK(cudaStreamSynchronize(stream));

            d_corrected_candidates.resize(decodedSequencePitchInBytes * (*h_num_total_corrected_candidates), stream);
            d_numEditsPerCorrectedCandidate.resize((*h_num_total_corrected_candidates), stream);
            std::size_t numEditsCandidates = SDIV(editsPitchInBytes * (*h_num_total_corrected_candidates), sizeof(EncodedCorrectionEdit));
            d_editsPerCorrectedCandidate.resize(numEditsCandidates, stream);

            currentOutput->h_candidate_sequences_lengths.resize((*h_num_total_corrected_candidates));
            currentOutput->h_corrected_candidates.resize((*h_num_total_corrected_candidates) * decodedSequencePitchInBytes);
            currentOutput->h_editsPerCorrectedCandidate.resize(numEditsCandidates);
            currentOutput->h_numEditsPerCorrectedCandidate.resize((*h_num_total_corrected_candidates));
            currentOutput->h_candidateEditOffsets.resize((*h_num_total_corrected_candidates));
            currentOutput->h_indices_of_corrected_candidates.resize((*h_num_total_corrected_candidates));
            currentOutput->h_alignment_shifts.resize((*h_num_total_corrected_candidates));
            currentOutput->h_candidate_read_ids.resize((*h_num_total_corrected_candidates));
            currentOutput->h_correctedCandidatesOffsets.resize((*h_num_total_corrected_candidates) * decodedSequencePitchInBytes);
            
            CUDACHECK(cudaMemsetAsync(
                d_numEditsPerCorrectedCandidate.data(),
                0,
                sizeof(int) * (*h_num_total_corrected_candidates),
                stream
            ));

            #if 0

            callCorrectCandidatesAndComputeEditsKernel(
                d_corrected_candidates.data(),
                d_editsPerCorrectedCandidate.data(),
                d_numEditsPerCorrectedCandidate.data(),              
                managedgpumsa->multiMSAView(),
                d_alignment_shifts.data(),
                d_alignment_best_alignment_flags.data(),
                d_candidate_sequences_data.data(),
                d_candidate_sequences_lengths.data(),
                d_candidateContainsN.data(),
                d_indices_of_corrected_candidates.data(),
                d_num_total_corrected_candidates.data(),
                d_anchorIndicesOfCandidates.data(),
                d_numAnchors,
                d_numCandidates,
                getDoNotUseEditsValue(),
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,
                gpuReadStorage->getSequenceLengthUpperBound(),
                stream
            );
            #else




            callCorrectCandidatesKernel(
                d_corrected_candidates.data(),            
                managedgpumsa->multiMSAView(),
                d_alignment_shifts.data(),
                d_alignment_best_alignment_flags.data(),
                d_candidate_sequences_data.data(),
                d_candidate_sequences_lengths.data(),
                d_candidateContainsN.data(),
                d_indices_of_corrected_candidates.data(),
                d_num_total_corrected_candidates.data(),
                d_anchorIndicesOfCandidates.data(),
                d_numAnchors.data(),
                d_numCandidates.data(),                
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                gpuReadStorage->getSequenceLengthUpperBound(),
                stream
            );            

            callConstructSequenceCorrectionResultsKernel(
                d_editsPerCorrectedCandidate.data(),
                d_numEditsPerCorrectedCandidate.data(),
                getDoNotUseEditsValue(),
                d_indices_of_corrected_candidates.data(),
                d_num_total_corrected_candidates.data(),
                d_candidateContainsN.data(),
                d_candidate_sequences_data.data(),
                d_candidate_sequences_lengths.data(),
                d_corrected_candidates.data(),
                (*h_num_total_corrected_candidates),
                true,
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,      
                stream
            );

            #endif
  
        }

        void correctCandidatesForestGpu(cudaStream_t stream){

            const float min_support_threshold = 1.0f-3.0f*programOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                programOptions->m_coverage / 6.0f * programOptions->estimatedCoverage);
            const int new_columns_to_correct = programOptions->new_columns_to_correct;

            rmm::device_uvector<bool> d_candidateCanBeCorrected(maxCandidates, stream, mr);

            cub::TransformInputIterator<bool, IsHqAnchor, AnchorHighQualityFlag*>
                d_isHqanchor(d_is_high_quality_anchor.data(), IsHqAnchor{});

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_high_quality_anchor_indices.data(),
                d_num_high_quality_anchor_indices.data(),
                d_isHqanchor,
                d_numAnchors.data()
            ); CUDACHECKASYNC;

            gpucorrectorkernels::initArraysBeforeCandidateCorrectionKernel<<<SDIV(maxCandidates, 128), 128, 0, stream>>>(
                maxCandidates,
                d_numAnchors.data(),
                d_num_corrected_candidates_per_anchor.data(),
                d_candidateCanBeCorrected.data()
            ); CUDACHECKASYNC;

   
            #if 1
                rmm::device_uvector<bool> d_flagsCandidates(currentNumCandidates, stream, mr);
                bool* d_excludeFlags = d_flagsCandidates.data();
                bool* h_excludeFlags = h_flagsCandidates.data();

                //corrections of candidates for which a high quality anchor correction exists will not be used
                //-> don't compute them
                for(int i = 0; i < currentNumCandidates; i++){
                    const read_number candidateReadId = currentInput->h_candidate_read_ids[i];
                    h_excludeFlags[i] = correctionFlags->isCorrectedAsHQAnchor(candidateReadId);
                }

                cudaMemcpyAsync(
                    d_excludeFlags,
                    h_excludeFlags,
                    sizeof(bool) * currentNumCandidates,
                    H2D,
                    stream
                );

                callFlagCandidatesToBeCorrectedWithExcludeFlagsKernel(
                    d_candidateCanBeCorrected.data(),
                    d_num_corrected_candidates_per_anchor.data(),
                    managedgpumsa->multiMSAView(),
                    d_excludeFlags,
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
            #else
            callFlagCandidatesToBeCorrectedKernel_async(
                d_candidateCanBeCorrected.data(),
                d_num_corrected_candidates_per_anchor.data(),
                managedgpumsa->multiMSAView(),
                d_alignment_shifts.data(),
                d_candidate_sequences_lengths.data(),
                d_anchorIndicesOfCandidates.data(),
                d_is_high_quality_anchor.data(),
                d_candidates_per_anchor_prefixsum,
                d_indices.data(),
                d_indices_per_anchor.data(),
                d_numAnchors,
                d_numCandidates,
                min_support_threshold,
                min_coverage_threshold,
                new_columns_to_correct,
                stream
            );
            #endif

            CubCallWrapper(mr).cubSelectFlagged(
                cub::CountingInputIterator<int>(0),
                d_candidateCanBeCorrected.data(),
                d_indices_of_corrected_candidates.data(),
                d_num_total_corrected_candidates.data(),
                maxCandidates,
                stream
            );

            //int h_num_total_corrected_candidates = 0;
            CUDACHECK(cudaMemcpyAsync(
                h_num_total_corrected_candidates.data(),
                d_num_total_corrected_candidates.data(),
                sizeof(int),
                D2H,
                stream
            ));
            CUDACHECK(cudaStreamSynchronize(stream));

            d_corrected_candidates.resize(decodedSequencePitchInBytes * (*h_num_total_corrected_candidates), stream);
            d_numEditsPerCorrectedCandidate.resize((*h_num_total_corrected_candidates), stream);
            std::size_t numEditsCandidates = SDIV(editsPitchInBytes * (*h_num_total_corrected_candidates), sizeof(EncodedCorrectionEdit));
            d_editsPerCorrectedCandidate.resize(numEditsCandidates, stream);

            currentOutput->h_candidate_sequences_lengths.resize((*h_num_total_corrected_candidates));
            currentOutput->h_corrected_candidates.resize((*h_num_total_corrected_candidates) * decodedSequencePitchInBytes);
            currentOutput->h_editsPerCorrectedCandidate.resize(numEditsCandidates);
            currentOutput->h_numEditsPerCorrectedCandidate.resize((*h_num_total_corrected_candidates));
            currentOutput->h_candidateEditOffsets.resize((*h_num_total_corrected_candidates));
            currentOutput->h_indices_of_corrected_candidates.resize((*h_num_total_corrected_candidates));
            currentOutput->h_alignment_shifts.resize((*h_num_total_corrected_candidates));
            currentOutput->h_candidate_read_ids.resize((*h_num_total_corrected_candidates));
            currentOutput->h_correctedCandidatesOffsets.resize((*h_num_total_corrected_candidates) * decodedSequencePitchInBytes);
            
            CUDACHECK(cudaMemsetAsync(
                d_numEditsPerCorrectedCandidate.data(),
                0,
                sizeof(int) * (*h_num_total_corrected_candidates),
                stream
            ));

            callMsaCorrectCandidatesWithForestKernel(
                d_corrected_candidates.data(),            
                managedgpumsa->multiMSAView(),
                *gpuForestCandidate,
                programOptions->thresholdCands,
                programOptions->estimatedCoverage,
                d_alignment_shifts.data(),
                d_alignment_best_alignment_flags.data(),
                d_candidate_sequences_data.data(),
                d_candidate_sequences_lengths.data(),
                d_indices_of_corrected_candidates.data(),
                d_num_total_corrected_candidates.data(),
                d_anchorIndicesOfCandidates.data(),
                currentNumCandidates, //*h_num_total_corrected_candidates,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                gpuReadStorage->getSequenceLengthUpperBound(),
                stream
            );  

            callConstructSequenceCorrectionResultsKernel(
                d_editsPerCorrectedCandidate.data(),
                d_numEditsPerCorrectedCandidate.data(),
                getDoNotUseEditsValue(),
                d_indices_of_corrected_candidates.data(),
                d_num_total_corrected_candidates.data(),
                d_candidateContainsN.data(),
                d_candidate_sequences_data.data(),
                d_candidate_sequences_lengths.data(),
                d_corrected_candidates.data(),
                (*h_num_total_corrected_candidates),
                true,
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,      
                stream
            );
            
        }

        static constexpr int getDoNotUseEditsValue() noexcept{
            return -1;
        }

    private:

        int deviceId;
        std::array<CudaEvent, 2> events;

        CudaEvent previousBatchFinishedEvent;

        std::size_t msaColumnPitchInElements;
        std::size_t encodedSequencePitchInInts;
        std::size_t decodedSequencePitchInBytes;
        std::size_t qualityPitchInBytes;
        std::size_t editsPitchInBytes;

        int maxAnchors;
        int maxCandidates;
        int maxNumEditsPerSequence;
        int currentNumAnchors;
        int currentNumCandidates;

        std::map<int, int> numCandidatesPerReadMap{};

        const ReadCorrectionFlags* correctionFlags;

        const GpuReadStorage* gpuReadStorage;

        const ProgramOptions* programOptions;

        GpuErrorCorrectorInput* currentInput;
        GpuErrorCorrectorRawOutput* currentOutput;
        GpuErrorCorrectorRawOutput currentOutputData;

        rmm::mr::device_memory_resource* mr;
        ThreadPool* threadPool;
        ThreadPool::ParallelForHandle pforHandle;

        const GpuForest* gpuForestAnchor{};
        const GpuForest* gpuForestCandidate{};

        ReadStorageHandle readstorageHandle;

        PinnedBuffer<int> h_num_total_corrected_candidates;
        PinnedBuffer<int> h_num_indices;
        PinnedBuffer<int> h_numSelected;
        PinnedBuffer<int> h_numRemainingCandidatesAfterAlignment;
        PinnedBuffer<int> h_managedmsa_tmp;

        PinnedBuffer<read_number> h_indicesForGather;
        rmm::device_uvector<read_number> d_indicesForGather;

        //rmm::device_uvector<char> d_candidate_qualities_compact;

        rmm::device_uvector<bool> d_anchorContainsN;
        rmm::device_uvector<bool> d_candidateContainsN;
        rmm::device_uvector<int> d_candidate_sequences_lengths;
        rmm::device_uvector<unsigned int> d_candidate_sequences_data;
        //rmm::device_uvector<unsigned int> d_transposedCandidateSequencesData;
        rmm::device_uvector<char> d_anchor_qualities;
        rmm::device_uvector<char> d_candidate_qualities;
        rmm::device_uvector<int> d_anchorIndicesOfCandidates;
        rmm::device_uvector<int> d_alignment_overlaps;
        rmm::device_uvector<int> d_alignment_shifts;
        rmm::device_uvector<int> d_alignment_nOps;
        rmm::device_uvector<AlignmentOrientation> d_alignment_best_alignment_flags; 
        rmm::device_uvector<int> d_indices;
        rmm::device_uvector<int> d_indices_per_anchor;
        rmm::device_uvector<int> d_indices_per_anchor_prefixsum;
        rmm::device_uvector<int> d_num_indices;
        rmm::device_uvector<char> d_corrected_anchors;
        rmm::device_uvector<char> d_corrected_candidates;
        rmm::device_uvector<int> d_num_corrected_candidates_per_anchor;
        rmm::device_uvector<int> d_num_corrected_candidates_per_anchor_prefixsum;
        rmm::device_uvector<int> d_num_total_corrected_candidates;
        rmm::device_uvector<bool> d_anchor_is_corrected;
        rmm::device_uvector<AnchorHighQualityFlag> d_is_high_quality_anchor;
        rmm::device_uvector<int> d_high_quality_anchor_indices;
        rmm::device_uvector<int> d_num_high_quality_anchor_indices; 
        rmm::device_uvector<EncodedCorrectionEdit> d_editsPerCorrectedanchor;
        rmm::device_uvector<int> d_numEditsPerCorrectedanchor;
        rmm::device_uvector<EncodedCorrectionEdit> d_editsPerCorrectedCandidate;
        
        rmm::device_uvector<int> d_numEditsPerCorrectedCandidate;
        rmm::device_uvector<int> d_indices_of_corrected_anchors;
        rmm::device_uvector<int> d_num_indices_of_corrected_anchors;
        rmm::device_uvector<int> d_indices_of_corrected_candidates;
        rmm::device_uvector<int> d_totalNumEdits;
        rmm::device_uvector<bool> d_isPairedCandidate;
        PinnedBuffer<bool> h_isPairedCandidate;

        rmm::device_uvector<int> d_numAnchors;
        rmm::device_uvector<int> d_numCandidates;
        rmm::device_uvector<read_number> d_anchorReadIds;
        rmm::device_uvector<unsigned int> d_anchor_sequences_data;
        rmm::device_uvector<int> d_anchor_sequences_lengths;
        rmm::device_uvector<read_number> d_candidate_read_ids;
        rmm::device_uvector<int> d_candidates_per_anchor;
        rmm::device_uvector<int> d_candidates_per_anchor_prefixsum; 

        PinnedBuffer<int> h_candidates_per_anchor_prefixsum; 
        PinnedBuffer<int> h_indices;

        PinnedBuffer<bool> h_flagsCandidates;

        std::unique_ptr<ManagedGPUMultiMSA> managedgpumsa;

        
        std::map<GpuErrorCorrectorRawOutput*, CudaGraph> graphMap;
    };



}
}






#endif
