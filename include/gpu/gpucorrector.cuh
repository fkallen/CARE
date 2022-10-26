#ifndef CARE_GPUCORRECTOR_CUH
#define CARE_GPUCORRECTOR_CUH

//#define CANDS_FOREST_FLAGS_DEFAULT 
#define CANDS_FOREST_FLAGS_COMPUTE_AHEAD


#include <hpc_helpers.cuh>
#include <hpc_helpers/include/nvtx_markers.cuh>

#include <gpu/gpuminhasher.cuh>
#include <gpu/kernels.hpp>
#include <gpu/gpucorrectorkernels.cuh>
#include <gpu/gpureadstorage.cuh>
#include <gpu/asyncresult.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/cubwrappers.cuh>
#include <gpu/gpumsamanaged.cuh>
#include <gpu/global_cuda_stream_pool.cuh>
#include <gpu/minhashqueryfilter.cuh>

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

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/exec_policy.hpp>
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
            d_anchor_sequences_lengths(0, cudaStreamPerThread, mr),
            d_candidate_read_ids(0, cudaStreamPerThread, mr),
            d_candidate_sequences_data(0, cudaStreamPerThread, mr),
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

        bool nothingToDo = true;
        int numAnchors = 0;
        int numCorrectedCandidates = 0;
        PinnedBuffer<int> h_numCorrectedAnchors{};
        PinnedBuffer<read_number> h_anchorReadIds{};
        PinnedBuffer<bool> h_anchor_is_corrected{};
        PinnedBuffer<AnchorHighQualityFlag> h_is_high_quality_anchor{};
        PinnedBuffer<std::uint8_t> serializedAnchorResults{};
        PinnedBuffer<std::uint32_t> serializedAnchorOffsets{};
        PinnedBuffer<std::uint8_t> serializedCandidateResults{};
        PinnedBuffer<std::uint32_t> serializedCandidateOffsets{};
        
        CudaEvent event{cudaEventDisableTiming};
        

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            auto handleHost = [&](const auto& h){
                info.host += h.sizeInBytes();
            };

            handleHost(h_numCorrectedAnchors);
            handleHost(h_anchorReadIds);
            handleHost(h_anchor_is_corrected);
            handleHost(h_is_high_quality_anchor);
            handleHost(serializedAnchorResults);
            handleHost(serializedAnchorOffsets);
            handleHost(serializedCandidateResults);
            handleHost(serializedCandidateOffsets);            

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

            CUDACHECK(cudaMemcpyAsync(
                ecinput.d_anchorReadIds.data(),
                ecinput.h_anchorReadIds.data(),
                sizeof(read_number) * (*ecinput.h_numAnchors.data()),
                H2D,
                stream
            ));

            if(numIds > 0){
                nvtx::push_range("getAnchorReads", 0);
                getAnchorReads(ecinput, useQualityScores, stream);
                nvtx::pop_range();

                nvtx::push_range("getCandidateReadIdsWithMinhashing", 1);
                getCandidateReadIdsWithMinhashing(ecinput, stream);
                nvtx::pop_range();

                CUDACHECK(cudaStreamSynchronize(stream));

                getCandidateReads(ecinput, useQualityScores, stream);
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

            gpuReadStorage->gatherContiguousSequences(
                readstorageHandle,
                ecinput.d_anchor_sequences_data.data(),
                encodedSequencePitchInInts,
                ecinput.h_anchorReadIds[0],
                numAnchors,
                stream,
                mr
            );

            gpuReadStorage->gatherSequenceLengths(
                readstorageHandle,
                ecinput.d_anchor_sequences_lengths.data(),
                ecinput.d_anchorReadIds.data(),
                numAnchors,
                stream
            );
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

            ecinput.d_candidate_sequences_lengths.resize(numCandidates, stream);

            gpuReadStorage->gatherSequenceLengths(
                readstorageHandle,
                ecinput.d_candidate_sequences_lengths.data(),
                ecinput.d_candidate_read_ids.data(),
                numCandidates,
                stream
            );
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
                (*ecinput.h_numAnchors.data()),                
                totalNumValues,
                ecinput.d_candidate_read_ids.data(),
                ecinput.d_candidates_per_anchor.data(),
                ecinput.d_candidates_per_anchor_prefixsum.data(),
                stream,
                mr
            );

            rmm::device_uvector<read_number> d_candidate_read_ids2(totalNumValues, stream, mr);
            rmm::device_uvector<int> d_candidates_per_anchor2((*ecinput.h_numAnchors), stream, mr);
            rmm::device_uvector<int> d_candidates_per_anchor_prefixsum2(1 + (*ecinput.h_numAnchors), stream, mr);

            cub::DoubleBuffer<read_number> d_items{ecinput.d_candidate_read_ids.data(), d_candidate_read_ids2.data()};
            cub::DoubleBuffer<int> d_numItemsPerSegment{ecinput.d_candidates_per_anchor.data(), d_candidates_per_anchor2.data()};
            cub::DoubleBuffer<int> d_numItemsPerSegmentPrefixSum{ecinput.d_candidates_per_anchor_prefixsum.data(), d_candidates_per_anchor_prefixsum2.data()};

            GpuMinhashQueryFilter::keepDistinctAndNotMatching(
                ecinput.d_anchorReadIds.data(),
                d_items,
                d_numItemsPerSegment,
                d_numItemsPerSegmentPrefixSum, //numSegments + 1
                (*ecinput.h_numAnchors),
                totalNumValues,
                stream,
                mr
            );

            if(d_items.Current() != ecinput.d_candidate_read_ids.data()){
                //std::cerr << "swap d_candidate_read_ids\n";
                std::swap(ecinput.d_candidate_read_ids, d_candidate_read_ids2);
            }
            if(d_numItemsPerSegment.Current() != ecinput.d_candidates_per_anchor.data()){
                //std::cerr << "swap d_candidates_per_anchor\n";
                std::swap(ecinput.d_candidates_per_anchor, d_candidates_per_anchor2);
            }
            if(d_numItemsPerSegmentPrefixSum.Current() != ecinput.d_candidates_per_anchor_prefixsum.data()){
                //std::cerr << "swap d_candidates_per_anchor_prefixsum\n";
                std::swap(ecinput.d_candidates_per_anchor_prefixsum, d_candidates_per_anchor_prefixsum2);
            }

            gpucorrectorkernels::copyMinhashResultsKernel<<<640, 256, 0, stream>>>(
                ecinput.d_numCandidates.data(),
                ecinput.h_numCandidates.data(),
                ecinput.h_candidate_read_ids.data(),
                ecinput.d_candidates_per_anchor_prefixsum.data(),
                ecinput.d_candidate_read_ids.data(),
                *ecinput.h_numAnchors.data()
            ); CUDACHECKASYNC;

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


        SerializedEncodedCorrectionOutput constructSerializedEncodedResults(const GpuErrorCorrectorRawOutput& currentOutput) const{
            //assert(cudaSuccess == currentOutput.event.query());

            if(currentOutput.nothingToDo){
                return SerializedEncodedCorrectionOutput{};
            }

            SerializedEncodedCorrectionOutput serializedEncodedCorrectionOutput;

            //currentOutput.numCorrectedAnchors = *currentOutput.h_numCorrectedAnchors;
            const int numCorrectedAnchors = *currentOutput.h_numCorrectedAnchors;

            serializedEncodedCorrectionOutput.numAnchors = numCorrectedAnchors;
            serializedEncodedCorrectionOutput.numCandidates = 0;
            //serializedEncodedCorrectionOutput.serializedEncodedAnchorCorrections.resize(currentOutput.serializedAnchorResults.size());
            serializedEncodedCorrectionOutput.serializedEncodedAnchorCorrections.resize(currentOutput.serializedAnchorOffsets[numCorrectedAnchors]);
            serializedEncodedCorrectionOutput.beginOffsetsAnchors.resize(numCorrectedAnchors+1);
            serializedEncodedCorrectionOutput.beginOffsetsAnchors[0] = 0;           

            {
                nvtx::ScopedRange sr("copySerializedAnchors", 1);

                // for(int anchor_index = 0; anchor_index < currentOutput.numAnchors; anchor_index++){
                //     const read_number readId = currentOutput.h_anchorReadIds[anchor_index];
                //     const bool isCorrected = currentOutput.h_anchor_is_corrected[anchor_index];
                //     const bool isHQ = currentOutput.h_is_high_quality_anchor[anchor_index].hq();

                //     if(isHQ){
                //         correctionFlags->setCorrectedAsHqAnchor(readId);
                //     }

                //     if(isCorrected){
                //         ; //nothing
                //     }else{
                //         correctionFlags->setCouldNotBeCorrectedAsAnchor(readId);
                //     }

                //     assert(!(isHQ && !isCorrected));                   
                // }

                //currentOutput.serializedAnchorResults only contains data of corrected anchors. can copy directly.
                std::copy(
                    currentOutput.serializedAnchorResults.data(),
                    currentOutput.serializedAnchorResults.data() + currentOutput.serializedAnchorOffsets[numCorrectedAnchors],
                    serializedEncodedCorrectionOutput.serializedEncodedAnchorCorrections.data()
                );
                std::copy(
                    currentOutput.serializedAnchorOffsets.data(),
                    currentOutput.serializedAnchorOffsets.data() + numCorrectedAnchors + 1,
                    serializedEncodedCorrectionOutput.beginOffsetsAnchors.data()
                );
                serializedEncodedCorrectionOutput.numAnchors = numCorrectedAnchors;
            }

            if(programOptions->correctCandidates){
                serializedEncodedCorrectionOutput.serializedEncodedCandidateCorrections.resize(currentOutput.serializedCandidateResults.size());
                serializedEncodedCorrectionOutput.beginOffsetsCandidates.resize(currentOutput.numCorrectedCandidates+1);
                serializedEncodedCorrectionOutput.beginOffsetsCandidates[0] = 0;   

                nvtx::ScopedRange sr("copySerializedCands", 1);

                std::copy(
                    currentOutput.serializedCandidateResults.data(),
                    currentOutput.serializedCandidateResults.data() + currentOutput.serializedCandidateOffsets[currentOutput.numCorrectedCandidates],
                    serializedEncodedCorrectionOutput.serializedEncodedCandidateCorrections.data()
                );
                std::copy(
                    currentOutput.serializedCandidateOffsets.data(),
                    currentOutput.serializedCandidateOffsets.data() + currentOutput.numCorrectedCandidates + 1,
                    serializedEncodedCorrectionOutput.beginOffsetsCandidates.data()
                );
                serializedEncodedCorrectionOutput.numCandidates = currentOutput.numCorrectedCandidates;
            }


            return serializedEncodedCorrectionOutput;
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
            correctionFlags{&correctionFlags_},
            gpuReadStorage{&gpuReadStorage_},
            programOptions{&programOptions_},
            mr{mr_},
            threadPool{threadPool_},
            gpuForestAnchor{gpuForestAnchor_},
            gpuForestCandidate{gpuForestCandidate_},
            readstorageHandle{gpuReadStorage->makeHandle()},
            d_indicesForGather{0, cudaStreamPerThread, mr},
            d_anchorContainsN{0, cudaStreamPerThread, mr},
            d_candidateContainsN_{0, cudaStreamPerThread, mr},
            d_candidate_sequences_lengths_{0, cudaStreamPerThread, mr},
            d_candidate_sequences_data_{0, cudaStreamPerThread, mr},
            d_anchorIndicesOfCandidates_{0, cudaStreamPerThread, mr},
            d_alignment_overlaps_{0, cudaStreamPerThread, mr},
            d_alignment_shifts_{0, cudaStreamPerThread, mr},
            d_alignment_nOps_{0, cudaStreamPerThread, mr},
            d_alignment_best_alignment_flags_{0, cudaStreamPerThread, mr}, 
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
            d_hqAnchorCorrectionOfCandidateExists_{0, cudaStreamPerThread, mr},
            d_allCandidateData{0, cudaStreamPerThread, mr},
            d_numEditsPerCorrectedCandidate{0, cudaStreamPerThread, mr},
            d_indices_of_corrected_anchors{0, cudaStreamPerThread, mr},
            d_num_indices_of_corrected_anchors{0, cudaStreamPerThread, mr},
            d_indices_of_corrected_candidates_{0, cudaStreamPerThread, mr},
            d_totalNumEdits{0, cudaStreamPerThread, mr},
            d_isPairedCandidate_{0, cudaStreamPerThread, mr},
            d_numAnchors{0, cudaStreamPerThread, mr},
            d_numCandidates{0, cudaStreamPerThread, mr},
            d_anchorReadIds{0, cudaStreamPerThread, mr},
            d_anchor_sequences_data{0, cudaStreamPerThread, mr},
            d_anchor_sequences_lengths{0, cudaStreamPerThread, mr},
            d_candidate_read_ids_{0, cudaStreamPerThread, mr},
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

            inputCandidateDataIsReadyEvent = CudaEvent{cudaEventDisableTiming};
            previousBatchFinishedEvent = CudaEvent{cudaEventDisableTiming};

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
            extraStream = streampool::get_current_device_pool()->get_stream();
        }

        ~GpuErrorCorrector(){
            gpuReadStorage->destroyHandle(readstorageHandle);
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

            if(currentNumCandidates == 0 || currentNumAnchors == 0){
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
                d_candidate_read_ids,
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

            CUDACHECK(cudaMemcpyAsync(
                d_candidate_sequences_data,
                currentInput->d_candidate_sequences_data.data(),
                sizeof(unsigned int) * encodedSequencePitchInInts * currentNumCandidates,
                D2D,
                stream
            ));

            CUDACHECK(cudaMemcpyAsync(
                d_candidate_sequences_lengths,
                currentInput->d_candidate_sequences_lengths.data(),
                sizeof(int) * currentNumCandidates,
                D2D,
                stream
            ));

            CUDACHECK(cudaEventRecord(inputCandidateDataIsReadyEvent, stream));

            //after gpu data has been copied to local working set, the gpu data of currentInput can be reused
            CUDACHECK(currentInput->event.record(stream));

            gpucorrectorkernels::setAnchorIndicesOfCandidateskernel
                    <<<currentNumAnchors, 128, 0, stream>>>(
                d_anchorIndicesOfCandidates,
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

            nvtx::push_range("getCandidateAlignments", 5);
            getCandidateAlignments(stream); 
            nvtx::pop_range();

            nvtx::push_range("buildMultipleSequenceAlignment", 6);
            buildAndRefineMultipleSequenceAlignment(stream);
            nvtx::pop_range();

            nvtx::push_range("correctanchors", 8);
            correctAnchors(stream);
            nvtx::pop_range();
            
            if(programOptions->correctCandidates) {
                //CUDACHECK(cudaEventRecord(events[0], stream));

                //#ifdef CANDS_FOREST_FLAGS_COMPUTE_AHEAD
                //if(programOptions->correctionType == CorrectionType::Forest){
                    {
                    nvtx::ScopedRange sr("candidate hq flags", 7);
                    bool* h_excludeFlags = h_flagsCandidates.data();

                    //corrections of candidates for which a high quality anchor correction exists will not be used
                    //-> don't compute them
                    for(int i = 0; i < currentNumCandidates; i++){
                        const read_number candidateReadId = currentInput->h_candidate_read_ids[i];
                        h_excludeFlags[i] = correctionFlags->isCorrectedAsHQAnchor(candidateReadId);
                    }

                    cudaMemcpyAsync(
                        d_hqAnchorCorrectionOfCandidateExists,
                        h_excludeFlags,
                        sizeof(bool) * currentNumCandidates,
                        H2D,
                        //extraStream
                        stream
                    );
                    }

                    nvtx::push_range("correctCandidates", 9);
                    correctCandidates(stream);
                    nvtx::pop_range();

                //}
                //#endif
            }
            

            nvtx::push_range("copyAnchorResultsFromDeviceToHost", 3);
            copyAnchorResultsFromDeviceToHost(stream);
            nvtx::pop_range();

            if(programOptions->correctCandidates) {
                // cudaStream_t candsStream = extraStream;
                // CUDACHECK(cudaStreamWaitEvent(candsStream, events[0], 0)); 

                // nvtx::push_range("correctCandidates", 9);
                // correctCandidates(candsStream);
                // nvtx::pop_range();

                // nvtx::push_range("copyCandidateResultsFromDeviceToHost", 4);
                // copyCandidateResultsFromDeviceToHost(candsStream);
                // nvtx::pop_range(); 

                // CUDACHECK(cudaEventRecord(events[0], candsStream)); 
                // CUDACHECK(cudaStreamWaitEvent(stream, events[0], 0));    

                nvtx::push_range("copyCandidateResultsFromDeviceToHost", 4);
                copyCandidateResultsFromDeviceToHost(stream);
                nvtx::pop_range();   
            }

            managedgpumsa = nullptr;

            currentOutput->h_anchorReadIds.resize(currentNumAnchors);
            std::copy_n(currentInput->h_anchorReadIds.data(), currentNumAnchors, currentOutput->h_anchorReadIds.data());            

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
            handleHost(h_managedmsa_tmp);

            handleHost(h_indicesForGather);
            handleHost(h_isPairedCandidate);
            handleHost(h_candidates_per_anchor_prefixsum);
            handleHost(h_indices);
            handleHost(h_flagsCandidates);

            handleDevice(d_anchorContainsN);
            handleDevice(d_candidateContainsN_);
            handleDevice(d_candidate_sequences_lengths_);
            handleDevice(d_candidate_sequences_data_);
            handleDevice(d_anchorIndicesOfCandidates_);

            handleDevice(d_allCandidateData);

            handleDevice(d_alignment_overlaps_);
            handleDevice(d_alignment_shifts_);
            handleDevice(d_alignment_nOps_);
            handleDevice(d_alignment_best_alignment_flags_);

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
            handleDevice(d_hqAnchorCorrectionOfCandidateExists_);            
            handleDevice(d_numEditsPerCorrectedCandidate);
            handleDevice(d_indices_of_corrected_anchors);
            handleDevice(d_num_indices_of_corrected_anchors);
            handleDevice(d_indices_of_corrected_candidates_);
            handleDevice(d_numEditsPerCorrectedanchor);
            handleDevice(d_numAnchors);
            handleDevice(d_numCandidates);
            handleDevice(d_anchorReadIds);
            handleDevice(d_anchor_sequences_data);
            handleDevice(d_anchor_sequences_lengths);
            handleDevice(d_candidate_read_ids_);
            handleDevice(d_candidates_per_anchor);
            handleDevice(d_candidates_per_anchor_prefixsum);

            return info;
        } 

        void releaseMemory(cudaStream_t stream){
            auto handleDevice = [&](auto& d){
                ::destroy(d, stream);
            };

            handleDevice(d_anchorContainsN);
            handleDevice(d_candidateContainsN_);
            handleDevice(d_candidate_sequences_lengths_);
            handleDevice(d_candidate_sequences_data_);
            handleDevice(d_anchorIndicesOfCandidates_);

            handleDevice(d_allCandidateData);

            handleDevice(d_alignment_overlaps_);
            handleDevice(d_alignment_shifts_);
            handleDevice(d_alignment_nOps_);
            handleDevice(d_alignment_best_alignment_flags_);
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
            handleDevice(d_hqAnchorCorrectionOfCandidateExists_);
            handleDevice(d_numEditsPerCorrectedCandidate);
            handleDevice(d_indices_of_corrected_anchors);
            handleDevice(d_num_indices_of_corrected_anchors);
            handleDevice(d_indices_of_corrected_candidates_);
            handleDevice(d_numEditsPerCorrectedanchor);
            handleDevice(d_numAnchors);
            handleDevice(d_numCandidates);
            handleDevice(d_anchorReadIds);
            handleDevice(d_anchor_sequences_data);
            handleDevice(d_anchor_sequences_lengths);
            handleDevice(d_candidate_read_ids_);
        } 

        void releaseCandidateMemory(cudaStream_t stream){
            auto handleDevice = [&](auto& d){
                ::destroy(d, stream);
            };

            handleDevice(d_candidateContainsN_);
            handleDevice(d_candidate_sequences_lengths_);
            handleDevice(d_candidate_sequences_data_);
            handleDevice(d_anchorIndicesOfCandidates_);

            handleDevice(d_allCandidateData);

            handleDevice(d_alignment_overlaps_);
            handleDevice(d_alignment_shifts_);
            handleDevice(d_alignment_nOps_);
            handleDevice(d_alignment_best_alignment_flags_);
            handleDevice(d_indices);
            handleDevice(d_corrected_candidates);
            handleDevice(d_editsPerCorrectedCandidate);
            handleDevice(d_hqAnchorCorrectionOfCandidateExists_);
            handleDevice(d_numEditsPerCorrectedCandidate);
            handleDevice(d_indices_of_corrected_candidates_);
            handleDevice(d_candidate_read_ids_);
        } 

        


    public: //private:

        void initFixedSizeBuffers(cudaStream_t stream){
            const std::size_t numEditsAnchors = SDIV(editsPitchInBytes * maxAnchors, sizeof(EncodedCorrectionEdit));          

            h_num_total_corrected_candidates.resize(1);
            h_num_indices.resize(1);
            h_numSelected.resize(1);
            h_managedmsa_tmp.resize(1);

            d_anchorContainsN.resize(maxAnchors, stream);

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
 
        void resizeBuffers(int /*numReads*/, int numCandidates, cudaStream_t stream){  
            //assert(numReads <= maxAnchors);
                       
            h_isPairedCandidate.resize(numCandidates);
            h_flagsCandidates.resize(numCandidates);
            
            d_indices.resize(numCandidates + 1, stream);


            #if 0
            d_alignment_overlaps_.resize(numCandidates, stream);
            d_alignment_shifts_.resize(numCandidates, stream);
            d_alignment_nOps_.resize(numCandidates, stream);
            d_alignment_best_alignment_flags_.resize(numCandidates, stream);            
            d_anchorIndicesOfCandidates_.resize(numCandidates, stream);
            d_candidateContainsN_.resize(numCandidates, stream);
            d_isPairedCandidate_.resize(numCandidates, stream);
            d_indices_of_corrected_candidates_.resize(numCandidates, stream);
            d_hqAnchorCorrectionOfCandidateExists_.resize(numCandidates, stream);
            d_candidate_read_ids_.resize(numCandidates, stream);
            d_candidate_sequences_lengths_.resize(numCandidates, stream);
            d_candidate_sequences_data_.resize(numCandidates * encodedSequencePitchInInts, stream);

            d_alignment_overlaps = d_alignment_overlaps_.data();
            d_alignment_shifts = d_alignment_shifts_.data();
            d_alignment_nOps = d_alignment_nOps_.data();
            d_alignment_best_alignment_flags = d_alignment_best_alignment_flags_.data();
            d_anchorIndicesOfCandidates = d_anchorIndicesOfCandidates_.data();
            d_candidateContainsN = d_candidateContainsN_.data();
            d_isPairedCandidate = d_isPairedCandidate_.data();
            d_indices_of_corrected_candidates = d_indices_of_corrected_candidates_.data();
            d_hqAnchorCorrectionOfCandidateExists = d_hqAnchorCorrectionOfCandidateExists_.data();
            d_candidate_read_ids = d_candidate_read_ids_.data();
            d_candidate_sequences_lengths = d_candidate_sequences_lengths_.data();
            d_candidate_sequences_data = d_candidate_sequences_data_.data();


            #else 

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

            void* allocations[12];

            size_t temp_storage_bytes = 0;

            CUDACHECK(cub::AliasTemporaries(
                nullptr,
                temp_storage_bytes,
                allocations,
                allocation_sizes
            ));

            resizeUninitialized(d_allCandidateData, temp_storage_bytes, stream);

            CUDACHECK(cub::AliasTemporaries(
                d_allCandidateData.data(),
                temp_storage_bytes,
                allocations,
                allocation_sizes
            ));

            d_alignment_overlaps = reinterpret_cast<int*>(allocations[0]);
            d_alignment_shifts = reinterpret_cast<int*>(allocations[1]);
            d_alignment_nOps = reinterpret_cast<int*>(allocations[2]);
            d_alignment_best_alignment_flags = reinterpret_cast<AlignmentOrientation*>(allocations[3]);
            d_anchorIndicesOfCandidates = reinterpret_cast<int*>(allocations[4]);
            d_candidateContainsN = reinterpret_cast<bool*>(allocations[5]);
            d_isPairedCandidate = reinterpret_cast<bool*>(allocations[6]);
            d_indices_of_corrected_candidates = reinterpret_cast<int*>(allocations[7]);
            d_hqAnchorCorrectionOfCandidateExists = reinterpret_cast<bool*>(allocations[8]);
            d_candidate_read_ids = reinterpret_cast<read_number*>(allocations[9]);
            d_candidate_sequences_lengths = reinterpret_cast<int*>(allocations[10]);
            d_candidate_sequences_data = reinterpret_cast<unsigned int*>(allocations[11]);

            #endif
            
        }

        void flagPairedCandidates(cudaStream_t stream){

            if(gpuReadStorage->isPairedEnd()){

                assert(currentNumAnchors % 2 == 0);
                assert(currentNumAnchors != 0);

                //d_isPairedCandidate.resize(currentNumCandidates, stream);

                helpers::call_fill_kernel_async(d_isPairedCandidate, currentNumCandidates, false, stream);                   

                dim3 block = 128;
                dim3 grid = currentNumAnchors / 2;
                constexpr int staticSmemBytes = 4096;

                gpucorrectorkernels::flagPairedCandidatesKernel<128,staticSmemBytes>
                <<<grid, block, 0, stream>>>(
                    currentNumAnchors / 2,
                    d_candidates_per_anchor.data(),
                    d_candidates_per_anchor_prefixsum.data(),
                    d_candidate_read_ids,
                    d_isPairedCandidate
                ); CUDACHECKASYNC;
            }else{
                CUDACHECK(cudaMemsetAsync(
                    d_isPairedCandidate,
                    0,
                    sizeof(bool) * currentNumCandidates,
                    stream
                ));
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

        void copyAnchorResultsFromDeviceToHostClassic_serialized(cudaStream_t stream){
            nvtx::ScopedRange sr("constructSerializedAnchorResults-gpu", 5);

            const std::uint32_t maxSerializedBytesPerAnchor = 
                sizeof(read_number) 
                + sizeof(std::uint32_t) 
                + sizeof(short) 
                + sizeof(char) * gpuReadStorage->getSequenceLengthUpperBound();

            const std::uint32_t maxResultBytes = maxSerializedBytesPerAnchor * currentNumAnchors;

            #if 1
            size_t allocation_sizes[3]{};
            allocation_sizes[0] = sizeof(std::uint32_t) * currentNumAnchors; // d_numBytesPerSerializedAnchor
            allocation_sizes[1] = sizeof(std::uint32_t) * (currentNumAnchors+1); // d_numBytesPerSerializedAnchorPrefixSum
            allocation_sizes[2] = sizeof(uint8_t) * maxResultBytes; // d_serializedAnchorResults
            void* allocations[3]{};
            std::size_t tempbytes = 0;

            CUDACHECK(cub::AliasTemporaries(
                nullptr,
                tempbytes,
                allocations,
                allocation_sizes
            ));

            rmm::device_uvector<char> d_tmp(tempbytes, stream, mr);

            CUDACHECK(cub::AliasTemporaries(
                d_tmp.data(),
                tempbytes,
                allocations,
                allocation_sizes
            ));

            std::uint32_t* d_numBytesPerSerializedAnchor = reinterpret_cast<std::uint32_t*>(allocations[0]);
            std::uint32_t* d_numBytesPerSerializedAnchorPrefixSum = reinterpret_cast<std::uint32_t*>(allocations[1]);
            std::uint8_t* d_serializedAnchorResults = reinterpret_cast<std::uint8_t*>(allocations[2]);

            #else

            rmm::device_uvector<std::uint32_t> d_numBytesPerSerializedAnchor_(currentNumAnchors, stream, mr);
            rmm::device_uvector<std::uint32_t> d_numBytesPerSerializedAnchorPrefixSum_(currentNumAnchors+1, stream, mr);
            rmm::device_uvector<std::uint8_t> d_serializedAnchorResults_(maxResultBytes, stream, mr);

            std::uint32_t* d_numBytesPerSerializedAnchor = d_numBytesPerSerializedAnchor_.data();
            std::uint32_t* d_numBytesPerSerializedAnchorPrefixSum = d_numBytesPerSerializedAnchorPrefixSum_.data();
            std::uint8_t* d_serializedAnchorResults = d_serializedAnchorResults_.data();

            #endif


            //compute bytes per anchor
            helpers::lambda_kernel<<<SDIV(currentNumAnchors, 128), 128, 0, stream>>>(
                [
                    d_numBytesPerSerializedAnchor = d_numBytesPerSerializedAnchor,
                    d_numBytesPerSerializedAnchorPrefixSum = d_numBytesPerSerializedAnchorPrefixSum,
                    d_numEditsPerCorrectedanchor = d_numEditsPerCorrectedanchor.data(),
                    d_anchor_sequences_lengths = d_anchor_sequences_lengths.data(),
                    currentNumAnchors = currentNumAnchors,
                    dontUseEditsValue = getDoNotUseEditsValue(),
                    d_num_indices_of_corrected_anchors = d_num_indices_of_corrected_anchors.data(),
                    d_indices_of_corrected_anchors = d_indices_of_corrected_anchors.data(),
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
                rmm::exec_policy_nosync(stream, mr),
                d_numBytesPerSerializedAnchor,
                d_numBytesPerSerializedAnchor + currentNumAnchors,
                d_numBytesPerSerializedAnchorPrefixSum + 1
            );           

            //compute serialized anchors
            helpers::lambda_kernel<<<std::max(1, SDIV(currentNumAnchors, 128)), 128, 0, stream>>>(
                [
                    d_numBytesPerSerializedAnchor = d_numBytesPerSerializedAnchor,
                    d_numBytesPerSerializedAnchorPrefixSum = d_numBytesPerSerializedAnchorPrefixSum,
                    d_serializedAnchorResults = d_serializedAnchorResults,
                    d_numEditsPerCorrectedanchor = d_numEditsPerCorrectedanchor.data(),
                    d_anchor_sequences_lengths = d_anchor_sequences_lengths.data(),
                    d_num_indices_of_corrected_anchors = d_num_indices_of_corrected_anchors.data(),
                    dontUseEditsValue = getDoNotUseEditsValue(),
                    d_is_high_quality_anchor = d_is_high_quality_anchor.data(),
                    d_anchorReadIds = d_anchorReadIds.data(),
                    d_corrected_anchors = d_corrected_anchors.data(),
                    decodedSequencePitchInBytes = this->decodedSequencePitchInBytes,
                    d_editsPerCorrectedanchor = d_editsPerCorrectedanchor.data(),
                    editsPitchInBytes = editsPitchInBytes,
                    d_indices_of_corrected_anchors = d_indices_of_corrected_anchors.data()
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

                        assert(ptr == d_serializedAnchorResults + d_numBytesPerSerializedAnchorPrefixSum[outputIndex+1]);
                    }
                }
            ); CUDACHECKASYNC;

            currentOutput->serializedAnchorResults.resize(maxResultBytes);
            currentOutput->serializedAnchorOffsets.resize(currentNumAnchors + 1);
            currentOutput->h_numCorrectedAnchors.resize(1);
            currentOutput->h_anchor_is_corrected.resize(currentNumAnchors);
            currentOutput->h_is_high_quality_anchor.resize(currentNumAnchors);

            //copy data to host. since number of output bytes of serialized results is only available 
            // on the device, use a kernel

            helpers::lambda_kernel<<<480,128,0,stream>>>(
                [
                    h_numCorrectedAnchors = currentOutput->h_numCorrectedAnchors.data(),
                    d_numCorrectedAnchors = d_num_indices_of_corrected_anchors.data(),
                    h_serializedAnchorOffsets = currentOutput->serializedAnchorOffsets.data(),
                    d_numBytesPerSerializedAnchorPrefixSum = d_numBytesPerSerializedAnchorPrefixSum,
                    h_anchor_is_corrected = currentOutput->h_anchor_is_corrected.data(),
                    d_anchor_is_corrected = d_anchor_is_corrected.data(),
                    h_is_high_quality_anchor = currentOutput->h_is_high_quality_anchor.data(),
                    d_is_high_quality_anchor = d_is_high_quality_anchor.data(),
                    h_serializedAnchorResults = currentOutput->serializedAnchorResults.data(),
                    d_serializedAnchorResults = d_serializedAnchorResults,
                    currentNumAnchors = currentNumAnchors
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

                    for(int i = tid; i < currentNumAnchors; i += stride){
                        h_anchor_is_corrected[i] = d_anchor_is_corrected[i];
                        h_is_high_quality_anchor[i] = d_is_high_quality_anchor[i];
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

        void copyAnchorResultsFromDeviceToHostClassic(cudaStream_t stream){
            copyAnchorResultsFromDeviceToHostClassic_serialized(stream);
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

        void copyCandidateResultsFromDeviceToHostClassic_serialized(cudaStream_t stream){
            nvtx::ScopedRange sr("constructSerializedCandidateResults-gpu", 5);

            int numCorrectedCandidates = (*h_num_total_corrected_candidates);

            const std::uint32_t maxSerializedBytesPerCandidate = 
                sizeof(read_number) 
                + sizeof(std::uint32_t) 
                + sizeof(short) 
                + sizeof(char) * gpuReadStorage->getSequenceLengthUpperBound()
                + sizeof(short);

            const std::uint32_t maxResultBytes = maxSerializedBytesPerCandidate * numCorrectedCandidates;

            #if 1
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

            rmm::device_uvector<char> d_tmp(tempbytes, stream, mr);

            CUDACHECK(cub::AliasTemporaries(
                d_tmp.data(),
                tempbytes,
                allocations,
                allocation_sizes
            ));

            std::uint32_t* d_numBytesPerSerializedCandidate = reinterpret_cast<std::uint32_t*>(allocations[0]);
            std::uint32_t* d_numBytesPerSerializedCandidatePrefixSum = reinterpret_cast<std::uint32_t*>(allocations[1]);
            std::uint8_t* d_serializedCandidateResults = reinterpret_cast<std::uint8_t*>(allocations[2]);

            #else

            rmm::device_uvector<std::uint32_t> d_numBytesPerSerializedCandidate_(numCorrectedCandidates, stream, mr);
            rmm::device_uvector<std::uint32_t> d_numBytesPerSerializedCandidatePrefixSum_(numCorrectedCandidates+1, stream, mr);
            rmm::device_uvector<std::uint8_t> d_serializedCandidateResults_(maxResultBytes, stream, mr);

            std::uint32_t* d_numBytesPerSerializedCandidate = d_numBytesPerSerializedCandidate_.data();
            std::uint32_t* d_numBytesPerSerializedCandidatePrefixSum = d_numBytesPerSerializedCandidatePrefixSum_.data();
            std::uint8_t* d_serializedCandidateResults = d_serializedCandidateResults_.data();

            #endif

            //compute bytes per numCorrectedCandidates
            helpers::lambda_kernel<<<std::max(1, SDIV(numCorrectedCandidates, 128)), 128, 0, stream>>>(
                [
                    d_numBytesPerSerializedCandidate = d_numBytesPerSerializedCandidate,
                    d_numBytesPerSerializedCandidatePrefixSum = d_numBytesPerSerializedCandidatePrefixSum,
                    d_numEditsPerCorrectedCandidate = d_numEditsPerCorrectedCandidate.data(),
                    d_candidate_sequences_lengths = d_candidate_sequences_lengths,
                    numCorrectedCandidates = numCorrectedCandidates,
                    dontUseEditsValue = getDoNotUseEditsValue(),
                    d_indices_of_corrected_candidates = d_indices_of_corrected_candidates,
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
                rmm::exec_policy_nosync(stream, mr),
                d_numBytesPerSerializedCandidate,
                d_numBytesPerSerializedCandidate + numCorrectedCandidates,
                d_numBytesPerSerializedCandidatePrefixSum + 1
            );

            //compute serialized candidates
            helpers::lambda_kernel<<<std::max(1, SDIV(numCorrectedCandidates, 128)), 128, 0, stream>>>(
                [
                    d_numBytesPerSerializedCandidate = d_numBytesPerSerializedCandidate,
                    d_numBytesPerSerializedCandidatePrefixSum = d_numBytesPerSerializedCandidatePrefixSum,
                    d_serializedCandidateResults = d_serializedCandidateResults,
                    d_numEditsPerCorrectedCandidate = d_numEditsPerCorrectedCandidate.data(),
                    d_candidate_sequences_lengths = d_candidate_sequences_lengths,
                    numCorrectedCandidates = numCorrectedCandidates,
                    dontUseEditsValue = getDoNotUseEditsValue(),
                    d_candidate_read_ids = d_candidate_read_ids,
                    d_corrected_candidates = d_corrected_candidates.data(),
                    d_alignment_shifts = d_alignment_shifts,
                    decodedSequencePitchInBytes = this->decodedSequencePitchInBytes,
                    d_editsPerCorrectedCandidate = d_editsPerCorrectedCandidate.data(),
                    editsPitchInBytes = editsPitchInBytes,
                    d_indices_of_corrected_candidates = d_indices_of_corrected_candidates
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

                        assert(ptr == d_serializedCandidateResults + d_numBytesPerSerializedCandidatePrefixSum[outputIndex+1]);
                    }
                }
            ); CUDACHECKASYNC;

            currentOutput->serializedCandidateResults.resize(maxResultBytes);
            currentOutput->serializedCandidateOffsets.resize(numCorrectedCandidates + 1);
            currentOutput->numCorrectedCandidates = numCorrectedCandidates;

            //copy data to host. since number of output bytes of serialized results is only available 
            // on the device, use a kernel

            helpers::lambda_kernel<<<480,128,0,stream>>>(
                [
                    h_serializedCandidateOffsets = currentOutput->serializedCandidateOffsets.data(),
                    d_numBytesPerSerializedCandidatePrefixSum = d_numBytesPerSerializedCandidatePrefixSum,
                    h_serializedCandidateResults = currentOutput->serializedCandidateResults.data(),
                    d_serializedCandidateResults = d_serializedCandidateResults,
                    numCorrectedCandidates = numCorrectedCandidates
                ] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    for(int i = tid; i < numCorrectedCandidates + 1; i += stride){
                        h_serializedCandidateOffsets[i] = d_numBytesPerSerializedCandidatePrefixSum[i];
                    }

                    int serializedBytes = d_numBytesPerSerializedCandidatePrefixSum[numCorrectedCandidates];
                    const size_t numIntCopies = serializedBytes / sizeof(int);
                    const char* src = reinterpret_cast<const char*>(d_serializedCandidateResults);
                    char* dst = reinterpret_cast<char*>(h_serializedCandidateResults);
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

        void copyCandidateResultsFromDeviceToHostClassic(cudaStream_t stream){
            copyCandidateResultsFromDeviceToHostClassic_serialized(stream);
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
                d_candidateContainsN, 
                d_candidate_read_ids, 
                currentNumCandidates,
                stream
            ); 
        }

        void getCandidateAlignments(cudaStream_t stream){

            const bool removeAmbiguousAnchors = programOptions->excludeAmbiguousReads;
            const bool removeAmbiguousCandidates = programOptions->excludeAmbiguousReads;
   
            callShiftedHammingDistanceKernel(
                d_alignment_overlaps,
                d_alignment_shifts,
                d_alignment_nOps,
                d_alignment_best_alignment_flags,
                d_anchor_sequences_data.data(),
                d_candidate_sequences_data,
                d_anchor_sequences_lengths.data(),
                d_candidate_sequences_lengths,
                d_anchorIndicesOfCandidates,
                currentNumAnchors,
                currentNumCandidates,
                d_anchorContainsN.data(),
                removeAmbiguousAnchors,
                d_candidateContainsN,
                removeAmbiguousCandidates,
                gpuReadStorage->getSequenceLengthUpperBound(),
                gpuReadStorage->getSequenceLengthUpperBound(),
                encodedSequencePitchInInts,
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
                    d_alignment_best_alignment_flags,
                    d_alignment_nOps,
                    d_alignment_overlaps,
                    d_candidates_per_anchor_prefixsum.data(),
                    currentNumAnchors,
                    currentNumCandidates,
                    programOptions->estimatedErrorrate,
                    programOptions->estimatedCoverage * programOptions->m_coverage,
                    stream
                );
            }else{
                helpers::lambda_kernel<<<SDIV(currentNumCandidates, 128), 128, 0, stream>>>(
                    [
                        bestAlignmentFlags = d_alignment_best_alignment_flags,
                        nOps = d_alignment_nOps,
                        overlaps = d_alignment_overlaps,
                        currentNumCandidates = currentNumCandidates,
                        d_isPairedCandidate = d_isPairedCandidate,
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
                    d_alignment_best_alignment_flags.data(),
                    d_alignment_nOps.data(),
                    d_alignment_overlaps.data(),
                    d_candidates_per_anchor_prefixsum.data(),
                    d_numAnchors.data(),
                    d_numCandidates.data(),
                    maxAnchors,
                    currentNumCandidates,
                    programOptions->estimatedErrorrate,
                    programOptions->estimatedCoverage * programOptions->m_coverage,
                    stream
                );
            #endif

            callSelectIndicesOfGoodCandidatesKernelAsync(
                d_indices.data(),
                d_indices_per_anchor.data(),
                d_num_indices.data(),
                d_alignment_best_alignment_flags,
                d_candidates_per_anchor.data(),
                d_candidates_per_anchor_prefixsum.data(),
                d_anchorIndicesOfCandidates,
                currentNumAnchors,
                currentNumCandidates,
                stream
            );
        }

        void buildAndRefineMultipleSequenceAlignment(cudaStream_t stream){
            size_t allQualDataBytes = 0;
            char* d_allQualData = nullptr;
            char* d_anchor_qual = nullptr;
            char* d_cand_qual = nullptr;

            if(programOptions->useQualityScores){
                #if 1

                cudaStream_t qualityStream = extraStream;

                CUDACHECK(cudaStreamWaitEvent(qualityStream, inputCandidateDataIsReadyEvent, 0));

                size_t allocation_sizes[2]{};
                allocation_sizes[0] = currentNumAnchors * qualityPitchInBytes; // d_anchor_qual
                allocation_sizes[1] = currentNumCandidates * qualityPitchInBytes; // d_cand_qual
                void* allocations[2]{};

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    allQualDataBytes,
                    allocations,
                    allocation_sizes
                ));

                d_allQualData = reinterpret_cast<char*>(mr->allocate(allQualDataBytes, qualityStream));

                CUDACHECK(cub::AliasTemporaries(
                    d_allQualData,
                    allQualDataBytes,
                    allocations,
                    allocation_sizes
                ));

                d_anchor_qual = reinterpret_cast<char*>(allocations[0]);
                d_cand_qual = reinterpret_cast<char*>(allocations[1]);

                gpuReadStorage->gatherContiguousQualities(
                    readstorageHandle,
                    d_anchor_qual,
                    qualityPitchInBytes,
                    currentInput->h_anchorReadIds[0],
                    currentNumAnchors,
                    qualityStream,
                    mr
                );

                gpuReadStorage->gatherQualities(
                    readstorageHandle,
                    d_cand_qual,
                    qualityPitchInBytes,
                    makeAsyncConstBufferWrapper(currentInput->h_candidate_read_ids.data()),
                    d_candidate_read_ids,
                    currentNumCandidates,
                    qualityStream,
                    mr
                );
                
                CUDACHECK(cudaEventRecord(events[0], qualityStream));
                CUDACHECK(cudaStreamWaitEvent(stream, events[0], 0));

                #else 

                static_assert(false, "Untested code branch");

                d_indicesForGather.resize(currentNumCandidates, stream);
                rmm::device_uvector<int> d_prefixsum(maxAnchors + 1, stream, mr);

                CubCallWrapper(mr).cubExclusiveSum(
                    d_indices_per_anchor.data(),
                    d_prefixsum.data(),
                    maxAnchors,
                    stream
                );

                h_indicesForGather.resize(numCandidates);

                auto zippedValid = thrust::make_zip_iterator(thrust::make_tuple(
                    h_indicesForGather.data(),
                    d_indicesForGather.data()
                ));

                auto duplicateInput = [] __host__ __device__ (read_number id){
                    return thrust::make_tuple(id, id);
                };
                auto duplicatedIds = thrust::make_transform_iterator(
                    d_candidate_read_ids.data(),
                    duplicateInput
                );
                auto copyifend = thrust::copy_if(
                    rmm::exec_policy_nosync(stream, mr),
                    duplicatedIds,
                    duplicatedIds + currentNumCandidates,
                    d_alignment_best_alignment_flags.data(),
                    zippedValid,
                    [] __host__ __device__ (const AlignmentOrientation& o){
                        return o != AlignmentOrientation::None;
                    }
                );

                const int hNumIndices = thrust::distance(zippedValid, copyifend);

                size_t allocation_sizes[2];
                allocation_sizes[0] = currentNumAnchors * qualityPitchInBytes; // d_anchor_qual
                allocation_sizes[1] = currentNumCandidates * qualityPitchInBytes; // d_cand_qual
                void* allocations[2];

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    allQualDataBytes,
                    allocations,
                    allocation_sizes
                ));

                d_allQualData = reinterpret_cast<char*>(mr->allocate(allQualDataBytes, qualityStream));

                CUDACHECK(cub::AliasTemporaries(
                    d_allQualData,
                    allQualDataBytes,
                    allocations,
                    allocation_sizes
                ));

                d_anchor_qual = reinterpret_cast<char*>(allocations[0]);
                d_cand_qual = reinterpret_cast<char*>(allocations[1]);

                gpuReadStorage->gatherContiguousQualities(
                    readstorageHandle,
                    d_anchor_qual,
                    qualityPitchInBytes,
                    currentInput->h_anchorReadIds[0],
                    currentNumAnchors,
                    qualityStream,
                    mr
                );               

                rmm::device_uvector<char> d_candidate_qualities_compact(hNumIndices * qualityPitchInBytes, stream, mr);

                nvtx::push_range("get compact qscores " + std::to_string(hNumIndices) + " " + std::to_string(currentNumCandidates), 6);
                gpuReadStorage->gatherQualities(
                    readstorageHandle,
                    d_candidate_qualities_compact.data(),
                    qualityPitchInBytes,
                    makeAsyncConstBufferWrapper(h_indicesForGather.data()),
                    d_indicesForGather.data(),
                    hNumIndices,
                    stream,
                    mr
                );
                nvtx::pop_range();

                #if 0
                //scatter compact quality scores to correct positions
                helpers::lambda_kernel<<<SDIV(hNumIndices, 256 / 8), 256, 0, stream>>>(
                    [
                        d_candidate_qualities_compact = d_candidate_qualities_compact.data(),
                        d_candidate_qualities = d_cand_qual,
                        d_candidate_sequences_lengths = d_candidate_sequences_lengths.data(),
                        qualityPitchInBytes = qualityPitchInBytes,
                        d_indices = d_indices.data(),
                        d_indices_per_anchor = d_indices_per_anchor.data(),
                        d_indices_per_anchor_prefixsum = d_prefixsum.data(),
                        d_num_indices = d_num_indices.data(),
                        d_candidates_per_anchor_prefixsum = d_candidates_per_anchor_prefixsum.data(),
                        currentNumAnchors = currentNumAnchors,
                        hNumIndices = hNumIndices
                    ] __device__ (){
                        constexpr int groupsize = 8;
                        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

                        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
                        const int numGroups = (blockDim.x * gridDim.x) / groupsize;

                        assert(qualityPitchInBytes % sizeof(int) == 0);

                        for(int c = groupId; c < hNumIndices; c += numGroups){
                            const int anchorIndex = thrust::distance(
                                d_indices_per_anchor_prefixsum,
                                thrust::lower_bound(
                                    thrust::seq,
                                    d_indices_per_anchor_prefixsum,
                                    d_indices_per_anchor_prefixsum + currentNumAnchors,
                                    c + 1
                                )
                            )-1;

                            const int segmentOffset = d_candidates_per_anchor_prefixsum[anchorIndex];
                            const int* const myIndices = d_indices + segmentOffset;
                            const int localCandidatePositionInAnchor = groupId - d_indices_per_anchor_prefixsum[anchorIndex];
                            const int outputCandidateIndex = segmentOffset + myIndices[localCandidatePositionInAnchor];

                            const int candidateLength = d_candidate_sequences_lengths[outputCandidateIndex];
                            const int iters = SDIV(candidateLength, sizeof(int));

                            const int* const input = (const int*)(d_candidate_qualities_compact + size_t(c) * qualityPitchInBytes);
                            int* const output = (int*)(d_candidate_qualities + size_t(outputCandidateIndex) * qualityPitchInBytes);

                            for(int k = group.thread_rank(); k < iters; k += group.size()){
                                output[k] = input[k];
                            }
                        }
                    }
                ); CUDACHECKASYNC;

                #else

                //scatter compact quality scores to correct positions
                helpers::lambda_kernel<<<maxAnchors, 256, 0, stream>>>(
                    [
                        d_candidate_qualities_compact = d_candidate_qualities_compact.data(),
                        d_candidate_qualities = d_cand_qual,
                        d_candidate_sequences_lengths = d_candidate_sequences_lengths.data(),
                        qualityPitchInBytes = qualityPitchInBytes,
                        d_indices = d_indices.data(),
                        d_indices_per_anchor = d_indices_per_anchor.data(),
                        d_indices_per_anchor_prefixsum = d_prefixsum.data(),
                        d_num_indices = d_num_indices.data(),
                        d_candidates_per_anchor_prefixsum = d_candidates_per_anchor_prefixsum.data(),
                        currentNumAnchors = currentNumAnchors
                    ] __device__ (){
                        constexpr int groupsize = 8;
                        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

                        const int groupId = threadIdx.x / groupsize;
                        const int numgroups = blockDim.x / groupsize;

                        assert(qualityPitchInBytes % sizeof(int) == 0);

                        for(int anchor = blockIdx.x; anchor < currentNumAnchors; anchor += gridDim.x){

                            const int globalCandidateOffset = d_candidates_per_anchor_prefixsum[anchor];
                            const int* const myIndices = d_indices + globalCandidateOffset;
                            const int numIndices = d_indices_per_anchor[anchor];
                            const int offset = d_indices_per_anchor_prefixsum[anchor];

                            for(int c = groupId; c < numIndices; c += numgroups){
                                const int outputpos = globalCandidateOffset + myIndices[c];
                                const int inputpos = offset + c;
                                const int length = d_candidate_sequences_lengths[outputpos];

                                const int iters = SDIV(length, sizeof(int));

                                const int* const input = (const int*)(d_candidate_qualities_compact + size_t(inputpos) * qualityPitchInBytes);
                                int* const output = (int*)(d_candidate_qualities + size_t(outputpos) * qualityPitchInBytes);

                                for(int k = group.thread_rank(); k < iters; k += group.size()){
                                    output[k] = input[k];
                                }
                            }
                        }
                    }
                ); CUDACHECKASYNC;
                #endif

                #endif

            }

            managedgpumsa = std::make_unique<ManagedGPUMultiMSA>(stream, mr, h_managedmsa_tmp.data());

            #if 0

            if(useMsaRefinement()){
                rmm::device_uvector<int> d_indices_tmp(currentNumCandidates+1, stream, mr);
                rmm::device_uvector<int> d_indices_per_anchor_tmp(maxAnchors+1, stream, mr);
                rmm::device_uvector<int> d_num_indices_tmp(1, stream, mr);
     
                managedgpumsa->constructAndRefine(
                    d_indices_tmp.data(),
                    d_indices_per_anchor_tmp.data(),
                    d_num_indices_tmp.data(),
                    d_alignment_overlaps.data(),
                    d_alignment_shifts.data(),
                    d_alignment_nOps.data(),
                    d_alignment_best_alignment_flags.data(),
                    d_anchor_sequences_lengths.data(),
                    d_candidate_sequences_lengths.data(),
                    d_indices.data(),
                    d_indices_per_anchor.data(),
                    d_candidates_per_anchor_prefixsum.data(),
                    d_anchor_sequences_data.data(),
                    d_candidate_sequences_data.data(),
                    d_isPairedCandidate.data(),
                    d_anchor_qual,
                    d_cand_qual,
                    currentNumAnchors,
                    currentNumCandidates,
                    programOptions->maxErrorRate,
                    programOptions->useQualityScores,
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    programOptions->estimatedCoverage,
                    getNumRefinementIterations(),
                    MSAColumnCount{static_cast<int>(msaColumnPitchInElements)},
                    stream
                );
                std::swap(d_indices_tmp, d_indices);
                std::swap(d_indices_per_anchor_tmp, d_indices_per_anchor);
                std::swap(d_num_indices_tmp, d_num_indices);
            }else{
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
                    d_anchor_qual,
                    currentNumAnchors,
                    d_candidate_sequences_lengths.data(),
                    d_candidate_sequences_data.data(),
                    d_cand_qual,
                    d_isPairedCandidate.data(),
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    programOptions->useQualityScores,
                    programOptions->maxErrorRate,
                    MSAColumnCount{static_cast<int>(msaColumnPitchInElements)},
                    stream
                );
            }

            #else

            managedgpumsa->construct(
                d_alignment_overlaps,
                d_alignment_shifts,
                d_alignment_nOps,
                d_alignment_best_alignment_flags,
                d_indices.data(),
                d_indices_per_anchor.data(),
                d_candidates_per_anchor_prefixsum.data(),
                d_anchor_sequences_lengths.data(),
                d_anchor_sequences_data.data(),
                d_anchor_qual,
                currentNumAnchors,
                d_candidate_sequences_lengths,
                d_candidate_sequences_data,
                d_cand_qual,
                d_isPairedCandidate,
                encodedSequencePitchInInts,
                qualityPitchInBytes,
                programOptions->useQualityScores,
                programOptions->maxErrorRate,
                MSAColumnCount{static_cast<int>(msaColumnPitchInElements)},
                stream
            );

            if(useMsaRefinement()){
                
                rmm::device_uvector<int> d_indices_tmp(currentNumCandidates+1, stream, mr);
                rmm::device_uvector<int> d_indices_per_anchor_tmp(maxAnchors+1, stream, mr);
                rmm::device_uvector<int> d_num_indices_tmp(1, stream, mr);

                managedgpumsa->refine(
                    d_indices_tmp.data(),
                    d_indices_per_anchor_tmp.data(),
                    d_num_indices_tmp.data(),
                    d_alignment_overlaps,
                    d_alignment_shifts,
                    d_alignment_nOps,
                    d_alignment_best_alignment_flags,
                    d_indices.data(),
                    d_indices_per_anchor.data(),
                    d_candidates_per_anchor_prefixsum.data(),
                    d_anchor_sequences_lengths.data(),
                    d_anchor_sequences_data.data(),
                    d_anchor_qual,
                    currentNumAnchors,
                    d_candidate_sequences_lengths,
                    d_candidate_sequences_data,
                    d_cand_qual,
                    d_isPairedCandidate,
                    currentNumCandidates,
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    programOptions->useQualityScores,
                    programOptions->maxErrorRate,
                    programOptions->estimatedCoverage,
                    getNumRefinementIterations(),
                    stream
                );

                std::swap(d_indices_tmp, d_indices);
                std::swap(d_indices_per_anchor_tmp, d_indices_per_anchor);
                std::swap(d_num_indices_tmp, d_num_indices);

            }

            #endif

            if(programOptions->useQualityScores){
                mr->deallocate(d_allQualData, allQualDataBytes, stream);
            }
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

            rmm::device_uvector<bool> d_candidateCanBeCorrected(currentNumCandidates, stream, mr);

            cub::TransformInputIterator<bool, IsHqAnchor, AnchorHighQualityFlag*>
                d_isHqanchor(d_is_high_quality_anchor.data(), IsHqAnchor{});

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_high_quality_anchor_indices.data(),
                d_num_high_quality_anchor_indices.data(),
                d_isHqanchor,
                d_numAnchors.data()
            ); CUDACHECKASYNC;

            gpucorrectorkernels::initArraysBeforeCandidateCorrectionKernel<<<SDIV(currentNumCandidates, 128), 128, 0, stream>>>(
                currentNumCandidates,
                d_numAnchors.data(),
                d_num_corrected_candidates_per_anchor.data(),
                d_candidateCanBeCorrected.data()
            ); CUDACHECKASYNC;

            #if 1

            bool* d_excludeFlags = d_hqAnchorCorrectionOfCandidateExists;

            callFlagCandidatesToBeCorrectedWithExcludeFlagsKernel(
                d_candidateCanBeCorrected.data(),
                d_num_corrected_candidates_per_anchor.data(),
                managedgpumsa->multiMSAView(),
                d_excludeFlags,
                d_alignment_shifts,
                d_candidate_sequences_lengths,
                d_anchorIndicesOfCandidates,
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
                d_indices_of_corrected_candidates,
                d_num_total_corrected_candidates.data(),
                currentNumCandidates,
                stream
            );

            CUDACHECK(cudaMemcpyAsync(
                h_num_total_corrected_candidates.data(),
                d_num_total_corrected_candidates.data(),
                sizeof(int),
                D2H,
                stream
            ));
            CUDACHECK(cudaStreamSynchronize(stream));

            if((*h_num_total_corrected_candidates) > 0){

                d_corrected_candidates.resize(decodedSequencePitchInBytes * (*h_num_total_corrected_candidates), stream);
                d_numEditsPerCorrectedCandidate.resize((*h_num_total_corrected_candidates), stream);
                std::size_t numEditsCandidates = SDIV(editsPitchInBytes * (*h_num_total_corrected_candidates), sizeof(EncodedCorrectionEdit));
                d_editsPerCorrectedCandidate.resize(numEditsCandidates, stream);
                
                callCorrectCandidatesKernel(
                    d_corrected_candidates.data(),            
                    managedgpumsa->multiMSAView(),
                    d_alignment_shifts,
                    d_alignment_best_alignment_flags,
                    d_candidate_sequences_data,
                    d_candidate_sequences_lengths,
                    d_candidateContainsN,
                    d_indices_of_corrected_candidates,
                    d_num_total_corrected_candidates.data(),
                    d_anchorIndicesOfCandidates,
                    *h_num_total_corrected_candidates,
                    encodedSequencePitchInInts,
                    decodedSequencePitchInBytes,
                    gpuReadStorage->getSequenceLengthUpperBound(),
                    stream
                );            

                callConstructSequenceCorrectionResultsKernel(
                    d_editsPerCorrectedCandidate.data(),
                    d_numEditsPerCorrectedCandidate.data(),
                    getDoNotUseEditsValue(),
                    d_indices_of_corrected_candidates,
                    d_num_total_corrected_candidates.data(),
                    d_candidateContainsN,
                    d_candidate_sequences_data,
                    d_candidate_sequences_lengths,
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
        }

        void correctCandidatesForestGpu(cudaStream_t stream){

            const float min_support_threshold = 1.0f-3.0f*programOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                programOptions->m_coverage / 6.0f * programOptions->estimatedCoverage);
            const int new_columns_to_correct = programOptions->new_columns_to_correct;

            rmm::device_uvector<bool> d_candidateCanBeCorrected(currentNumCandidates, stream, mr);

            cub::TransformInputIterator<bool, IsHqAnchor, AnchorHighQualityFlag*>
                d_isHqanchor(d_is_high_quality_anchor.data(), IsHqAnchor{});

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_high_quality_anchor_indices.data(),
                d_num_high_quality_anchor_indices.data(),
                d_isHqanchor,
                d_numAnchors.data()
            ); CUDACHECKASYNC;

            gpucorrectorkernels::initArraysBeforeCandidateCorrectionKernel<<<SDIV(currentNumCandidates, 128), 128, 0, stream>>>(
                currentNumCandidates,
                d_numAnchors.data(),
                d_num_corrected_candidates_per_anchor.data(),
                d_candidateCanBeCorrected.data()
            ); CUDACHECKASYNC;

   
            #if 1
                bool* d_excludeFlags = d_hqAnchorCorrectionOfCandidateExists;

                callFlagCandidatesToBeCorrectedWithExcludeFlagsKernel(
                    d_candidateCanBeCorrected.data(),
                    d_num_corrected_candidates_per_anchor.data(),
                    managedgpumsa->multiMSAView(),
                    d_excludeFlags,
                    d_alignment_shifts,
                    d_candidate_sequences_lengths,
                    d_anchorIndicesOfCandidates,
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
                d_alignment_shifts,
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
                d_indices_of_corrected_candidates,
                d_num_total_corrected_candidates.data(),
                currentNumCandidates,
                stream
            );

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

            callMsaCorrectCandidatesWithForestKernel(
                d_corrected_candidates.data(),            
                managedgpumsa->multiMSAView(),
                *gpuForestCandidate,
                programOptions->thresholdCands,
                programOptions->estimatedCoverage,
                d_alignment_shifts,
                d_alignment_best_alignment_flags,
                d_candidate_sequences_data,
                d_candidate_sequences_lengths,
                d_indices_of_corrected_candidates,
                d_anchorIndicesOfCandidates,
                *h_num_total_corrected_candidates,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                gpuReadStorage->getSequenceLengthUpperBound(),
                stream
            );  

            callConstructSequenceCorrectionResultsKernel(
                d_editsPerCorrectedCandidate.data(),
                d_numEditsPerCorrectedCandidate.data(),
                getDoNotUseEditsValue(),
                d_indices_of_corrected_candidates,
                d_num_total_corrected_candidates.data(),
                d_candidateContainsN,
                d_candidate_sequences_data,
                d_candidate_sequences_lengths,
                d_corrected_candidates.data(),
                *h_num_total_corrected_candidates,
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

        void updateCorrectionFlags(const GpuErrorCorrectorRawOutput& currentOutput) const{
            if(!currentOutput.nothingToDo){
                for(int anchor_index = 0; anchor_index < currentOutput.numAnchors; anchor_index++){
                    const read_number readId = currentOutput.h_anchorReadIds[anchor_index];
                    const bool isCorrected = currentOutput.h_anchor_is_corrected[anchor_index];
                    const bool isHQ = currentOutput.h_is_high_quality_anchor[anchor_index].hq();

                    if(isHQ){
                        correctionFlags->setCorrectedAsHqAnchor(readId);
                    }

                    if(isCorrected){
                        ; //nothing
                    }else{
                        correctionFlags->setCouldNotBeCorrectedAsAnchor(readId);
                    }

                    assert(!(isHQ && !isCorrected));                   
                }
            }
        }

    private:

        int deviceId;
        std::array<CudaEvent, 2> events;
        cudaStream_t extraStream;

        CudaEvent previousBatchFinishedEvent;
        CudaEvent inputCandidateDataIsReadyEvent;

        std::size_t msaColumnPitchInElements;
        std::size_t encodedSequencePitchInInts;
        std::size_t decodedSequencePitchInBytes;
        std::size_t qualityPitchInBytes;
        std::size_t editsPitchInBytes;

        int maxAnchors;
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
        PinnedBuffer<int> h_managedmsa_tmp;

        PinnedBuffer<read_number> h_indicesForGather;
        rmm::device_uvector<read_number> d_indicesForGather;

        rmm::device_uvector<bool> d_anchorContainsN;
        rmm::device_uvector<bool> d_candidateContainsN_;
        rmm::device_uvector<int> d_candidate_sequences_lengths_;
        rmm::device_uvector<unsigned int> d_candidate_sequences_data_;
        rmm::device_uvector<int> d_anchorIndicesOfCandidates_;
        rmm::device_uvector<int> d_alignment_overlaps_;
        rmm::device_uvector<int> d_alignment_shifts_;
        rmm::device_uvector<int> d_alignment_nOps_;
        rmm::device_uvector<AlignmentOrientation> d_alignment_best_alignment_flags_; 
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
        rmm::device_uvector<bool> d_hqAnchorCorrectionOfCandidateExists_;

        rmm::device_uvector<char> d_allCandidateData;

        int* d_alignment_overlaps = nullptr;
        int* d_alignment_shifts = nullptr;
        int* d_alignment_nOps = nullptr;
        AlignmentOrientation* d_alignment_best_alignment_flags = nullptr;
        int* d_anchorIndicesOfCandidates = nullptr;
        bool* d_candidateContainsN = nullptr;
        bool* d_isPairedCandidate = nullptr;
        int* d_indices_of_corrected_candidates = nullptr;
        bool* d_hqAnchorCorrectionOfCandidateExists = nullptr;

        read_number* d_candidate_read_ids = nullptr;
        int* d_candidate_sequences_lengths = nullptr;
        unsigned int* d_candidate_sequences_data = nullptr;

        
        rmm::device_uvector<int> d_numEditsPerCorrectedCandidate;
        rmm::device_uvector<int> d_indices_of_corrected_anchors;
        rmm::device_uvector<int> d_num_indices_of_corrected_anchors;
        rmm::device_uvector<int> d_indices_of_corrected_candidates_;
        rmm::device_uvector<int> d_totalNumEdits;
        rmm::device_uvector<bool> d_isPairedCandidate_;
        PinnedBuffer<bool> h_isPairedCandidate;

        rmm::device_uvector<int> d_numAnchors;
        rmm::device_uvector<int> d_numCandidates;
        rmm::device_uvector<read_number> d_anchorReadIds;
        rmm::device_uvector<unsigned int> d_anchor_sequences_data;
        rmm::device_uvector<int> d_anchor_sequences_lengths;
        rmm::device_uvector<read_number> d_candidate_read_ids_;
        rmm::device_uvector<int> d_candidates_per_anchor;
        rmm::device_uvector<int> d_candidates_per_anchor_prefixsum; 

        PinnedBuffer<int> h_candidates_per_anchor_prefixsum; 
        PinnedBuffer<int> h_indices;

        PinnedBuffer<bool> h_flagsCandidates;

        std::unique_ptr<ManagedGPUMultiMSA> managedgpumsa;
    };



}
}






#endif
