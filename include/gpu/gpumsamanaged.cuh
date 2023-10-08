#ifndef CARE_GPU_MSA_MANAGED_CUH
#define CARE_GPU_MSA_MANAGED_CUH

#include <gpu/gpumsa.cuh>
#include <gpu/cubvector.cuh>
#include <gpu/kernels.hpp>
#include <hpc_helpers.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <memorymanagement.hpp>
#include <cub/cub.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <gpu/rmm_utilities.cuh>


#include <cstdint>
#include <limits>

namespace care{
namespace gpu{

    #define GPUMSAMANAGED_SINGLE_DATA_BUFFER
    
    struct MSAColumnCount{
    public:
        constexpr MSAColumnCount(int c) noexcept : value(c){}

        constexpr int count() const noexcept{
            return value;
        }

        static constexpr int unknown() noexcept{
            return std::numeric_limits<int>::max();
        }
    private:
        int value;
    };

    bool operator==(MSAColumnCount l, MSAColumnCount r) noexcept{
        return l.count() == r.count();
    }

    bool operator!=(MSAColumnCount l, MSAColumnCount r) noexcept{
        return !(operator==(l,r));
    }


    struct ManagedGPUMultiMSA{
    public:
        ManagedGPUMultiMSA(cudaStream_t stream, rmm::mr::device_memory_resource* mr_, int* h_tempstorage = nullptr) 
            : mr(mr_),
            #ifdef GPUMSAMANAGED_SINGLE_DATA_BUFFER
            d_data(0, stream, mr){
            #else
            d_consensusEncoded_(0, stream, mr),
            d_counts_(0, stream, mr),
            d_coverages_(0, stream, mr),
            d_origCoverages_(0, stream, mr),
            d_weights_(0, stream, mr),
            d_support_(0, stream, mr),
            d_origWeights_(0, stream, mr),
            d_columnProperties_(0, stream, mr){
            #endif

            CUDACHECK(cudaGetDevice(&deviceId));

            if(h_tempstorage != nullptr){
                tempvalue = h_tempstorage;
            }else{
                pinnedValue.resize(1);
                tempvalue = pinnedValue.data();
            }
        }

        ~ManagedGPUMultiMSA(){
            cub::SwitchDevice sd(deviceId);
            #ifdef GPUMSAMANAGED_SINGLE_DATA_BUFFER
            d_data.release();
            #else
            d_consensusEncoded_.release();
            d_counts_.release();
            d_coverages_.release();
            d_origCoverages_.release();
            d_weights_.release();
            d_support_.release();
            d_origWeights_.release();
            d_columnProperties_.release();
            #endif
        }

        void constructAndRefine(
            int* d_newIndices,
            int* d_newNumIndicesPerAnchor,
            int* d_newNumIndices,
            const int* d_overlaps,
            const int* d_shifts,
            const int* d_nOps,
            const AlignmentOrientation* d_bestAlignmentFlags,
            const int* d_anchorLengths,
            const int* d_candidateLengths,
            int* d_candidatePositionsInSegments,
            int* d_numCandidatePositionsInSegments,
            const int* d_segmentBeginOffsets,            
            const unsigned int* d_anchorSequencesData,
            const unsigned int* d_candidateSequencesData,
            const bool* d_isPairedCandidate,
            const char* d_anchorQualities,
            const char* d_candidateQualities,
            int numAnchors,
            int maxNumCandidates,
            float desiredAlignmentMaxErrorRate,
            bool canUseQualityScores,
            int encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            int dataset_coverage,
            int numRefinementIterations,
            MSAColumnCount maximumMsaWidth, // upper bound for number of columns in a single msa. must be large enough to actually fit the data.
            cudaStream_t stream
        ){
            initializeBuffers(
                maximumMsaWidth, 
                numAnchors, 
                d_shifts,
                d_anchorLengths,
                d_candidateLengths,
                d_candidatePositionsInSegments,
                d_numCandidatePositionsInSegments,
                d_segmentBeginOffsets,
                stream
            );

            rmm::device_uvector<bool> d_temp(maxNumCandidates, stream, mr);

            callConstructAndRefineMultipleSequenceAlignmentsKernel(
                d_newIndices,
                d_newNumIndicesPerAnchor,
                d_newNumIndices,
                multiMSA,
                d_overlaps,
                d_shifts,
                d_nOps,
                d_bestAlignmentFlags,
                d_anchorLengths,
                d_candidateLengths,
                d_candidatePositionsInSegments,
                d_numCandidatePositionsInSegments,
                d_segmentBeginOffsets,            
                d_anchorSequencesData,
                d_candidateSequencesData,
                d_isPairedCandidate,
                d_anchorQualities,
                d_candidateQualities,
                d_temp.data(),
                numAnchors,
                desiredAlignmentMaxErrorRate,
                canUseQualityScores,
                encodedSequencePitchInInts,
                qualityPitchInBytes,
                dataset_coverage,
                numRefinementIterations,
                stream
            );
        }

        void construct(
            const int* d_alignment_overlaps,
            const int* d_alignment_shifts,
            const int* d_alignment_nOps,
            const AlignmentOrientation* d_alignment_best_alignment_flags,
            const int* d_candidatePositionsInSegments,
            const int* d_numCandidatePositionsInSegments,
            const int* d_segmentBeginOffsets,
            const int* d_anchorSequencesLength,
            const unsigned int* d_anchorSequences,
            const char* d_anchorQualities,
            int numAnchors,
            const int* d_candidateSequencesLength,
            const unsigned int* d_candidateSequences,
            const char* d_candidateQualities,
            const bool* d_isPairedCandidate,
            std::size_t encodedSequencePitchInInts,
            std::size_t qualityPitchInBytes,
            bool useQualityScores,
            float desiredAlignmentMaxErrorRate,
            MSAColumnCount maximumMsaWidth, // upper bound for number of columns in a single msa. must be large enough to actually fit the data.
            cudaStream_t stream
        ){
            initializeBuffers(
                maximumMsaWidth, 
                numAnchors, 
                d_alignment_shifts,
                d_anchorSequencesLength,
                d_candidateSequencesLength,
                d_candidatePositionsInSegments,
                d_numCandidatePositionsInSegments,
                d_segmentBeginOffsets,
                stream
            );

            callConstructMultipleSequenceAlignmentsKernel_async(
                multiMSA,
                d_alignment_overlaps,
                d_alignment_shifts,
                d_alignment_nOps,
                d_alignment_best_alignment_flags,
                d_anchorSequencesLength,
                d_candidateSequencesLength,
                d_candidatePositionsInSegments,
                d_numCandidatePositionsInSegments,
                d_segmentBeginOffsets,
                d_anchorSequences,
                d_candidateSequences,
                d_isPairedCandidate,
                d_anchorQualities,
                d_candidateQualities,
                desiredAlignmentMaxErrorRate,
                numAnchors,
                useQualityScores,
                encodedSequencePitchInInts,
                qualityPitchInBytes,
                stream
            );
        }

        void refine(
            int* d_newCandidatePositionsInSegments,
            int* d_newNumCandidatePositionsInSegments,
            int* d_newNumCandidates,
            const int* d_alignment_overlaps,
            const int* d_alignment_shifts,
            const int* d_alignment_nOps,
            const AlignmentOrientation* d_alignment_best_alignment_flags,
            int* d_candidatePositionsInSegments,
            int* d_numCandidatePositionsInSegments,
            const int* d_segmentBeginOffsets,
            const int* d_anchorSequencesLength,
            const unsigned int* d_anchorSequences,
            const char* d_anchorQualities,
            int numAnchors,
            const int* d_candidateSequencesLength,
            const unsigned int* d_candidateSequences,
            const char* d_candidateQualities,
            const bool* d_isPairedCandidate,
            int maxNumCandidates,
            std::size_t encodedSequencePitchInInts,
            std::size_t qualityPitchInBytes,
            bool useQualityScores,
            float desiredAlignmentMaxErrorRate,
            int dataset_coverage,
            int numIterations,
            cudaStream_t stream
        ){
            //std::cerr << "thread " << std::this_thread::get_id() << " msa refine, stream " << stream << "\n";

            rmm::device_uvector<bool> d_temp(maxNumCandidates, stream, mr);

            callMsaCandidateRefinementKernel_multiiter_async(
                d_newCandidatePositionsInSegments,
                d_newNumCandidatePositionsInSegments,
                d_newNumCandidates,
                multiMSA,
                d_alignment_best_alignment_flags,
                d_alignment_shifts,
                d_alignment_nOps,
                d_alignment_overlaps,
                d_anchorSequences,
                d_candidateSequences,
                d_isPairedCandidate,
                d_anchorSequencesLength,
                d_candidateSequencesLength,
                d_anchorQualities,
                d_candidateQualities,
                d_temp.data(),
                d_segmentBeginOffsets,
                desiredAlignmentMaxErrorRate,
                numAnchors,
                useQualityScores,
                encodedSequencePitchInInts,
                qualityPitchInBytes,
                d_candidatePositionsInSegments,
                d_numCandidatePositionsInSegments,
                dataset_coverage,
                numIterations,
                stream
            );
        }

        void refine(
            char* d_temp, //sizeof(bool) * maxNumCandidates,
            int* d_newCandidatePositionsInSegments,
            int* d_newNumCandidatePositionsInSegments,
            int* d_newNumCandidates,
            const int* d_alignment_overlaps,
            const int* d_alignment_shifts,
            const int* d_alignment_nOps,
            const AlignmentOrientation* d_alignment_best_alignment_flags,
            int* d_candidatePositionsInSegments,
            int* d_numCandidatePositionsInSegments,
            const int* d_segmentBeginOffsets,
            const int* d_anchorSequencesLength,
            const unsigned int* d_anchorSequences,
            const char* d_anchorQualities,
            int numAnchors,
            const int* d_candidateSequencesLength,
            const unsigned int* d_candidateSequences,
            const char* d_candidateQualities,
            const bool* d_isPairedCandidate,
            int /*maxNumCandidates*/,
            std::size_t encodedSequencePitchInInts,
            std::size_t qualityPitchInBytes,
            bool useQualityScores,
            float desiredAlignmentMaxErrorRate,
            int dataset_coverage,
            int numIterations,
            cudaStream_t stream
        ){
            //std::cerr << "thread " << std::this_thread::get_id() << " msa refine, stream " << stream << "\n";

            bool* d_tmpflags = reinterpret_cast<bool*>(d_temp);

            callMsaCandidateRefinementKernel_multiiter_async(
                d_newCandidatePositionsInSegments,
                d_newNumCandidatePositionsInSegments,
                d_newNumCandidates,
                multiMSA,
                d_alignment_best_alignment_flags,
                d_alignment_shifts,
                d_alignment_nOps,
                d_alignment_overlaps,
                d_anchorSequences,
                d_candidateSequences,
                d_isPairedCandidate,
                d_anchorSequencesLength,
                d_candidateSequencesLength,
                d_anchorQualities,
                d_candidateQualities,
                d_tmpflags,
                d_segmentBeginOffsets,
                desiredAlignmentMaxErrorRate,
                numAnchors,
                useQualityScores,
                encodedSequencePitchInInts,
                qualityPitchInBytes,
                d_candidatePositionsInSegments,
                d_numCandidatePositionsInSegments,
                dataset_coverage,
                numIterations,
                stream
            );
        }

        void computeConsensusQuality(
            char* d_consensusQuality,
            int consensusQualityPitchInBytes,
            cudaStream_t stream
        ){
            callComputeMsaConsensusQualityKernel(
                d_consensusQuality,
                consensusQualityPitchInBytes,
                multiMSA,
                stream
            );
        }

        void computeConsensus(
            char* d_consensus,
            int consensusPitchInBytes,
            cudaStream_t stream
        ){
            callComputeDecodedMsaConsensusKernel(
                d_consensus,
                consensusPitchInBytes,
                multiMSA,
                stream
            );
        }

        void computeMsaSizes(
            int* d_sizes,
            cudaStream_t stream
        ){
            callComputeMsaSizesKernel(
                d_sizes,
                multiMSA,
                stream
            );
        }

        MemoryUsage getMemoryInfo() const{

            MemoryUsage info{};
            // auto handleHost = [&](const auto& h){
            //     info.host += h.sizeInBytes();
            // };
            auto handleDevice = [&](const auto& d){
                using ElementType = typename std::remove_reference<decltype(d)>::type::value_type;
                info.device[deviceId] += d.size() * sizeof(ElementType);
            };

            #ifdef GPUMSAMANAGED_SINGLE_DATA_BUFFER
            handleDevice(d_data);
            #else
            handleDevice(d_consensusEncoded_);
            handleDevice(d_counts_);
            handleDevice(d_coverages_);
            handleDevice(d_origCoverages_);
            handleDevice(d_weights_);
            handleDevice(d_support_);
            handleDevice(d_origWeights_);
            handleDevice(d_columnProperties_);
            #endif

            return info;
        }

        void destroy(cudaStream_t stream){
            #ifdef GPUMSAMANAGED_SINGLE_DATA_BUFFER
            ::destroy(d_data, stream);
            #else
            ::destroy(d_consensusEncoded_, stream);
            ::destroy(d_counts_, stream);
            ::destroy(d_coverages_, stream);
            ::destroy(d_origCoverages_, stream);
            ::destroy(d_weights_, stream);
            ::destroy(d_support_, stream);
            ::destroy(d_origWeights_, stream);
            ::destroy(d_columnProperties_, stream);
            #endif


            multiMSA.numMSAs = 0;
            multiMSA.columnPitchInElements = 0;
            multiMSA.counts = nullptr;
            multiMSA.weights = nullptr;
            multiMSA.coverages = nullptr;
            multiMSA.consensus = nullptr;
            multiMSA.support = nullptr;
            multiMSA.origWeights = nullptr;
            multiMSA.origCoverages = nullptr;
            multiMSA.columnProperties = nullptr;
        }

        GPUMultiMSA multiMSAView() const{
            return multiMSA;
        }

        int getMaximumMsaWidth() const{
            return columnPitchInElements;
        }

        int* getCounts() noexcept{
            return multiMSA.counts;
        }

        float* getWeights() noexcept{
            return multiMSA.weights;
        }

        int* getCoverages() noexcept{
            return multiMSA.coverages;
        }

        std::uint8_t* getEncodedConsensus() noexcept{
            return multiMSA.consensus;
        }

        float* getSupport() noexcept{
            return multiMSA.support;
        }

        float* getOrigWeights() noexcept{
            return multiMSA.origWeights;
        }

        int* getOrigCoverages() noexcept{
            return multiMSA.origCoverages;
        }

        MSAColumnProperties* getColumnProperties() noexcept{
            return multiMSA.columnProperties;
        }

        int getNumMSAs() noexcept{
            return numMSAs;
        }

    private:
        void initializeBuffers(
            MSAColumnCount maximumMsaWidth, 
            int numAnchors, 
            const int* d_alignment_shifts,
            const int* d_anchorSequencesLength,
            const int* d_candidateSequencesLength,
            const int* d_candidatePositionsInSegments,
            const int* d_numCandidatePositionsInSegments,
            const int* d_segmentBeginOffsets,
            cudaStream_t stream
        ){
            if(maximumMsaWidth == MSAColumnCount::unknown()){
                rmm::device_scalar<int> d_maxMsaWidth(stream, mr);

                gpu::callComputeMaximumMsaWidthKernel(
                    d_maxMsaWidth.data(),
                    d_alignment_shifts,
                    d_anchorSequencesLength,
                    d_candidateSequencesLength,
                    d_candidatePositionsInSegments,
                    d_numCandidatePositionsInSegments,
                    d_segmentBeginOffsets,
                    numAnchors,
                    stream
                );

                CUDACHECK(cudaMemcpyAsync(
                    pinnedValue.data(),
                    d_maxMsaWidth.data(),
                    sizeof(int),
                    D2H,
                    stream
                ));

                CUDACHECK(cudaStreamSynchronize(stream));

                columnPitchInElements = *pinnedValue;
            }else{
                columnPitchInElements = maximumMsaWidth.count();
            }

            numMSAs = numAnchors;

            //pad to 128
            columnPitchInElements = SDIV(columnPitchInElements, 128) * 128;
            #ifdef GPUMSAMANAGED_SINGLE_DATA_BUFFER
            size_t allocation_sizes[8];
            allocation_sizes[0] = sizeof(std::uint8_t) * columnPitchInElements * numAnchors; // consensus
            allocation_sizes[1] = sizeof(int) * 4 * columnPitchInElements * numAnchors; // counts
            allocation_sizes[2] = sizeof(int) * columnPitchInElements * numAnchors; // coverages
            allocation_sizes[3] = sizeof(float) * 4 * columnPitchInElements * numAnchors; // weights
            allocation_sizes[4] = sizeof(float) * columnPitchInElements * numAnchors; // support
            allocation_sizes[5] = sizeof(float) * columnPitchInElements * numAnchors; // origweights
            allocation_sizes[6] = sizeof(int) * columnPitchInElements * numAnchors; // origcoverages
            allocation_sizes[7] = sizeof(MSAColumnProperties) * numAnchors; // column properties

            void* allocations[8];

            size_t temp_storage_bytes = 0;

            CUDACHECK(cub::AliasTemporaries(
                nullptr,
                temp_storage_bytes,
                allocations,
                allocation_sizes
            ));

            resizeUninitialized(d_data, temp_storage_bytes, stream);

            CUDACHECK(cub::AliasTemporaries(
                d_data.data(),
                temp_storage_bytes,
                allocations,
                allocation_sizes
            ));

            multiMSA.numMSAs = numMSAs;
            multiMSA.columnPitchInElements = columnPitchInElements;
            multiMSA.consensus = reinterpret_cast<std::uint8_t*>(allocations[0]);
            multiMSA.counts = reinterpret_cast<int*>(allocations[1]);
            multiMSA.coverages = reinterpret_cast<int*>(allocations[2]);
            multiMSA.weights = reinterpret_cast<float*>(allocations[3]);
            multiMSA.support = reinterpret_cast<float*>(allocations[4]);
            multiMSA.origWeights = reinterpret_cast<float*>(allocations[5]);
            multiMSA.origCoverages = reinterpret_cast<int*>(allocations[6]);
            multiMSA.columnProperties = reinterpret_cast<MSAColumnProperties*>(allocations[7]);

            #else 

            resizeUninitialized(d_consensusEncoded_, columnPitchInElements * numAnchors, stream);
            resizeUninitialized(d_counts_, 4 * columnPitchInElements * numAnchors, stream);
            resizeUninitialized(d_coverages_, columnPitchInElements * numAnchors, stream);
            resizeUninitialized(d_origCoverages_, columnPitchInElements * numAnchors, stream);
            resizeUninitialized(d_weights_, 4 * columnPitchInElements * numAnchors, stream);
            resizeUninitialized(d_support_, columnPitchInElements * numAnchors, stream);
            resizeUninitialized(d_origWeights_, columnPitchInElements * numAnchors, stream);
            resizeUninitialized(d_columnProperties_, numAnchors, stream);

            // CUDACHECK(cudaMemsetAsync(d_consensusEncoded.data(), 0, sizeof(std::uint8_t) * columnPitchInElements * numAnchors, stream));
            // CUDACHECK(cudaMemsetAsync(d_counts.data(), 0, sizeof(int) * 4*columnPitchInElements * numAnchors, stream));
            // CUDACHECK(cudaMemsetAsync(d_coverages.data(), 0, sizeof(int) * columnPitchInElements * numAnchors, stream));
            // CUDACHECK(cudaMemsetAsync(d_origCoverages.data(), 0, sizeof(int) * columnPitchInElements * numAnchors, stream));
            // CUDACHECK(cudaMemsetAsync(d_weights.data(), 0, sizeof(float) * 4*columnPitchInElements * numAnchors, stream));
            // CUDACHECK(cudaMemsetAsync(d_support.data(), 0, sizeof(float) * columnPitchInElements * numAnchors, stream));
            // CUDACHECK(cudaMemsetAsync(d_origWeights.data(), 0, sizeof(float) * columnPitchInElements * numAnchors, stream));
            // CUDACHECK(cudaMemsetAsync(d_columnProperties.data(), 0, sizeof(MSAColumnProperties) * numAnchors, stream));

            multiMSA.numMSAs = numMSAs;
            multiMSA.columnPitchInElements = columnPitchInElements;
            multiMSA.counts = d_counts_.data();
            multiMSA.weights = d_weights_.data();
            multiMSA.coverages = d_coverages_.data();
            multiMSA.consensus = d_consensusEncoded_.data();
            multiMSA.support = d_support_.data();
            multiMSA.origWeights = d_origWeights_.data();
            multiMSA.origCoverages = d_origCoverages_.data();
            multiMSA.columnProperties = d_columnProperties_.data();

            #endif
        }

        int deviceId = 0;
        int numMSAs{};
        int columnPitchInElements{};
        int* tempvalue{};

        helpers::SimpleAllocationPinnedHost<int, 0> pinnedValue{};

        rmm::mr::device_memory_resource* mr;

        #ifdef GPUMSAMANAGED_SINGLE_DATA_BUFFER
        rmm::device_uvector<char> d_data;
        #else
        rmm::device_uvector<std::uint8_t> d_consensusEncoded_;
        rmm::device_uvector<int> d_counts_;
        rmm::device_uvector<int> d_coverages_;
        rmm::device_uvector<int> d_origCoverages_;
        rmm::device_uvector<float> d_weights_;
        rmm::device_uvector<float> d_support_;
        rmm::device_uvector<float> d_origWeights_;
        rmm::device_uvector<MSAColumnProperties> d_columnProperties_;
        #endif

        GPUMultiMSA multiMSA{}; //4319
    };


} //namespace gpu
} //namespace care

#endif