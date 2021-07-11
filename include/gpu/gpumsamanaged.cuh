#ifndef CARE_GPU_MSA_MANAGED_CUH
#define CARE_GPU_MSA_MANAGED_CUH

#include <gpu/gpumsa.cuh>
#include <gpu/cubvector.cuh>
#include <gpu/kernellaunch.hpp>
#include <gpu/kernels.hpp>
#include <hpc_helpers.cuh>

#include <cub/cub.cuh>

#include <cstdint>
#include <limits>

namespace care{
namespace gpu{


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
        ManagedGPUMultiMSA(cub::CachingDeviceAllocator& alloc) 
            : cubAllocator(&alloc),
            d_consensusEncoded(alloc),
            d_counts(alloc),
            d_coverages(alloc),
            d_origCoverages(alloc),
            d_weights(alloc),
            d_support(alloc),
            d_origWeights(alloc),
            d_columnProperties(alloc){

            int deviceId;
            cudaGetDevice(&deviceId); CUERR;
            kernelLaunchHandle = gpu::make_kernel_launch_handle(deviceId);

            pinnedValue.resize(1);
        }

        void construct(
            const int* d_alignment_overlaps,
            const int* d_alignment_shifts,
            const int* d_alignment_nOps,
            const BestAlignment_t* d_alignment_best_alignment_flags,
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
            int maxNumCandidates,
            const int* d_numAnchors,
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
                d_numAnchors,
                desiredAlignmentMaxErrorRate,
                numAnchors,
                maxNumCandidates,
                useQualityScores,
                encodedSequencePitchInInts,
                qualityPitchInBytes,
                stream,
                kernelLaunchHandle
            );
        }

        void refine(
            int* d_newCandidatePositionsInSegments,
            int* d_newNumCandidatePositionsInSegments,
            int* d_newNumCandidates,
            const int* d_alignment_overlaps,
            const int* d_alignment_shifts,
            const int* d_alignment_nOps,
            const BestAlignment_t* d_alignment_best_alignment_flags,
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
            const int* d_numAnchors,
            std::size_t encodedSequencePitchInInts,
            std::size_t qualityPitchInBytes,
            bool useQualityScores,
            float desiredAlignmentMaxErrorRate,
            int dataset_coverage,
            int numIterations,
            cudaStream_t stream
        ){
            CachedDeviceUVector<bool> d_temp(maxNumCandidates, stream, *cubAllocator);

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
                d_numAnchors,
                desiredAlignmentMaxErrorRate,
                numAnchors,
                maxNumCandidates,
                useQualityScores,
                encodedSequencePitchInInts,
                qualityPitchInBytes,
                d_candidatePositionsInSegments,
                d_numCandidatePositionsInSegments,
                dataset_coverage,
                numIterations,
                stream,
                kernelLaunchHandle
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

        void destroy(){
            d_consensusEncoded.destroy();
            d_counts.destroy();
            d_coverages.destroy();
            d_origCoverages.destroy();
            d_weights.destroy();
            d_support.destroy();
            d_origWeights.destroy();
            d_columnProperties.destroy();

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
                CachedDeviceUVector<int> d_maxMsaWidth(1, stream, *cubAllocator);

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

                cudaMemcpyAsync(
                    pinnedValue.data(),
                    d_maxMsaWidth.data(),
                    sizeof(int),
                    D2H,
                    stream
                ); CUERR;

                cudaStreamSynchronize(stream); CUERR;

                columnPitchInElements = *pinnedValue;
            }else{
                columnPitchInElements = maximumMsaWidth.count();
            }

            numMSAs = numAnchors;

            //pad to 128
            columnPitchInElements = SDIV(columnPitchInElements, 128) * 128;

            d_consensusEncoded.resizeUninitialized(columnPitchInElements * numAnchors, stream);
            d_counts.resizeUninitialized(4 * columnPitchInElements * numAnchors, stream);
            d_coverages.resizeUninitialized(columnPitchInElements * numAnchors, stream);
            d_origCoverages.resizeUninitialized(columnPitchInElements * numAnchors, stream);
            d_weights.resizeUninitialized(4 * columnPitchInElements * numAnchors, stream);
            d_support.resizeUninitialized(columnPitchInElements * numAnchors, stream);
            d_origWeights.resizeUninitialized(columnPitchInElements * numAnchors, stream);
            d_columnProperties.resizeUninitialized(numAnchors, stream);

            multiMSA.numMSAs = numMSAs;
            multiMSA.columnPitchInElements = columnPitchInElements;
            multiMSA.counts = d_counts.data();
            multiMSA.weights = d_weights.data();
            multiMSA.coverages = d_coverages.data();
            multiMSA.consensus = d_consensusEncoded.data();
            multiMSA.support = d_support.data();
            multiMSA.origWeights = d_origWeights.data();
            multiMSA.origCoverages = d_origCoverages.data();
            multiMSA.columnProperties = d_columnProperties.data();
        }
        public:  

        int numMSAs{};
        int columnPitchInElements{};

        helpers::SimpleAllocationPinnedHost<int, 0> pinnedValue{};
        CudaEvent event{cudaEventDisableTiming};

        cub::CachingDeviceAllocator* cubAllocator{};

        CachedDeviceUVector<std::uint8_t> d_consensusEncoded;
        CachedDeviceUVector<int> d_counts;
        CachedDeviceUVector<int> d_coverages;
        CachedDeviceUVector<int> d_origCoverages;
        CachedDeviceUVector<float> d_weights;
        CachedDeviceUVector<float> d_support;
        CachedDeviceUVector<float> d_origWeights;
        CachedDeviceUVector<MSAColumnProperties> d_columnProperties;

        GPUMultiMSA multiMSA{};
        gpu::KernelLaunchHandle kernelLaunchHandle{};
    };


} //namespace gpu
} //namespace care

#endif