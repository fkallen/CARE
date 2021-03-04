#ifndef READ_EXTENDER_GPU_HPP
#define READ_EXTENDER_GPU_HPP

#include <readextenderbase.hpp>
#include <config.hpp>
#include <hpc_helpers.cuh>

#include <gpu/gpumsa.cuh>
#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <gpu/gpuminhasher.cuh>
#include <sequencehelpers.hpp>

#include <algorithm>
#include <vector>
#include <numeric>

#include <cub/cub.cuh>

#include <thrust/device_new_allocator.h>


namespace care{


namespace readextendergpukernels{

    template<int dummy=0>
    __global__
    void setAnchorIndicesOfCandidateskernel(
        int* __restrict__ d_anchorIndicesOfCandidates,
        const int* __restrict__ numAnchorsPtr,
        const int* __restrict__ d_candidates_per_anchor,
        const int* __restrict__ d_candidates_per_anchor_prefixsum
    ){
        for(int anchorIndex = blockIdx.x; anchorIndex < *numAnchorsPtr; anchorIndex += gridDim.x){
            const int offset = d_candidates_per_anchor_prefixsum[anchorIndex];
            const int numCandidatesOfAnchor = d_candidates_per_anchor[anchorIndex];
            int* const beginptr = &d_anchorIndicesOfCandidates[offset];

            for(int localindex = threadIdx.x; localindex < numCandidatesOfAnchor; localindex += blockDim.x){
                beginptr[localindex] = anchorIndex;
            }
        }
    }
    

    template<int blocksize>
    __global__
    void reverseComplement2bitKernel(
        const int* __restrict__ lengths,
        const unsigned int* __restrict__ forward,
        unsigned int* __restrict__ reverse,
        int num,
        int encodedSequencePitchInInts
    ){

        for(int s = threadIdx.x + blockIdx.x * blockDim.x; s < num; s += blockDim.x * gridDim.x){
            const unsigned int* input = forward + encodedSequencePitchInInts * s;
            unsigned int* output = reverse + encodedSequencePitchInInts * s;
            const int length = lengths[s];

            SequenceHelpers::reverseComplementSequence2Bit(
                output,
                input,
                length,
                [](auto i){return i;},
                [](auto i){return i;}
            );
        }

        // constexpr int smemsizeints = blocksize * 16;
        // __shared__ unsigned int sharedsequences[smemsizeints]; //sequences will be stored transposed

        // const int sequencesPerSmem = std::min(blocksize, smemsizeints / encodedSequencePitchInInts);
        // assert(sequencesPerSmem > 0);

        // const int smemiterations = SDIV(num, sequencesPerSmem);

        // for(int smemiteration = blockIdx.x; smemiteration < smemiterations; smemiteration += gridDim.x){

        //     const int idBegin = smemiteration * sequencesPerSmem;
        //     const int idEnd = std::min((smemiteration+1) * sequencesPerSmem, num);

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         for(int intindex = 0; intindex < encodedSequencePitchInInts; intindex++){ //load intindex-th element of sequence s
        //             sharedsequences[intindex * sequencesPerSmem + s] = forward[encodedSequencePitchInInts * s + intindex];
        //         }
        //     }

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         SequenceHelpers::reverseComplementSequenceInplace2Bit(&sharedsequences[s], lengths[s], [&](auto i){return i * sequencesPerSmem;});
        //     }

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         for(int intindex = 0; intindex < encodedSequencePitchInInts; intindex++){ //load intindex-th element of sequence s
        //             reverse[encodedSequencePitchInInts * s + intindex] = sharedsequences[intindex * sequencesPerSmem + s];
        //         }
        //     }
        // }
    }


    template<int blocksize, int groupsize>
    __global__
    void filtermatekernel(
        const unsigned int* __restrict__ anchormatedata,
        const unsigned int* __restrict__ candidatefwddata,
        //const unsigned int* __restrict__ candidatefwddata2,
        int encodedSequencePitchInInts,
        const int* __restrict__ numCandidatesPerAnchor,
        const int* __restrict__ numCandidatesPerAnchorPrefixSum,
        const int* __restrict__ activeTaskIndices,
        int numTasksWithRemovedMate,
        bool* __restrict__ output_keepflags
    ){

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupindex = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int numgroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupindexinblock = threadIdx.x / groupsize;

        static_assert(blocksize % groupsize == 0);
        //constexpr int groupsperblock = blocksize / groupsize;

        extern __shared__ unsigned int smem[]; //sizeof(unsigned int) * groupsperblock * encodedSequencePitchInInts

        unsigned int* sharedMate = smem + groupindexinblock * encodedSequencePitchInInts;

        for(int task = groupindex; task < numTasksWithRemovedMate; task += numgroups){

            const int globalTaskIndex = activeTaskIndices[task];
            const int numCandidates = numCandidatesPerAnchor[globalTaskIndex];
            const int candidatesOffset = numCandidatesPerAnchorPrefixSum[globalTaskIndex];

            for(int p = group.thread_rank(); p < encodedSequencePitchInInts; p++){
                sharedMate[p] = anchormatedata[encodedSequencePitchInInts * task + p];
            }
            group.sync();

            //compare mate to candidates. 1 thread per candidate
            for(int c = group.thread_rank(); c < numCandidates; c += group.size()){
                bool doKeep = false;
                const unsigned int* const candidateptr = candidatefwddata + encodedSequencePitchInInts * (candidatesOffset + c);

                for(int p = 0; p < encodedSequencePitchInInts; p++){
                    const unsigned int aaa = sharedMate[p];
                    const unsigned int bbb = candidateptr[p];

                    if(aaa != bbb){
                        doKeep = true;
                        break;
                    }
                }

                output_keepflags[(candidatesOffset + c)] = doKeep;
            }
        }
    }
}


struct ReadExtenderGpu final : public ReadExtenderBase{
public:

    static constexpr int primary_stream_index = 0;

    template<class T>
    using DeviceBuffer = helpers::SimpleAllocationDevice<T>;
    //using DeviceBuffer = helpers::SimpleAllocationPinnedHost<T>;

    template<class T>
    using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;


    ReadExtenderGpu(
        int insertSize,
        int insertSizeStddev,
        int maxextensionPerStep,
        int maximumSequenceLength,
        int kmerLength_,
        const gpu::GpuReadStorage& rs, 
        const gpu::GpuMinhasher& gmh,
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap,
        cub::CachingDeviceAllocator& cubAllocator_
    ) 
    : ReadExtenderBase(insertSize, insertSizeStddev, maxextensionPerStep, maximumSequenceLength, coropts, gap),
        kmerLength(kmerLength_),
        gpuReadStorage(&rs),
        gpuMinhasher(&gmh),
        readStorageHandle(gpuReadStorage->makeHandle()),
        minhashHandle(gpuMinhasher->makeQueryHandle()),
        cubAllocator(&cubAllocator_){

        cudaGetDevice(&deviceId); CUERR;

        kernelLaunchHandle = gpu::make_kernel_launch_handle(deviceId);

        const std::size_t min_overlap = std::max(
            1, 
            std::max(
                goodAlignmentProperties.min_overlap, 
                int(gpuReadStorage->getSequenceLengthUpperBound() * goodAlignmentProperties.min_overlap_ratio)
            )
        );
        const std::size_t msa_max_column_count = (3*gpuReadStorage->getSequenceLengthUpperBound() - 2*min_overlap);
        //round up to 32 elements
        msaColumnPitchInElements = SDIV(msa_max_column_count, 32) * 32;
    }

    ~ReadExtenderGpu(){
        gpuReadStorage->destroyHandle(readStorageHandle);
        gpuMinhasher->destroyHandle(minhashHandle);
    }

    struct BatchData{
        bool pairedEnd = false;
        int numTasks = 0;
        int numTasksWithMateRemoved = 0;

        int totalNumCandidates = 0;
        int totalNumberOfUsedIds = 0;

        PinnedBuffer<read_number> h_anchorReadIds{};
        DeviceBuffer<read_number> d_anchorReadIds{};
        PinnedBuffer<read_number> h_mateReadIds{};
        DeviceBuffer<read_number> d_mateReadIds{};
        PinnedBuffer<read_number> h_candidateReadIds{};
        DeviceBuffer<read_number> d_candidateReadIds{};

        PinnedBuffer<int> h_anchorIndicesOfCandidates{};
        PinnedBuffer<int> h_anchorIndicesOfCandidates2{};

        PinnedBuffer<int> h_segmentIds2{};
        PinnedBuffer<int> h_segmentIds4{};

        DeviceBuffer<int> d_anchorIndicesOfCandidates{};
        DeviceBuffer<int> d_anchorIndicesOfCandidates2{};
        DeviceBuffer<int> d_segmentIdsOfUsedReadIds{};
        DeviceBuffer<int> d_segmentIds4{};

        PinnedBuffer<int> h_segmentIdsOfUsedReadIds{};
        PinnedBuffer<int> h_segmentIdsOfReadIds{};

        PinnedBuffer<unsigned int> h_anchormatedata{};
        DeviceBuffer<unsigned int> d_anchormatedata{};


        PinnedBuffer<unsigned int> h_inputanchormatedata{};
        DeviceBuffer<unsigned int> d_inputanchormatedata{};

        DeviceBuffer<int> d_anchorIndicesWithRemovedMates{};

        DeviceBuffer<int> d_indexlist1{};

        PinnedBuffer<int> h_numCandidatesPerAnchor{};
        PinnedBuffer<int> h_numCandidatesPerAnchor2{};
        DeviceBuffer<int> d_numCandidatesPerAnchor{};
        DeviceBuffer<int> d_numCandidatesPerAnchor2{};
        PinnedBuffer<int> h_numCandidatesPerAnchorPrefixSum{};
        DeviceBuffer<int> d_numCandidatesPerAnchorPrefixSum{};
        DeviceBuffer<int> d_numCandidatesPerAnchorPrefixSum2{};
        PinnedBuffer<int> h_alignment_overlaps{};
        PinnedBuffer<int> h_alignment_shifts{};
        PinnedBuffer<int> h_alignment_nOps{};

        PinnedBuffer<BestAlignment_t> h_alignment_best_alignment_flags{};

        DeviceBuffer<int> d_alignment_overlaps{};
        DeviceBuffer<int> d_alignment_shifts{};
        DeviceBuffer<int> d_alignment_nOps{};
        DeviceBuffer<bool> d_alignment_isValid{};
        DeviceBuffer<BestAlignment_t> d_alignment_best_alignment_flags{};

        DeviceBuffer<int> d_alignment_overlaps2{};
        DeviceBuffer<int> d_alignment_shifts2{};
        DeviceBuffer<int> d_alignment_nOps2{};
        DeviceBuffer<BestAlignment_t> d_alignment_best_alignment_flags2{};

        PinnedBuffer<int> h_numAnchors{};
        PinnedBuffer<int> h_numCandidates{};
        DeviceBuffer<int> d_numAnchors{};
        DeviceBuffer<int> d_numCandidates{};
        DeviceBuffer<int> d_numCandidates2{};
        PinnedBuffer<int> h_numAnchorsWithRemovedMates{};

        PinnedBuffer<int> h_anchorSequencesLength{};
        DeviceBuffer<int> d_anchorSequencesLength{};
        PinnedBuffer<int> h_candidateSequencesLength{};
        DeviceBuffer<int> d_candidateSequencesLength{};
        PinnedBuffer<unsigned int> h_subjectSequencesData{};
        PinnedBuffer<unsigned int> h_candidateSequencesData{};
        PinnedBuffer<unsigned int> h_candidateSequencesRevcData{};

        DeviceBuffer<int> d_candidateSequencesLength2{};
        DeviceBuffer<unsigned int> d_candidateSequencesData2{};
        DeviceBuffer<unsigned int> d_candidateSequencesRevcData2{};
        DeviceBuffer<read_number> d_candidateReadIds2{};
        

        DeviceBuffer<unsigned int> d_subjectSequencesData{};
        DeviceBuffer<unsigned int> d_candidateSequencesData{};
        DeviceBuffer<unsigned int> d_candidateSequencesRevcData{};

        DeviceBuffer<int> d_activeTaskIndices{};
        PinnedBuffer<int> h_activeTaskIndices{};

        DeviceBuffer<int> d_intbuffercandidates{};

        DeviceBuffer<char> d_tempstorage{};

        DeviceBuffer<bool> d_flagsanchors{};

        DeviceBuffer<bool> d_flagscandidates{};

        PinnedBuffer<read_number> h_usedReadIds{};
        PinnedBuffer<int> h_numUsedReadIdsPerAnchor{};
        PinnedBuffer<int> h_numUsedReadIdsPerAnchorPrefixSum{};

        DeviceBuffer<read_number> d_usedReadIds{};
        DeviceBuffer<int> d_numUsedReadIdsPerAnchor{};
        DeviceBuffer<int> d_numUsedReadIdsPerAnchorPrefixSum{};

        PinnedBuffer<char> h_consensus;
        PinnedBuffer<gpu::MSAColumnProperties> h_msa_column_properties;

        DeviceBuffer<std::uint8_t> d_consensus; //encoded , 0-4
        DeviceBuffer<float> d_support;
        DeviceBuffer<int> d_coverage;
        DeviceBuffer<float> d_origWeights;
        DeviceBuffer<int> d_origCoverages;
        DeviceBuffer<gpu::MSAColumnProperties> d_msa_column_properties;
        DeviceBuffer<int> d_counts;
        DeviceBuffer<float> d_weights;

        PinnedBuffer<MultipleSequenceAlignment::PossibleSplitColumn> h_possibleSplitColumns;
        PinnedBuffer<int> h_numPossibleSplitColumnsPerAnchor;
        DeviceBuffer<MultipleSequenceAlignment::PossibleSplitColumn> d_possibleSplitColumns;
        DeviceBuffer<int> d_numPossibleSplitColumnsPerAnchor;

        
    };

    static constexpr int getNumRefinementIterations() noexcept{
        return 5;
    }
     
public: //private:

    std::vector<ExtendResult> processPairedEndTasks(std::vector<Task>& tasks) override;

    std::vector<ExtendResult> processSingleEndTasks(std::vector<Task>& tasks) override;


public:
    void getCandidateReadIds(BatchData& batchData, cudaStream_t stream) const;
    void removeUsedIdsAndMateIds(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const;
    void loadCandidateSequenceData(BatchData& batchData, cudaStream_t stream) const;
    void eraseDataOfRemovedMates(BatchData& batchData, cudaStream_t stream) const;
    void calculateAlignments(BatchData& batchData, cudaStream_t stream) const;
    void filterAlignments(BatchData& batchData, cudaStream_t stream) const;
    void computeMSAs(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const;
    void copyBuffersToHost(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const;

    void copyBatchDataIntoTask(Task& task, int taskindex, const BatchData& data) const{
        const int numCandidates = data.h_numCandidatesPerAnchor[taskindex];
        const int offset = data.h_numCandidatesPerAnchorPrefixSum[taskindex];

        task.candidateReadIds.resize(numCandidates);
        std::copy_n(data.h_candidateReadIds.data() + offset, numCandidates, task.candidateReadIds.begin());

        task.candidateSequenceLengths.resize(numCandidates);
        std::copy_n(data.h_candidateSequencesLength.data() + offset, numCandidates, task.candidateSequenceLengths.begin());

        task.candidateSequenceData.resize(numCandidates * encodedSequencePitchInInts);
        std::copy_n(
            data.h_candidateSequencesData.data() + offset * encodedSequencePitchInInts, 
            numCandidates * encodedSequencePitchInInts, 
            task.candidateSequenceData.begin()
        );

        task.alignmentFlags.resize(numCandidates);
        task.alignments.resize(numCandidates);

        for(int c = 0; c < numCandidates; c++){
            task.alignments[c].shift = data.h_alignment_shifts[offset + c];
            task.alignments[c].overlap = data.h_alignment_overlaps[offset + c];
            task.alignments[c].nOps = data.h_alignment_nOps[offset + c];
            task.alignmentFlags[c] = data.h_alignment_best_alignment_flags[offset + c];
        }

        task.numRemainingCandidates = numCandidates;

        if(task.numRemainingCandidates == 0){
            task.abort = true;
            task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;
        }
    }

    MultipleSequenceAlignment constructMsaWithDataFromTask(Task& task) const{
        const std::string& decodedAnchor = task.totalDecodedAnchors.back();

        MultipleSequenceAlignment msa;

        auto build = [&](){

            task.candidateShifts.resize(task.numRemainingCandidates);
            task.candidateOverlapWeights.resize(task.numRemainingCandidates);

            //gather data required for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                task.candidateShifts[c] = task.alignments[c].shift;

                task.candidateOverlapWeights[c] = calculateOverlapWeight(
                    task.currentAnchorLength, 
                    task.alignments[c].nOps,
                    task.alignments[c].overlap,
                    goodAlignmentProperties.maxErrorRate
                );
            }

            task.candidateStrings.resize(decodedSequencePitchInBytes * task.numRemainingCandidates, '\0');

            //decode the candidates for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                SequenceHelpers::decode2BitSequence(
                    task.candidateStrings.data() + c * decodedSequencePitchInBytes,
                    task.candidateSequenceData.data() + c * encodedSequencePitchInInts,
                    task.candidateSequenceLengths[c]
                );

                if(task.alignmentFlags[c] == BestAlignment_t::ReverseComplement){
                    SequenceHelpers::reverseComplementSequenceDecodedInplace(
                        task.candidateStrings.data() + c * decodedSequencePitchInBytes, 
                        task.candidateSequenceLengths[c]
                    );
                }
            }

            MultipleSequenceAlignment::InputData msaInput;
            msaInput.useQualityScores = false;
            msaInput.subjectLength = task.currentAnchorLength;
            msaInput.nCandidates = task.numRemainingCandidates;
            msaInput.candidatesPitch = decodedSequencePitchInBytes;
            msaInput.candidateQualitiesPitch = 0;
            msaInput.subject = decodedAnchor.c_str();
            msaInput.candidates = task.candidateStrings.data();
            msaInput.subjectQualities = nullptr;
            msaInput.candidateQualities = nullptr;
            msaInput.candidateLengths = task.candidateSequenceLengths.data();
            msaInput.candidateShifts = task.candidateShifts.data();
            msaInput.candidateDefaultWeightFactors = task.candidateOverlapWeights.data();                    

            msa.build(msaInput);
        };

        build();

        #if 1

        auto removeCandidatesOfDifferentRegion = [&](const auto& minimizationResult){
            const int numCandidates = task.candidateReadIds.size();

            int insertpos = 0;
            for(int i = 0; i < numCandidates; i++){
                if(!minimizationResult.differentRegionCandidate[i]){               
                    //keep candidate

                    task.candidateReadIds[insertpos] = task.candidateReadIds[i];

                    std::copy_n(
                        task.candidateSequenceData.data() + i * size_t(encodedSequencePitchInInts),
                        encodedSequencePitchInInts,
                        task.candidateSequenceData.data() + insertpos * size_t(encodedSequencePitchInInts)
                    );

                    task.candidateSequenceLengths[insertpos] = task.candidateSequenceLengths[i];
                    task.alignmentFlags[insertpos] = task.alignmentFlags[i];
                    task.alignments[insertpos] = task.alignments[i];
                    task.candidateOverlapWeights[insertpos] = task.candidateOverlapWeights[i];
                    task.candidateShifts[insertpos] = task.candidateShifts[i];

                    std::copy_n(
                        task.candidateStrings.data() + i * size_t(decodedSequencePitchInBytes),
                        decodedSequencePitchInBytes,
                        task.candidateStrings.data() + insertpos * size_t(decodedSequencePitchInBytes)
                    );

                    insertpos++;
                }
            }

            task.numRemainingCandidates = insertpos;

            task.candidateReadIds.erase(
                task.candidateReadIds.begin() + insertpos, 
                task.candidateReadIds.end()
            );
            task.candidateSequenceData.erase(
                task.candidateSequenceData.begin() + encodedSequencePitchInInts * insertpos, 
                task.candidateSequenceData.end()
            );
            task.candidateSequenceLengths.erase(
                task.candidateSequenceLengths.begin() + insertpos, 
                task.candidateSequenceLengths.end()
            );
            task.alignmentFlags.erase(
                task.alignmentFlags.begin() + insertpos, 
                task.alignmentFlags.end()
            );
            task.alignments.erase(
                task.alignments.begin() + insertpos, 
                task.alignments.end()
            );

            task.candidateStrings.erase(
                task.candidateStrings.begin() + decodedSequencePitchInBytes * insertpos, 
                task.candidateStrings.end()
            );
            task.candidateOverlapWeights.erase(
                task.candidateOverlapWeights.begin() + insertpos, 
                task.candidateOverlapWeights.end()
            );
            task.candidateShifts.erase(
                task.candidateShifts.begin() + insertpos, 
                task.candidateShifts.end()
            );
            
        };

        if(getNumRefinementIterations() > 0){                

            for(int numIterations = 0; numIterations < getNumRefinementIterations(); numIterations++){
                const auto minimizationResult = msa.findCandidatesOfDifferentRegion(
                    correctionOptions.estimatedCoverage
                );

                if(minimizationResult.performedMinimization){
                    removeCandidatesOfDifferentRegion(minimizationResult);

                    //build minimized multiple sequence alignment
                    build();
                }else{
                    break;
                }               
                
            }
        }   

        #endif

        return msa;
    }

private:
    int deviceId;
    int kmerLength;
    std::size_t msaColumnPitchInElements = 0;

    thrust::device_new_allocator<char> thrustallocator{};
    cub::CachingDeviceAllocator* cubAllocator{};

    std::array<CudaStream, 4> streams{};
    std::array<CudaEvent, 1> events{};

    mutable gpu::KernelLaunchHandle kernelLaunchHandle;

    const gpu::GpuReadStorage* gpuReadStorage;
    const gpu::GpuMinhasher* gpuMinhasher;

    mutable ReadStorageHandle readStorageHandle;
    mutable gpu::GpuMinhasher::QueryHandle minhashHandle;


    BatchData batchData{};

};


}


#endif