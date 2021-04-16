#ifndef READ_EXTENDER_GPU_HPP
#define READ_EXTENDER_GPU_HPP

#include <readextenderbase.hpp>
#include <config.hpp>
#include <hpc_helpers.cuh>

#include <gpu/gpumsa.cuh>
#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/segmented_set_operations.cuh>
#include <gpu/cachingallocator.cuh>
#include <sequencehelpers.hpp>
#include <hostdevicefunctions.cuh>

#include <algorithm>
#include <vector>
#include <numeric>

#include <cub/cub.cuh>

#include <thrust/device_new_allocator.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>




namespace care{




namespace readextendergpukernels{

    __global__
    void flagCandidateIdsWhichAreEqualToAnchorOrMateKernel(
        const read_number* __restrict__ candidateReadIds,
        const read_number* __restrict__ anchorReadIds,
        const read_number* __restrict__ mateReadIds,
        const int* __restrict__ numCandidatesPerAnchorPrefixSum,
        const int* __restrict__ numCandidatesPerAnchor,
        bool* __restrict__ keepflags, // size numCandidates
        bool* __restrict__ mateRemovedFlags, //size numTasks
        int* __restrict__ numCandidatesPerAnchorOut,
        int numTasks,
        bool isPairedEnd
    );

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

struct BatchData{
    template<class T>
    using DeviceBuffer = helpers::SimpleAllocationDevice<T>;
    //using DeviceBuffer = helpers::SimpleAllocationPinnedHost<T>;

    template<class T>
    using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

    enum class State{
        BeforePrepare,
        BeforeHash,
        BeforeStep,
        BeforeExtend,
        BeforeOutput,
        None
    };

    std::string to_string(State s) const{
        switch(s){
            case State::BeforePrepare: return "BeforePrepare";
            case State::BeforeHash: return "BeforeHash";
            case State::BeforeStep: return "BeforeStep";
            case State::BeforeExtend: return "BeforeExtend";
            case State::BeforeOutput: return "BeforeOutput";
            case State::None: return "None";
            default: return "Missing case BatchData::to_string(State)\n";
        };
    }

    void init(std::vector<ReadExtenderBase::Task> tasks_,
        std::size_t encodedSequencePitchInInts_,
        std::size_t decodedSequencePitchInBytes_,
        std::size_t msaColumnPitchInElements_
    ){
        tasks = std::move(tasks_);

        if(tasks.empty()) return;

        encodedSequencePitchInInts = encodedSequencePitchInInts_;
        decodedSequencePitchInBytes = decodedSequencePitchInBytes_;
        msaColumnPitchInElements = msaColumnPitchInElements_;

        indicesOfActiveTasks.resize(tasks.size());
        std::iota(indicesOfActiveTasks.begin(), indicesOfActiveTasks.end(), 0);

        splitTracker.clear();
        for(const auto& t : tasks){
            splitTracker[t.myReadId] = 1;
        }

        //set input string as current anchor
        for(auto& task : tasks){
            std::string decodedAnchor(task.currentAnchorLength, '\0');

            SequenceHelpers::decode2BitSequence(
                &decodedAnchor[0],
                task.currentAnchor.data(),
                task.currentAnchorLength
            );

            task.totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
            task.totalAnchorBeginInExtendedRead.emplace_back(0);
        }

        pairedEnd = tasks[indicesOfActiveTasks[0]].pairedEnd;
    }

    bool isEmpty() const noexcept{
        return indicesOfActiveTasks.empty();
    }

    void setState(State newstate){      
        #if 0
        std::cerr << "batchdata " << someId << " statechange " << to_string(state) << " -> " << to_string(newstate) << "\n";
        #endif

        state = newstate;
    }


    bool needPrepareStep = false;
    bool pairedEnd = false;
    State state = State::None;
    int numTasks = 0;
    int numTasksWithMateRemoved = 0;
    int someId = 0;

    int totalNumCandidates = 0;
    int totalNumberOfUsedIds = 0;

    std::size_t encodedSequencePitchInInts = 0;
    std::size_t decodedSequencePitchInBytes = 0;
    std::size_t msaColumnPitchInElements = 0;
    std::size_t qualityPitchInBytes = 0;

    PinnedBuffer<read_number> h_anchorReadIds{};
    DeviceBuffer<read_number> d_anchorReadIds{};
    PinnedBuffer<read_number> h_mateReadIds{};
    DeviceBuffer<read_number> d_mateReadIds{};
    PinnedBuffer<read_number> h_candidateReadIds{};
    PinnedBuffer<read_number> h_candidateReadIds2{};
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
    PinnedBuffer<int> h_numCandidatesPerAnchorPrefixSum2{};
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

    
    std::array<CudaEvent, 1> events{};
    std::array<CudaStream, 4> streams{};
    std::vector<int> indicesOfActiveTasks{};
    std::vector<ReadExtenderBase::Task> tasks;
    std::map<read_number, int> splitTracker{}; //counts number of tasks per read id, which can change by splitting a task

};




struct GpuReadHasher{
public:
    GpuReadHasher(
        const gpu::GpuMinhasher& mh
    ) : gpuMinhasher(&mh),
        minhashHandle(gpuMinhasher->makeQueryHandle()) {
    }

    void getCandidateReadIds(BatchData& batchData) const{

        int totalNumValues = 0;

        gpuMinhasher->determineNumValues(
            minhashHandle,
            batchData.d_subjectSequencesData.get(),
            batchData.encodedSequencePitchInInts,
            batchData.d_anchorSequencesLength.get(),
            batchData.numTasks,
            batchData.d_numCandidatesPerAnchor.get(),
            totalNumValues,
            batchData.streams[0]
        );

        cudaStreamSynchronize(batchData.streams[0]); CUERR;

        batchData.d_candidateReadIds.resize(totalNumValues);        

        if(totalNumValues == 0){
            cudaMemsetAsync(batchData.d_numCandidatesPerAnchor.get(), 0, sizeof(int) * batchData.numTasks , batchData.streams[0]); CUERR;
            cudaMemsetAsync(batchData.d_numCandidatesPerAnchorPrefixSum.get(), 0, sizeof(int) * (1 + batchData.numTasks), batchData.streams[0]); CUERR;
            batchData.totalNumCandidates = 0;
            return;
        }

        gpuMinhasher->retrieveValues(
            minhashHandle,
            nullptr,
            batchData.numTasks,              
            totalNumValues,
            batchData.d_candidateReadIds.get(),
            batchData.d_numCandidatesPerAnchor.get(),
            batchData.d_numCandidatesPerAnchorPrefixSum.get(),
            batchData.streams[0]
        );

        cudaMemcpyAsync(
            &batchData.totalNumCandidates,
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + batchData.numTasks,
            sizeof(int),
            D2H,
            batchData.streams[0]
        ); CUERR;

        cudaStreamSynchronize(batchData.streams[0]); CUERR;
    }
    
    void getCandidateReadIds(BatchData& batchData, cudaStream_t stream) const{

        int totalNumValues = 0;

        gpuMinhasher->determineNumValues(
            minhashHandle,
            batchData.d_subjectSequencesData.get(),
            batchData.encodedSequencePitchInInts,
            batchData.d_anchorSequencesLength.get(),
            batchData.numTasks,
            batchData.d_numCandidatesPerAnchor.get(),
            totalNumValues,
            stream
        );

        cudaStreamSynchronize(stream); CUERR;

        batchData.d_candidateReadIds.resize(totalNumValues);        

        if(totalNumValues == 0){
            cudaMemsetAsync(batchData.d_numCandidatesPerAnchor.get(), 0, sizeof(int) * batchData.numTasks , stream); CUERR;
            cudaMemsetAsync(batchData.d_numCandidatesPerAnchorPrefixSum.get(), 0, sizeof(int) * (1 + batchData.numTasks), stream); CUERR;
            batchData.totalNumCandidates = 0;
            return;
        }

        gpuMinhasher->retrieveValues(
            minhashHandle,
            nullptr,
            batchData.numTasks,              
            totalNumValues,
            batchData.d_candidateReadIds.get(),
            batchData.d_numCandidatesPerAnchor.get(),
            batchData.d_numCandidatesPerAnchorPrefixSum.get(),
            stream
        );

        cudaMemcpyAsync(
            &batchData.totalNumCandidates,
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + batchData.numTasks,
            sizeof(int),
            D2H,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;
    }

    const gpu::GpuMinhasher* gpuMinhasher{};
    mutable gpu::GpuMinhasher::QueryHandle minhashHandle;
};


struct GpuExtensionStepper{
public:
    int deviceId{};
    int insertSize{};
    int insertSizeStddev{};
    int maxextensionPerStep{};
    cub::CachingDeviceAllocator* cubAllocator{};
    const gpu::GpuReadStorage* gpuReadStorage{};
    const CorrectionOptions* correctionOptions{};
    const GoodAlignmentProperties* goodAlignmentProperties{};
    mutable ReadStorageHandle readStorageHandle{};
    mutable gpu::KernelLaunchHandle kernelLaunchHandle{};
    
    GpuExtensionStepper() = default;

    GpuExtensionStepper(
        const gpu::GpuReadStorage& rs, 
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap,
        int insertSize_,
        int insertSizeStddev_,
        int maxextensionPerStep_,
        cub::CachingDeviceAllocator& cubAllocator_
    ) : insertSize(insertSize_),
        insertSizeStddev(insertSizeStddev_),
        maxextensionPerStep(maxextensionPerStep_),
        cubAllocator(&cubAllocator_),
        gpuReadStorage(&rs),
        correctionOptions(&coropts),
        goodAlignmentProperties(&gap),
        readStorageHandle(gpuReadStorage->makeHandle()){

        cudaGetDevice(&deviceId); CUERR;

        kernelLaunchHandle = gpu::make_kernel_launch_handle(deviceId);
    }

    constexpr int getNumRefinementIterations() const noexcept{
        return 5;
    }

    void prepareStep(BatchData& batchData) const{
        const int numActiveTasks = batchData.indicesOfActiveTasks.size();
        batchData.numTasks = numActiveTasks;

        batchData.h_numAnchors.resize(1);
        batchData.d_numAnchors.resize(1);
        batchData.h_numCandidates.resize(1);
        batchData.d_numCandidates.resize(1);
        batchData.d_numCandidates2.resize(1);
        batchData.h_numAnchorsWithRemovedMates.resize(1);

        batchData.h_anchorReadIds.resize(numActiveTasks);
        batchData.d_anchorReadIds.resize(numActiveTasks);
        batchData.h_mateReadIds.resize(numActiveTasks);
        batchData.d_mateReadIds.resize(numActiveTasks);
        
        batchData.h_subjectSequencesData.resize(batchData.encodedSequencePitchInInts * numActiveTasks);
        batchData.d_subjectSequencesData.resize(batchData.encodedSequencePitchInInts * numActiveTasks);
        batchData.h_anchorSequencesLength.resize(numActiveTasks);
        batchData.d_anchorSequencesLength.resize(numActiveTasks);

        batchData.h_anchormatedata.resize(numActiveTasks * batchData.encodedSequencePitchInInts);
        batchData.d_anchormatedata.resize(numActiveTasks * batchData.encodedSequencePitchInInts);

        batchData.h_inputanchormatedata.resize(numActiveTasks * batchData.encodedSequencePitchInInts);
        batchData.d_inputanchormatedata.resize(numActiveTasks * batchData.encodedSequencePitchInInts);

        batchData.h_numCandidatesPerAnchor.resize(numActiveTasks);
        batchData.d_numCandidatesPerAnchor.resize(numActiveTasks);
        batchData.h_numCandidatesPerAnchor2.resize(numActiveTasks);
        batchData.d_numCandidatesPerAnchor2.resize(numActiveTasks);
        batchData.h_numCandidatesPerAnchorPrefixSum.resize(numActiveTasks+1);
        batchData.d_numCandidatesPerAnchorPrefixSum.resize(numActiveTasks+1);
        batchData.d_numCandidatesPerAnchorPrefixSum2.resize(numActiveTasks+1);

        batchData.d_anchorIndicesWithRemovedMates.resize(numActiveTasks);

        batchData.h_numCandidatesPerAnchorPrefixSum[0] = 0;

        batchData.totalNumberOfUsedIds = 0;

        for(int t = 0; t < numActiveTasks; t++){
            auto& task = batchData.tasks[batchData.indicesOfActiveTasks[t]];
            task.dataIsAvailable = false;

            batchData.h_anchorReadIds[t] = task.myReadId;
            batchData.h_mateReadIds[t] = task.mateReadId;
            batchData.totalNumberOfUsedIds += task.allUsedCandidateReadIdPairs.size();

            // std::copy(
            //     task.encodedMate.begin(),
            //     task.encodedMate.end(),
            //     batchData.h_inputanchormatedata.begin() + t * batchData.encodedSequencePitchInInts
            // );

            //if(task.iteration >= 0){

                batchData.h_anchorSequencesLength[t] = task.currentAnchorLength;

                std::copy(
                    task.currentAnchor.begin(),
                    task.currentAnchor.end(),
                    batchData.h_subjectSequencesData.begin() + t * batchData.encodedSequencePitchInInts
                );
            // }else{
            //     //only hash kmers which include extended positions

            //     const int extendedPositionsPreviousIteration 
            //         = task.totalAnchorBeginInExtendedRead.at(task.iteration) - task.totalAnchorBeginInExtendedRead.at(task.iteration-1);

            //     const int lengthToHash = std::min(task.currentAnchorLength, kmerLength + extendedPositionsPreviousIteration - 1);
            //     batchData.h_anchorSequencesLength[t] = lengthToHash;

            //     //std::cerr << "lengthToHash = " << lengthToHash << "\n";

            //     std::vector<char> buf(task.currentAnchorLength);
            //     SequenceHelpers::decode2BitSequence(buf.data(), task.currentAnchor.data(), task.currentAnchorLength);
            //     SequenceHelpers::encodeSequence2Bit(
            //         batchData.h_subjectSequencesData.get() + t * batchData.encodedSequencePitchInInts, 
            //         buf.data() + task.currentAnchorLength - lengthToHash, 
            //         lengthToHash
            //     );
            // }
        }

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                //batchData.h_inputanchormatedata.data(),
                batchData.h_subjectSequencesData.data()
            )),
            batchData.numTasks * batchData.encodedSequencePitchInInts,
            thrust::make_zip_iterator(thrust::make_tuple(
                //batchData.d_inputanchormatedata.data(),
                batchData.d_subjectSequencesData.data()
            )),
            batchData.streams[0]
        );

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.h_anchorSequencesLength.data(),
                batchData.h_anchorReadIds.data(),
                batchData.h_mateReadIds.data()
            )),
            batchData.numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_anchorSequencesLength.data(),
                batchData.d_anchorReadIds.data(),
                batchData.d_mateReadIds.data()
            )),
            batchData.streams[0]
        );

        if(0){
            batchData.h_usedReadIds.resize(batchData.totalNumberOfUsedIds);
            batchData.h_numUsedReadIdsPerAnchor.resize(batchData.numTasks);
            batchData.h_numUsedReadIdsPerAnchorPrefixSum.resize(batchData.numTasks);
            batchData.d_usedReadIds.resize(batchData.totalNumberOfUsedIds);
            batchData.d_numUsedReadIdsPerAnchor.resize(batchData.numTasks);
            batchData.d_numUsedReadIdsPerAnchorPrefixSum.resize(batchData.numTasks);

            batchData.d_segmentIdsOfUsedReadIds.resize(batchData.totalNumberOfUsedIds);
            batchData.h_segmentIdsOfUsedReadIds.resize(batchData.totalNumberOfUsedIds);
            
            batchData.h_numUsedReadIdsPerAnchorPrefixSum[0] = 0;

            auto segmentIdsIter = batchData.h_segmentIdsOfUsedReadIds.begin();
            auto h_usedReadIdsIter = batchData.h_usedReadIds.begin();

            for(int i = 0; i < batchData.numTasks; i++){
                auto& task = batchData.tasks[batchData.indicesOfActiveTasks[i]];

                const int numUsedIds = task.allUsedCandidateReadIdPairs.size();

                std::fill(segmentIdsIter, segmentIdsIter + numUsedIds, i);
                segmentIdsIter += numUsedIds;

                h_usedReadIdsIter = std::copy(
                    task.allUsedCandidateReadIdPairs.begin(),
                    task.allUsedCandidateReadIdPairs.end(),
                    h_usedReadIdsIter
                );
                batchData.h_numUsedReadIdsPerAnchor[i] = numUsedIds;

                if(i < batchData.numTasks - 1){
                    batchData.h_numUsedReadIdsPerAnchorPrefixSum[i+1] 
                        = batchData.h_numUsedReadIdsPerAnchorPrefixSum[i] + batchData.h_numUsedReadIdsPerAnchor[i];
                }
            }

            assert(std::distance(batchData.h_usedReadIds.data(), h_usedReadIdsIter) == batchData.totalNumberOfUsedIds);

            // cudaMemcpyAsync(
            //     batchData.d_segmentIdsOfUsedReadIds.data(),
            //     batchData.h_segmentIdsOfUsedReadIds.data(),
            //     sizeof(int) * batchData.totalNumberOfUsedIds,
            //     H2D,
            //     batchData.streams[1]
            // ); CUERR;

            // cudaMemcpyAsync(
            //     batchData.d_usedReadIds.data(),
            //     batchData.h_usedReadIds.data(),
            //     sizeof(read_number) * batchData.totalNumberOfUsedIds,
            //     H2D,
            //     batchData.streams[1]
            // ); CUERR;

            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.h_segmentIdsOfUsedReadIds.data(),
                    batchData.h_usedReadIds.data()
                )),
                batchData.totalNumberOfUsedIds,
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.d_segmentIdsOfUsedReadIds.data(),
                    batchData.d_usedReadIds.data()
                )),
                batchData.streams[1]
            );

            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.h_numUsedReadIdsPerAnchorPrefixSum.data(),
                    batchData.h_numUsedReadIdsPerAnchor.data()
                )),
                batchData.numTasks,
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(),
                    batchData.d_numUsedReadIdsPerAnchor.data()
                )),
                batchData.streams[1]
            );
        }
    }

    void step(BatchData& batchData) const{
        #if 1
        //undo: replace vecAccess\(([a-zA-z]+), ([a-zA-z]+)\) by $1[$2]
        auto vecAccess = [](auto& vec, auto index) -> decltype(vec[index]){
            return vec[index];
        };
        #else 
        auto vecAccess = [](auto& vec, auto index) -> decltype(vec.at(index)){
            return vec.at(index);
        };
        #endif 

        const int numActiveTasks = batchData.indicesOfActiveTasks.size();

        nvtx::push_range("removeUsedIdsAndMateIds", 1);

        removeUsedIdsAndMateIdsCPU(batchData, batchData.streams[0], batchData.streams[1]);    
        // cudaStreamSynchronize(batchData.streams[0]);    
        // cudaStreamSynchronize(batchData.streams[1]);

        // removeUsedIdsAndMateIds(batchData, batchData.streams[0], batchData.streams[1]);  
        // cudaStreamSynchronize(batchData.streams[0]);    
        // cudaStreamSynchronize(batchData.streams[1]);  

        // std::vector<read_number> foo(batchData.totalNumCandidates);
        // cudaMemcpy(
        //     foo.data(),
        //     batchData.d_candidateReadIds.data(),
        //     sizeof(read_number) * batchData.totalNumCandidates,
        //     D2H
        // );

        // std::vector<int> foo2(batchData.numTasks);
        // cudaMemcpy(
        //     foo2.data(),
        //     batchData.d_numCandidatesPerAnchor.data(),
        //     sizeof(int) * batchData.numTasks,
        //     D2H
        // );

        // std::vector<int> foo3(batchData.numTasks+1);
        // cudaMemcpy(
        //     foo3.data(),
        //     batchData.d_numCandidatesPerAnchorPrefixSum.data(),
        //     sizeof(int) * (batchData.numTasks+1),
        //     D2H
        // );

        // std::cerr << "foo\n";
        // std::copy(foo.begin(), foo.end(), std::ostream_iterator<read_number>(std::cerr, " "));
        // std::cerr << "\n";
        // std::cerr << "foo2\n";
        // std::copy(foo2.begin(), foo2.end(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";
        // std::cerr << "foo3\n";
        // std::copy(foo3.begin(), foo3.end(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";

        nvtx::pop_range();

        //std::exit(0);


        //collectTimer.start();

        nvtx::push_range("loadCandidateSequenceData", 2);

        loadCandidateSequenceData(batchData, batchData.streams[0]);

        nvtx::pop_range();
        
        if(batchData.numTasksWithMateRemoved > 0){

            //for those tasks where a mate id has been removed, remove candidates whose sequence is equal to the mate sequence.
            //Sets batchData.totalNumCandidates to the sum of number of candidates for all tasks.

            nvtx::push_range("eraseDataOfRemovedMates", 3);

            eraseDataOfRemovedMates(batchData, batchData.streams[0]);

            nvtx::pop_range();

        }

        //collectTimer.stop();

        /*
            Compute alignments
        */

        //alignmentTimer.start();

        nvtx::push_range("calculateAlignments", 4);

        calculateAlignments(batchData, batchData.streams[0]);

        nvtx::pop_range();

        //alignmentTimer.stop();
        
        nvtx::push_range("filterAlignments", 5);
    
        //Sets batchData.totalNumCandidates to the sum of number of candidates for all tasks.
        filterAlignments(batchData, batchData.streams[0]);

        nvtx::pop_range();

        nvtx::push_range("computeMSAs", 6);

        //Sets batchData.totalNumCandidates to the sum of number of candidates for all tasks. (msa refinement can remove candidates)
        computeMSAs(batchData, batchData.streams[0], batchData.streams[1]);

        nvtx::pop_range();

        //copy all necessary buffers to host
            
        nvtx::push_range("copyBuffersToHost", 7);

        copyBuffersToHost(batchData, batchData.streams[0], batchData.streams[1]);

        nvtx::pop_range();
        

        cudaStreamSynchronize(batchData.streams[0]); CUERR;
        cudaStreamSynchronize(batchData.streams[1]); CUERR;

        for(int i = 0; i < numActiveTasks; i++){
            auto& task = batchData.tasks[batchData.indicesOfActiveTasks[i]];

            task.numRemainingCandidates = batchData.h_numCandidatesPerAnchor[i];

            if(task.numRemainingCandidates == 0){
                task.abort = true;
                task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;
            }
        }


        
    }

    void extendAfterStep(BatchData& batchData) const{
        #if 1
        //undo: replace vecAccess\(([a-zA-z]+), ([a-zA-z]+)\) by $1[$2]
        auto vecAccess = [](auto& vec, auto index) -> decltype(vec[index]){
            return vec[index];
        };
        #else 
        auto vecAccess = [](auto& vec, auto index) -> decltype(vec.at(index)){
            return vec.at(index);
        };
        #endif 

        const int numActiveTasks = batchData.indicesOfActiveTasks.size();

        std::vector<ReadExtenderBase::Task> newTasksFromSplit;
        std::vector<int> newTaskIndices;        

        auto constructMsa = [&](auto& task, int taskindex){
            assert(task.dataIsAvailable);
            return constructMsaWithDataFromTask(task, batchData);
        };

        auto extendWithMsa = [&](auto& task, const char* consensus, int consensusLength, int taskIndex){

            //can extend by at most maxextensionPerStep bps
            int extendBy = std::min(
                consensusLength - task.currentAnchorLength, 
                maxextensionPerStep
            );
            //cannot extend over fragment 
            extendBy = std::min(extendBy, (insertSize + insertSizeStddev - task.mateLength) - task.accumExtensionLengths);

            auto makeAnchorForNextIteration = [&](){
                if(extendBy == 0){
                    task.abort = true;
                    task.abortReason = AbortReason::MsaNotExtended;
                }else{
                    task.accumExtensionLengths += extendBy;

                    //update data for next iteration of outer while loop                           

                    std::string decodedAnchor(consensus + extendBy, task.currentAnchorLength);

                    const int numInts = SequenceHelpers::getEncodedNumInts2Bit(task.currentAnchorLength);

                    task.currentAnchor.resize(numInts);

                    SequenceHelpers::encodeSequence2Bit(
                        task.currentAnchor.data(), 
                        decodedAnchor.data(), 
                        task.currentAnchorLength
                    );

                    task.totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
                    task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                    // task.resultsequence.insert(
                    //     task.resultsequence.end(), 
                    //     consensus + task.currentAnchorLength, 
                    //     consensus + task.currentAnchorLength + extendBy
                    // );


                    // std::string tmp(task.currentAnchorLength, '\0');

                    // decode2BitSequence(
                    //     &tmp[0],
                    //     task.currentAnchor.data(),
                    //     task.currentAnchorLength
                    // );

                    // auto sub = task.resultsequence.substr(task.resultsequence.length() - task.currentAnchorLength);

                    // assert(sub == tmp);
                }
            };

            constexpr int requiredOverlapMate = 70; //TODO relative overlap 
            constexpr int numMismatchesUpperBound = 2;

            if(task.pairedEnd && task.accumExtensionLengths + consensusLength - requiredOverlapMate + task.mateLength >= insertSize - insertSizeStddev){
                //check if mate can be overlapped with consensus 

                //hamMap[i] stores possible starting positions of overlaps which would have hamming distance i
                std::map<int, std::vector<int>> hamMap;

                //longmatchMap[i] stores possible starting positions of overlaps which would have a longest match of length i between mate and msa consensus
                //std::map<int, std::vector<int>> longmatchMap; //map length of longest match to list start positions

                //for each possibility to overlap the mate and consensus such that the merged sequence would end in the desired range [insertSize - insertSizeStddev, insertSize + insertSizeStddev]

                const int firstStartpos = std::max(0, insertSize - insertSizeStddev - task.accumExtensionLengths - task.mateLength);
                const int lastStartposExcl = std::min(
                    std::max(0, insertSize + insertSizeStddev - task.accumExtensionLengths - task.mateLength) + 1,
                    consensusLength - requiredOverlapMate
                );

                for(int startpos = firstStartpos; startpos < lastStartposExcl; startpos++){
                    //compute metrics of overlap
                        
                    const int ham = cpu::hammingDistanceOverlap(
                        consensus + startpos, consensus + consensusLength, 
                        task.decodedMateRevC.begin(), task.decodedMateRevC.end()
                    );

                    hamMap[ham].emplace_back(startpos);

                    // const int longest = cpu::longestMatch(
                    //     consensus + startpos, consensus + consensusLength, 
                    //     task.decodedMateRevC.begin(), task.decodedMateRevC.end()
                    // );

                    // longmatchMap[longest].emplace_back(startpos);
                }
                
                std::vector<std::pair<int, std::vector<int>>> flatMap(hamMap.begin(), hamMap.end());
                //sort by hamming distance, ascending
                std::sort(flatMap.begin(), flatMap.end(), [](const auto& p1, const auto& p2){return p1.first < p2.first;});

                //std::vector<std::pair<int, std::vector<int>>> flatMap2(longmatchMap.begin(), longmatchMap.end());
                //sort by length of longest match, descending
                //std::sort(flatMap2.begin(), flatMap2.end(), [](const auto& p1, const auto& p2){return p2.first < p1.first;});

                //if there exists an overlap between msa consensus and mate which would end the merge, use the best one
                if(flatMap.size() > 0 && flatMap[0].first <= numMismatchesUpperBound){
                //if(flatMap2.size() > 0 && flatMap2[0].first >= 40){
                    const int mateStartposInConsensus = flatMap[0].second.front();
                    const int missingPositionsBetweenAnchorEndAndMateBegin = std::max(0, mateStartposInConsensus - task.currentAnchorLength);

                    if(missingPositionsBetweenAnchorEndAndMateBegin > 0){
                        //bridge the gap between current anchor and mate
                        task.totalDecodedAnchors.emplace_back(
                            consensus + missingPositionsBetweenAnchorEndAndMateBegin,
                            consensus + missingPositionsBetweenAnchorEndAndMateBegin + mateStartposInConsensus
                        );
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths + missingPositionsBetweenAnchorEndAndMateBegin);
                    }


                    task.mateHasBeenFound = true;

                    //const int currentAccumExtensionLengths = task.accumExtensionLengths;
                    
                    task.accumExtensionLengths += mateStartposInConsensus;
                    std::string decodedAnchor(task.decodedMateRevC);

                    task.totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
                    task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                    // const int startpos = mateStartposInConsensus;
                    // task.resultsequence.resize(currentAccumExtensionLengths + startpos + task.decodedMateRevC.length());
                    // const auto replaceBegin = task.resultsequence.begin() + currentAccumExtensionLengths + startpos;
                    // task.resultsequence.replace(
                    //     replaceBegin, 
                    //     replaceBegin + task.decodedMateRevC.length(), 
                    //     task.decodedMateRevC.begin(), 
                    //     task.decodedMateRevC.end()
                    // );

                }else{
                    makeAnchorForNextIteration();
                }
            }else{
                makeAnchorForNextIteration();
            }
        };

        auto keepSelectedCandidates = [&](auto& task, const auto& selectedCandidateIndices, int taskIndex){
            assert(task.dataIsAvailable);

            const int numCandidateIndices = selectedCandidateIndices.size();
            assert(numCandidateIndices <= task.numRemainingCandidates);

            for(int i = 0; i < numCandidateIndices; i++){
                const int c = vecAccess(selectedCandidateIndices, i);
                // if(!(0 <= c && c < task.candidateReadIds.size())){
                //     std::cerr << "c = " << c << ", candidateReadIds.size() = " << task.candidateReadIds.size() << "\n";
                // }

                // assert(0 <= c && c < task.candidateReadIds.size());
                // assert(0 <= c && c < task.candidateSequenceLengths.size());
                // assert(0 <= c && c < task.alignments.size());
                // assert(0 <= c && c < task.alignmentFlags.size());

                // assert(0 <= c && c*encodedSequencePitchInInts < task.candidateSequencesFwdData.size());
                // assert(0 <= c && c*encodedSequencePitchInInts < task.candidateSequencesRevcData.size());
                // assert(0 <= c && c*encodedSequencePitchInInts < task.candidateSequenceData.size());

                vecAccess(task.candidateReadIds, i) = vecAccess(task.candidateReadIds, c);
                vecAccess(task.candidateSequenceLengths , i) = vecAccess(task.candidateSequenceLengths, c);
                vecAccess(task.alignments, i) = vecAccess(task.alignments, c);
                vecAccess(task.alignmentFlags, i) = vecAccess(task.alignmentFlags, c);
                vecAccess(task.candidateShifts, i) = vecAccess(task.candidateShifts, c);
                vecAccess(task.candidateOverlapWeights, i) = vecAccess(task.candidateOverlapWeights, c);

                // std::copy_n(
                //     task.candidateSequencesFwdData.begin() + c * encodedSequencePitchInInts,
                //     encodedSequencePitchInInts,
                //     task.candidateSequencesFwdData.begin() + i * encodedSequencePitchInInts
                // );

                // std::copy_n(
                //     task.candidateSequencesRevcData.begin() + c * encodedSequencePitchInInts,
                //     encodedSequencePitchInInts,
                //     task.candidateSequencesRevcData.begin() + i * encodedSequencePitchInInts
                // );

                std::copy_n(
                    task.candidateSequenceData.begin() + c * batchData.encodedSequencePitchInInts,
                    batchData.encodedSequencePitchInInts,
                    task.candidateSequenceData.begin() + i * batchData.encodedSequencePitchInInts
                );

                std::copy_n(
                    task.candidateStrings.begin() + c * batchData.decodedSequencePitchInBytes,
                    batchData.decodedSequencePitchInBytes,
                    task.candidateStrings.begin() + i * batchData.decodedSequencePitchInBytes
                );
            }

            task.candidateReadIds.erase(
                task.candidateReadIds.begin() + numCandidateIndices,
                task.candidateReadIds.end()
            );
            task.candidateSequenceLengths.erase(
                task.candidateSequenceLengths.begin() + numCandidateIndices,
                task.candidateSequenceLengths.end()
            );
            task.alignments.erase(
                task.alignments.begin() + numCandidateIndices,
                task.alignments.end()
            );
            task.alignmentFlags.erase(
                task.alignmentFlags.begin() + numCandidateIndices,
                task.alignmentFlags.end()
            );
            task.candidateSequenceData.erase(
                task.candidateSequenceData.begin() + numCandidateIndices * batchData.encodedSequencePitchInInts,
                task.candidateSequenceData.end()
            );
            if(task.pairedEnd){
                task.mateIdLocationIter = std::lower_bound(
                    task.candidateReadIds.begin(),
                    task.candidateReadIds.end(),
                    task.mateReadId
                );

                task.mateHasBeenFound = (task.mateIdLocationIter != task.candidateReadIds.end() 
                    && *task.mateIdLocationIter == task.mateReadId);
            }
            task.numRemainingCandidates = numCandidateIndices;
        };

        nvtx::push_range("MSA", 6);
        //msaTimer.start();

        for(int i = 0; i < numActiveTasks; i++){ 
            const int indexOfActiveTask = batchData.indicesOfActiveTasks[i];
            auto& task = batchData.tasks[indexOfActiveTask];

            if(task.numRemainingCandidates == 0){
                continue;
            }
            assert(task.numRemainingCandidates > 0);

            const std::size_t splitcolumnsPitchElements = 32;
            const auto* const possibleSplitColumns = batchData.h_possibleSplitColumns.data() + splitcolumnsPitchElements * i;
            const int numPossibleSplitColumns = batchData.h_numPossibleSplitColumnsPerAnchor[i];
            const gpu::MSAColumnProperties msaProps = batchData.h_msa_column_properties[i];

            const int consensusLength = msaProps.lastColumn_excl - msaProps.firstColumn_incl;
            const char* const consensus = batchData.h_consensus.data() + i * batchData.msaColumnPitchInElements;

            
#if 1
            //if(task.splitDepth == 0){
            if(batchData.splitTracker[task.myReadId] <= 4 && batchData.h_numPossibleSplitColumnsPerAnchor[i] > 0){
                
                const int numCandidates = task.numRemainingCandidates;
                const int offset = batchData.h_numCandidatesPerAnchorPrefixSum[i];
                const int* const candidateShifts = batchData.h_alignment_shifts.data() + offset;
                const int* const candidateLengths = batchData.h_candidateSequencesLength.data() + offset;
                const auto* const alignmentFlags = batchData.h_alignment_best_alignment_flags.data() + offset;

                const unsigned int* const myCandidateSequencesData = &batchData.h_candidateSequencesData[offset * batchData.encodedSequencePitchInInts];

                task.candidateStrings.resize(batchData.decodedSequencePitchInBytes * numCandidates, '\0');

                //decode the candidates for msa
                for(int c = 0; c < task.numRemainingCandidates; c++){
                    SequenceHelpers::decode2BitSequence(
                        task.candidateStrings.data() + c * batchData.decodedSequencePitchInBytes,
                        myCandidateSequencesData + c * batchData.encodedSequencePitchInInts,
                        candidateLengths[c]
                    );

                    if(alignmentFlags[c] == BestAlignment_t::ReverseComplement){
                        SequenceHelpers::reverseComplementSequenceDecodedInplace(
                            task.candidateStrings.data() + c * batchData.decodedSequencePitchInBytes, 
                            candidateLengths[c]
                        );
                    }
                }

                auto possibleSplits = inspectColumnsRegionSplit(
                    possibleSplitColumns,
                    numPossibleSplitColumns,
                    task.currentAnchorLength, 
                    msaProps.lastColumn_excl - msaProps.firstColumn_incl,
                    msaProps.subjectColumnsBegin_incl,
                    numCandidates,
                    task.candidateStrings.data(),
                    batchData.decodedSequencePitchInBytes,
                    candidateShifts,
                    candidateLengths
                );

                if(possibleSplits.splits.size() > 1){
                    //nvtx::push_range("split msa", 8);
                    //auto& task = batchData.tasks[indexOfActiveTask];
                    #if 1
                    std::sort(
                        possibleSplits.splits.begin(), 
                        possibleSplits.splits.end(),
                        [](const auto& split1, const auto& split2){
                            //sort by size, descending
                            return split2.listOfCandidates.size() < split1.listOfCandidates.size();
                        }
                    );
                    #else
                    std::nth_element(
                        possibleSplits.splits.begin(), 
                        possibleSplits.splits.begin() + 1, 
                        possibleSplits.splits.end(),
                        [](const auto& split1, const auto& split2){
                            //sort by size, descending
                            return split2.listOfCandidates.size() < split1.listOfCandidates.size();
                        }
                    );
                    #endif

                    // std::cerr << "split[0] = ";
                    // for(auto x : possibleSplits.splits[0].listOfCandidates) std::cerr << x << " ";
                    // std::cerr << "\nsplit[1] = ";
                    // for(auto x : possibleSplits.splits[1].listOfCandidates) std::cerr << x << " ";
                    // std::cerr << "\n";

                    // auto printColumnInfo = [](const auto& x){
                    //     std::cerr << "(" << x.column << ", " << x.letter << ", " << x.ratio << ") ";
                    // };

                    // std::cerr << "columns[0] = ";
                    // for(auto x : possibleSplits.splits[0].columnInfo) printColumnInfo(x);
                    // std::cerr << "\ncolumns[1] = ";
                    // for(auto x : possibleSplits.splits[1].columnInfo) printColumnInfo(x);
                    // std::cerr << "\n";


                    //copy task's data from batchData into task

                    copyBatchDataIntoTask(task, i, batchData);

                    //create the separate shifts array
                    //create defaultweights, which is split in keepSelectedCandidates

                    task.candidateShifts.resize(task.alignments.size());
                    task.candidateOverlapWeights.resize(task.numRemainingCandidates);

                    for(int c = 0; c < task.numRemainingCandidates; c++){
                        task.candidateShifts[c] = task.alignments[c].shift;

                        task.candidateOverlapWeights[c] = calculateOverlapWeight(
                            task.currentAnchorLength, 
                            task.alignments[c].nOps,
                            task.alignments[c].overlap,
                            goodAlignmentProperties->maxErrorRate
                        );
                    }                       

                    task.dataIsAvailable = true;
                    //create a copy of task, and only keep candidates of first split
                    
                    auto taskCopy = task;
                    taskCopy.splitDepth++;

                    // std::cerr << "split\n";
                    // msa.print(std::cerr); 
                    // std::cerr << "\n into \n";

                    keepSelectedCandidates(taskCopy, possibleSplits.splits[0].listOfCandidates, -1);
                    const MultipleSequenceAlignment msaOfCopy = constructMsa(taskCopy, -1);

                    // msaOfCopy.print(std::cerr); 
                    // std::cerr << "\n and \n";

                    extendWithMsa(taskCopy, msaOfCopy.consensus.data(), msaOfCopy.consensus.size(), -1);

                    //only keep canddiates of second split
                    keepSelectedCandidates(task, possibleSplits.splits[1].listOfCandidates, -1); 
                    const MultipleSequenceAlignment newMsa = constructMsa(task, -1);

                    // newMsa.print(std::cerr); 
                    // std::cerr << "\n";

                    extendWithMsa(task, newMsa.consensus.data(), newMsa.consensus.size(), -1);

                    //if extension was not possible in task, replace task by task copy
                    if(task.abort && task.abortReason == AbortReason::MsaNotExtended){
                        //replace task by taskCopy
                        task = std::move(taskCopy);
                    }else if(!taskCopy.abort){
                        //if extension was possible in both task and taskCopy, taskCopy will be added to tasks and list of active tasks
                        newTaskIndices.emplace_back(batchData.tasks.size() + newTasksFromSplit.size());
                        newTasksFromSplit.emplace_back(std::move(taskCopy));

                        batchData.splitTracker[task.myReadId]++;


                    }
                    //nvtx::pop_range();                     
                }else{
                    extendWithMsa(task, consensus, consensusLength, indexOfActiveTask);
                }
            }else{
                extendWithMsa(task, consensus, consensusLength, indexOfActiveTask);
            }
#else 
            extendWithMsa(task, consensus, consensusLength, indexOfActiveTask);
#endif

        }

        //msaTimer.stop();

        nvtx::pop_range();

        if(newTasksFromSplit.size() > 0){
            //std::cerr << "Added " << newTasksFromSplit.size() << " tasks\n";
            batchData.tasks.insert(
                batchData.tasks.end(), 
                std::make_move_iterator(newTasksFromSplit.begin()), 
                std::make_move_iterator(newTasksFromSplit.end())
            );
            batchData.indicesOfActiveTasks.insert(
                batchData.indicesOfActiveTasks.end(), 
                newTaskIndices.begin(), 
                newTaskIndices.end()
            );
        }           

        /*
            update book-keeping of used candidates
        */  

        for(int i = 0; i < numActiveTasks; i++){
            auto& task = batchData.tasks[batchData.indicesOfActiveTasks[i]];

                                    
            {
                if(task.dataIsAvailable){
                    std::vector<read_number> tmp(task.allUsedCandidateReadIdPairs.size() + task.candidateReadIds.size());
                    auto tmp_end = std::merge(
                        task.allUsedCandidateReadIdPairs.begin(),
                        task.allUsedCandidateReadIdPairs.end(),
                        task.candidateReadIds.begin(),
                        task.candidateReadIds.end(),
                        tmp.begin()
                    );

                    tmp.erase(tmp_end, tmp.end());

                    std::swap(task.allUsedCandidateReadIdPairs, tmp);
                }else{
                    const int numCandidates = batchData.h_numCandidatesPerAnchor[i];
                    const int offset = batchData.h_numCandidatesPerAnchorPrefixSum[i];
                    const read_number* ids = &batchData.h_candidateReadIds[offset];

                    std::vector<read_number> tmp(task.allUsedCandidateReadIdPairs.size() + numCandidates);
                    auto tmp_end = std::merge(
                        task.allUsedCandidateReadIdPairs.begin(),
                        task.allUsedCandidateReadIdPairs.end(),
                        ids,
                        ids + numCandidates,
                        tmp.begin()
                    );

                    tmp.erase(tmp_end, tmp.end());

                    std::swap(task.allUsedCandidateReadIdPairs, tmp);
                }
            }

            // task.usedCandidateReadIdsPerIteration.emplace_back(std::move(task.candidateReadIds));
            // task.usedAlignmentsPerIteration.emplace_back(std::move(task.alignments));
            // task.usedAlignmentFlagsPerIteration.emplace_back(std::move(task.alignmentFlags));

            task.iteration++;
        }
        
        //update list of active task indices

        batchData.indicesOfActiveTasks.erase(
            std::remove_if(
                batchData.indicesOfActiveTasks.begin(), 
                batchData.indicesOfActiveTasks.end(),
                [&](int index){
                    return !batchData.tasks[index].isActive(insertSize, insertSizeStddev);
                }
            ),
            batchData.indicesOfActiveTasks.end()
        );
    }

    void removeUsedIdsAndMateIdsCPU(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const{
        batchData.h_candidateReadIds.resize(batchData.totalNumCandidates);
        batchData.h_numCandidatesPerAnchor.resize(batchData.numTasks);
        batchData.h_numCandidatesPerAnchorPrefixSum.resize(batchData.numTasks + 1);
        batchData.h_candidateReadIds2.resize(batchData.totalNumCandidates);
        batchData.h_numCandidatesPerAnchor2.resize(batchData.numTasks);
        batchData.h_numCandidatesPerAnchorPrefixSum2.resize(batchData.numTasks + 1);

        cudaMemcpyAsync(
            batchData.h_candidateReadIds.data(),
            batchData.d_candidateReadIds.data(),
            sizeof(read_number) * batchData.totalNumCandidates,
            D2H,
            firstStream
        );
        
        // cudaMemcpyAsync(
        //     batchData.h_numCandidatesPerAnchor.data(),
        //     batchData.d_numCandidatesPerAnchor.data(),
        //     sizeof(int) * (batchData.numTasks),
        //     D2H,
        //     firstStream
        // );
        // cudaMemcpyAsync(
        //     batchData.h_numCandidatesPerAnchorPrefixSum.data(),
        //     batchData.d_numCandidatesPerAnchorPrefixSum.data(),
        //     sizeof(int) * (batchData.numTasks + 1),
        //     D2H,
        //     firstStream
        // );
        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.d_numCandidatesPerAnchor.data()
            )),
            batchData.numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.h_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.h_numCandidatesPerAnchor.data()
            )),
            firstStream
        ); 
        cudaStreamSynchronize(firstStream);

        read_number* destids = batchData.h_candidateReadIds2.data();
        batchData.h_numCandidatesPerAnchorPrefixSum[0] = 0;
        batchData.h_numCandidatesPerAnchorPrefixSum2[0] = 0;
        batchData.numTasksWithMateRemoved = 0;

        /*
            Remove anchor ids and mate ids from candidates
        */

        for(int k = 0; k < batchData.numTasks; k++){
            auto& task = batchData.tasks[batchData.indicesOfActiveTasks[k]];

            int num = batchData.h_numCandidatesPerAnchor[k];
            const int offset = batchData.h_numCandidatesPerAnchorPrefixSum[k];
            read_number* myIds = batchData.h_candidateReadIds.data() + offset;

            auto readIdPos = std::lower_bound(
                myIds,                                            
                myIds + num,
                task.myReadId
            );

            if(readIdPos != myIds + num && *readIdPos == task.myReadId){
                std::copy(readIdPos + 1, myIds + num, readIdPos);
                num--;
            }

            if(task.pairedEnd){                

                //remove mate of input from candidate list
                auto mateReadIdPos = std::lower_bound(
                    myIds,                                            
                    myIds + num,
                    task.mateReadId
                );

                if(mateReadIdPos != myIds + num && *mateReadIdPos == task.mateReadId){
                    batchData.h_numCandidatesPerAnchor2[k] = num - 1;
                    destids = std::copy(myIds, mateReadIdPos, destids);
                    destids = std::copy(mateReadIdPos + 1, myIds + num, destids);
                    task.mateRemovedFromCandidates = true;
                    batchData.numTasksWithMateRemoved++;
                }else{
                    batchData.h_numCandidatesPerAnchor2[k] = num;
                    destids = std::copy(myIds, myIds + num, destids);
                    task.mateRemovedFromCandidates = false;
                }

                batchData.h_numCandidatesPerAnchorPrefixSum2[k+1] =
                    batchData.h_numCandidatesPerAnchorPrefixSum2[k] + batchData.h_numCandidatesPerAnchor2[k];
            }else{
                batchData.h_numCandidatesPerAnchor2[k] = num;
                destids = std::copy(myIds, myIds + num, destids);
                task.mateRemovedFromCandidates = false;
                batchData.h_numCandidatesPerAnchorPrefixSum2[k+1] =
                    batchData.h_numCandidatesPerAnchorPrefixSum2[k] + batchData.h_numCandidatesPerAnchor2[k];
            }
        }

        /*
            Remove candidate pairs which have already been used for extension
        */

        destids = batchData.h_candidateReadIds.data();

        for(int k = 0; k < batchData.numTasks; k++){
            auto& task = batchData.tasks[batchData.indicesOfActiveTasks[k]];

            const int num = batchData.h_numCandidatesPerAnchor2[k];
            const int offset = batchData.h_numCandidatesPerAnchorPrefixSum2[k];
            read_number* myIds = batchData.h_candidateReadIds2.data() + offset;

            std::vector<read_number> tmp(task.candidateReadIds.size());

            #if 0
            auto end = std::set_difference(
                myIds,
                myIds + num,
                task.allUsedCandidateReadIdPairs.begin(),
                task.allUsedCandidateReadIdPairs.end(),
                destids
            );
            #else
            auto end = std::copy(myIds, myIds + num, destids);
            #endif

            batchData.h_numCandidatesPerAnchor[k] = std::distance(destids, end);
            destids = end;

            batchData.h_numCandidatesPerAnchorPrefixSum[k+1] =
                    batchData.h_numCandidatesPerAnchorPrefixSum[k] + batchData.h_numCandidatesPerAnchor[k];
        }

        batchData.totalNumCandidates = batchData.h_numCandidatesPerAnchorPrefixSum[batchData.numTasks];

        //std::cerr << "old numTasksWithMateRemoved = " << batchData.numTasksWithMateRemoved << ", totalNumCandidates = " << batchData.totalNumCandidates << "\n";

        cudaMemcpyAsync(
            batchData.d_candidateReadIds.data(),
            batchData.h_candidateReadIds.data(),
            sizeof(read_number) * batchData.totalNumCandidates,
            H2D,
            firstStream
        );
        // cudaMemcpyAsync(
        //     batchData.d_numCandidatesPerAnchor.data(),
        //     batchData.h_numCandidatesPerAnchor.data(),
        //     sizeof(int) * (batchData.numTasks),
        //     H2D,
        //     firstStream
        // );
        // cudaMemcpyAsync(
        //     batchData.d_numCandidatesPerAnchorPrefixSum.data(),
        //     batchData.h_numCandidatesPerAnchorPrefixSum.data(),
        //     sizeof(int) * (batchData.numTasks + 1),
        //     H2D,
        //     firstStream
        // );

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.h_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.h_numCandidatesPerAnchor.data()
            )),
            batchData.numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.d_numCandidatesPerAnchor.data()
            )),
            firstStream
        ); 

        batchData.d_anchorIndicesOfCandidates.resize(batchData.totalNumCandidates);
        batchData.d_anchorIndicesOfCandidates2.resize(batchData.totalNumCandidates);
        batchData.h_anchorIndicesOfCandidates.resize(batchData.totalNumCandidates);

        for(int i = 0, sum = 0; i < batchData.numTasks; i++){
            std::fill(
                batchData.h_anchorIndicesOfCandidates.data() + sum,
                batchData.h_anchorIndicesOfCandidates.data() + sum + batchData.h_numCandidatesPerAnchor[i],
                i
            );
            sum += batchData.h_numCandidatesPerAnchor[i];
        }

        cudaMemcpyAsync(
            batchData.d_anchorIndicesOfCandidates.data(),
            batchData.h_anchorIndicesOfCandidates.data(),
            sizeof(int) * batchData.totalNumCandidates,
            H2D,
            firstStream
        ); CUERR;

        if(batchData.numTasksWithMateRemoved > 0){

            batchData.h_inputanchormatedata.resize(batchData.encodedSequencePitchInInts * batchData.numTasksWithMateRemoved);
            batchData.h_segmentIdsOfReadIds.resize(batchData.numTasksWithMateRemoved);

            unsigned int* destmatedata = batchData.h_inputanchormatedata.data();
            int* destids = batchData.h_segmentIdsOfReadIds.data();

            for(int k = 0; k < batchData.numTasks; k++){
                auto& task = batchData.tasks[batchData.indicesOfActiveTasks[k]];
                if(task.mateRemovedFromCandidates){
                    destmatedata = std::copy(
                        task.encodedMate.begin(),
                        task.encodedMate.end(),
                        destmatedata
                    );
                    *destids = k;
                    destids++;
                }
            }

            cudaMemcpyAsync(
                batchData.d_anchormatedata.data(),
                batchData.h_inputanchormatedata.data(),
                sizeof(unsigned int) * (batchData.encodedSequencePitchInInts * batchData.numTasksWithMateRemoved),
                H2D,
                firstStream
            );

            cudaMemcpyAsync(
                batchData.d_anchorIndicesWithRemovedMates.data(),
                batchData.h_segmentIdsOfReadIds.data(),
                sizeof(int) * (batchData.numTasksWithMateRemoved),
                H2D,
                firstStream
            );
        }

    }


    void removeUsedIdsAndMateIds(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const{
        //std::cerr << "\n" << batchData.totalNumCandidates << "\n";
        
        batchData.d_anchorIndicesOfCandidates.resize(batchData.totalNumCandidates);
        batchData.d_anchorIndicesOfCandidates2.resize(batchData.totalNumCandidates);
        batchData.d_flagscandidates.resize(batchData.totalNumCandidates);
        batchData.d_flagsanchors.resize(batchData.numTasks);
        batchData.d_candidateReadIds2.resize(batchData.totalNumCandidates);

        batchData.h_segmentIdsOfReadIds.resize(batchData.totalNumCandidates);

        batchData.h_numCandidatesPerAnchor.resize(batchData.numTasks);

        //determine required temp bytes for following cub calls, and allocate temp storage

        cudaError_t cubstatus = cudaSuccess;
        std::size_t cubBytes = 0;
        std::size_t cubBytes2 = 0;

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes2,
            batchData.d_candidateReadIds.data(),
            batchData.d_flagscandidates.data(),
            batchData.d_candidateReadIds2.data(),
            batchData.h_numCandidates.data(),
            batchData.totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);
        
        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr, 
            cubBytes2, 
            (int*)nullptr,
            (int*)nullptr,
            batchData.numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);
        
        cubstatus = cub::DeviceScan::InclusiveScan(
            nullptr, 
            cubBytes2, 
            batchData.d_anchorIndicesOfCandidates.data(), 
            batchData.d_anchorIndicesOfCandidates.data(), 
            cub::Max{},
            batchData.totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);
        
        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes2,
            thrust::make_counting_iterator(0),
            batchData.d_flagsanchors.data(),
            batchData.d_anchorIndicesWithRemovedMates.data(),
            batchData.h_numAnchorsWithRemovedMates.data(),
            batchData.numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);
        
        void* cubtempstorage; cubAllocator->DeviceAllocate((void**)&cubtempstorage, cubBytes, firstStream);   
        
        helpers::call_fill_kernel_async(batchData.d_flagscandidates.data(), batchData.d_flagscandidates.size(), false, firstStream);
        
        //flag candidates ids to remove because they are equal to anchor id or equal to mate id
        readextendergpukernels::flagCandidateIdsWhichAreEqualToAnchorOrMateKernel<<<4096, 128, 0, firstStream>>>(
            batchData.d_candidateReadIds.data(),
            batchData.d_anchorReadIds.data(),
            batchData.d_mateReadIds.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum.data(),
            batchData.d_numCandidatesPerAnchor.data(),
            batchData.d_flagscandidates.data(),
            batchData.d_flagsanchors.data(),
            batchData.d_numCandidatesPerAnchor2.data(),
            batchData.numTasks,
            batchData.pairedEnd
        );
        CUERR;

        cudaEventRecord(batchData.events[0], firstStream);
        cudaStreamWaitEvent(secondStream, batchData.events[0], 0); CUERR;

        cudaMemcpyAsync(
            batchData.h_numCandidatesPerAnchor.data(),
            batchData.d_numCandidatesPerAnchor2.data(),
            sizeof(int) * batchData.numTasks,
            D2H,
            secondStream
        ); CUERR;

        cudaEventRecord(batchData.events[0], secondStream);

        //determine task ids with removed mates

        assert(batchData.d_anchorIndicesWithRemovedMates.data() != nullptr);
        assert(batchData.h_numAnchorsWithRemovedMates.data() != nullptr);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtempstorage,
            cubBytes,
            thrust::make_counting_iterator(0),
            batchData.d_flagsanchors.data(),
            batchData.d_anchorIndicesWithRemovedMates.data(),
            batchData.h_numAnchorsWithRemovedMates.data(),
            batchData.numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        //copy selected candidate ids

        assert(batchData.d_candidateReadIds2.data() != nullptr);
        assert(batchData.h_numCandidates.data() != nullptr);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtempstorage,
            cubBytes,
            batchData.d_candidateReadIds.data(),
            batchData.d_flagscandidates.data(),
            batchData.d_candidateReadIds2.data(),
            batchData.h_numCandidates.data(),
            batchData.totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        cudaStreamSynchronize(firstStream); CUERR; //wait for h_numCandidates   and h_numAnchorsWithRemovedMates
        batchData.numTasksWithMateRemoved = *batchData.h_numAnchorsWithRemovedMates;
        batchData.totalNumCandidates = *batchData.h_numCandidates;

        //std::cerr << "new numTasksWithMateRemoved = " << batchData.numTasksWithMateRemoved << ", totalNumCandidates = " << batchData.totalNumCandidates << "\n";

        if(batchData.numTasksWithMateRemoved > 0){

            //copy mate sequence data of removed mates

            std::size_t cubtempstream2bytes = 0;
            cubstatus = cub::DeviceSelect::Flagged(
                nullptr,
                cubtempstream2bytes,
                batchData.d_inputanchormatedata.data(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    SequenceFlagMultiplier{batchData.d_flagsanchors.data(), int(batchData.encodedSequencePitchInInts)}
                ),
                batchData.d_anchormatedata.data(),
                thrust::make_discard_iterator(),
                batchData.numTasks * batchData.encodedSequencePitchInInts,
                secondStream
            );
            assert(cubstatus == cudaSuccess);
    
            void* cubtempstream2 = nullptr; cubAllocator->DeviceAllocate((void**)&cubtempstream2, cubtempstream2bytes, secondStream);
                
            assert(batchData.d_anchormatedata.data() != nullptr);

            cubstatus = cub::DeviceSelect::Flagged(
                cubtempstream2,
                cubtempstream2bytes,
                batchData.d_inputanchormatedata.data(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    SequenceFlagMultiplier{batchData.d_flagsanchors.data(), int(batchData.encodedSequencePitchInInts)}
                ),
                batchData.d_anchormatedata.data(),
                thrust::make_discard_iterator(),
                batchData.numTasks * batchData.encodedSequencePitchInInts,
                secondStream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtempstream2);
        }

        cudaEventSynchronize(batchData.events[0]); CUERR;

        for(int i = 0, sum = 0; i < batchData.numTasks; i++){
            std::fill(
                batchData.h_segmentIdsOfReadIds.data() + sum,
                batchData.h_segmentIdsOfReadIds.data() + sum + batchData.h_numCandidatesPerAnchor[i],
                i
            );
            sum += batchData.h_numCandidatesPerAnchor[i];
        }

        cudaMemcpyAsync(
            batchData.d_anchorIndicesOfCandidates.data(),
            batchData.h_segmentIdsOfReadIds.data(),
            sizeof(int) * batchData.totalNumCandidates,
            H2D,
            firstStream
        ); CUERR;

        

        // //compute prefix sum of number of candidates per anchor
        // helpers::call_set_kernel_async(batchData.d_numCandidatesPerAnchorPrefixSum2.data(), 0, 0, firstStream);

        // cubstatus = cub::DeviceScan::InclusiveSum(
        //     cubtempstorage, 
        //     cubBytes, 
        //     batchData.d_numCandidatesPerAnchor2.data(), 
        //     batchData.d_numCandidatesPerAnchorPrefixSum2.data() + 1, 
        //     batchData.numTasks,
        //     firstStream
        // );
        // assert(cubstatus == cudaSuccess);

        // //compute segment ids for candidate read ids

        // helpers::call_fill_kernel_async(batchData.d_anchorIndicesOfCandidates.data(), batchData.totalNumCandidates, 0, firstStream);

        // setFirstSegmentIdsKernel<<<SDIV(batchData.numTasks, 256), 256, 0, firstStream>>>(
        //     batchData.d_numCandidatesPerAnchor2.data(),
        //     batchData.d_anchorIndicesOfCandidates.data(),
        //     batchData.d_numCandidatesPerAnchorPrefixSum2.data(),
        //     batchData.numTasks
        // );

        // cubstatus = cub::DeviceScan::InclusiveScan(
        //     cubtempstorage, 
        //     cubBytes, 
        //     batchData.d_anchorIndicesOfCandidates.data(), 
        //     batchData.d_anchorIndicesOfCandidates.data(), 
        //     cub::Max{},
        //     batchData.totalNumCandidates,
        //     firstStream
        // );
        // assert(cubstatus == cudaSuccess);
   

        ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, firstStream);

        //cudaStreamWaitEvent(firstStream, batchData.events[0], 0); CUERR;

        
        //compute segmented set difference between candidate read ids and used candidate read ids
        auto d_candidateReadIds_end = GpuSegmentedSetOperation{}.difference(
            thrustCachingAllocator1,
            batchData.d_candidateReadIds2.data(),
            batchData.d_numCandidatesPerAnchor2.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum2.data(),
            batchData.d_anchorIndicesOfCandidates.data(),
            batchData.totalNumCandidates,
            batchData.d_usedReadIds.data(),
            batchData.d_numUsedReadIdsPerAnchor.data(),
            batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(),
            batchData.d_segmentIdsOfUsedReadIds.data(),
            batchData.totalNumberOfUsedIds,
            batchData.numTasks,        
            batchData.d_candidateReadIds.data(),
            batchData.d_numCandidatesPerAnchor.data(),
            batchData.d_anchorIndicesOfCandidates2.data(),
            firstStream
        );

        std::swap(batchData.d_anchorIndicesOfCandidates, batchData.d_anchorIndicesOfCandidates2);

        batchData.totalNumCandidates = std::distance(batchData.d_candidateReadIds.data(), d_candidateReadIds_end);

        //compute prefix sum of new segment sizes
    
        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtempstorage, 
            cubBytes, 
            batchData.d_numCandidatesPerAnchor.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        if(batchData.numTasksWithMateRemoved > 0){
            cudaEventRecord(batchData.events[0], secondStream);
            cudaStreamWaitEvent(firstStream, batchData.events[0], 0); CUERR;
        }

        cubAllocator->DeviceFree(cubtempstorage);
    }

    void loadCandidateSequenceData(BatchData& batchData, cudaStream_t stream) const{

        const int totalNumCandidates = batchData.totalNumCandidates;

        batchData.d_candidateSequencesLength.resize(batchData.totalNumCandidates);
        batchData.d_candidateSequencesData.resize(batchData.encodedSequencePitchInInts * batchData.totalNumCandidates);

        gpuReadStorage->gatherSequences(
            readStorageHandle,
            batchData.d_candidateSequencesData.get(),
            batchData.encodedSequencePitchInInts,
            batchData.h_candidateReadIds.get(),
            batchData.d_candidateReadIds.get(), //device accessible
            totalNumCandidates,
            stream
        );

        gpuReadStorage->gatherSequenceLengths(
            readStorageHandle,
            batchData.d_candidateSequencesLength.get(),
            batchData.d_candidateReadIds.get(),
            totalNumCandidates,
            stream
        );
    }

    void eraseDataOfRemovedMates(BatchData& batchData, cudaStream_t stream) const{

        const int totalNumCandidates = batchData.totalNumCandidates;

        batchData.d_intbuffercandidates.resize(batchData.totalNumCandidates);
        batchData.d_flagscandidates.resize(batchData.totalNumCandidates);

        batchData.d_candidateSequencesLength2.resize(batchData.totalNumCandidates);
        batchData.d_candidateSequencesData2.resize(batchData.encodedSequencePitchInInts * batchData.totalNumCandidates);
        batchData.d_candidateReadIds2.resize(batchData.totalNumCandidates);

        constexpr int groupsize = 32;
        constexpr int blocksize = 128;
        constexpr int groupsperblock = blocksize / groupsize;
        dim3 block(blocksize,1,1);
        dim3 grid(SDIV(batchData.numTasksWithMateRemoved * groupsize, blocksize), 1, 1);
        const std::size_t smembytes = sizeof(unsigned int) * groupsperblock * batchData.encodedSequencePitchInInts;

        bool* const d_keepflags = batchData.d_flagscandidates.data();

        helpers::call_fill_kernel_async(d_keepflags, totalNumCandidates, true, stream);

        readextendergpukernels::filtermatekernel<blocksize,groupsize><<<grid, block, smembytes, stream>>>(
            batchData.d_anchormatedata.data(),
            batchData.d_candidateSequencesData.data(),
            batchData.encodedSequencePitchInInts,
            batchData.d_numCandidatesPerAnchor.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum.data(),
            batchData.d_anchorIndicesWithRemovedMates.data(),
            batchData.numTasksWithMateRemoved,
            d_keepflags
        ); CUERR;

        // auto negate = [] __device__ (bool b){
        //     return !b;
        // };

        // cub::TransformInputIterator<bool, decltype(negate), bool*> d_keepflags(batchData.d_flagscandidates.data(), negate);

        std::size_t requiredCubSize = 0;
        std::size_t requiredCubSize1 = 0;
        std::size_t requiredCubSize2 = 0;
        cudaError_t cubstatus = cub::DeviceScan::ExclusiveSum(
            nullptr,
            requiredCubSize1,
            d_keepflags, 
            batchData.d_intbuffercandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr,
            requiredCubSize2,
            batchData.d_numCandidatesPerAnchor.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        requiredCubSize = std::max(requiredCubSize1, requiredCubSize2);

        void* cubtemp; cubAllocator->DeviceAllocate((void**)&cubtemp, requiredCubSize, stream);

        cubstatus = cub::DeviceScan::ExclusiveSum(
            cubtemp,
            requiredCubSize,
            d_keepflags, 
            batchData.d_intbuffercandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        helpers::lambda_kernel<<<4096, 128, 0, stream>>>(
            [
                numTasks = batchData.numTasks,
                encodedSequencePitchInInts = batchData.encodedSequencePitchInInts,
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                d_keepflags,
                d_outputpositions = batchData.d_intbuffercandidates.data(),
                d_candidateReadIds = batchData.d_candidateReadIds.data(),
                d_candidateSequencesLength = batchData.d_candidateSequencesLength.data(),
                d_candidateSequencesData = batchData.d_candidateSequencesData.data(),
                d_anchorIndicesOfCandidates = batchData.d_anchorIndicesOfCandidates.data(),
                d_candidateReadIdsOut = batchData.d_candidateReadIds2.data(),
                d_candidateSequencesLengthOut = batchData.d_candidateSequencesLength2.data(),
                d_candidateSequencesDataOut = batchData.d_candidateSequencesData2.data(),
                d_anchorIndicesOfCandidatesOut = batchData.d_anchorIndicesOfCandidates2.data()
            ] __device__ (){

                constexpr int elementsPerIteration = 128;

                using BlockReduce = cub::BlockReduce<int, elementsPerIteration>;
                __shared__ typename BlockReduce::TempStorage temp_storage;

                for(int t = blockIdx.x; t < numTasks; t += gridDim.x){
                    const int numCandidates = d_numCandidatesPerAnchor[t];
                    const int inputOffset = d_numCandidatesPerAnchorPrefixSum[t];

                    int numSelected = 0;

                    for(int i = threadIdx.x; i < numCandidates; i += blockDim.x){
                        if(d_keepflags[inputOffset + i]){
                            const int outputLocation = d_outputpositions[inputOffset + i];

                            d_candidateReadIdsOut[outputLocation] = d_candidateReadIds[inputOffset + i];
                            d_candidateSequencesLengthOut[outputLocation] = d_candidateSequencesLength[inputOffset + i];
                            d_anchorIndicesOfCandidatesOut[outputLocation] = d_anchorIndicesOfCandidates[inputOffset + i];

                            numSelected++;
                        }
                    }

                    for(int i = threadIdx.x; i < numCandidates * encodedSequencePitchInInts; i += blockDim.x){
                        const int which = i / encodedSequencePitchInInts;
                        const int what = i % encodedSequencePitchInInts;

                        if(d_keepflags[inputOffset + which]){
                            d_candidateSequencesDataOut[d_outputpositions[inputOffset + which] * encodedSequencePitchInInts + what] = d_candidateSequencesData[(inputOffset + which) * encodedSequencePitchInInts + what];
                        }
                    }

                    numSelected = BlockReduce(temp_storage).Sum(numSelected);
                    __syncthreads();
                    
                    if(threadIdx.x == 0){
                        if(numSelected != numCandidates){
                            assert(numSelected < numCandidates);
                            d_numCandidatesPerAnchor[t] = numSelected;
                            //printf("task %d, removed %d\n", t, numCandidates - numSelected);
                        }
                    }

                }
            }
        ); CUERR;

        //update prefix sum
        
        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtemp,
            requiredCubSize,
            batchData.d_numCandidatesPerAnchor.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);       

        cubAllocator->DeviceFree(cubtemp);

        std::swap(batchData.d_candidateReadIds2, batchData.d_candidateReadIds);
        std::swap(batchData.d_candidateSequencesLength2, batchData.d_candidateSequencesLength);
        std::swap(batchData.d_candidateSequencesData2, batchData.d_candidateSequencesData);
        std::swap(batchData.d_anchorIndicesOfCandidates2, batchData.d_anchorIndicesOfCandidates);

        cudaMemcpyAsync(
            &batchData.totalNumCandidates,
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + batchData.numTasks,
            sizeof(int),
            D2H,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;
       
    }

    void calculateAlignments(BatchData& batchData, cudaStream_t stream) const{

        // cudaStreamSynchronize(stream); CUERR; //DEBUG
        // std::vector<int> vec1(batchData.d_numCandidatesPerAnchor.size());
        // cudaMemcpyAsync(vec1.data(), batchData.d_numCandidatesPerAnchor.data(), batchData.d_numCandidatesPerAnchor.sizeInBytes(), D2H, stream); CUERR;
        // cudaStreamSynchronize(stream); CUERR; //DEBUG
        // std::cerr << "Iteration: " << batchData.tasks[0].iteration << "\n";
        // for(int i = 0; i < batchData.numTasks; i++){
        //     std::cerr << vec1[i] << " ";
        // }
        // std::cerr << "\n";

        batchData.d_alignment_overlaps.resize(batchData.totalNumCandidates);
        batchData.d_alignment_shifts.resize(batchData.totalNumCandidates);
        batchData.d_alignment_nOps.resize(batchData.totalNumCandidates);
        batchData.d_alignment_isValid.resize(batchData.totalNumCandidates);
        batchData.d_alignment_best_alignment_flags.resize(batchData.totalNumCandidates);
        
        batchData.h_numAnchors[0] = batchData.numTasks;

        const bool* const d_anchorContainsN = nullptr;
        const bool* const d_candidateContainsN = nullptr;
        const bool removeAmbiguousAnchors = false;
        const bool removeAmbiguousCandidates = false;
        const int maxNumAnchors = batchData.numTasks;
        const int maxNumCandidates = batchData.totalNumCandidates; //this does not need to be exact, but it must be >= batchData.d_numCandidatesPerAnchorPrefixSum[batchData.numTasks]
        const int maximumSequenceLength = batchData.encodedSequencePitchInInts * 16;
        const int encodedSequencePitchInInts2Bit = batchData.encodedSequencePitchInInts;
        const int min_overlap = goodAlignmentProperties->min_overlap;
        const float maxErrorRate = goodAlignmentProperties->maxErrorRate;
        const float min_overlap_ratio = goodAlignmentProperties->min_overlap_ratio;
        const float estimatedNucleotideErrorRate = correctionOptions->estimatedErrorrate;

        auto callAlignmentKernel = [&](void* d_tempstorage, size_t& tempstoragebytes){

            call_popcount_rightshifted_hamming_distance_kernel_async(
                d_tempstorage,
                tempstoragebytes,
                batchData.d_alignment_overlaps.get(),
                batchData.d_alignment_shifts.get(),
                batchData.d_alignment_nOps.get(),
                batchData.d_alignment_isValid.get(),
                batchData.d_alignment_best_alignment_flags.get(),
                batchData.d_subjectSequencesData.get(),
                batchData.d_candidateSequencesData.get(),
                batchData.d_anchorSequencesLength.get(),
                batchData.d_candidateSequencesLength.get(),
                batchData.d_numCandidatesPerAnchorPrefixSum.get(),
                batchData.d_numCandidatesPerAnchor.get(),
                batchData.d_anchorIndicesOfCandidates.get(),
                batchData.h_numAnchors.get(),
                &batchData.d_numCandidatesPerAnchorPrefixSum[batchData.numTasks],
                d_anchorContainsN,
                removeAmbiguousAnchors,
                d_candidateContainsN,
                removeAmbiguousCandidates,
                maxNumAnchors,
                maxNumCandidates,
                maximumSequenceLength,
                encodedSequencePitchInInts2Bit,
                min_overlap,
                maxErrorRate,
                min_overlap_ratio,
                estimatedNucleotideErrorRate,
                stream,
                kernelLaunchHandle
            );
        };

        size_t tempstoragebytes = 0;
        callAlignmentKernel(nullptr, tempstoragebytes);

        batchData.d_tempstorage.resize(tempstoragebytes);

        callAlignmentKernel(batchData.d_tempstorage.get(), tempstoragebytes);
    }

    void filterAlignments(BatchData& batchData, cudaStream_t stream) const{

        const int totalNumCandidates = batchData.totalNumCandidates;
        const int numAnchors = batchData.numTasks;

        batchData.d_alignment_overlaps2.resize(batchData.totalNumCandidates);
        batchData.d_alignment_shifts2.resize(batchData.totalNumCandidates);
        batchData.d_alignment_nOps2.resize(batchData.totalNumCandidates);
        batchData.d_alignment_best_alignment_flags2.resize(batchData.totalNumCandidates);
        batchData.d_candidateSequencesLength2.resize(batchData.totalNumCandidates);
        batchData.d_candidateSequencesData2.resize(batchData.encodedSequencePitchInInts * batchData.totalNumCandidates);
        batchData.d_candidateReadIds2.resize(batchData.totalNumCandidates);

        batchData.d_flagscandidates.resize(batchData.totalNumCandidates);

        helpers::call_fill_kernel_async(batchData.d_flagscandidates.data(), batchData.d_flagscandidates.size(), true, stream);



        bool* const d_keepflags = batchData.d_flagscandidates.data();

        dim3 block(128,1,1);
        dim3 grid(numAnchors, 1, 1);

        //filter alignments of candidates. d_keepflags[i] will be set to false if candidate[i] should be removed
        //batchData.d_numCandidatesPerAnchor2[i] contains new number of candidates for anchor i
        helpers::lambda_kernel<<<grid, block, 0, stream>>>(
            [
                d_alignment_best_alignment_flags = batchData.d_alignment_best_alignment_flags.data(),
                d_alignment_shifts = batchData.d_alignment_shifts.data(),
                d_alignment_overlaps = batchData.d_alignment_overlaps.data(),
                d_anchorSequencesLength = batchData.d_anchorSequencesLength.data(),
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchor2 = batchData.d_numCandidatesPerAnchor2.data(),
                d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                d_keepflags,
                min_overlap_ratio = goodAlignmentProperties->min_overlap_ratio,
                numAnchors
            ] __device__ (){

                using BlockReduceFloat = cub::BlockReduce<float, 128>;
                using BlockReduceInt = cub::BlockReduce<int, 128>;

                __shared__ union {
                    typename BlockReduceFloat::TempStorage floatreduce;
                    typename BlockReduceInt::TempStorage intreduce;
                } cubtemp;

                __shared__ int intbroadcast;
                __shared__ float floatbroadcast;

                for(int a = blockIdx.x; a < numAnchors; a += gridDim.x){
                    const int num = d_numCandidatesPerAnchor[a];
                    const int offset = d_numCandidatesPerAnchorPrefixSum[a];
                    const float anchorLength = d_anchorSequencesLength[a];
                    int removed = 0;

                    int threadReducedGoodAlignmentExists = 0;
                    float threadReducedRelativeOverlapThreshold = 0.0f;

                    //loop over candidates to compute relative overlap threshold

                    for(int c = threadIdx.x; c < num; c += blockDim.x){
                        const auto alignmentflag = d_alignment_best_alignment_flags[offset + c];
                        const int shift = d_alignment_shifts[offset + c];

                        if(alignmentflag != BestAlignment_t::None && shift >= 0){
                            const float overlap = d_alignment_overlaps[offset + c];                            
                            const float relativeOverlap = overlap / anchorLength;
                            
                            if(relativeOverlap < 1.0f && fgeq(relativeOverlap, min_overlap_ratio)){
                                threadReducedGoodAlignmentExists = 1;
                                const float tmp = floorf(relativeOverlap * 10.0f) / 10.0f;
                                threadReducedRelativeOverlapThreshold = fmaxf(threadReducedRelativeOverlapThreshold, tmp);
                            }
                        }else{
                            //remove alignment with negative shift or bad alignments
                            d_keepflags[offset + c] = false;
                            removed++;
                        }                       
                    }

                    int blockreducedGoodAlignmentExists = BlockReduceInt(cubtemp.intreduce)
                        .Sum(threadReducedGoodAlignmentExists);
                    if(threadIdx.x == 0){
                        intbroadcast = blockreducedGoodAlignmentExists;
                        //printf("task %d good: %d\n", a, blockreducedGoodAlignmentExists);
                    }
                    __syncthreads();

                    blockreducedGoodAlignmentExists = intbroadcast;

                    if(blockreducedGoodAlignmentExists > 0){
                        float blockreducedRelativeOverlapThreshold = BlockReduceFloat(cubtemp.floatreduce)
                            .Reduce(threadReducedRelativeOverlapThreshold, cub::Max());
                        if(threadIdx.x == 0){
                            floatbroadcast = blockreducedRelativeOverlapThreshold;
                            //printf("task %d thresh: %f\n", a, blockreducedRelativeOverlapThreshold);
                        }
                        __syncthreads();

                        blockreducedRelativeOverlapThreshold = floatbroadcast;

                        // loop over candidates and remove those with an alignment overlap threshold smaller than the computed threshold
                        for(int c = threadIdx.x; c < num; c += blockDim.x){
    
                            if(d_keepflags[offset + c]){
                                const float overlap = d_alignment_overlaps[offset + c];                            
                                const float relativeOverlap = overlap / anchorLength;                 
    
                                if(!fgeq(relativeOverlap, blockreducedRelativeOverlapThreshold)){
                                    d_keepflags[offset + c] = false;
                                    removed++;
                                }
                            }
                        }
                    }else{
                        //NOOP.
                        //if no good alignment exists, no candidate is removed. we will try to work with the not-so-good alignments
                    }

                    removed = BlockReduceInt(cubtemp.intreduce).Sum(removed);

                    if(threadIdx.x == 0){
                        d_numCandidatesPerAnchor2[a] = num - removed;
                    }
                    __syncthreads();
                }
            }
        ); CUERR;

        // cudaDeviceSynchronize(); CUERR; //DEBUG

        // std::vector<BestAlignment_t> vec3(batchData.d_alignment_best_alignment_flags.size());
        // std::vector<int> vec4(batchData.d_alignment_shifts.size());
        // std::vector<int> vec5(batchData.d_alignment_best_alignment_flags.size());

        // cudaMemcpyAsync(vec3.data(), batchData.d_alignment_best_alignment_flags.data(), batchData.d_alignment_best_alignment_flags.sizeInBytes(), D2H, stream); CUERR;
        // cudaMemcpyAsync(vec4.data(), batchData.d_alignment_shifts.data(), batchData.d_alignment_shifts.sizeInBytes(), D2H, stream); CUERR;
        // cudaMemcpyAsync(vec5.data(), batchData.d_alignment_overlaps.data(), batchData.d_alignment_overlaps.sizeInBytes(), D2H, stream); CUERR;

        // cudaDeviceSynchronize(); CUERR; //DEBUG

        // for(int k = 0; k < batchData.totalNumCandidates; k++){
        //     std::cerr << int(vec3[k]) << " " << vec4[k] << " " << vec5[k] << "\n";
        // }
        // std::cerr << "\n";

        // cudaStreamSynchronize(stream); CUERR; //DEBUG
        // std::vector<int> vec2(batchData.d_numCandidatesPerAnchor2.size());
        // cudaMemcpyAsync(vec2.data(), batchData.d_numCandidatesPerAnchor2.data(), batchData.d_numCandidatesPerAnchor2.sizeInBytes(), D2H, stream); CUERR;
        // cudaStreamSynchronize(stream); CUERR; //DEBUG
        // for(int i = 0; i < batchData.numTasks; i++){
        //     std::cerr << vec2[i] << " ";
        // }
        // std::cerr << "\n";

        // std::exit(0);


        //setup cub 
        auto d_zip_input = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_alignment_nOps.data(),
                batchData.d_alignment_overlaps.data(),
                batchData.d_alignment_shifts.data(),
                batchData.d_alignment_best_alignment_flags.data(),
                batchData.d_candidateReadIds.data(),
                batchData.d_candidateSequencesLength.data()
            )
        );

        assert(batchData.d_alignment_nOps2.data() != nullptr);
        assert(batchData.d_alignment_overlaps2.data() != nullptr);
        assert(batchData.d_alignment_shifts2.data() != nullptr);
        assert(batchData.d_alignment_best_alignment_flags2.data() != nullptr);
        assert(batchData.d_candidateReadIds2.data() != nullptr);
        assert(batchData.d_candidateSequencesLength2.data() != nullptr);

        auto d_zip_output = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_alignment_nOps2.data(),
                batchData.d_alignment_overlaps2.data(),
                batchData.d_alignment_shifts2.data(),
                batchData.d_alignment_best_alignment_flags2.data(),
                batchData.d_candidateReadIds2.data(),
                batchData.d_candidateSequencesLength2.data()
            )
        );

        std::size_t requiredCubSize1 = 0;
        cudaError_t cubstatus = cub::DeviceSelect::Flagged(
            nullptr, 
            requiredCubSize1, 
            d_zip_input, 
            d_keepflags, 
            d_zip_output, 
            batchData.d_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        std::size_t requiredCubSize2 = 0;
        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            requiredCubSize2,
            batchData.d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                SequenceFlagMultiplier{d_keepflags, int(batchData.encodedSequencePitchInInts)}
            ),
            batchData.d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            totalNumCandidates * batchData.encodedSequencePitchInInts,
            stream
        );
        assert(cubstatus == cudaSuccess);

        std::size_t requiredCubSize3 = 0;
        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr,
            requiredCubSize3,
            batchData.d_numCandidatesPerAnchor2.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        std::size_t requiredCubSize = std::max(std::max(requiredCubSize1, requiredCubSize2), requiredCubSize3);
        void* cubtemp; cubAllocator->DeviceAllocate((void**)&cubtemp, requiredCubSize, stream);

        //compact zip data
        assert(batchData.d_numCandidates.data() != nullptr);
        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp, 
            requiredCubSize, 
            d_zip_input, 
            d_keepflags, 
            d_zip_output, 
            batchData.d_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        //compact sequence data.

        assert(batchData.d_candidateSequencesData2.data() != nullptr);
        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            requiredCubSize,
            batchData.d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                SequenceFlagMultiplier{d_keepflags, int(batchData.encodedSequencePitchInInts)}
            ),
            batchData.d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            totalNumCandidates * batchData.encodedSequencePitchInInts,
            stream
        );
        assert(cubstatus == cudaSuccess);

        //update prefix sum
        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtemp, 
            requiredCubSize, 
            batchData.d_numCandidatesPerAnchor2.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);


        cubAllocator->DeviceFree(cubtemp);

        std::swap(batchData.d_alignment_nOps2, batchData.d_alignment_nOps);
        std::swap(batchData.d_alignment_overlaps2, batchData.d_alignment_overlaps);
        std::swap(batchData.d_alignment_shifts2, batchData.d_alignment_shifts);
        std::swap(batchData.d_alignment_best_alignment_flags2, batchData.d_alignment_best_alignment_flags);
        std::swap(batchData.d_candidateReadIds2, batchData.d_candidateReadIds);
        std::swap(batchData.d_candidateSequencesLength2, batchData.d_candidateSequencesLength);
        std::swap(batchData.d_numCandidatesPerAnchor2, batchData.d_numCandidatesPerAnchor);
        std::swap(batchData.d_candidateSequencesData2, batchData.d_candidateSequencesData);

        cudaMemcpyAsync(
            &batchData.totalNumCandidates,
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + batchData.numTasks,
            sizeof(int),
            D2H,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;

        // cudaStreamSynchronize(stream); CUERR; //DEBUG
        // std::vector<int> vec1(batchData.d_numCandidatesPerAnchor.size());
        // cudaMemcpyAsync(vec1.data(), batchData.d_numCandidatesPerAnchor.data(), batchData.d_numCandidatesPerAnchor.sizeInBytes(), D2H, stream); CUERR;
        // cudaStreamSynchronize(stream); CUERR; //DEBUG
        // for(int i = 0; i < batchData.numTasks; i++){
        //     std::cerr << vec1[i] << " ";
        // }
        // std::cerr << "\n";
    }

    void computeMSAs(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const{
        batchData.d_consensus.resize(batchData.numTasks * batchData.msaColumnPitchInElements);
        batchData.d_support.resize(batchData.numTasks * batchData.msaColumnPitchInElements);
        batchData.d_coverage.resize(batchData.numTasks * batchData.msaColumnPitchInElements);
        batchData.d_origWeights.resize(batchData.numTasks * batchData.msaColumnPitchInElements);
        batchData.d_origCoverages.resize(batchData.numTasks * batchData.msaColumnPitchInElements);
        batchData.d_msa_column_properties.resize(batchData.numTasks);
        batchData.d_counts.resize(batchData.numTasks * 4 * batchData.msaColumnPitchInElements);
        batchData.d_weights.resize(batchData.numTasks * 4 * batchData.msaColumnPitchInElements);

        batchData.d_intbuffercandidates.resize(batchData.totalNumCandidates);
        batchData.d_indexlist1.resize(batchData.totalNumCandidates);
        batchData.d_numCandidatesPerAnchor2.resize(batchData.numTasks);
        batchData.d_flagscandidates.resize(batchData.totalNumCandidates);

        int* const indices1 = batchData.d_intbuffercandidates.data();
        int* const indices2 = batchData.d_indexlist1.data();

        helpers::lambda_kernel<<<batchData.numTasks, 128, 0, firstStream>>>(
            [
                indices1,
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.get(),
                d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.get()
            ] __device__ (){
                const int num = d_numCandidatesPerAnchor[blockIdx.x];
                const int offset = d_numCandidatesPerAnchorPrefixSum[blockIdx.x];
                
                for(int i = threadIdx.x; i < num; i += blockDim.x){
                    indices1[offset + i] = i;
                }
            }
        );

        gpu::GPUMultiMSA multiMSA;

        *batchData.h_numAnchors = batchData.numTasks;

        multiMSA.numMSAs = batchData.numTasks;
        multiMSA.columnPitchInElements = batchData.msaColumnPitchInElements;
        multiMSA.counts = batchData.d_counts.get();
        multiMSA.weights = batchData.d_weights.get();
        multiMSA.coverages = batchData.d_coverage.get();
        multiMSA.consensus = batchData.d_consensus.get();
        multiMSA.support = batchData.d_support.get();
        multiMSA.origWeights = batchData.d_origWeights.get();
        multiMSA.origCoverages = batchData.d_origCoverages.get();
        multiMSA.columnProperties = batchData.d_msa_column_properties.get();

        callConstructMultipleSequenceAlignmentsKernel_async(
            multiMSA,
            batchData.d_alignment_overlaps.get(),
            batchData.d_alignment_shifts.get(),
            batchData.d_alignment_nOps.get(),
            batchData.d_alignment_best_alignment_flags.get(),
            batchData.d_anchorSequencesLength.get(),
            batchData.d_candidateSequencesLength.get(),
            indices1, //d_indices,
            batchData.d_numCandidatesPerAnchor.get(),
            batchData.d_numCandidatesPerAnchorPrefixSum.get(),
            batchData.d_subjectSequencesData.get(),
            batchData.d_candidateSequencesData.get(),
            nullptr, //d_anchor_qualities.get(),
            nullptr, //d_candidate_qualities.get(),
            batchData.h_numAnchors.get(), //d_numAnchors
            goodAlignmentProperties->maxErrorRate,
            batchData.numTasks,
            batchData.totalNumCandidates,
            false, //correctionOptions->useQualityScores,
            batchData.encodedSequencePitchInInts,
            batchData.qualityPitchInBytes,
            firstStream,
            kernelLaunchHandle
        );

        //refine msa
        bool* const d_shouldBeKept = (bool*)batchData.d_flagscandidates.get();

        callMsaCandidateRefinementKernel_multiiter_async(
            indices2,
            batchData.d_numCandidatesPerAnchor2.data(),
            batchData.d_numCandidates2.get(),
            multiMSA,
            batchData.d_alignment_best_alignment_flags.get(),
            batchData.d_alignment_shifts.get(),
            batchData.d_alignment_nOps.get(),
            batchData.d_alignment_overlaps.get(),
            batchData.d_subjectSequencesData.get(),
            batchData.d_candidateSequencesData.get(),
            batchData.d_anchorSequencesLength.get(),
            batchData.d_candidateSequencesLength.get(),
            nullptr, //d_anchor_qualities.get(),
            nullptr, //d_candidate_qualities.get(),
            d_shouldBeKept,
            batchData.d_numCandidatesPerAnchorPrefixSum.get(),
            batchData.h_numAnchors.get(),
            goodAlignmentProperties->maxErrorRate,
            batchData.numTasks,
            batchData.totalNumCandidates,
            false, //correctionOptions->useQualityScores,
            batchData.encodedSequencePitchInInts,
            batchData.qualityPitchInBytes,
            indices1, //d_indices,
            batchData.d_numCandidatesPerAnchor.get(),
            correctionOptions->estimatedCoverage,
            getNumRefinementIterations(),
            firstStream,
            kernelLaunchHandle
        );

        cudaEventRecord(batchData.events[0], firstStream); CUERR;
        cudaStreamWaitEvent(secondStream, batchData.events[0], 0); CUERR;

        cudaMemcpyAsync(
            batchData.h_numCandidates.data(),
            batchData.d_numCandidates.data(),
            sizeof(int),
            D2H,
            secondStream
        ); CUERR;

        cudaEventRecord(batchData.events[0], secondStream); CUERR;

        bool cubdebugsync = false;
        cudaError_t cubstatus = cudaSuccess;

        //allocate cub storage
        std::size_t cubBytes = 0;
        std::size_t cubBytes2 = 0;
        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr,
            cubBytes,
            batchData.d_numCandidatesPerAnchor2.get(), 
            batchData.d_numCandidatesPerAnchorPrefixSum2.get() + 1, 
            batchData.numTasks, 
            firstStream,
            cubdebugsync
        );
        assert(cubstatus == cudaSuccess);

        auto in_zipped_begin = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_candidateReadIds.data(),
                batchData.d_candidateSequencesLength.data(),
                batchData.d_alignment_overlaps.data(),
                batchData.d_alignment_shifts.data(),
                batchData.d_alignment_nOps.data(),
                batchData.d_alignment_best_alignment_flags.data()
            )
        );

        auto out_zipped_begin = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_candidateReadIds2.data(),
                batchData.d_candidateSequencesLength2.data(),
                batchData.d_alignment_overlaps2.data(),
                batchData.d_alignment_shifts2.data(),
                batchData.d_alignment_nOps2.data(),
                batchData.d_alignment_best_alignment_flags2.data()
            )
        );

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes2,
            in_zipped_begin,
            batchData.d_flagscandidates.data(),
            out_zipped_begin,
            thrust::make_discard_iterator(),
            batchData.totalNumCandidates,
            firstStream,
            cubdebugsync
        );

        assert(cubstatus == cudaSuccess);
        cubBytes = std::max(cubBytes, cubBytes2);

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes2,
            batchData.d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                SequenceFlagMultiplier{batchData.d_flagscandidates.data(), int(batchData.encodedSequencePitchInInts)}
            ),
            batchData.d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            batchData.totalNumCandidates * batchData.encodedSequencePitchInInts,
            firstStream,
            cubdebugsync
        );

        assert(cubstatus == cudaSuccess);
        cubBytes = std::max(cubBytes, cubBytes2);

        void* cubtemp; cubAllocator->DeviceAllocate((void**)&cubtemp, cubBytes, firstStream);

        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtemp,
            cubBytes,
            batchData.d_numCandidatesPerAnchor2.get(), 
            batchData.d_numCandidatesPerAnchorPrefixSum2.get() + 1, 
            batchData.numTasks, 
            firstStream,
            cubdebugsync
        );
        assert(cubstatus == cudaSuccess);
      

        helpers::call_fill_kernel_async(batchData.d_flagscandidates.data(), batchData.totalNumCandidates, false, firstStream); CUERR;

        //convert output indices from task-local indices to global flags
        helpers::lambda_kernel<<<batchData.numTasks, 128, 0, firstStream>>>(
            [
                d_flagscandidates = batchData.d_flagscandidates.data(),
                indices2,
                d_numCandidatesPerAnchor2 = batchData.d_numCandidatesPerAnchor2.get(),
                d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.get()
            ] __device__ (){
                /*
                    Input:
                    indices2: 0,1,2,0,0,0,0,3,5,0
                    d_numCandidatesPerAnchorPrefixSum: 0,6,10

                    Output:
                    d_flagscandidates: 1,1,1,0,0,0,1,0,0,1,0,1
                */
                const int num = d_numCandidatesPerAnchor2[blockIdx.x];
                const int offset = d_numCandidatesPerAnchorPrefixSum[blockIdx.x];
                
                for(int i = threadIdx.x; i < num; i += blockDim.x){
                    const int globalIndex = indices2[offset + i] + offset;
                    d_flagscandidates[globalIndex] = true;
                }
            }
        ); CUERR;

        //compact candidate sequences according to flags                

        assert(batchData.d_candidateSequencesData2.data() != nullptr);
        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            cubBytes,
            batchData.d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                SequenceFlagMultiplier{batchData.d_flagscandidates.data(), int(batchData.encodedSequencePitchInInts)}
            ),
            batchData.d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            batchData.totalNumCandidates * batchData.encodedSequencePitchInInts,
            firstStream,
            cubdebugsync
        );

        assert(cubstatus == cudaSuccess);

        //compact other candidate buffers according to flags

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            cubBytes,
            in_zipped_begin,
            batchData.d_flagscandidates.data(),
            out_zipped_begin,
            thrust::make_discard_iterator(),
            batchData.totalNumCandidates,
            firstStream,
            cubdebugsync
        );

        assert(cubstatus == cudaSuccess);

        std::swap(batchData.d_numCandidatesPerAnchor, batchData.d_numCandidatesPerAnchor2);
        std::swap(batchData.d_numCandidatesPerAnchorPrefixSum, batchData.d_numCandidatesPerAnchorPrefixSum2);                
        std::swap(batchData.d_candidateSequencesData, batchData.d_candidateSequencesData2);
        std::swap(batchData.d_candidateReadIds, batchData.d_candidateReadIds2);
        std::swap(batchData.d_candidateSequencesLength, batchData.d_candidateSequencesLength2);
        std::swap(batchData.d_alignment_overlaps, batchData.d_alignment_overlaps2);
        std::swap(batchData.d_alignment_shifts, batchData.d_alignment_shifts2);
        std::swap(batchData.d_alignment_nOps, batchData.d_alignment_nOps2);
        std::swap(batchData.d_alignment_best_alignment_flags, batchData.d_alignment_best_alignment_flags2);

        //compute possible msa splits
        batchData.d_possibleSplitColumns.resize(32 * batchData.numTasks);
        batchData.d_numPossibleSplitColumnsPerAnchor.resize(batchData.numTasks);

        helpers::lambda_kernel<<<batchData.numTasks, 128, 0, firstStream>>>(
            [
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                d_counts = batchData.d_counts.data(),
                d_coverage = batchData.d_coverage.data(),
                msaColumnPitchInElements = batchData.msaColumnPitchInElements,
                d_msa_column_properties = batchData.d_msa_column_properties.data(),
                numTasks = batchData.numTasks,
                splitcolumnsPitchElements = 32,
                d_possibleSplitColumns = batchData.d_possibleSplitColumns.data(),
                d_numPossibleSplitColumnsPerTask = batchData.d_numPossibleSplitColumnsPerAnchor.data()
            ] __device__ (){

                using PSC = MultipleSequenceAlignment::PossibleSplitColumn;
                constexpr int maxColumnsPerTask = 32;

                __shared__ PSC sharedPSC[maxColumnsPerTask];
                __shared__ int broadcastint;

                using BlockReduce = cub::BlockReduce<int, 128>;
                using BlockScan = cub::BlockScan<int, 128>;
                __shared__ typename BlockReduce::TempStorage blockreducetemp;
                __shared__ typename BlockScan::TempStorage blockscantemp;

                for(int task = blockIdx.x; task < numTasks; task += gridDim.x){

                    int* const numSplitColumnsPtr = d_numPossibleSplitColumnsPerTask + task;
                    PSC* const splitColumnsPtr = d_possibleSplitColumns + splitcolumnsPitchElements * task;

                    if(d_numCandidatesPerAnchor[task] > 0){

                        //only check columns to the right of anchor
                        const int firstColumn = d_msa_column_properties[task].subjectColumnsEnd_excl;
                        const int lastColumnExcl = d_msa_column_properties[task].lastColumn_excl;                                

                        int* myCountsPtr[4];
                        myCountsPtr[0] = d_counts + 4 * msaColumnPitchInElements * task + 0 * msaColumnPitchInElements;
                        myCountsPtr[1] = d_counts + 4 * msaColumnPitchInElements * task + 1 * msaColumnPitchInElements;
                        myCountsPtr[2] = d_counts + 4 * msaColumnPitchInElements * task + 2 * msaColumnPitchInElements;
                        myCountsPtr[3] = d_counts + 4 * msaColumnPitchInElements * task + 3 * msaColumnPitchInElements;   
                        
                        int* myCoveragesPtr = d_coverage + msaColumnPitchInElements * task;

                        int totalNumResults = 0;

                        const int numIterations = SDIV(lastColumnExcl - firstColumn, blockDim.x);

                        for(int iteration = 0; iteration < numIterations; iteration++){
                            const int col = firstColumn + iteration * blockDim.x + threadIdx.x;

                            PSC myresults[3];
                            int myNumResults = 0;

                            if(col < lastColumnExcl){                                   

                                auto checkNuc = [&](const auto& counts, const char nuc){
                                    if(myNumResults < 3){

                                        const float ratio = float(counts[col]) / float(myCoveragesPtr[col]);
                                        //if((counts[col] == 2 && fgeq(ratio, 0.4f) && fleq(ratio, 0.6f)) || counts[col] > 2){
                                        if(counts[col] >= 2 && fgeq(ratio, 0.4f) && fleq(ratio, 0.6f)){

                                            #pragma unroll
                                            for(int k = 0; k < 3; k++){
                                                if(myNumResults == k){
                                                    myresults[k] = {nuc, col, ratio};
                                                    myNumResults++;
                                                    break;
                                                }
                                            }
                                            
                                        }
                                    }
                                };

                                checkNuc(myCountsPtr[0], 'A');
                                checkNuc(myCountsPtr[1], 'C');
                                checkNuc(myCountsPtr[2], 'G');
                                checkNuc(myCountsPtr[3], 'T');

                                if(myNumResults != 2){
                                    myNumResults = 0;
                                }    
                            }

                            int totalNumResultsIteration = BlockReduce(blockreducetemp).Sum(myNumResults);
                            if(threadIdx.x == 0){
                                broadcastint = totalNumResultsIteration;
                            }
                            __syncthreads();
                            totalNumResultsIteration = broadcastint;

                            if(totalNumResultsIteration + totalNumResults > maxColumnsPerTask){
                                totalNumResults = 0;
                                break;
                            }else{
                                int outputoffset = 0;

                                BlockScan(blockscantemp).ExclusiveSum(myNumResults, outputoffset);

                                if(myNumResults == 2){
                                    sharedPSC[totalNumResults + outputoffset + 0] = myresults[0];
                                    sharedPSC[totalNumResults + outputoffset + 1] = myresults[1];
                                }

                                totalNumResults += totalNumResultsIteration;
                            }

                            __syncthreads();
                        }

                        for(int i = threadIdx.x; i < totalNumResults; i += blockDim.x){
                            splitColumnsPtr[i] = sharedPSC[i];
                        }

                        if(threadIdx.x == 0){
                            *numSplitColumnsPtr = totalNumResults;
                        }
                    }else{
                        if(threadIdx.x == 0){
                            *numSplitColumnsPtr = 0;
                        }
                    }
                }
            }
        ); CUERR;

        cudaEventSynchronize(batchData.events[0]); CUERR; //wait for h_numCandidates

        batchData.totalNumCandidates = *batchData.h_numCandidates; 

        cubAllocator->DeviceFree(cubtemp);

        // cudaStreamSynchronize(firstStream); CUERR; //DEBUG
        // cudaDeviceSynchronize(); CUERR; //DEBUG
        // std::vector<int> vec1(batchData.d_numCandidatesPerAnchor.size());
        // cudaMemcpyAsync(vec1.data(), batchData.d_numCandidatesPerAnchor.data(), batchData.d_numCandidatesPerAnchor.sizeInBytes(), D2H, firstStream); CUERR;
        // cudaStreamSynchronize(firstStream); CUERR; //DEBUG
        // for(int i = 0; i < batchData.numTasks; i++){
        //     std::cerr << vec1[i] << " ";
        // }
        // std::cerr << "\n";
    }

    void copyBuffersToHost(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const{
        batchData.h_candidateReadIds.resize(batchData.totalNumCandidates);
        batchData.h_candidateSequencesLength.resize(batchData.totalNumCandidates);
        batchData.h_candidateSequencesData.resize(batchData.encodedSequencePitchInInts * batchData.totalNumCandidates);
        batchData.h_candidateSequencesRevcData.resize(batchData.encodedSequencePitchInInts * batchData.totalNumCandidates);

        batchData.h_alignment_overlaps.resize(batchData.totalNumCandidates);
        batchData.h_alignment_shifts.resize(batchData.totalNumCandidates);
        batchData.h_alignment_nOps.resize(batchData.totalNumCandidates);
        batchData.h_alignment_best_alignment_flags.resize(batchData.totalNumCandidates);

        batchData.h_possibleSplitColumns.resize(32 * batchData.numTasks);
        batchData.h_numPossibleSplitColumnsPerAnchor.resize(batchData.numTasks);
        batchData.h_consensus.resize(batchData.numTasks * batchData.msaColumnPitchInElements);
        batchData.h_msa_column_properties.resize(batchData.numTasks);

        //convert encoded consensus to characters and copy to host
        //copy column properties to host
        helpers::lambda_kernel<<<batchData.numTasks, 128, 0, secondStream>>>(
            [
                msaColumnPitchInElements = batchData.msaColumnPitchInElements,
                d_consensus = batchData.d_consensus.data(),
                h_consensus = batchData.h_consensus.data(),
                h_msa_column_properties = batchData.h_msa_column_properties.data(),
                d_msa_column_properties = batchData.d_msa_column_properties.data(),
                numTasks = batchData.numTasks
            ] __device__ (){

                auto decode = [](const std::uint8_t encoded){
                    char decoded = 'F';
                    if(encoded == std::uint8_t{0}){
                        decoded = 'A';
                    }else if(encoded == std::uint8_t{1}){
                        decoded = 'C';
                    }else if(encoded == std::uint8_t{2}){
                        decoded = 'G';
                    }else if(encoded == std::uint8_t{3}){
                        decoded = 'T';
                    }
                    return decoded;
                };

                const std::uint8_t* inputConsensus = &d_consensus[blockIdx.x * msaColumnPitchInElements];
                char* outputConsensus = &h_consensus[blockIdx.x * msaColumnPitchInElements];

                const int numVecIters = msaColumnPitchInElements / sizeof(char4);

                for(int i = threadIdx.x; i < numVecIters; i += blockDim.x){
                    char4 data = ((const char4*)inputConsensus)[i];

                    char4 result;
                    result.x = decode(data.x);
                    result.y = decode(data.y);
                    result.z = decode(data.z);
                    result.w = decode(data.w);

                    ((char4*)outputConsensus)[i] = result;
                }

                const int remaining = msaColumnPitchInElements - (numVecIters * sizeof(char4));
                
                for(int i = threadIdx.x; i < remaining; i += blockDim.x){
                    std::uint8_t encoded = d_consensus[blockIdx.x * msaColumnPitchInElements + numVecIters * sizeof(char4) + i];

                    char decoded = decode(encoded);

                    h_consensus[blockIdx.x * msaColumnPitchInElements + numVecIters * sizeof(char4) + i] = decoded;
                }

                //copy column properties
                for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < numTasks; i += blockDim.x * gridDim.x){
                    h_msa_column_properties[i] = d_msa_column_properties[i];
                }
            }
        );

        cudaMemcpyAsync(
            batchData.h_possibleSplitColumns.get(),
            batchData.d_possibleSplitColumns.get(),
            sizeof(MultipleSequenceAlignment::PossibleSplitColumn) * 32 * batchData.numTasks,
            D2H,
            firstStream
        ); CUERR;

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.d_numCandidatesPerAnchor.data(),
                batchData.d_numPossibleSplitColumnsPerAnchor.get()
            )),
            batchData.numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.h_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.h_numCandidatesPerAnchor.data(),
                batchData.h_numPossibleSplitColumnsPerAnchor.get()
            )),
            firstStream
        );          

        cudaMemcpyAsync(
            batchData.h_candidateSequencesData.get(),
            batchData.d_candidateSequencesData.get(),
            sizeof(unsigned int) * batchData.totalNumCandidates * batchData.encodedSequencePitchInInts,
            D2H,
            firstStream
        ); CUERR;

        auto d_zipped_begin = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_candidateReadIds.data(),
                batchData.d_candidateSequencesLength.data(),
                batchData.d_alignment_overlaps.data(),
                batchData.d_alignment_shifts.data(),
                batchData.d_alignment_nOps.data(),
                batchData.d_alignment_best_alignment_flags.data()
            )
        );

        auto h_zipped_begin = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.h_candidateReadIds.data(),
                batchData.h_candidateSequencesLength.data(),
                batchData.h_alignment_overlaps.data(),
                batchData.h_alignment_shifts.data(),
                batchData.h_alignment_nOps.data(),
                batchData.h_alignment_best_alignment_flags.data()
            )
        );

        helpers::call_copy_n_kernel(
            d_zipped_begin,
            batchData.totalNumCandidates,
            h_zipped_begin,
            firstStream
        );
    }

    void copyBatchDataIntoTask(ReadExtenderBase::Task& task, int taskindex, const BatchData& batchData) const{
        const int numCandidates = batchData.h_numCandidatesPerAnchor[taskindex];
        const int offset = batchData.h_numCandidatesPerAnchorPrefixSum[taskindex];

        task.candidateReadIds.resize(numCandidates);
        std::copy_n(batchData.h_candidateReadIds.data() + offset, numCandidates, task.candidateReadIds.begin());

        task.candidateSequenceLengths.resize(numCandidates);
        std::copy_n(batchData.h_candidateSequencesLength.data() + offset, numCandidates, task.candidateSequenceLengths.begin());

        task.candidateSequenceData.resize(numCandidates * batchData.encodedSequencePitchInInts);
        std::copy_n(
            batchData.h_candidateSequencesData.data() + offset * batchData.encodedSequencePitchInInts, 
            numCandidates * batchData.encodedSequencePitchInInts, 
            task.candidateSequenceData.begin()
        );

        task.alignmentFlags.resize(numCandidates);
        task.alignments.resize(numCandidates);

        for(int c = 0; c < numCandidates; c++){
            task.alignments[c].shift = batchData.h_alignment_shifts[offset + c];
            task.alignments[c].overlap = batchData.h_alignment_overlaps[offset + c];
            task.alignments[c].nOps = batchData.h_alignment_nOps[offset + c];
            task.alignmentFlags[c] = batchData.h_alignment_best_alignment_flags[offset + c];
        }

        task.numRemainingCandidates = numCandidates;

        if(task.numRemainingCandidates == 0){
            task.abort = true;
            task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;
        }
    }

    MultipleSequenceAlignment constructMsaWithDataFromTask(ReadExtenderBase::Task& task, const BatchData& batchData) const{
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
                    goodAlignmentProperties->maxErrorRate
                );
            }

            task.candidateStrings.resize(batchData.decodedSequencePitchInBytes * task.numRemainingCandidates, '\0');

            //decode the candidates for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                SequenceHelpers::decode2BitSequence(
                    task.candidateStrings.data() + c * batchData.decodedSequencePitchInBytes,
                    task.candidateSequenceData.data() + c * batchData.encodedSequencePitchInInts,
                    task.candidateSequenceLengths[c]
                );

                if(task.alignmentFlags[c] == BestAlignment_t::ReverseComplement){
                    SequenceHelpers::reverseComplementSequenceDecodedInplace(
                        task.candidateStrings.data() + c * batchData.decodedSequencePitchInBytes, 
                        task.candidateSequenceLengths[c]
                    );
                }
            }

            MultipleSequenceAlignment::InputData msaInput;
            msaInput.useQualityScores = false;
            msaInput.subjectLength = task.currentAnchorLength;
            msaInput.nCandidates = task.numRemainingCandidates;
            msaInput.candidatesPitch = batchData.decodedSequencePitchInBytes;
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
                        task.candidateSequenceData.data() + i * size_t(batchData.encodedSequencePitchInInts),
                        batchData.encodedSequencePitchInInts,
                        task.candidateSequenceData.data() + insertpos * size_t(batchData.encodedSequencePitchInInts)
                    );

                    task.candidateSequenceLengths[insertpos] = task.candidateSequenceLengths[i];
                    task.alignmentFlags[insertpos] = task.alignmentFlags[i];
                    task.alignments[insertpos] = task.alignments[i];
                    task.candidateOverlapWeights[insertpos] = task.candidateOverlapWeights[i];
                    task.candidateShifts[insertpos] = task.candidateShifts[i];

                    std::copy_n(
                        task.candidateStrings.data() + i * size_t(batchData.decodedSequencePitchInBytes),
                        batchData.decodedSequencePitchInBytes,
                        task.candidateStrings.data() + insertpos * size_t(batchData.decodedSequencePitchInBytes)
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
                task.candidateSequenceData.begin() + batchData.encodedSequencePitchInInts * insertpos, 
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
                task.candidateStrings.begin() + batchData.decodedSequencePitchInBytes * insertpos, 
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
                    correctionOptions->estimatedCoverage
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
};




}


#endif