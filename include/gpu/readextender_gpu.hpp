#ifndef READ_EXTENDER_GPU_HPP
#define READ_EXTENDER_GPU_HPP

#include <readextender.hpp>
#include <config.hpp>
#include <hpc_helpers.cuh>

#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <gpu/gpuminhasher.cuh>
#include <sequencehelpers.hpp>

#include <algorithm>
#include <vector>
#include <numeric>


namespace care{


namespace readextendergpukernels{

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
        bool* __restrict__ outputflags,
        bool printme
    ){

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupindex = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int numgroups = (gridDim.x * blockDim.x) / groupsize;

        for(int task = groupindex; task < numTasksWithRemovedMate; task += numgroups){

            const int globalTaskIndex = activeTaskIndices[task];
            const int numCandidates = numCandidatesPerAnchor[globalTaskIndex];
            const int candidatesOffset = numCandidatesPerAnchorPrefixSum[globalTaskIndex];
            const unsigned int* const mateptr = anchormatedata + encodedSequencePitchInInts * task;

            if(printme && threadIdx.x == 0){
            //if(threadIdx.x == 0){
                printf("task %d, globalTaskIndex %d, numCandidates %d, candidatesOffset %d\n", task, globalTaskIndex, numCandidates, candidatesOffset);
            }

            //compare mate to candidates. 1 thread per candidate
            for(int c = group.thread_rank(); c < numCandidates; c += group.size()){
                bool doRemove = true;
                const unsigned int* const candidateptr = candidatefwddata + encodedSequencePitchInInts * (candidatesOffset + c);

                // if(printme){
                //     printf("c=%d\n", c);
                //     for(int i = 0; i < encodedSequencePitchInInts; i++){
                //         printf("%u ", candidateptr[i]);
                //         //assert(candidatefwddata2[encodedSequencePitchInInts * (candidatesOffset + c) + i] == candidateptr[i]);
                //     }
                //     printf("\n");
                
                //     for(int i = 0; i < encodedSequencePitchInInts; i++){
                //         printf("%u ", candidatefwddata2[encodedSequencePitchInInts * (candidatesOffset + c) + i]);
                //         //assert(candidatefwddata2[encodedSequencePitchInInts * (candidatesOffset + c) + i] == candidateptr[i]);
                //     }
                // }
                // if(c == 0){
                //     printf(
                //         "encodedSequencePitchInInts %d, candidatesOffset %d, globalTaskIndex %d, numCandidates %d\n", 
                //         encodedSequencePitchInInts, 
                //         candidatesOffset,
                //         globalTaskIndex,
                //         numCandidates
                //     );
                // }
                //__syncthreads();
                for(int p = 0; p < encodedSequencePitchInInts; p++){
                    const unsigned int aaa = mateptr[p];
                    const unsigned int bbb = candidateptr[p];

                    if(aaa != bbb){
                        if(printme){
                            printf("t %d c %d p %d, aaa %u, bbb %u\n", task, c, p, aaa, bbb);
                        }
                        doRemove = false;
                        break;
                    }
                }

                outputflags[(candidatesOffset + c)] = doRemove;
            }
        }
    }
}


struct ReadExtenderGpu final : public ReadExtenderBase{
public:

    static constexpr int primary_stream_index = 0;

    template<class T>
    using DeviceBuffer = helpers::SimpleAllocationDevice<T>;

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
        const GoodAlignmentProperties& gap        
    ) 
    : ReadExtenderBase(insertSize, insertSizeStddev, maxextensionPerStep, maximumSequenceLength, coropts, gap),
        kmerLength(kmerLength_),
        gpuReadStorage(&rs),
        gpuMinhasher(&gmh),
        readStorageHandle(gpuReadStorage->makeHandle()),
        minhashHandle(gpuMinhasher->makeQueryHandle()){


        cudaGetDevice(&deviceId); CUERR;

        h_numAnchors.resize(1);
        h_numCandidates.resize(1);

        kernelLaunchHandle = gpu::make_kernel_launch_handle(deviceId);
    }

    ~ReadExtenderGpu(){
        gpuReadStorage->destroyHandle(readStorageHandle);
        gpuMinhasher->destroyHandle(minhashHandle);
    }
     
private:

    void getCandidateReadIdsSingle(
        std::vector<read_number>& result, 
        const unsigned int* encodedRead, 
        int readLength, 
        read_number readId,
        int beginPos = 0 // only positions [beginPos, readLength] are hashed
    ){
        assert(false);

        // result.clear();

        // const bool containsN = readStorage->readContainsN(readId);

        // //exclude anchors with ambiguous bases
        // if(!(correctionOptions.excludeAmbiguousReads && containsN)){

        //     const int length = readLength;
        //     std::string sequence(length, '0');

        //     decode2BitSequence(
        //         &sequence[0],
        //         encodedRead,
        //         length
        //     );

        //     minhasher->getCandidates_any_map(
        //         minhashHandle,
        //         sequence.c_str() + beginPos,
        //         std::max(0, readLength - beginPos),
        //         0
        //     );

        //     auto minhashResultsEnd = minhashHandle.result().end();
        //     //exclude candidates with ambiguous bases

        //     if(correctionOptions.excludeAmbiguousReads){
        //         minhashResultsEnd = std::remove_if(
        //             minhashHandle.result().begin(),
        //             minhashHandle.result().end(),
        //             [&](read_number readId){
        //                 return readStorage->readContainsN(readId);
        //             }
        //         );
        //     }            

        //     result.insert(
        //         result.begin(),
        //         minhashHandle.result().begin(),
        //         minhashResultsEnd
        //     );
        // }else{
        //     ; // no candidates
        // }
    }

    void getCandidateReadIds(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) override{
        // for(int indexOfActiveTask : indicesOfActiveTasks){
        //     auto& task = tasks[indexOfActiveTask];

        //     getCandidateReadIdsSingle(
        //         task.candidateReadIds, 
        //         task.currentAnchor.data(), 
        //         task.currentAnchorLength,
        //         task.currentAnchorReadId
        //     );

        // }

        getCandidateReadIdsGpu(tasks, indicesOfActiveTasks);
    }

#if 1
    void getCandidateReadIdsGpu(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks){
        nvtx::push_range("gpu_hashing", 2);

        const int numIndices = indicesOfActiveTasks.size();
        const std::size_t encodedSequencePitchInInts2Bit = encodedSequencePitchInInts;

        cudaStream_t stream = streams[primary_stream_index];

        //input buffers
        h_subjectSequencesData.resize(encodedSequencePitchInInts2Bit * numIndices);
        d_subjectSequencesData.resize(encodedSequencePitchInInts2Bit * numIndices);
        h_anchorSequencesLength.resize(numIndices);
        d_anchorSequencesLength.resize(numIndices);

        //output buffers
        h_numCandidatesPerAnchor.resize(numIndices);
        d_numCandidatesPerAnchor.resize(numIndices);
        h_numCandidatesPerAnchorPrefixSum.resize(numIndices+1);
        d_numCandidatesPerAnchorPrefixSum.resize(numIndices+1);

        #if 0
        constexpr int batchsize = 64;
        const int batches = SDIV(numIndices, batchsize);

        for(int batch = 0; batch < batches; batch++){
            const int begin = batch * batchsize;
            const int end = std::min((batch + 1) * batchsize, numIndices);
            const int num = end - begin;

            for(int t = begin; t < end; t++){
                const auto& task = tasks[indicesOfActiveTasks[t]];

                if(task.iteration >= 0){

                    h_anchorSequencesLength[t] = task.currentAnchorLength;

                    std::copy(
                        task.currentAnchor.begin(),
                        task.currentAnchor.end(),
                        h_subjectSequencesData.get() + t * encodedSequencePitchInInts2Bit
                    );
                }else{
                    //only hash kmers which include extended positions

                    const int extendedPositionsPreviousIteration 
                        = task.totalAnchorBeginInExtendedRead.at(task.iteration) - task.totalAnchorBeginInExtendedRead.at(task.iteration-1);

                    const int lengthToHash = std::min(task.currentAnchorLength, kmerLength + extendedPositionsPreviousIteration - 1);
                    h_anchorSequencesLength[t] = lengthToHash;

                    //std::cerr << "lengthToHash = " << lengthToHash << "\n";

                    std::vector<char> buf(task.currentAnchorLength);
                    SequenceHelpers::decode2BitSequence(buf.data(), task.currentAnchor.data(), task.currentAnchorLength);
                    SequenceHelpers::encodeSequence2Bit(
                        h_subjectSequencesData.get() + t * encodedSequencePitchInInts2Bit, 
                        buf.data() + task.currentAnchorLength - lengthToHash, 
                        lengthToHash
                    );

                }
            }

            cudaMemcpyAsync(
                d_subjectSequencesData.get() + encodedSequencePitchInInts2Bit * begin ,
                h_subjectSequencesData.get() + encodedSequencePitchInInts2Bit * begin,
                sizeof(unsigned int) * num * encodedSequencePitchInInts2Bit,
                H2D,
                streams[batch % streams.size()]
            ); CUERR;

            cudaMemcpyAsync(
                d_anchorSequencesLength.get() + begin,
                h_anchorSequencesLength.get() + begin,
                sizeof(int) * num,
                H2D,
                stream
            ); CUERR;
        }

        #else

        for(int t = 0; t < numIndices; t++){
            const auto& task = tasks[indicesOfActiveTasks[t]];

            if(task.iteration >= 0){

                h_anchorSequencesLength[t] = task.currentAnchorLength;

                std::copy(
                    task.currentAnchor.begin(),
                    task.currentAnchor.end(),
                    h_subjectSequencesData.get() + t * encodedSequencePitchInInts2Bit
                );
            }else{
                //only hash kmers which include extended positions

                const int extendedPositionsPreviousIteration 
                    = task.totalAnchorBeginInExtendedRead.at(task.iteration) - task.totalAnchorBeginInExtendedRead.at(task.iteration-1);

                const int lengthToHash = std::min(task.currentAnchorLength, kmerLength + extendedPositionsPreviousIteration - 1);
                h_anchorSequencesLength[t] = lengthToHash;

                //std::cerr << "lengthToHash = " << lengthToHash << "\n";

                std::vector<char> buf(task.currentAnchorLength);
                SequenceHelpers::decode2BitSequence(buf.data(), task.currentAnchor.data(), task.currentAnchorLength);
                SequenceHelpers::encodeSequence2Bit(
                    h_subjectSequencesData.get() + t * encodedSequencePitchInInts2Bit, 
                    buf.data() + task.currentAnchorLength - lengthToHash, 
                    lengthToHash
                );

            }
        }

        #endif

        cudaMemcpyAsync(
            d_subjectSequencesData.get(),
            h_subjectSequencesData.get(),
            sizeof(unsigned int) * numIndices * encodedSequencePitchInInts2Bit,
            H2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            d_anchorSequencesLength.get(),
            h_anchorSequencesLength.get(),
            sizeof(int) * numIndices,
            H2D,
            stream
        ); CUERR;

        int totalNumValues = 0;

        gpuMinhasher->determineNumValues(
            minhashHandle,
            d_subjectSequencesData.get(),
            encodedSequencePitchInInts2Bit,
            d_anchorSequencesLength.get(),
            numIndices,
            d_numCandidatesPerAnchor.get(),
            totalNumValues,
            stream
        );

        cudaStreamSynchronize(stream); CUERR;

        d_candidateReadIds.resize(totalNumValues);
        h_candidateReadIds.resize(totalNumValues);

        if(totalNumValues == 0){
            cudaMemsetAsync(d_numCandidatesPerAnchor.get(), 0, sizeof(int) * numIndices, stream); CUERR;
            cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.get(), 0, sizeof(int) * (1 + numIndices), stream); CUERR;
            return;
        }

        gpuMinhasher->retrieveValues(
            minhashHandle,
            nullptr,
            numIndices,                
            totalNumValues,
            d_candidateReadIds.get(),
            d_numCandidatesPerAnchor.get(),
            d_numCandidatesPerAnchorPrefixSum.get(),
            stream
        );

        //d_numCandidatesPerAnchor not copied to host because unused

        cudaMemcpyAsync(
            h_numCandidatesPerAnchorPrefixSum.get(),
            d_numCandidatesPerAnchorPrefixSum.get(),
            sizeof(int) * (numIndices+1),
            D2H,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;

        cudaMemcpyAsync(
            h_candidateReadIds.get(),
            d_candidateReadIds.get(),
            sizeof(read_number) * h_numCandidatesPerAnchorPrefixSum[numIndices],
            D2H,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;

        for(int t = 0; t < numIndices; t++){
            auto& task = tasks[indicesOfActiveTasks[t]];

            const int offsetBegin = h_numCandidatesPerAnchorPrefixSum[t];
            const int offsetEnd = h_numCandidatesPerAnchorPrefixSum[t+1];

            task.candidateReadIds.clear();
            
            task.candidateReadIds.insert(
                task.candidateReadIds.begin(),
                h_candidateReadIds.get() + offsetBegin,
                h_candidateReadIds.get() + offsetEnd
            );

            //std::cerr << "task " << task.myReadId << ", iteration " << task.iteration << ", candidates " << task.candidateReadIds.size();
        }

        nvtx::pop_range();
    }
#endif

    void loadCandidateSequenceData(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) override{

        nvtx::push_range("gpu_loadCandidates", 2);

        const int numIndices = indicesOfActiveTasks.size();

        h_numCandidatesPerAnchorPrefixSum.resize(numIndices+1);
        h_numCandidatesPerAnchorPrefixSum[0] = 0;        

        for(int t = 0; t < numIndices; t++){
            const auto& task = tasks[indicesOfActiveTasks[t]];

            const int numCandidates = task.candidateReadIds.size();

            h_numCandidatesPerAnchorPrefixSum[t+1] = numCandidates + h_numCandidatesPerAnchorPrefixSum[t];
        }

        const int totalNumCandidates = h_numCandidatesPerAnchorPrefixSum[numIndices];

        //input buffers
        h_candidateReadIds.resize(totalNumCandidates);
        d_candidateReadIds.resize(totalNumCandidates);

        //output buffers
        h_candidateSequencesLength.resize(totalNumCandidates);
        h_candidateSequencesData.resize(encodedSequencePitchInInts * totalNumCandidates);
        h_candidateSequencesRevcData.resize(encodedSequencePitchInInts * totalNumCandidates);

        d_candidateSequencesLength.resize(totalNumCandidates);
        d_candidateSequencesData.resize(encodedSequencePitchInInts * totalNumCandidates);
        d_candidateSequencesRevcData.resize(encodedSequencePitchInInts * totalNumCandidates);

        for(int t = 0; t < numIndices; t++){
            const auto& task = tasks[indicesOfActiveTasks[t]];

            const int numCandidates = task.candidateReadIds.size();

            const int offset = h_numCandidatesPerAnchorPrefixSum[t];

            std::copy(task.candidateReadIds.begin(), task.candidateReadIds.end(), h_candidateReadIds.get() + offset);
        }

        cudaStream_t stream = streams[primary_stream_index];

        cudaMemcpyAsync(
            d_candidateReadIds.get(),
            h_candidateReadIds.get(),
            sizeof(read_number) * totalNumCandidates,
            H2D,
            stream
        ); CUERR;

        gpuReadStorage->gatherSequences(
            readStorageHandle,
            d_candidateSequencesData.get(),
            encodedSequencePitchInInts,
            h_candidateReadIds.get(),
            d_candidateReadIds.get(), //device accessible
            totalNumCandidates,
            stream
        );

        gpuReadStorage->gatherSequenceLengths(
            readStorageHandle,
            d_candidateSequencesLength.get(),
            d_candidateReadIds.get(),
            totalNumCandidates,
            stream
        );

        readextendergpukernels::reverseComplement2bitKernel<128><<<320,128,0,stream>>>(
            d_candidateSequencesLength.get(),
            d_candidateSequencesData.get(),
            d_candidateSequencesRevcData.get(),
            totalNumCandidates,
            encodedSequencePitchInInts
        ); CUERR;

        cudaMemcpyAsync(
            h_candidateSequencesLength.get(),
            d_candidateSequencesLength.get(),
            sizeof(int) * totalNumCandidates,
            H2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            h_candidateSequencesData.get(),
            d_candidateSequencesData.get(),
            sizeof(unsigned int) * totalNumCandidates * encodedSequencePitchInInts,
            H2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            h_candidateSequencesRevcData.get(),
            d_candidateSequencesRevcData.get(),
            sizeof(unsigned int) * totalNumCandidates * encodedSequencePitchInInts,
            H2D,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;



        for(int t = 0; t < numIndices; t++){
            auto& task = tasks[indicesOfActiveTasks[t]];

            const int numCandidates = task.candidateReadIds.size();

            task.candidateSequenceLengths.resize(numCandidates);
            task.candidateSequencesFwdData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
            task.candidateSequencesRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

            const int offset = h_numCandidatesPerAnchorPrefixSum[t];

            if(task.myReadId == 2720 && task.iteration == 4){
                std::cerr << h_candidateSequencesData[(offset * encodedSequencePitchInInts) + 0] << "\n";
            }

            std::copy_n(
                h_candidateSequencesLength.get() + offset,
                numCandidates,
                task.candidateSequenceLengths.begin()
            );

            std::copy_n(
                h_candidateSequencesData.get() + (offset * encodedSequencePitchInInts),
                (numCandidates * encodedSequencePitchInInts),
                task.candidateSequencesFwdData.begin()
            );

            std::copy_n(
                h_candidateSequencesRevcData.get() + (offset * encodedSequencePitchInInts),
                (numCandidates * encodedSequencePitchInInts),
                task.candidateSequencesRevcData.begin()
            );
        }

        nvtx::pop_range();
    }

    void eraseDataOfRemovedMates(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) override{
        auto vecAccess = [](auto& vec, auto index) -> decltype(vec.at(index)){
            return vec.at(index);
        };

#if 1
        // std::vector<Task> tasksorig = tasks;
        // std::vector<Task> taskscopy = tasks;
        // std::vector<int> indicesOfActiveTaskscopy = indicesOfActiveTasks;

        const int numActiveTasks = indicesOfActiveTasks.size();
        assert(std::all_of(indicesOfActiveTasks.begin(), indicesOfActiveTasks.end(), 
            [&](auto i){return i < tasks.size();}));

        h_anchormatedata.resize(numActiveTasks * encodedSequencePitchInInts);
        d_anchormatedata.resize(numActiveTasks * encodedSequencePitchInInts);

        h_activeTaskIndices.resize(numActiveTasks);
        d_activeTaskIndices.resize(numActiveTasks);

        h_numCandidatesPerAnchor.resize(numActiveTasks);
        h_numCandidatesPerAnchorPrefixSum.resize(numActiveTasks+1);                

        h_numCandidatesPerAnchorPrefixSum[0] = 0;

        int numTasksWithMateRemoved = 0;  

        for(int t = 0; t < numActiveTasks; t++){
            const int indexOfActiveTask = indicesOfActiveTasks[t];
            auto& task = vecAccess(tasks, indexOfActiveTask);

            const int numCandidates = task.candidateReadIds.size();

            h_numCandidatesPerAnchor[t] = numCandidates;
            h_numCandidatesPerAnchorPrefixSum[t+1] = numCandidates + h_numCandidatesPerAnchorPrefixSum[t];

            if(task.mateRemovedFromCandidates){
                h_activeTaskIndices[numTasksWithMateRemoved] = t;

                std::copy(task.encodedMate.begin(), task.encodedMate.end(), h_anchormatedata.begin() + numTasksWithMateRemoved * encodedSequencePitchInInts);

                numTasksWithMateRemoved++;
            }
        }

        if(numTasksWithMateRemoved > 0){

            h_flags.resize(h_numCandidatesPerAnchorPrefixSum[numActiveTasks]);
            std::fill(h_flags.begin(), h_flags.end(), 0);

            cudaStream_t stream = streams[primary_stream_index];

            cudaMemcpyAsync(
                d_anchormatedata.data(),
                h_anchormatedata.data(),
                sizeof(unsigned int) * numTasksWithMateRemoved * encodedSequencePitchInInts,
                H2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_activeTaskIndices.data(),
                h_activeTaskIndices.data(),
                sizeof(int) * numTasksWithMateRemoved,
                H2D,
                stream
            ); CUERR;

            // std::cerr <<"d_candidateSequencesData.size() = " << d_candidateSequencesData.size() << "\n";
            // std::cerr << "maxcandidates = " << h_numCandidatesPerAnchorPrefixSum[numActiveTasks] << "\n";

            constexpr int groupsize = 32;
            dim3 block(128,1,1);
            dim3 grid(SDIV(numTasksWithMateRemoved * groupsize, 128), 1, 1);

            readextendergpukernels::filtermatekernel<128,groupsize><<<grid, block, 0, stream>>>(
            //readextendergpukernels::filtermatekernel<1,1><<<1, 1, 0, stream>>>(
                d_anchormatedata.data(),
                d_candidateSequencesData.data(),
                encodedSequencePitchInInts,
                h_numCandidatesPerAnchor.data(),
                h_numCandidatesPerAnchorPrefixSum.data(),
                d_activeTaskIndices.data(),
                numTasksWithMateRemoved,
                h_flags.data(),
                false//(tasks.size() > 1 && tasks[1].myReadId == 10243 && tasks[1].iteration == 2)
            ); CUERR;

            cudaStreamSynchronize(stream); CUERR;

            for(int t = 0; t < numActiveTasks; t++){
                const int indexOfActiveTask = indicesOfActiveTasks[t];
                auto& task = vecAccess(tasks, indexOfActiveTask);

                if(task.mateRemovedFromCandidates){
                    const int numCandidates = task.candidateReadIds.size();
                    const std::size_t offset = h_numCandidatesPerAnchorPrefixSum[t];

                    const bool* const removalflags = h_flags.data() + offset;
                    int numremaining = 0;

                    for(int c = 0; c < numCandidates; c++){
                        if(!removalflags[c]){

                            vecAccess(task.candidateReadIds, numremaining) = vecAccess(task.candidateReadIds, c);
                            vecAccess(task.candidateSequenceLengths, numremaining) = vecAccess(task.candidateSequenceLengths, c);                     

                            std::copy_n(
                                task.candidateSequencesFwdData.data() + c * encodedSequencePitchInInts,
                                encodedSequencePitchInInts,
                                task.candidateSequencesFwdData.data() + numremaining * encodedSequencePitchInInts
                            );

                            std::copy_n(
                                task.candidateSequencesRevcData.data() + c * encodedSequencePitchInInts,
                                encodedSequencePitchInInts,
                                task.candidateSequencesRevcData.data() + numremaining * encodedSequencePitchInInts
                            );

                            numremaining++;
                        }
                    }

                    task.candidateReadIds.erase(
                        task.candidateReadIds.begin() + numremaining, 
                        task.candidateReadIds.end()
                    );
                    task.candidateSequenceLengths.erase(
                        task.candidateSequenceLengths.begin() + numremaining, 
                        task.candidateSequenceLengths.end()
                    );
                    task.candidateSequencesFwdData.erase(
                        task.candidateSequencesFwdData.begin() + numremaining * encodedSequencePitchInInts, 
                        task.candidateSequencesFwdData.end()
                    );
                    task.candidateSequencesRevcData.erase(
                        task.candidateSequencesRevcData.begin() + numremaining * encodedSequencePitchInInts, 
                        task.candidateSequencesRevcData.end()
                    );

                    task.mateRemovedFromCandidates = false;
                }
            }

        }

#else

        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = vecAccess(tasks, indexOfActiveTask);

            if(task.mateRemovedFromCandidates){
                const int numCandidates = task.candidateReadIds.size();

                std::vector<int> positionsOfCandidatesToKeep;
                positionsOfCandidatesToKeep.reserve(numCandidates);

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = task.candidateSequencesFwdData.data() 
                                                    + std::size_t(encodedSequencePitchInInts) * c;

                    auto mismatchIters = std::mismatch(
                        task.encodedMate.begin(), task.encodedMate.end(),
                        seqPtr, seqPtr + encodedSequencePitchInInts
                    );

                    //candidate differs from mate
                    if(mismatchIters.first != task.encodedMate.end()){                            
                        positionsOfCandidatesToKeep.emplace_back(c);
                    }else{
                        ;//std::cerr << "";
                    }
                }

                //compact
                const int toKeep = positionsOfCandidatesToKeep.size();
                for(int c = 0; c < toKeep; c++){
                    const int index = vecAccess(positionsOfCandidatesToKeep, c);

                    vecAccess(task.candidateReadIds, c) = vecAccess(task.candidateReadIds, index);
                    vecAccess(task.candidateSequenceLengths, c) = vecAccess(task.candidateSequenceLengths, index);                        

                    std::copy_n(
                        task.candidateSequencesFwdData.data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequencesFwdData.data() + c * encodedSequencePitchInInts
                    );

                    std::copy_n(
                        task.candidateSequencesRevcData.data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequencesRevcData.data() + c * encodedSequencePitchInInts
                    );

                    
                }

                //erase
                task.candidateReadIds.erase(
                    task.candidateReadIds.begin() + toKeep, 
                    task.candidateReadIds.end()
                );
                task.candidateSequenceLengths.erase(
                    task.candidateSequenceLengths.begin() + toKeep, 
                    task.candidateSequenceLengths.end()
                );
                task.candidateSequencesFwdData.erase(
                    task.candidateSequencesFwdData.begin() + toKeep * encodedSequencePitchInInts, 
                    task.candidateSequencesFwdData.end()
                );
                task.candidateSequencesRevcData.erase(
                    task.candidateSequencesRevcData.begin() + toKeep * encodedSequencePitchInInts, 
                    task.candidateSequencesRevcData.end()
                );

                task.mateRemovedFromCandidates = false;

                //assert(task == taskscopy[indexOfActiveTask]);
            }else{
                //assert(task == taskscopy[indexOfActiveTask]);
            }

            

        }
#endif        
    }

    void calculateAlignments(std::vector<ReadExtenderBase::Task>& tasks, const std::vector<int>& indicesOfActiveTasks) override{
        nvtx::push_range("gpu_alignment", 2);

        const int numIndices = indicesOfActiveTasks.size();

        h_numCandidatesPerAnchor.resize(numIndices);
        h_numCandidatesPerAnchorPrefixSum.resize(numIndices+1);                

        h_numCandidatesPerAnchorPrefixSum[0] = 0;        

        for(int t = 0; t < numIndices; t++){
            const auto& task = tasks[indicesOfActiveTasks[t]];

            const int numCandidates = task.candidateReadIds.size();

            h_numCandidatesPerAnchor[t] = numCandidates;
            h_numCandidatesPerAnchorPrefixSum[t+1] = numCandidates + h_numCandidatesPerAnchorPrefixSum[t];
        }

        const int totalNumCandidates = h_numCandidatesPerAnchorPrefixSum[numIndices];

        h_alignment_overlaps.resize(totalNumCandidates);
        h_alignment_shifts.resize(totalNumCandidates);
        h_alignment_nOps.resize(totalNumCandidates);
        h_alignment_isValid.resize(totalNumCandidates);
        h_alignment_best_alignment_flags.resize(totalNumCandidates);


        h_anchorIndicesOfCandidates.resize(totalNumCandidates);
        h_anchorSequencesLength.resize(numIndices);
        h_candidateSequencesLength.resize(totalNumCandidates);
        h_subjectSequencesData.resize(numIndices * encodedSequencePitchInInts);
        h_candidateSequencesData.resize(totalNumCandidates * encodedSequencePitchInInts);

        d_subjectSequencesData.resize(numIndices * encodedSequencePitchInInts);
        d_candidateSequencesData.resize(totalNumCandidates * encodedSequencePitchInInts);

        auto* anchorcpyptr = h_subjectSequencesData.get();
        auto* candcpyptr = h_candidateSequencesData.get();

        int maxLength = 0;

        for(int t = 0; t < numIndices; t++){
            const auto& task = tasks[indicesOfActiveTasks[t]];
            const int numCandidates = task.candidateReadIds.size();

            const auto offset = h_numCandidatesPerAnchorPrefixSum[t];

            h_anchorSequencesLength[t] = task.currentAnchorLength;

            std::fill_n(
                h_anchorIndicesOfCandidates.get() + offset, 
                numCandidates, 
                t
            );

            std::copy_n(
                task.candidateSequenceLengths.begin(),
                numCandidates,
                h_candidateSequencesLength.get() + offset
            );

            auto localmax = std::max_element(task.candidateSequenceLengths.begin(), task.candidateSequenceLengths.end());
            if(localmax != task.candidateSequenceLengths.end()){
                maxLength = std::max(maxLength, *localmax);
            }

            anchorcpyptr = std::copy(
                task.currentAnchor.begin(), 
                task.currentAnchor.end(),
                anchorcpyptr
            );

            candcpyptr = std::copy(
                task.candidateSequencesFwdData.begin(), 
                task.candidateSequencesFwdData.end(),
                candcpyptr
            );
        }

        h_numAnchors[0] = numIndices;
        h_numCandidates[0] = totalNumCandidates;

        const bool* const d_anchorContainsN = nullptr;
        const bool* const d_candidateContainsN = nullptr;
        const bool removeAmbiguousAnchors = false;
        const bool removeAmbiguousCandidates = false;
        const int maxNumAnchors = numIndices;
        const int maxNumCandidates = totalNumCandidates;
        const int maximumSequenceLength = maxLength;
        const int encodedSequencePitchInInts2Bit = encodedSequencePitchInInts;
        const int min_overlap = goodAlignmentProperties.min_overlap;
        const float maxErrorRate = goodAlignmentProperties.maxErrorRate;
        const float min_overlap_ratio = goodAlignmentProperties.min_overlap_ratio;
        const float estimatedNucleotideErrorRate = correctionOptions.estimatedErrorrate;
        cudaStream_t stream = streams[primary_stream_index];

        auto callAlignmentKernel = [&](void* d_tempstorage, size_t& tempstoragebytes){

            call_popcount_rightshifted_hamming_distance_kernel_async(
                d_tempstorage,
                tempstoragebytes,
                h_alignment_overlaps.get(),
                h_alignment_shifts.get(),
                h_alignment_nOps.get(),
                h_alignment_isValid.get(),
                h_alignment_best_alignment_flags.get(),
                d_subjectSequencesData.get(),
                d_candidateSequencesData.get(),
                h_anchorSequencesLength.get(),
                h_candidateSequencesLength.get(),
                h_numCandidatesPerAnchorPrefixSum.get(),
                h_numCandidatesPerAnchor.get(),
                h_anchorIndicesOfCandidates.get(),
                h_numAnchors.get(),
                h_numCandidates.get(),
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

        cudaMemcpyAsync(
            d_subjectSequencesData.get(),
            h_subjectSequencesData.get(),
            sizeof(unsigned int) * numIndices * encodedSequencePitchInInts,
            H2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            d_candidateSequencesData.get(),
            h_candidateSequencesData.get(),
            sizeof(unsigned int) * totalNumCandidates * encodedSequencePitchInInts,
            H2D,
            stream
        ); CUERR;

        size_t tempstoragebytes = 0;
        callAlignmentKernel(nullptr, tempstoragebytes);

        d_tempstorage.resize(tempstoragebytes);

        callAlignmentKernel(d_tempstorage.get(), tempstoragebytes);

        for(int t = 0; t < numIndices; t++){
            auto& task = tasks[indicesOfActiveTasks[t]];

            const auto numCandidates = task.candidateReadIds.size();

            task.alignmentFlags.resize(numCandidates);
            task.alignments.resize(numCandidates);
        }

        cudaStreamSynchronize(stream); CUERR;

        for(int t = 0; t < numIndices; t++){
            auto& task = tasks[indicesOfActiveTasks[t]];
            const int numCandidates = task.candidateReadIds.size();

            const auto offset = h_numCandidatesPerAnchorPrefixSum[t];

            for(int c = 0; c < numCandidates; c++){
                task.alignments[c].shift = h_alignment_shifts[offset + c];
                task.alignments[c].overlap = h_alignment_overlaps[offset + c];
                task.alignments[c].nOps = h_alignment_nOps[offset + c];
                task.alignments[c].isValid = h_alignment_isValid[offset + c];
                task.alignmentFlags[c] = h_alignment_best_alignment_flags[offset + c];
            }
        }

        nvtx::pop_range();
    }

    int deviceId;
    int kmerLength;

    PinnedBuffer<read_number> h_readIds;
    DeviceBuffer<read_number> d_readIds;
    PinnedBuffer<read_number> h_candidateReadIds;
    DeviceBuffer<read_number> d_candidateReadIds;

    PinnedBuffer<unsigned int> h_anchormatedata;
    DeviceBuffer<unsigned int> d_anchormatedata;

    PinnedBuffer<int> h_numCandidatesPerAnchor;
    DeviceBuffer<int> d_numCandidatesPerAnchor;
    PinnedBuffer<int> h_numCandidatesPerAnchorPrefixSum;
    DeviceBuffer<int> d_numCandidatesPerAnchorPrefixSum;
    PinnedBuffer<int> h_alignment_overlaps;
    PinnedBuffer<int> h_alignment_shifts;
    PinnedBuffer<int> h_alignment_nOps;
    PinnedBuffer<bool> h_alignment_isValid;
    PinnedBuffer<BestAlignment_t> h_alignment_best_alignment_flags;

    PinnedBuffer<int> h_numAnchors;
    PinnedBuffer<int> h_numCandidates;

    PinnedBuffer<int> h_anchorIndicesOfCandidates;
    PinnedBuffer<int> h_anchorSequencesLength;
    DeviceBuffer<int> d_anchorSequencesLength;
    PinnedBuffer<int> h_candidateSequencesLength;
    DeviceBuffer<int> d_candidateSequencesLength;
    PinnedBuffer<unsigned int> h_subjectSequencesData;
    PinnedBuffer<unsigned int> h_candidateSequencesData;
    PinnedBuffer<unsigned int> h_candidateSequencesRevcData;
    

    DeviceBuffer<unsigned int> d_subjectSequencesData;
    DeviceBuffer<unsigned int> d_candidateSequencesData;
    DeviceBuffer<unsigned int> d_candidateSequencesRevcData;

    DeviceBuffer<int> d_activeTaskIndices;
    PinnedBuffer<int> h_activeTaskIndices;

    DeviceBuffer<char> d_tempstorage;
    DeviceBuffer<bool> d_flags;
    PinnedBuffer<bool> h_flags;

    std::array<CudaStream, 4> streams{};

    gpu::KernelLaunchHandle kernelLaunchHandle;

    const gpu::GpuReadStorage* gpuReadStorage;
    const gpu::GpuMinhasher* gpuMinhasher;

    ReadStorageHandle readStorageHandle;
    gpu::GpuMinhasher::QueryHandle minhashHandle;

};


}


#endif