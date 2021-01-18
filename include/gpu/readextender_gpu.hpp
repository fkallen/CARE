#ifndef READ_EXTENDER_GPU_HPP
#define READ_EXTENDER_GPU_HPP

#include <readextender.hpp>
#include <config.hpp>
#include <hpc_helpers.cuh>

#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <gpu/gpuminhasher.cuh>

#include <algorithm>
#include <vector>
#include <numeric>


namespace care{


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
            h_candidateSequencesData.get(),
            encodedSequencePitchInInts,
            h_candidateReadIds.get(),
            d_candidateReadIds.get(), //device accessible
            totalNumCandidates,
            stream
        );

        gpuReadStorage->gatherSequenceLengths(
            readStorageHandle,
            h_candidateSequencesLength.get(),
            d_candidateReadIds.get(),
            totalNumCandidates,
            stream
        );

        cudaStreamSynchronize(stream); CUERR;



        for(int t = 0; t < numIndices; t++){
            auto& task = tasks[indicesOfActiveTasks[t]];

            const int numCandidates = task.candidateReadIds.size();

            task.candidateSequenceLengths.resize(numCandidates);
            task.candidateSequencesFwdData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

            const int offset = h_numCandidatesPerAnchorPrefixSum[t];

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
        }

        nvtx::pop_range();
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
    PinnedBuffer<unsigned int> h_subjectSequencesData;
    PinnedBuffer<unsigned int> h_candidateSequencesData;

    DeviceBuffer<unsigned int> d_subjectSequencesData;
    DeviceBuffer<unsigned int> d_candidateSequencesData;

    DeviceBuffer<char> d_tempstorage;

    std::array<CudaStream, 4> streams{};

    gpu::KernelLaunchHandle kernelLaunchHandle;

    const gpu::GpuReadStorage* gpuReadStorage;
    const gpu::GpuMinhasher* gpuMinhasher;

    ReadStorageHandle readStorageHandle;
    gpu::GpuMinhasher::QueryHandle minhashHandle;

};


}


#endif