#ifndef READ_EXTENDER_GPU_HPP
#define READ_EXTENDER_GPU_HPP

#include <readextender.hpp>
#include <config.hpp>
#include <hpc_helpers.cuh>

#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <gpu/simpleallocation.cuh>

#include <algorithm>
#include <vector>
#include <numeric>


namespace care{

template<class T>
using DeviceBuffer = SimpleAllocationDevice<T>;

template<class T>
using PinnedBuffer = SimpleAllocationPinnedHost<T>;

struct ReadExtenderGpu final : public ReadExtenderBase{
public:

    static constexpr int primary_stream_index = 0;


    ReadExtenderGpu(
        int insertSize,
        int insertSizeStddev,
        int maximumSequenceLength,
        const cpu::ContiguousReadStorage& rs, 
        const Minhasher& mh,
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap        
    ) : ReadExtenderBase(insertSize, insertSizeStddev, maximumSequenceLength, rs, coropts, gap),
        minhasher(&mh){


        cudaGetDevice(&deviceId); CUERR;

        h_numAnchors.resize(1);
        h_numCandidates.resize(1);

        kernelLaunchHandle = gpu::make_kernel_launch_handle(deviceId);
    }
     
private:

    void getCandidatesSingle(
        std::vector<read_number>& result, 
        const unsigned int* encodedRead, 
        int readLength, 
        read_number readId,
        int beginPos = 0 // only positions [beginPos, readLength] are hashed
    ){

        result.clear();

        const bool containsN = readStorage->readContainsN(readId);

        //exclude anchors with ambiguous bases
        if(!(correctionOptions.excludeAmbiguousReads && containsN)){

            const int length = readLength;
            std::string sequence(length, '0');

            decode2BitSequence(
                &sequence[0],
                encodedRead,
                length
            );

            minhasher->getCandidates_any_map(
                minhashHandle,
                sequence.c_str() + beginPos,
                std::max(0, readLength - beginPos),
                0
            );

            auto minhashResultsEnd = minhashHandle.result().end();
            //exclude candidates with ambiguous bases

            if(correctionOptions.excludeAmbiguousReads){
                minhashResultsEnd = std::remove_if(
                    minhashHandle.result().begin(),
                    minhashHandle.result().end(),
                    [&](read_number readId){
                        return readStorage->readContainsN(readId);
                    }
                );
            }            

            result.insert(
                result.begin(),
                minhashHandle.result().begin(),
                minhashResultsEnd
            );
        }else{
            ; // no candidates
        }
    }

    void getCandidates(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) override{
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            getCandidatesSingle(
                task.candidateReadIds, 
                task.currentAnchor.data(), 
                task.currentAnchorLength,
                task.currentAnchorReadId
            );

        }
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
        const int maximumSequenceLength = 100;
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

    PinnedBuffer<int> h_numCandidatesPerAnchor;
    PinnedBuffer<int> h_numCandidatesPerAnchorPrefixSum;
    PinnedBuffer<int> h_alignment_overlaps;
    PinnedBuffer<int> h_alignment_shifts;
    PinnedBuffer<int> h_alignment_nOps;
    PinnedBuffer<bool> h_alignment_isValid;
    PinnedBuffer<BestAlignment_t> h_alignment_best_alignment_flags;

    PinnedBuffer<int> h_numAnchors;
    PinnedBuffer<int> h_numCandidates;

    PinnedBuffer<int> h_anchorIndicesOfCandidates;
    PinnedBuffer<int> h_anchorSequencesLength;
    PinnedBuffer<int> h_candidateSequencesLength;
    PinnedBuffer<unsigned int> h_subjectSequencesData;
    PinnedBuffer<unsigned int> h_candidateSequencesData;

    DeviceBuffer<unsigned int> d_subjectSequencesData;
    DeviceBuffer<unsigned int> d_candidateSequencesData;

    DeviceBuffer<char> d_tempstorage;

    std::array<CudaStream, 2> streams{};

    gpu::KernelLaunchHandle kernelLaunchHandle;

    const Minhasher* minhasher;
    Minhasher::Handle minhashHandle;

};


}


#endif