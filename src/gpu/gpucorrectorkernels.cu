#include <correctionresultprocessing.hpp>
#include <config.hpp>

#include <gpu/gpucorrectorkernels.cuh>


namespace care{
    namespace gpu{
    
    namespace gpucorrectorkernels{
    
        __global__
        void copyCorrectionInputDeviceData(
            int* __restrict__ output_numAnchors,
            int* __restrict__ output_numCandidates,
            read_number* __restrict__ output_anchor_read_ids,
            unsigned int* __restrict__ output_anchor_sequences_data,
            int* __restrict__ output_anchor_sequences_lengths,
            read_number* __restrict__ output_candidate_read_ids,
            int* __restrict__ output_candidates_per_anchor,
            int* __restrict__ output_candidates_per_anchor_prefixsum,
            const int encodedSequencePitchInInts,
            const int input_numAnchors,
            const int input_numCandidates,
            const read_number* __restrict__ input_anchor_read_ids,
            const unsigned int* __restrict__ input_anchor_sequences_data,
            const int* __restrict__ input_anchor_sequences_lengths,
            const read_number* __restrict__ input_candidate_read_ids,
            const int* __restrict__ input_candidates_per_anchor,
            const int* __restrict__ input_candidates_per_anchor_prefixsum
        ){
            const int numAnchors = input_numAnchors;
            const int numCandidates = input_numCandidates;
    
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const int stride = blockDim.x * gridDim.x;
    
            if(tid == 0){
                *output_numAnchors = numAnchors;
                *output_numCandidates = numCandidates;
            }

            const int s = max(numAnchors + 1, max(numCandidates, numAnchors * encodedSequencePitchInInts));

            for(int i = tid; i < s; i += stride){
                if(i < numAnchors){
                    output_anchor_read_ids[i] = input_anchor_read_ids[i];
                    output_anchor_sequences_lengths[i] = input_anchor_sequences_lengths[i];
                    output_candidates_per_anchor[i] = input_candidates_per_anchor[i];
                }

                if(i < numAnchors + 1){
                    output_candidates_per_anchor_prefixsum[i] = input_candidates_per_anchor_prefixsum[i];
                }

                if(i < numAnchors * encodedSequencePitchInInts){
                    output_anchor_sequences_data[i] = input_anchor_sequences_data[i];
                }

                if(i < numCandidates){
                    output_candidate_read_ids[i] = input_candidate_read_ids[i];
                }
            }
        }
    
        __global__ 
        void copyMinhashResultsKernel(
            int* __restrict__ d_numCandidates,
            int* __restrict__ h_numCandidates,
            read_number* __restrict__ h_candidate_read_ids,
            const int* __restrict__ d_candidates_per_anchor_prefixsum,
            const read_number* __restrict__ d_candidate_read_ids,
            const int numAnchors
        ){
            const int numCandidates = d_candidates_per_anchor_prefixsum[numAnchors];
    
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const int stride = blockDim.x * gridDim.x;
    
            if(tid == 0){
                *d_numCandidates = numCandidates;
                *h_numCandidates = numCandidates;
            }
    
            for(int i = tid; i < numCandidates; i += stride){
                h_candidate_read_ids[i] = d_candidate_read_ids[i];
            }
        }
    
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
    
        __global__ 
        void initArraysBeforeCandidateCorrectionKernel(
            int maxNumCandidates,
            const int* __restrict__ d_numAnchors,
            int* __restrict__ d_num_corrected_candidates_per_anchor,
            bool* __restrict__ d_candidateCanBeCorrected
        ){
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const int stride = blockDim.x * gridDim.x;
    
            const int numAnchors = *d_numAnchors;
    
            for(int i = tid; i < numAnchors; i += stride){
                d_num_corrected_candidates_per_anchor[i] = 0;
            }
    
            for(int i = tid; i < maxNumCandidates; i += stride){
                d_candidateCanBeCorrected[i] = 0;
            }
        }

        __global__
        void compactEditsKernel(
            const care::TempCorrectedSequence::EncodedEdit* __restrict__ d_inputEdits,
            care::TempCorrectedSequence::EncodedEdit* __restrict__ d_outputEdits,
            const int* __restrict__ d_editsOutputOffsets,
            const int* __restrict__ d_numSequences,
            const int* __restrict__ d_numEditsPerSequence,
            int doNotUseEditsValue,
            std::size_t editsPitchInBytes
        ){
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const int stride = blockDim.x * gridDim.x;

            const int N = *d_numSequences;

            for(int c = tid; c < N; c += stride){
                const int numEdits = d_numEditsPerSequence[c];

                if(numEdits != doNotUseEditsValue && numEdits > 0){
                    const int outputOffset = d_editsOutputOffsets[c];

                    auto* outputPtr = d_outputEdits + outputOffset;
                    const auto* inputPtr = (const TempCorrectedSequence::EncodedEdit*)(((const char*)d_inputEdits) 
                        + c * editsPitchInBytes);
                    for(int e = 0; e < numEdits; e++){
                        outputPtr[e] = inputPtr[e];
                    }
                }
            }
        }
        
    } //namespace gpucorrectorkernels   

} //namespace gpu
} //namespace care