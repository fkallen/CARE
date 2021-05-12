#ifndef CARE_GPUCORRECTORKERNELS_CUH
#define CARE_GPUCORRECTORKERNELS_CUH


#include <config.hpp>
#include <correctionresultprocessing.hpp>

#include <gpu/cuda_block_select.cuh>

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
    );

    __global__ 
    void copyMinhashResultsKernel(
        int* __restrict__ d_numCandidates,
        int* __restrict__ h_numCandidates,
        read_number* __restrict__ h_candidate_read_ids,
        const int* __restrict__ d_candidates_per_anchor_prefixsum,
        const read_number* __restrict__ d_candidate_read_ids,
        const int numAnchors
    );

    __global__
    void setAnchorIndicesOfCandidateskernel(
        int* __restrict__ d_anchorIndicesOfCandidates,
        const int* __restrict__ numAnchorsPtr,
        const int* __restrict__ d_candidates_per_anchor,
        const int* __restrict__ d_candidates_per_anchor_prefixsum
    );


    template<int blocksize, class Flags>
    __global__
    void selectIndicesOfFlagsOneBlock(
        int* __restrict__ selectedIndices,
        int* __restrict__ numSelectedIndices,
        const Flags flags,
        const int* __restrict__ numFlagsPtr
    ){
        constexpr int ITEMS_PER_THREAD = 4;
        constexpr int itemsPerIteration = blocksize * ITEMS_PER_THREAD;

        using MyBlockSelect = BlockSelect<int, blocksize>;

        __shared__ typename MyBlockSelect::TempStorage temp_storage;

        int aggregate = 0;
        const int numFlags = *numFlagsPtr;
        const int iters = SDIV(numFlags, blocksize * ITEMS_PER_THREAD);
        const int threadoffset = ITEMS_PER_THREAD * threadIdx.x;

        int remainingItems = numFlags;

        for(int iter = 0; iter < iters; iter++){
            const int validItems = min(remainingItems, itemsPerIteration);

            int data[ITEMS_PER_THREAD];

            const int iteroffset = itemsPerIteration * iter;

            #pragma unroll
            for(int k = 0; k < ITEMS_PER_THREAD; k++){
                if(iteroffset + threadoffset + k < numFlags){
                    data[k] = int(flags[iteroffset + threadoffset + k]);
                }else{
                    data[k] = 0;
                }
            }

            #pragma unroll
            for(int k = 0; k < ITEMS_PER_THREAD; k++){
                if(iteroffset + threadoffset + k < numFlags){
                    data[k] = data[k] != 0 ? 1 : 0;
                }
            }

            const int numSelected = MyBlockSelect(temp_storage).ForEachFlaggedPosition(data, validItems,
                [&](const auto& flaggedPosition, const int& outputpos){
                    selectedIndices[aggregate + outputpos] = iteroffset + flaggedPosition;
                }
            );

            aggregate += numSelected;
            remainingItems -= validItems;

            __syncthreads();
        }

        if(threadIdx.x == 0){
            *numSelectedIndices = aggregate;

            // for(int i = 0; i < aggregate; i++){
            //     printf("%d ", selectedIndices[i]);
            // }
            // printf("\n");
        }

    }

    __global__ 
    void initArraysBeforeCandidateCorrectionKernel(
        int maxNumCandidates,
        const int* __restrict__ d_numAnchors,
        int* __restrict__ d_num_corrected_candidates_per_anchor,
        bool* __restrict__ d_candidateCanBeCorrected
    );

} //namespace gpucorrectorkernels   

} //namespace gpu
} //namespace care


#endif