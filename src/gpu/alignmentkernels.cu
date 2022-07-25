//#define NDEBUG

#include <gpu/kernels.hpp>
#include <gpu/cudaerrorcheck.cuh>

#include <alignmentorientation.hpp>

#include <sequencehelpers.hpp>

#include <hostdevicefunctions.cuh>

#include <hpc_helpers.cuh>
#include <config.hpp>

#include <cassert>


#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <thrust/binary_search.h>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace cg = cooperative_groups;



namespace care{
namespace gpu{

namespace alignmentdetail{

    template<int BLOCKSIZE, int ITEMS_PER_THREAD, class InputIter>
    __global__
    void inclusivePrefixSumLeadingZeroSingleBlockKernel(
        int* __restrict__ output,
        InputIter input,
        int numItems
    ){
        using BlockScan = cub::BlockScan<int, BLOCKSIZE>;

        __shared__ typename BlockScan::TempStorage temp_storage;

        int aggregate = 0;

        const int iters = SDIV(numItems, BLOCKSIZE * ITEMS_PER_THREAD);

        const int threadoffset = ITEMS_PER_THREAD * threadIdx.x;

        if(threadIdx.x == 0){
            output[0] = 0;
        }

        for(int iter = 0; iter < iters; iter++){
            int thread_data[ITEMS_PER_THREAD];

            const int iteroffset = BLOCKSIZE * ITEMS_PER_THREAD * iter;

            #pragma unroll
            for(int k = 0; k < ITEMS_PER_THREAD; k++){
                if(iteroffset + threadoffset + k < numItems){
                    thread_data[k] = input[iteroffset + threadoffset + k];
                }else{
                    thread_data[k] = 0;
                }
            }

            int block_aggregate = 0;
            BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, block_aggregate);

            #pragma unroll
            for(int k = 0; k < ITEMS_PER_THREAD; k++){
                if(iteroffset + threadoffset + k < numItems){
                    output[1+iteroffset + threadoffset + k] = aggregate + thread_data[k];
                }
            }

            aggregate += block_aggregate;

            __syncthreads();
        } 
    }
}


    template<int blocksize, int tilesize>
    __launch_bounds__(blocksize)
    __global__
    void selectIndicesOfGoodCandidatesKernel(
        int* __restrict__ d_indicesOfGoodCandidates,
        int* __restrict__ d_numIndicesPerAnchor,
        int* __restrict__ d_totalNumIndices,
        const AlignmentOrientation* __restrict__ d_alignmentFlags,
        const int* __restrict__ d_candidates_per_anchor,
        const int* __restrict__ d_candidates_per_anchor_prefixsum,
        const int* __restrict__ d_anchorIndicesOfCandidates,
        int numAnchors
    ){

        static_assert(blocksize % tilesize == 0);
        static_assert(tilesize == 32);

        constexpr int numTilesPerBlock = blocksize / tilesize;

        const int numTiles = (gridDim.x * blocksize) / tilesize;
        const int tileId = (threadIdx.x + blockIdx.x * blocksize) / tilesize;
        const int tileIdInBlock = threadIdx.x / tilesize;

        __shared__ int totalIndices;
        __shared__ int counts[numTilesPerBlock];

        if(threadIdx.x == 0){
            totalIndices = 0;
        }
        __syncthreads();

        auto tile = cg::tiled_partition<tilesize>(cg::this_thread_block());

        for(int anchorIndex = tileId; anchorIndex < numAnchors; anchorIndex += numTiles){

            const int offset = d_candidates_per_anchor_prefixsum[anchorIndex];
            int* const indicesPtr = d_indicesOfGoodCandidates + offset;
            int* const numIndicesPtr = d_numIndicesPerAnchor + anchorIndex;
            const AlignmentOrientation* const myAlignmentFlagsPtr = d_alignmentFlags + offset;

            const int numCandidatesForAnchor = d_candidates_per_anchor[anchorIndex];

            if(tile.thread_rank() == 0){
                counts[tileIdInBlock] = 0;
            }
            tile.sync();

            for(int localCandidateIndex = tile.thread_rank(); 
                    localCandidateIndex < numCandidatesForAnchor; 
                    localCandidateIndex += tile.size()){
                
                const AlignmentOrientation alignmentflag = myAlignmentFlagsPtr[localCandidateIndex];

                if(alignmentflag != AlignmentOrientation::None){
                    cg::coalesced_group g = cg::coalesced_threads();
                    int outputPos;
                    if (g.thread_rank() == 0) {
                        outputPos = atomicAdd(&counts[tileIdInBlock], g.size());
                        atomicAdd(&totalIndices, g.size());
                    }
                    outputPos = g.thread_rank() + g.shfl(outputPos, 0);
                    indicesPtr[outputPos] = localCandidateIndex;
                }
            }

            tile.sync();
            if(tile.thread_rank() == 0){
                atomicAdd(numIndicesPtr, counts[tileIdInBlock]);
            }

        }

        __syncthreads();

        if(threadIdx.x == 0){
            atomicAdd(d_totalNumIndices, totalIndices);
        }
    }



    template<int BLOCKSIZE>
    __launch_bounds__(BLOCKSIZE)
    __global__
    void cuda_filter_alignments_by_mismatchratio_kernel(
        AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const int* __restrict__ nOps,
        const int* __restrict__ overlaps,
        const int* __restrict__ d_candidates_per_anchor_prefixsum,
        int numAnchors,
        float mismatchratioBaseFactor,
        float goodAlignmentsCountThreshold
    ){

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceInt::TempStorage intreduce;
            int broadcast[3];
        } temp_storage;

        const int n_anchors = numAnchors;

        for(int anchorindex = blockIdx.x; anchorindex < n_anchors; anchorindex += gridDim.x) {

            const int candidatesForAnchor = d_candidates_per_anchor_prefixsum[anchorindex+1]
                                            - d_candidates_per_anchor_prefixsum[anchorindex];

            const int firstIndex = d_candidates_per_anchor_prefixsum[anchorindex];

            int counts[3]{0,0,0};

            for(int index = threadIdx.x; index < candidatesForAnchor; index += blockDim.x) {

                const int candidate_index = firstIndex + index;
                if(bestAlignmentFlags[candidate_index] != AlignmentOrientation::None) {

                    const int alignment_overlap = overlaps[candidate_index];
                    const int alignment_nops = nOps[candidate_index];

                    const float mismatchratio = float(alignment_nops) / alignment_overlap;

                    if(mismatchratio >= 4 * mismatchratioBaseFactor) {
                        bestAlignmentFlags[candidate_index] = AlignmentOrientation::None;
                    }else{

                        #pragma unroll
                        for(int i = 2; i <= 4; i++) {
                            counts[i-2] += (mismatchratio < i * mismatchratioBaseFactor);
                        }
                    }

                }
            }

            //accumulate counts over block
            #pragma unroll
            for(int i = 0; i < 3; i++) {
                counts[i] = BlockReduceInt(temp_storage.intreduce).Sum(counts[i]);
                __syncthreads();
            }

            //broadcast accumulated counts to block
            if(threadIdx.x == 0) {
                #pragma unroll
                for(int i = 0; i < 3; i++) {
                    temp_storage.broadcast[i] = counts[i];
                }
            }

            __syncthreads();

            #pragma unroll
            for(int i = 0; i < 3; i++) {
                counts[i] = temp_storage.broadcast[i];
            }

            float mismatchratioThreshold = 0;
            if (counts[0] >= goodAlignmentsCountThreshold) {
                mismatchratioThreshold = 2 * mismatchratioBaseFactor;
            } else if (counts[1] >= goodAlignmentsCountThreshold) {
                mismatchratioThreshold = 3 * mismatchratioBaseFactor;
            } else if (counts[2] >= goodAlignmentsCountThreshold) {
                mismatchratioThreshold = 4 * mismatchratioBaseFactor;
            } else {
                mismatchratioThreshold = -1.0f;                         //this will invalidate all alignments for anchor
                //mismatchratioThreshold = 4 * mismatchratioBaseFactor; //use alignments from every bin
                //mismatchratioThreshold = 1.1f;
            }

            // Invalidate all alignments for anchor with mismatchratio >= mismatchratioThreshold
            for(int index = threadIdx.x; index < candidatesForAnchor; index += blockDim.x) {
                const int candidate_index = firstIndex + index;
                if(bestAlignmentFlags[candidate_index] != AlignmentOrientation::None) {

                    const int alignment_overlap = overlaps[candidate_index];
                    const int alignment_nops = nOps[candidate_index];

                    const float mismatchratio = float(alignment_nops) / alignment_overlap;

                    const bool doRemove = mismatchratio >= mismatchratioThreshold;
                    if(doRemove){
                        bestAlignmentFlags[candidate_index] = AlignmentOrientation::None;
                    }
                }
            }
        }
    }



    //compute hamming distance. lhi and llo will be shifted to the left by shift bits
    template<class IndexTransformation1,
                class IndexTransformation2,
                class PopcountFunc>
    __device__
    int hammingdistanceHiLoWithShift(
        const unsigned int* lhi_begin,
        const unsigned int* llo_begin,
        const unsigned int* rhi,
        const unsigned int* rlo,
        int lhi_bitcount,
        int rhi_bitcount,
        int numIntsL,
        int numIntsR,
        int shiftamount,
        int max_errors_excl,
        IndexTransformation1 indextrafoL,
        IndexTransformation2 indextrafoR,
        PopcountFunc popcount
    ){

        const int overlap_bitcount = std::min(std::max(0, lhi_bitcount - shiftamount), rhi_bitcount);

        if(overlap_bitcount == 0)
            return max_errors_excl+1;

        const int partitions = SDIV(overlap_bitcount, (8 * sizeof(unsigned int)));
        const int remaining_bitcount = partitions * sizeof(unsigned int) * 8 - overlap_bitcount;
        const int completeShiftInts = shiftamount / (8 * sizeof(unsigned int));
        const int remainingShift = shiftamount - completeShiftInts * 8 * sizeof(unsigned int);

        #if 0
        auto myfunnelshift = [](unsigned int a, unsigned int b, int shift){
            return (a << shift) | (b >> (8 * sizeof(unsigned int) - shift));
        };
        #else
        auto myfunnelshift = [](unsigned int a, unsigned int b, int shift){
            return __funnelshift_l(b, a, shift);
        };
        #endif

        int result = 0;

        for(int i = 0; i < partitions - 1 && result < max_errors_excl; i += 1) {
            //compute the shifted values of l
            const unsigned int aaa = lhi_begin[indextrafoL(completeShiftInts + i)];
            const unsigned int aab = lhi_begin[indextrafoL(completeShiftInts + i + 1)];
            const unsigned int a = myfunnelshift(aaa, aab, remainingShift);
            const unsigned int baa = llo_begin[indextrafoL(completeShiftInts + i)];
            const unsigned int bab = llo_begin[indextrafoL(completeShiftInts + i + 1)];
            const unsigned int b = myfunnelshift(baa, bab, remainingShift);
            const unsigned int hixor = a ^ rhi[indextrafoR(i)];
            const unsigned int loxor = b ^ rlo[indextrafoR(i)];
            const unsigned int bits = hixor | loxor;
            result += popcount(bits);
            // if(debug){
            //     printf("i %d, %u %u %u %u %d\n",
            //         i, a, rhi[indextrafoR(i)], b, rlo[indextrafoR(i)], result);
            // }
        }

        if(result >= max_errors_excl)
            return result;

        const unsigned int mask = remaining_bitcount == 0 ? 0xFFFFFFFF : 0xFFFFFFFF << (remaining_bitcount);
        
        unsigned int a = 0;
        unsigned int b = 0;
        if(completeShiftInts + partitions - 1 < numIntsL - 1){
            unsigned int aaa = lhi_begin[indextrafoL(completeShiftInts + partitions - 1)];
            unsigned int aab = lhi_begin[indextrafoL(completeShiftInts + partitions - 1 + 1)];
            a = myfunnelshift(aaa, aab, remainingShift);
            unsigned int baa = llo_begin[indextrafoL(completeShiftInts + partitions - 1)];
            unsigned int bab = llo_begin[indextrafoL(completeShiftInts + partitions - 1 + 1)];
            b = myfunnelshift(baa, bab, remainingShift);
        }else{
            a = (lhi_begin[indextrafoL(completeShiftInts + partitions - 1)]) << remainingShift;
            b = (llo_begin[indextrafoL(completeShiftInts + partitions - 1)]) << remainingShift;
        }
        const unsigned int hixor = a ^ rhi[indextrafoR(partitions - 1)];
        const unsigned int loxor = b ^ rlo[indextrafoR(partitions - 1)];
        const unsigned int bits = hixor | loxor;
        result += popcount(bits & mask);

        // if(debug){
        //     printf("i %d, %u %u %u %u %d\n",
        //     partitions - 1, a & mask, rhi[indextrafoR(partitions - 1)] & mask, b & mask, rlo[indextrafoR(partitions - 1)] & mask, result);
        // }

        return result;
    }


    /*

        read-to-read shifted hamming distance

        uses 1 thread per alignment
    */

    template<int blocksize, int encoderGroupSize, bool usePositiveShifts, bool useNegativeShifts>
    __launch_bounds__(blocksize)
    __global__
    void shiftedHammingDistanceKernelSmem1(
        int* __restrict__ d_alignment_overlaps,
        int* __restrict__ d_alignment_shifts,
        int* __restrict__ d_alignment_nOps,
        AlignmentOrientation* __restrict__ d_bestOrientations,
        const unsigned int* __restrict__ anchorData,
        const int* __restrict__ anchorSequencesLength,
        size_t encodedSequencePitchInInts2BitAnchor,
        const unsigned int* __restrict__ candidateData,
        const int* __restrict__ candidateSequencesLength,
        size_t encodedSequencePitchInInts2BitCandidate,
        const int* __restrict__ anchorIndicesOfCandidates,
        size_t smemEncodedSequencePitchInInts2BitHiloAnchor,
        size_t smemEncodedSequencePitchInInts2BitHiloCandidate,
        const bool* __restrict__ anchorContainsN,
        bool removeAmbiguousAnchors,
        const bool* __restrict__ candidateContainsN,
        bool removeAmbiguousCandidates,
        int numCandidates,
        int min_overlap,
        float maxErrorRate, //allow only less than (anchorLength * maxErrorRate) mismatches
        float min_overlap_ratio, //require at least overlap of (anchorLength * minOverlapReate)
        float estimatedNucleotideErrorRate
    ){

        auto block_transposed_index = [](int logical_index) -> int {
            return logical_index * blocksize;
        };

        auto alignmentComparator = [&] (int fwd_alignment_overlap,
            int revc_alignment_overlap,
            int fwd_alignment_nops,
            int revc_alignment_nops,
            bool fwd_alignment_isvalid,
            bool revc_alignment_isvalid,
            int anchorlength,
            int querylength)->AlignmentOrientation{

            return chooseBestAlignmentOrientation(
                fwd_alignment_overlap,
                revc_alignment_overlap,
                fwd_alignment_nops,
                revc_alignment_nops,
                fwd_alignment_isvalid,
                revc_alignment_isvalid,
                anchorlength,
                querylength,
                min_overlap_ratio,
                min_overlap,
                estimatedNucleotideErrorRate * 4.0f
            );
        };

        // each thread stores anchor and candidate
        // sizeof(unsigned int) * encodedSequencePitchInInts2BitHiLoCandidate * blocksize
        //  + sizeof(unsigned int) * encodedSequencePitchInInts2BitHiLoCandidate * blocksize 
        extern __shared__ unsigned int sharedmemory[];

        //set up shared memory pointers
        //data is stored block-transposed to avoid bank conflicts
        unsigned int* const sharedAnchors = sharedmemory;
        unsigned int* const mySharedAnchor = sharedAnchors + threadIdx.x;
        unsigned int* const sharedCandidates = sharedAnchors + smemEncodedSequencePitchInInts2BitHiloAnchor * blocksize;
        unsigned int* const mySharedCandidate = sharedCandidates + threadIdx.x;

        constexpr int numEncoderGroupsPerBlock = blocksize / encoderGroupSize;
        auto encoderGroup = cg::tiled_partition<encoderGroupSize>(cg::this_thread_block());

        const int numBlockIters = SDIV(numCandidates, blocksize);

        for(int blockIter = blockIdx.x; blockIter < numBlockIters; blockIter += gridDim.x){
            const int candidateBlockOffset = blockIter * blocksize;
            const int numCandidatesInBlockIteration = min(numCandidates - candidateBlockOffset, blocksize);

            //convert the candidate sequences to HiLo format and store in shared memory
            for(int s = encoderGroup.meta_group_rank(); s < numCandidatesInBlockIteration; s += numEncoderGroupsPerBlock){
                const int candidateIndex = candidateBlockOffset + s;
                const int length = candidateSequencesLength[candidateIndex];
                unsigned int* const out = sharedCandidates + s;
                const unsigned int* const in = candidateData + encodedSequencePitchInInts2BitCandidate * candidateIndex;
                auto inindextrafo = [](auto i){return i;};
                auto outindextrafo = block_transposed_index;
                convert2BitTo2BitHiLo(
                    encoderGroup,
                    out,
                    in,
                    length,
                    inindextrafo,
                    outindextrafo
                );
            }

            //convert the anchor sequences to HiLo format and store in shared memory
            for(int s = encoderGroup.meta_group_rank(); s < numCandidatesInBlockIteration; s += numEncoderGroupsPerBlock){
                const int anchorIndex = (anchorIndicesOfCandidates == nullptr ? 
                    candidateBlockOffset + s
                    : anchorIndicesOfCandidates[candidateBlockOffset + s]);
                const int length = anchorSequencesLength[anchorIndex];
                unsigned int* const out = sharedAnchors + s;
                const unsigned int* const in = anchorData + encodedSequencePitchInInts2BitAnchor * anchorIndex;
                auto inindextrafo = [](auto i){return i;};
                auto outindextrafo = block_transposed_index;
                convert2BitTo2BitHiLo(
                    encoderGroup,
                    out,
                    in,
                    length,
                    inindextrafo,
                    outindextrafo
                );
            }

            __syncthreads();

            const int candidateIndex = candidateBlockOffset + threadIdx.x;
            if(candidateIndex < numCandidates){
                const int anchorIndex = (anchorIndicesOfCandidates == nullptr ? 
                    candidateIndex
                    : anchorIndicesOfCandidates[candidateIndex]);

                if(!(removeAmbiguousAnchors && anchorContainsN[anchorIndex]) && !(removeAmbiguousCandidates && candidateContainsN[candidateIndex])){

                    const int anchorLength = anchorSequencesLength[anchorIndex];
                    const int anchorints = SequenceHelpers::getEncodedNumInts2BitHiLo(anchorLength);
                    assert(anchorints <= int(smemEncodedSequencePitchInInts2BitHiloAnchor));
                    const int candidateLength = candidateSequencesLength[candidateIndex];
                    const int candidateints = SequenceHelpers::getEncodedNumInts2BitHiLo(candidateLength);
                    assert(candidateints <= int(smemEncodedSequencePitchInInts2BitHiloCandidate));

                    const unsigned int* const anchor_hi = mySharedAnchor;
                    const unsigned int* const anchor_lo = mySharedAnchor + block_transposed_index(anchorints / 2);
                    const unsigned int* const candidate_hi = mySharedCandidate;
                    const unsigned int* const candidate_lo = mySharedCandidate + block_transposed_index(candidateints / 2);

                    const int minoverlap = max(min_overlap, int(float(anchorLength) * min_overlap_ratio));

                    int bestScore[2]{std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
                    int bestShift[2]{-candidateLength, -candidateLength};
                    int bestOverlap[2]{0,0};                    

                    for(int orientation = 0; orientation < 2; orientation++){
                        const bool isReverseComplement = orientation == 1;

                        if(isReverseComplement) {
                            SequenceHelpers::reverseComplementSequenceInplace2BitHiLo(mySharedCandidate, candidateLength, block_transposed_index);
                        }

                        auto updateBest = [&](int shift, int overlapsize, int hammingDistance, int max_errors_excl){
                            //treat non-overlapping positions as mismatches to prefer a greater overlap if hamming distance is equal for multiple shifts
                            const int nonoverlapping = anchorLength + candidateLength - 2 * overlapsize;
                            const int score = (hammingDistance < max_errors_excl ?
                                hammingDistance + nonoverlapping // non-overlapping regions count as mismatches
                                : std::numeric_limits<int>::max()); // too many errors, discard

                            if(score < bestScore[orientation]){
                                bestScore[orientation] = score;
                                bestShift[orientation] = shift;
                                bestOverlap[orientation] = overlapsize;
                            }                
                        };

                        if constexpr (usePositiveShifts){
                            //compute positive shifts ,i.e. shift anchor to the left

                            for(int shift = 0; shift < anchorLength - minoverlap + 1; shift += 1) {
                                const int overlapsize = min(anchorLength - shift, candidateLength);
                                const int max_errors_excl = min(int(float(overlapsize) * maxErrorRate),
                                    bestScore[orientation] - (anchorLength + candidateLength) + 2*overlapsize);
                            
                                if(max_errors_excl > 0){
                                    const int hammingDistance = hammingdistanceHiLoWithShift(
                                        anchor_hi,
                                        anchor_lo,
                                        candidate_hi,
                                        candidate_lo,
                                        anchorLength,
                                        candidateLength,
                                        anchorints / 2,
                                        candidateints / 2,
                                        shift,
                                        max_errors_excl,
                                        block_transposed_index,
                                        block_transposed_index,
                                        [](auto i){return __popc(i);}
                                    );

                                    updateBest(shift, overlapsize, hammingDistance, max_errors_excl);
                                }
                            }
                        }

                        if constexpr(useNegativeShifts){

                            //compute negative shifts ,i.e. shift candidate to the left
                            for(int shift = -1; shift >= - candidateLength + minoverlap; shift -= 1) {
                                const int overlapsize = min(anchorLength, candidateLength + shift);
                                const int max_errors_excl = min(int(float(overlapsize) * maxErrorRate),
                                    bestScore[orientation] - (anchorLength + candidateLength) + 2*overlapsize);

                                if(max_errors_excl > 0){
                                    const int hammingDistance = hammingdistanceHiLoWithShift(
                                        candidate_hi,
                                        candidate_lo,
                                        anchor_hi,
                                        anchor_lo,
                                        candidateLength,
                                        anchorLength,
                                        candidateints / 2,
                                        anchorints / 2,
                                        -shift,
                                        max_errors_excl,
                                        block_transposed_index,
                                        block_transposed_index,
                                        [](auto i){return __popc(i);}
                                    );

                                    updateBest(shift, overlapsize, hammingDistance, max_errors_excl); 
                                }
                            }
                        }
                    }


                    //compare fwd alignment and revc alignment.
                    const int hammingdistance0 = bestScore[0] - (anchorLength + candidateLength) + 2*bestOverlap[0];
                    const int hammingdistance1 = bestScore[1] - (anchorLength + candidateLength) + 2*bestOverlap[1];

                    const AlignmentOrientation flag = alignmentComparator(
                        bestOverlap[0],
                        bestOverlap[1],
                        hammingdistance0,
                        hammingdistance1,
                        bestShift[0] != -candidateLength,
                        bestShift[1] != -candidateLength,
                        anchorLength,
                        candidateLength
                    );

                    d_bestOrientations[candidateIndex] = flag;
                    if(flag != AlignmentOrientation::None){
                        d_alignment_overlaps[candidateIndex] = flag == AlignmentOrientation::Forward ? bestOverlap[0] : bestOverlap[1];
                        d_alignment_shifts[candidateIndex] = flag == AlignmentOrientation::Forward ? bestShift[0] : bestShift[1];
                        d_alignment_nOps[candidateIndex] = flag == AlignmentOrientation::Forward ? hammingdistance0 : hammingdistance1;
                    }

                }else{
                    d_bestOrientations[candidateIndex] = AlignmentOrientation::None;
                }
            }
            __syncthreads();
        }
    }



    //####################   KERNEL DISPATCH   ####################


    void call_cuda_filter_alignments_by_mismatchratio_kernel_async(
        AlignmentOrientation* d_bestAlignmentFlags,
        const int* d_nOps,
        const int* d_overlaps,
        const int* d_candidates_per_anchor_prefixsum,
        int numAnchors,
        int /*numCandidates*/,
        float mismatchratioBaseFactor,
        float goodAlignmentsCountThreshold,
        cudaStream_t stream
    ){


        constexpr int blocksize = 128;
        const std::size_t smem = 0;

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            cuda_filter_alignments_by_mismatchratio_kernel<blocksize>,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

        dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(numAnchors, maxBlocks));

        cuda_filter_alignments_by_mismatchratio_kernel<blocksize><<<grid, block, smem, stream>>>( 
            d_bestAlignmentFlags, 
            d_nOps, 
            d_overlaps, 
            d_candidates_per_anchor_prefixsum, 
            numAnchors,
            mismatchratioBaseFactor, 
            goodAlignmentsCountThreshold
        ); 
        CUDACHECKASYNC;


    }


    void callSelectIndicesOfGoodCandidatesKernelAsync(
        int* d_indicesOfGoodCandidates,
        int* d_numIndicesPerAnchor,
        int* d_totalNumIndices,
        const AlignmentOrientation* d_alignmentFlags,
        const int* d_candidates_per_anchor,
        const int* d_candidates_per_anchor_prefixsum,
        const int* d_anchorIndicesOfCandidates,
        int numAnchors,
        int numCandidates,
        cudaStream_t stream
    ){

        constexpr int blocksize = 128;
        constexpr int tilesize = 32;

        const std::size_t smem = 0;

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            selectIndicesOfGoodCandidatesKernel<blocksize, tilesize>,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

        helpers::lambda_kernel<<<SDIV(numCandidates, 256), 256, 0, stream>>>([=] __device__(){
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;

            if(tid < numCandidates){
                d_indicesOfGoodCandidates[tid] = -1;
            }

            if(tid < numAnchors){
                d_numIndicesPerAnchor[tid] = 0;
            }

            if(tid == 0){
                *d_totalNumIndices = 0;
            }
        }); CUDACHECKASYNC;

        dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(SDIV(numCandidates, blocksize), maxBlocks));

        selectIndicesOfGoodCandidatesKernel<blocksize, tilesize><<<grid, block, 0, stream>>>(
            d_indicesOfGoodCandidates,
            d_numIndicesPerAnchor,
            d_totalNumIndices,
            d_alignmentFlags,
            d_candidates_per_anchor,
            d_candidates_per_anchor_prefixsum,
            d_anchorIndicesOfCandidates,
            numAnchors
        ); CUDACHECKASYNC;
    }








    

    template<int blocksize, int groupsize, bool usePositiveShifts, bool useNegativeShifts>
    void callShiftedHammingDistanceKernelSmem1_impl(
        int* d_alignment_overlaps,
        int* d_alignment_shifts,
        int* d_alignment_nOps,
        AlignmentOrientation* d_alignment_best_alignment_flags,
        const unsigned int* d_anchorSequencesData,
        const unsigned int* d_candidateSequencesData,
        const int* d_anchorSequencesLength,
        const int* d_candidateSequencesLength,
        const int* d_anchorIndicesOfCandidates,
        int numAnchors,
        int numCandidates,
        const bool* d_anchorContainsN,
        bool removeAmbiguousAnchors,
        const bool* d_candidateContainsN,
        bool removeAmbiguousCandidates,
        int maximumSequenceLengthAnchor,
        int maximumSequenceLengthCandidate,
        std::size_t encodedSequencePitchInInts2BitAnchor,
        std::size_t encodedSequencePitchInInts2BitCandidate,
        int min_overlap,
        float maxErrorRate,
        float min_overlap_ratio,
        float estimatedNucleotideErrorRate,
        cudaStream_t stream
    ){
        if(numAnchors == 0 || numCandidates == 0) return;

        const std::size_t intsPerSequenceHiLoAnchor = SequenceHelpers::getEncodedNumInts2BitHiLo(maximumSequenceLengthAnchor);
        const std::size_t intsPerSequenceHiLoCandidates = SequenceHelpers::getEncodedNumInts2BitHiLo(maximumSequenceLengthCandidate);

        auto kernel = shiftedHammingDistanceKernelSmem1<blocksize, groupsize,usePositiveShifts,useNegativeShifts>;
        const std::size_t smem = sizeof(unsigned int) * (intsPerSequenceHiLoAnchor * blocksize + intsPerSequenceHiLoCandidates * blocksize);
        // std::cerr << "intsPerSequenceHiLoAnchor: " << intsPerSequenceHiLoAnchor << "\n";
        // std::cerr << "intsPerSequenceHiLoCandidates: " << intsPerSequenceHiLoCandidates << "\n";
        // std::cerr << "smem: " << smem << "\n";

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            kernel,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;  

        dim3 block(blocksize, 1, 1);
        const int numBlocks = SDIV(numCandidates, blocksize);
        dim3 grid(std::min(numBlocks, maxBlocks), 1, 1);

        kernel<<<grid, block, smem, stream>>>(
            d_alignment_overlaps,
            d_alignment_shifts,
            d_alignment_nOps,
            d_alignment_best_alignment_flags,
            d_anchorSequencesData,
            d_anchorSequencesLength,
            encodedSequencePitchInInts2BitAnchor,
            d_candidateSequencesData,
            d_candidateSequencesLength,
            encodedSequencePitchInInts2BitCandidate,
            d_anchorIndicesOfCandidates,
            intsPerSequenceHiLoAnchor,
            intsPerSequenceHiLoCandidates,
            d_anchorContainsN,
            removeAmbiguousAnchors,
            d_candidateContainsN,
            removeAmbiguousCandidates,
            numCandidates,
            min_overlap,
            maxErrorRate,
            min_overlap_ratio,
            estimatedNucleotideErrorRate
        );
        CUDACHECKASYNC;
    }


    void callShiftedHammingDistanceKernel(
        int* d_alignment_overlaps,
        int* d_alignment_shifts,
        int* d_alignment_nOps,
        AlignmentOrientation* d_alignment_best_alignment_flags,
        const unsigned int* d_anchorSequencesData,
        const unsigned int* d_candidateSequencesData,
        const int* d_anchorSequencesLength,
        const int* d_candidateSequencesLength,
        const int* d_anchorIndicesOfCandidates,
        int numAnchors,
        int numCandidates,
        const bool* d_anchorContainsN,
        bool removeAmbiguousAnchors,
        const bool* d_candidateContainsN,
        bool removeAmbiguousCandidates,
        int maximumSequenceLengthAnchor,
        int maximumSequenceLengthCandidate,
        std::size_t encodedSequencePitchInInts2BitAnchor,
        std::size_t encodedSequencePitchInInts2BitCandidate,
        int min_overlap,
        float maxErrorRate,
        float min_overlap_ratio,
        float estimatedNucleotideErrorRate,
        cudaStream_t stream
    ){
        constexpr bool usePositiveShifts = true;
        constexpr bool useNegativeShifts = true;
        callShiftedHammingDistanceKernelSmem1_impl<128, 2,usePositiveShifts,useNegativeShifts>(
            d_alignment_overlaps,
            d_alignment_shifts,
            d_alignment_nOps,
            d_alignment_best_alignment_flags,
            d_anchorSequencesData,
            d_candidateSequencesData,
            d_anchorSequencesLength,
            d_candidateSequencesLength,
            d_anchorIndicesOfCandidates,
            numAnchors,
            numCandidates,
            d_anchorContainsN,
            removeAmbiguousAnchors,
            d_candidateContainsN,
            removeAmbiguousCandidates,
            maximumSequenceLengthAnchor,
            maximumSequenceLengthCandidate,
            encodedSequencePitchInInts2BitAnchor,
            encodedSequencePitchInInts2BitCandidate,
            min_overlap,
            maxErrorRate,
            min_overlap_ratio,
            estimatedNucleotideErrorRate,
            stream
        );
    }

    void callRightShiftedHammingDistanceKernel(
        int* d_alignment_overlaps,
        int* d_alignment_shifts,
        int* d_alignment_nOps,
        AlignmentOrientation* d_alignment_best_alignment_flags,
        const unsigned int* d_anchorSequencesData,
        const unsigned int* d_candidateSequencesData,
        const int* d_anchorSequencesLength,
        const int* d_candidateSequencesLength,
        const int* d_anchorIndicesOfCandidates,
        int numAnchors,
        int numCandidates,
        const bool* d_anchorContainsN,
        bool removeAmbiguousAnchors,
        const bool* d_candidateContainsN,
        bool removeAmbiguousCandidates,
        int maximumSequenceLengthAnchor,
        int maximumSequenceLengthCandidate,
        std::size_t encodedSequencePitchInInts2BitAnchor,
        std::size_t encodedSequencePitchInInts2BitCandidate,
        int min_overlap,
        float maxErrorRate,
        float min_overlap_ratio,
        float estimatedNucleotideErrorRate,
        cudaStream_t stream
    ){
        constexpr bool usePositiveShifts = true;
        constexpr bool useNegativeShifts = false;
        callShiftedHammingDistanceKernelSmem1_impl<128, 2,usePositiveShifts,useNegativeShifts>(
            d_alignment_overlaps,
            d_alignment_shifts,
            d_alignment_nOps,
            d_alignment_best_alignment_flags,
            d_anchorSequencesData,
            d_candidateSequencesData,
            d_anchorSequencesLength,
            d_candidateSequencesLength,
            d_anchorIndicesOfCandidates,
            numAnchors,
            numCandidates,
            d_anchorContainsN,
            removeAmbiguousAnchors,
            d_candidateContainsN,
            removeAmbiguousCandidates,
            maximumSequenceLengthAnchor,
            maximumSequenceLengthCandidate,
            encodedSequencePitchInInts2BitAnchor,
            encodedSequencePitchInInts2BitCandidate,
            min_overlap,
            maxErrorRate,
            min_overlap_ratio,
            estimatedNucleotideErrorRate,
            stream
        );
    }






}
}
