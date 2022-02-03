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

namespace cg = cooperative_groups;



namespace care{
namespace gpu{


    template<int blocksize, int tilesize>
    __global__
    void selectIndicesOfGoodCandidatesKernel(
            int* __restrict__ d_indicesOfGoodCandidates,
            int* __restrict__ d_numIndicesPerAnchor,
            int* __restrict__ d_totalNumIndices,
            const AlignmentOrientation* __restrict__ d_alignmentFlags,
            const int* __restrict__ d_candidates_per_anchor,
            const int* __restrict__ d_candidates_per_anchor_prefixsum,
            const int* __restrict__ d_anchorIndicesOfCandidates,
            const int* __restrict__ d_numAnchors,
            const int* __restrict__ d_numCandidates
            ){

        static_assert(blocksize % tilesize == 0);
        static_assert(tilesize == 32);

        constexpr int numTilesPerBlock = blocksize / tilesize;

        const int numAnchors = *d_numAnchors;
        //const int numCandidates = *d_numCandidates;

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




    /*

        For each candidate, compute the alignment of anchor|candidate and anchor|revc-candidate
        Compares both alignments and keeps the better one

        Sequences are stored in dynamic sized shared memory.
        To reduce shared memory usage, the candidates belonging to the same anchor
        are processed by a set of tiles. Each tile only computes alignments for a single anchor.
        This anchor is stored in shared memory and shared by all threads within a tile
    */

    template<int tilesize>
    __global__
    void
    popcount_shifted_hamming_distance_smem_kernel(
                const unsigned int* __restrict__ anchorDataHiLo,
                const unsigned int* __restrict__ candidateDataHiLoTransposed,
                int* __restrict__ d_alignment_overlaps,
                int* __restrict__ d_alignment_shifts,
                int* __restrict__ d_alignment_nOps,
                bool* __restrict__ d_alignment_isValid,
                AlignmentOrientation* __restrict__ d_alignment_best_alignment_flags,
                const int* __restrict__ anchorSequencesLength,
                const int* __restrict__ candidateSequencesLength,
                const int* __restrict__ candidates_per_anchor_prefixsum,
                const int* __restrict__ tiles_per_anchor_prefixsum,
                const int* __restrict__ numAnchorsPtr,
                const int* __restrict__ numCandidatesPtr,
                const bool* __restrict__ anchorContainsN,
                bool removeAmbiguousAnchors,
                const bool* __restrict__ candidateContainsN,
                bool removeAmbiguousCandidates,
                int encodedSequencePitchInInts2BitHiLo,
                int min_overlap,
                float maxErrorRate,
                float min_overlap_ratio,
                float estimatedNucleotideErrorRate){

        const int n_anchors = *numAnchorsPtr;
        const int n_candidates = *numCandidatesPtr;

        auto no_bank_conflict_index = [](int logical_index) -> int {
            return logical_index * blockDim.x;
        };

        auto identity = [](auto logical_index){
            return logical_index;
        };

        auto popcount = [](auto i){return __popc(i);};

        auto hammingDistanceWithShift = [&](bool doShift, int overlapsize, int max_errors,
                                    unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                    int shiftptr_size,
                                    const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                    auto transfunc2){

            if(doShift){
                shiftBitArrayLeftBy<1>(shiftptr_hi, shiftptr_size / 2, transfunc1);
                shiftBitArrayLeftBy<1>(shiftptr_lo, shiftptr_size / 2, transfunc1);
            }

            const int score = hammingdistanceHiLo(shiftptr_hi,
                                                shiftptr_lo,
                                                otherptr_hi,
                                                otherptr_lo,
                                                overlapsize,
                                                overlapsize,
                                                max_errors,
                                                transfunc1,
                                                transfunc2,
                                                popcount);

            return score;
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

        // sizeof(char) * (max_sequence_bytes * num_tiles   // tiles share the anchor
        //                    + max_sequence_bytes * num_threads // each thread works with its own candidate
        //                    + max_sequence_bytes * num_threads) // each thread needs memory to shift a sequence
        extern __shared__ unsigned int sharedmemory[];

        //set up shared memory pointers

        const int tiles = (blockDim.x * gridDim.x) / tilesize;
        const int globalTileId = (blockDim.x * blockIdx.x + threadIdx.x) / tilesize;
        const int localTileId = (threadIdx.x) / tilesize;
        const int tilesPerBlock = blockDim.x / tilesize;
        const int laneInTile = threadIdx.x % tilesize;
        const int requiredTiles = tiles_per_anchor_prefixsum[n_anchors];

        unsigned int* const anchorBackupsBegin = sharedmemory; // per tile shared memory to store anchor
        unsigned int* const queryBackupsBegin = anchorBackupsBegin + encodedSequencePitchInInts2BitHiLo * tilesPerBlock; // per thread shared memory to store query
        unsigned int* const mySequencesBegin = queryBackupsBegin + encodedSequencePitchInInts2BitHiLo * blockDim.x; // per thread shared memory to store shifted sequence

        unsigned int* const anchorBackup = anchorBackupsBegin + encodedSequencePitchInInts2BitHiLo * localTileId; // accesed via identity
        unsigned int* const queryBackup = queryBackupsBegin + threadIdx.x; // accesed via no_bank_conflict_index
        unsigned int* const mySequence = mySequencesBegin + threadIdx.x; // accesed via no_bank_conflict_index

        for(int logicalTileId = globalTileId; logicalTileId < requiredTiles ; logicalTileId += tiles){

            const int anchorIndex = thrust::distance(tiles_per_anchor_prefixsum,
                                                    thrust::lower_bound(
                                                        thrust::seq,
                                                        tiles_per_anchor_prefixsum,
                                                        tiles_per_anchor_prefixsum + n_anchors + 1,
                                                        logicalTileId + 1))-1;

            const int candidatesBeforeThisAnchor = candidates_per_anchor_prefixsum[anchorIndex];
            const int maxCandidateIndex_excl = candidates_per_anchor_prefixsum[anchorIndex+1];
            //const int tilesForThisAnchor = tiles_per_anchor_prefixsum[anchorIndex + 1] - tiles_per_anchor_prefixsum[anchorIndex];
            const int tileForThisAnchor = logicalTileId - tiles_per_anchor_prefixsum[anchorIndex];
            const int candidateIndex = candidatesBeforeThisAnchor + tileForThisAnchor * tilesize + laneInTile;

            const int anchorbases = anchorSequencesLength[anchorIndex];
            const int anchorints = SequenceHelpers::getEncodedNumInts2BitHiLo(anchorbases);
            const unsigned int* anchorptr = anchorDataHiLo + std::size_t(anchorIndex) * encodedSequencePitchInInts2BitHiLo;

            //save anchor in shared memory (in parallel, per tile)
            for(int lane = laneInTile; lane < encodedSequencePitchInInts2BitHiLo; lane += tilesize) {
                anchorBackup[identity(lane)] = anchorptr[lane];
                //transposed
                //anchorBackup[identity(lane)] = ((unsigned int*)(anchorptr))[lane * n_anchors];
            }

            cg::tiled_partition<tilesize>(cg::this_thread_block()).sync();


            if(candidateIndex < maxCandidateIndex_excl){
                if(!(removeAmbiguousAnchors && anchorContainsN[anchorIndex]) && !(removeAmbiguousCandidates && candidateContainsN[candidateIndex])){
                    const int querybases = candidateSequencesLength[candidateIndex];
                    const int queryints = SequenceHelpers::getEncodedNumInts2BitHiLo(querybases);
                    const int totalbases = anchorbases + querybases;
                    const int minoverlap = max(min_overlap, int(float(anchorbases) * min_overlap_ratio));

                    const unsigned int* candidateptr = candidateDataHiLoTransposed + std::size_t(candidateIndex);

                    //save query in shared memory
                    for(int i = 0; i < encodedSequencePitchInInts2BitHiLo; i += 1) {
                        //queryBackup[no_bank_conflict_index(i)] = ((unsigned int*)(candidateptr))[i];
                        //transposed
                        queryBackup[no_bank_conflict_index(i)] = candidateptr[i * n_candidates];
                    }

                    const unsigned int* const anchorBackup_hi = anchorBackup;
                    const unsigned int* const anchorBackup_lo = anchorBackup + identity(anchorints/2);
                    const unsigned int* const queryBackup_hi = queryBackup;
                    const unsigned int* const queryBackup_lo = queryBackup + no_bank_conflict_index(queryints/2);

                    int bestScore[2];
                    int bestShift[2];
                    int overlapsize[2];
                    int opnr[2];

                    #pragma unroll
                    for(int orientation = 0; orientation < 2; orientation++){
                        const bool isReverseComplement = orientation == 1;

                        if(isReverseComplement) {
                            SequenceHelpers::reverseComplementSequenceInplace2BitHiLo(queryBackup, querybases, no_bank_conflict_index);
                        }

                        //begin SHD algorithm

                        bestScore[orientation] = totalbases;     // score is number of mismatches
                        bestShift[orientation] = -querybases;    // shift of query relative to anchor. shift < 0 if query begins before anchor

                        auto handle_shift = [&](int shift, int overlapsize,
                                                    unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                                    int shiftptr_size,
                                                    const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                                    auto transfunc2){

                            //const int max_errors = int(float(overlapsize) * maxErrorRate);
                            const int max_errors_excl = min(int(float(overlapsize) * maxErrorRate),
                            bestScore[orientation] - totalbases + 2*overlapsize);

                            if(max_errors_excl > 0){

                                int score = hammingDistanceWithShift(shift != 0, overlapsize, max_errors_excl,
                                                    shiftptr_hi,shiftptr_lo, transfunc1,
                                                    shiftptr_size,
                                                    otherptr_hi, otherptr_lo, transfunc2);

                                

                                // printf("%d, %d %d %d --- ", queryIndex, shift, overlapsize, score);

                                // printf("%d %d %d %d | %d %d %d %d --- ", 
                                //     shiftptr_hi[transfunc1(0)], shiftptr_hi[transfunc1(1)], shiftptr_hi[transfunc1(2)], shiftptr_hi[transfunc1(3)],
                                //     shiftptr_lo[transfunc1(0)], shiftptr_lo[transfunc1(1)], shiftptr_lo[transfunc1(2)], shiftptr_lo[transfunc1(3)]);

                                // printf("%d %d %d %d | %d %d %d %d\n", 
                                //     otherptr_hi[transfunc2(0)], otherptr_hi[transfunc2(1)], otherptr_hi[transfunc2(2)], otherptr_hi[transfunc2(3)],
                                //     otherptr_lo[transfunc2(0)], otherptr_lo[transfunc2(1)], otherptr_lo[transfunc2(2)], otherptr_lo[transfunc2(3)]);

                                score = (score < max_errors_excl ?
                                        score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                                        : std::numeric_limits<int>::max()); // too many errors, discard

                                if(score < bestScore[orientation]){
                                    bestScore[orientation] = score;
                                    bestShift[orientation] = shift;
                                }

                                return true;
                            }else{
                                //printf("%d, %d %d %d max_errors_excl\n", queryIndex, shift, overlapsize, max_errors_excl);
                                return false;
                            }
                        };

                        //initialize threadlocal smem array with anchor
                        for(int i = 0; i < encodedSequencePitchInInts2BitHiLo; i += 1) {
                            mySequence[no_bank_conflict_index(i)] = anchorBackup[identity(i)];
                        }

                        unsigned int* mySequence_hi = mySequence;
                        unsigned int* mySequence_lo = mySequence + no_bank_conflict_index(anchorints / 2);

                        for(int shift = 0; shift < anchorbases - minoverlap + 1; shift += 1) {
                            const int overlapsize = min(anchorbases - shift, querybases);

                            bool b = handle_shift(shift, overlapsize,
                                            mySequence_hi, mySequence_lo, no_bank_conflict_index,
                                            anchorints,
                                            queryBackup_hi, queryBackup_lo, no_bank_conflict_index);
                            if(!b){
                                break;
                            }
                        }

                        //initialize threadlocal smem array with query
                        for(int i = 0; i < encodedSequencePitchInInts2BitHiLo; i += 1) {
                            mySequence[no_bank_conflict_index(i)] = queryBackup[no_bank_conflict_index(i)];
                        }

                        mySequence_hi = mySequence;
                        mySequence_lo = mySequence + no_bank_conflict_index(queryints / 2);

                        for(int shift = -1; shift >= -querybases + minoverlap; shift -= 1) {
                            const int overlapsize = min(anchorbases, querybases + shift);

                            bool b = handle_shift(shift, overlapsize,
                                            mySequence_hi, mySequence_lo, no_bank_conflict_index,
                                            queryints,
                                            anchorBackup_hi, anchorBackup_lo, identity);
                            if(!b){
                                break;
                            }
                        }

                        const int queryoverlapbegin_incl = max(-bestShift[orientation], 0);
                        const int queryoverlapend_excl = min(querybases, anchorbases - bestShift[orientation]);
                        overlapsize[orientation] = queryoverlapend_excl - queryoverlapbegin_incl;
                        opnr[orientation] = bestScore[orientation] - totalbases + 2*overlapsize[orientation];
                    }

                    const AlignmentOrientation flag = alignmentComparator(
                        overlapsize[0],
                        overlapsize[1],
                        opnr[0],
                        opnr[1],
                        bestShift[0] != -querybases,
                        bestShift[1] != -querybases,
                        anchorbases,
                        querybases
                    );

                    d_alignment_best_alignment_flags[candidateIndex] = flag;
                    d_alignment_overlaps[candidateIndex] = flag == AlignmentOrientation::Forward ? overlapsize[0] : overlapsize[1];
                    d_alignment_shifts[candidateIndex] = flag == AlignmentOrientation::Forward ? bestShift[0] : bestShift[1];
                    d_alignment_nOps[candidateIndex] = flag == AlignmentOrientation::Forward ? opnr[0] : opnr[1];
                    d_alignment_isValid[candidateIndex] = flag == AlignmentOrientation::Forward ? bestShift[0] != -querybases : bestShift[1] != -querybases;
                }else{
                    d_alignment_best_alignment_flags[candidateIndex] = AlignmentOrientation::None;
                    d_alignment_isValid[candidateIndex] = false;
                }
            }
        }
    }

    template<int tilesize>
    __global__
    void
    popcount_rightshifted_hamming_distance_smem_kernel(
                const unsigned int* __restrict__ anchorDataHiLo,
                const unsigned int* __restrict__ candidateDataHiLoTransposed,
                int* __restrict__ d_alignment_overlaps,
                int* __restrict__ d_alignment_shifts,
                int* __restrict__ d_alignment_nOps,
                bool* __restrict__ d_alignment_isValid,
                AlignmentOrientation* __restrict__ d_alignment_best_alignment_flags,
                const int* __restrict__ anchorSequencesLength,
                const int* __restrict__ candidateSequencesLength,
                const int* __restrict__ candidates_per_anchor_prefixsum,
                const int* __restrict__ tiles_per_anchor_prefixsum,
                const int* __restrict__ numAnchorsPtr,
                const int* __restrict__ numCandidatesPtr,
                const bool* __restrict__ anchorContainsN,
                bool removeAmbiguousAnchors,
                const bool* __restrict__ candidateContainsN,
                bool removeAmbiguousCandidates,
                int encodedSequencePitchInInts2BitHiLo,
                int min_overlap,
                float maxErrorRate,
                float min_overlap_ratio,
                float estimatedNucleotideErrorRate){

        const int n_anchors = *numAnchorsPtr;
        const int n_candidates = *numCandidatesPtr;

        auto make_reverse_complement_inplace = [&](unsigned int* sequence, int sequencelength, auto indextrafo){
            SequenceHelpers::reverseComplementSequenceInplace2BitHiLo((unsigned int*)sequence, sequencelength, indextrafo);
        };

        auto no_bank_conflict_index = [](int logical_index) -> int {
            return logical_index * blockDim.x;
        };

        auto identity = [](auto logical_index){
            return logical_index;
        };

        auto popcount = [](auto i){return __popc(i);};

        auto hammingDistanceWithShift = [&](bool doShift, int overlapsize, int max_errors,
                                    unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                    int shiftptr_size,
                                    const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                    auto transfunc2){

            if(doShift){
                shiftBitArrayLeftBy<1>(shiftptr_hi, shiftptr_size / 2, transfunc1);
                shiftBitArrayLeftBy<1>(shiftptr_lo, shiftptr_size / 2, transfunc1);
            }

            const int score = hammingdistanceHiLo(shiftptr_hi,
                                                shiftptr_lo,
                                                otherptr_hi,
                                                otherptr_lo,
                                                overlapsize,
                                                overlapsize,
                                                max_errors,
                                                transfunc1,
                                                transfunc2,
                                                popcount);

            return score;
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

        // sizeof(char) * (max_sequence_bytes * num_tiles   // tiles share the anchor
        //                    + max_sequence_bytes * num_threads // each thread works with its own candidate
        //                    + max_sequence_bytes * num_threads) // each thread needs memory to shift a sequence
        extern __shared__ unsigned int sharedmemory[];

        //set up shared memory pointers

        const int tiles = (blockDim.x * gridDim.x) / tilesize;
        const int globalTileId = (blockDim.x * blockIdx.x + threadIdx.x) / tilesize;
        const int localTileId = (threadIdx.x) / tilesize;
        const int tilesPerBlock = blockDim.x / tilesize;
        const int laneInTile = threadIdx.x % tilesize;
        const int requiredTiles = tiles_per_anchor_prefixsum[n_anchors];

        unsigned int* const anchorBackupsBegin = sharedmemory; // per tile shared memory to store anchor
        unsigned int* const queryBackupsBegin = anchorBackupsBegin + encodedSequencePitchInInts2BitHiLo * tilesPerBlock; // per thread shared memory to store query
        unsigned int* const mySequencesBegin = queryBackupsBegin + encodedSequencePitchInInts2BitHiLo * blockDim.x; // per thread shared memory to store shifted sequence

        unsigned int* const anchorBackup = anchorBackupsBegin + encodedSequencePitchInInts2BitHiLo * localTileId; // accesed via identity
        unsigned int* const queryBackup = queryBackupsBegin + threadIdx.x; // accesed via no_bank_conflict_index
        unsigned int* const mySequence = mySequencesBegin + threadIdx.x; // accesed via no_bank_conflict_index

        for(int logicalTileId = globalTileId; logicalTileId < requiredTiles ; logicalTileId += tiles){

            const int anchorIndex = thrust::distance(tiles_per_anchor_prefixsum,
                                                    thrust::lower_bound(
                                                        thrust::seq,
                                                        tiles_per_anchor_prefixsum,
                                                        tiles_per_anchor_prefixsum + n_anchors + 1,
                                                        logicalTileId + 1))-1;

            const int candidatesBeforeThisAnchor = candidates_per_anchor_prefixsum[anchorIndex];
            const int maxCandidateIndex_excl = candidates_per_anchor_prefixsum[anchorIndex+1];
            //const int tilesForThisAnchor = tiles_per_anchor_prefixsum[anchorIndex + 1] - tiles_per_anchor_prefixsum[anchorIndex];
            const int tileForThisAnchor = logicalTileId - tiles_per_anchor_prefixsum[anchorIndex];
            const int candidateIndex = candidatesBeforeThisAnchor + tileForThisAnchor * tilesize + laneInTile;

            const int anchorbases = anchorSequencesLength[anchorIndex];
            const int anchorints = SequenceHelpers::getEncodedNumInts2BitHiLo(anchorbases);
            const unsigned int* anchorptr = anchorDataHiLo + std::size_t(anchorIndex) * encodedSequencePitchInInts2BitHiLo;

            //save anchor in shared memory (in parallel, per tile)
            for(int lane = laneInTile; lane < encodedSequencePitchInInts2BitHiLo; lane += tilesize) {
                anchorBackup[identity(lane)] = anchorptr[lane];
                //transposed
                //anchorBackup[identity(lane)] = ((unsigned int*)(anchorptr))[lane * n_anchors];
            }

            cg::tiled_partition<tilesize>(cg::this_thread_block()).sync();


            if(candidateIndex < maxCandidateIndex_excl){
                if(!(removeAmbiguousAnchors && anchorContainsN[anchorIndex]) && !(removeAmbiguousCandidates && candidateContainsN[candidateIndex])){
                    const int querybases = candidateSequencesLength[candidateIndex];
                    const int queryints = SequenceHelpers::getEncodedNumInts2BitHiLo(querybases);
                    const int totalbases = anchorbases + querybases;
                    const int minoverlap = max(min_overlap, int(float(anchorbases) * min_overlap_ratio));

                    const unsigned int* candidateptr = candidateDataHiLoTransposed + std::size_t(candidateIndex);

                    //save query in shared memory
                    for(int i = 0; i < encodedSequencePitchInInts2BitHiLo; i += 1) {
                        //queryBackup[no_bank_conflict_index(i)] = ((unsigned int*)(candidateptr))[i];
                        //transposed
                        queryBackup[no_bank_conflict_index(i)] = candidateptr[i * n_candidates];
                    }

                    const unsigned int* const queryBackup_hi = queryBackup;
                    const unsigned int* const queryBackup_lo = queryBackup + no_bank_conflict_index(queryints/2);

                    int bestScore[2];
                    int bestShift[2];
                    int overlapsize[2];
                    int opnr[2];

                    #pragma unroll
                    for(int orientation = 0; orientation < 2; orientation++){
                        const bool isReverseComplement = orientation == 1;

                        if(isReverseComplement) {
                            make_reverse_complement_inplace(queryBackup, querybases, no_bank_conflict_index);
                        }

                        //begin SHD algorithm

                        bestScore[orientation] = totalbases;     // score is number of mismatches
                        bestShift[orientation] = -querybases;    // shift of query relative to anchor. shift < 0 if query begins before anchor

                        auto handle_shift = [&](int shift, int overlapsize,
                                                    unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                                    int shiftptr_size,
                                                    const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                                    auto transfunc2){

                            //const int max_errors = int(float(overlapsize) * maxErrorRate);
                            const int max_errors_excl = min(int(float(overlapsize) * maxErrorRate),
                            bestScore[orientation] - totalbases + 2*overlapsize);

                            if(max_errors_excl > 0){

                                int score = hammingDistanceWithShift(shift != 0, overlapsize, max_errors_excl,
                                                    shiftptr_hi,shiftptr_lo, transfunc1,
                                                    shiftptr_size,
                                                    otherptr_hi, otherptr_lo, transfunc2);

                                

                                // printf("%d, %d %d %d --- ", queryIndex, shift, overlapsize, score);

                                // printf("%d %d %d %d | %d %d %d %d --- ", 
                                //     shiftptr_hi[transfunc1(0)], shiftptr_hi[transfunc1(1)], shiftptr_hi[transfunc1(2)], shiftptr_hi[transfunc1(3)],
                                //     shiftptr_lo[transfunc1(0)], shiftptr_lo[transfunc1(1)], shiftptr_lo[transfunc1(2)], shiftptr_lo[transfunc1(3)]);

                                // printf("%d %d %d %d | %d %d %d %d\n", 
                                //     otherptr_hi[transfunc2(0)], otherptr_hi[transfunc2(1)], otherptr_hi[transfunc2(2)], otherptr_hi[transfunc2(3)],
                                //     otherptr_lo[transfunc2(0)], otherptr_lo[transfunc2(1)], otherptr_lo[transfunc2(2)], otherptr_lo[transfunc2(3)]);

                                score = (score < max_errors_excl ?
                                        score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                                        : std::numeric_limits<int>::max()); // too many errors, discard

                                if(score < bestScore[orientation]){
                                    bestScore[orientation] = score;
                                    bestShift[orientation] = shift;
                                }

                                return true;
                            }else{
                                //printf("%d, %d %d %d max_errors_excl\n", queryIndex, shift, overlapsize, max_errors_excl);
                                return false;
                            }
                        };

                        //initialize threadlocal smem array with anchor
                        for(int i = 0; i < encodedSequencePitchInInts2BitHiLo; i += 1) {
                            mySequence[no_bank_conflict_index(i)] = anchorBackup[identity(i)];
                        }

                        unsigned int* mySequence_hi = mySequence;
                        unsigned int* mySequence_lo = mySequence + no_bank_conflict_index(anchorints / 2);

                        for(int shift = 0; shift < anchorbases - minoverlap + 1; shift += 1) {
                            const int overlapsize = min(anchorbases - shift, querybases);

                            bool b = handle_shift(shift, overlapsize,
                                            mySequence_hi, mySequence_lo, no_bank_conflict_index,
                                            anchorints,
                                            queryBackup_hi, queryBackup_lo, no_bank_conflict_index);
                            if(!b){
                                break;
                            }
                        }

                        const int queryoverlapbegin_incl = max(-bestShift[orientation], 0);
                        const int queryoverlapend_excl = min(querybases, anchorbases - bestShift[orientation]);
                        overlapsize[orientation] = queryoverlapend_excl - queryoverlapbegin_incl;
                        opnr[orientation] = bestScore[orientation] - totalbases + 2*overlapsize[orientation];
                    }

                    const AlignmentOrientation flag = alignmentComparator(
                        overlapsize[0],
                        overlapsize[1],
                        opnr[0],
                        opnr[1],
                        bestShift[0] != -querybases,
                        bestShift[1] != -querybases,
                        anchorbases,
                        querybases
                    );

                    d_alignment_best_alignment_flags[candidateIndex] = flag;
                    d_alignment_overlaps[candidateIndex] = flag == AlignmentOrientation::Forward ? overlapsize[0] : overlapsize[1];
                    d_alignment_shifts[candidateIndex] = flag == AlignmentOrientation::Forward ? bestShift[0] : bestShift[1];
                    d_alignment_nOps[candidateIndex] = flag == AlignmentOrientation::Forward ? opnr[0] : opnr[1];
                    d_alignment_isValid[candidateIndex] = flag == AlignmentOrientation::Forward ? bestShift[0] != -querybases : bestShift[1] != -querybases;
                }else{
                    d_alignment_best_alignment_flags[candidateIndex] = AlignmentOrientation::None;
                    d_alignment_isValid[candidateIndex] = false;
                }
            }
        }
    }


    /*
        Uses 1 thread per candidate to compute the alignment of anchor|candidate and anchor|revc-candidate
        Compares both alignments and keeps the better one

        Sequences are stored in registers
    */

    template<int blocksize, int maxValidIntsPerSequence>
    __global__
    void
    popcount_shifted_hamming_distance_reg_kernel(
                const unsigned int* __restrict__ anchorDataHiLoTransposed,
                const unsigned int* __restrict__ candidateDataHiLoTransposed,
                const int* __restrict__ anchorSequencesLength,
                const int* __restrict__ candidateSequencesLength,
                AlignmentOrientation* __restrict__ bestAlignmentFlags,
                int* __restrict__ alignment_overlaps,
                int* __restrict__ alignment_shifts,
                int* __restrict__ alignment_nOps,
                bool* __restrict__ alignment_isValid,
                const int* __restrict__ d_anchorIndicesOfCandidates,
                const int* __restrict__ numAnchorsPtr,
                const int* __restrict__ numCandidatesPtr,
                const bool* __restrict__ anchorContainsN,
                bool removeAmbiguousAnchors,
                const bool* __restrict__ candidateContainsN,
                bool removeAmbiguousCandidates,
                size_t encodedSequencePitchInInts2BitHiLo,
                int min_overlap,
                float maxErrorRate,
                float min_overlap_ratio,
                float estimatedNucleotideErrorRate){

        static_assert(maxValidIntsPerSequence % 2 == 0, ""); //2bithilo has even number of ints


        const int n_anchors = *numAnchorsPtr;
        const int n_candidates = *numCandidatesPtr;

        auto popcount = [](auto i){return __popc(i);};

        auto hammingdistanceHiLoReg = [&](
                            const auto& lhi,
                            const auto& llo,
                            const auto& rhi,
                            const auto& rlo,
                            int lhi_bitcount,
                            int rhi_bitcount,
                            int max_errors){

            constexpr int N = maxValidIntsPerSequence / 2;

            const int overlap_bitcount = std::min(lhi_bitcount, rhi_bitcount);

            if(overlap_bitcount == 0)
                return max_errors+1;

            const int partitions = SDIV(overlap_bitcount, (8 * sizeof(unsigned int)));
            const int remaining_bitcount = partitions * sizeof(unsigned int) * 8 - overlap_bitcount;

            int result = 0;

            #pragma unroll 
            for(int i = 0; i < N - 1; i++){
                if(i < partitions - 1 && result < max_errors){
                    const unsigned int hixor = lhi[i] ^ rhi[i];
                    const unsigned int loxor = llo[i] ^ rlo[i];
                    const unsigned int bits = hixor | loxor;
                    result += popcount(bits);
                }
            }

            if(result >= max_errors)
                return result;

            // i == partitions - 1

            #pragma unroll 
            for(int i = N-1; i >= 0; i--){
                if(partitions - 1 == i){
                    const unsigned int mask = remaining_bitcount == 0 ? 0xFFFFFFFF : 0xFFFFFFFF << (remaining_bitcount);
                    const unsigned int hixor = lhi[i] ^ rhi[i];
                    const unsigned int loxor = llo[i] ^ rlo[i];
                    const unsigned int bits = hixor | loxor;
                    result += popcount(bits & mask);
                }
            }

            return result;
        };

        auto maskBitArray = [](auto& uintarrayHi, auto& uintarrayLo, int keeplength){
            //only keep the first keeplength bits, set remaining bits to 0
            constexpr int N = maxValidIntsPerSequence / 2;

            const int unusedInts = N - SDIV(keeplength, 32);
            if(unusedInts > 0){
                #pragma unroll
                for(int i = 0; i < N; ++i){
                    if(i >= N-unusedInts){
                        uintarrayHi[i] = 0;
                        uintarrayLo[i] = 0;
                    }
                }
            }

            const int unusedBitsInt = SDIV(keeplength, 32) * 32 - keeplength;

            if(unusedBitsInt != 0){
                #pragma unroll
                for(int i = 0; i < N - 1; ++i){
                    if(i == N-unusedInts-1){
                        unsigned int mask = ~((1u << unusedBitsInt)-1);
                        uintarrayHi[i] &= mask;
                        uintarrayLo[i] &= mask;
                        break;
                    }
                }
            }
        };

        auto shiftBitArrayLeftBy1 = [](auto& uintarray){
            constexpr int shift = 1;
            static_assert(shift < 32, "");

            constexpr int N = maxValidIntsPerSequence / 2;    
            #pragma unroll
            for(int i = 0; i < N - 1; i += 1) {
                const unsigned int a = uintarray[i];
                const unsigned int b = uintarray[i+1];
    
                uintarray[i] = (a << shift) | (b >> (8 * sizeof(unsigned int) - shift));
            }
    
            uintarray[N-1] <<= shift;
        };

        auto hammingDistanceWithShift = [&](bool doShift, int overlapsize, int max_errors,
                                    auto& shiftptr_hi, auto& shiftptr_lo,
                                    const auto& otherptr_hi, const auto& otherptr_lo
                                    ){

            if(doShift){
                shiftBitArrayLeftBy1(shiftptr_hi);
                shiftBitArrayLeftBy1(shiftptr_lo);
            }

            const int score = hammingdistanceHiLoReg(shiftptr_hi,
                                                shiftptr_lo,
                                                otherptr_hi,
                                                otherptr_lo,
                                                overlapsize,
                                                overlapsize,
                                                max_errors);

            return score;
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


        unsigned int anchorBackupHi[maxValidIntsPerSequence / 2];
        unsigned int anchorBackupLo[maxValidIntsPerSequence / 2];
        unsigned int queryBackupHi[maxValidIntsPerSequence / 2];
        unsigned int queryBackupLo[maxValidIntsPerSequence / 2];
        unsigned int mySequenceHi[maxValidIntsPerSequence / 2];
        unsigned int mySequenceLo[maxValidIntsPerSequence / 2];

        auto reverseComplementQuery = [&](int querylength, int validInts){

            constexpr int N = maxValidIntsPerSequence / 2;

            #pragma unroll
            for(int i = 0; i < N/2; ++i){
                const unsigned int hifront = SequenceHelpers::reverseComplementInt2BitHiLoHalf(queryBackupHi[i]);
                const unsigned int hiback = SequenceHelpers::reverseComplementInt2BitHiLoHalf(queryBackupHi[N - 1 - i]);
                queryBackupHi[i] = hiback;
                queryBackupHi[N - 1 - i] = hifront;
    
                const unsigned int lofront = SequenceHelpers::reverseComplementInt2BitHiLoHalf(queryBackupLo[i]);
                const unsigned int loback = SequenceHelpers::reverseComplementInt2BitHiLoHalf(queryBackupLo[N - 1 - i]);
                queryBackupLo[i] = loback;
                queryBackupLo[N - 1 - i] = lofront;
            }

            if(N % 2 == 1){
                constexpr int middleindex = N/2;
                queryBackupHi[middleindex] = SequenceHelpers::reverseComplementInt2BitHiLoHalf(queryBackupHi[middleindex]);
                queryBackupLo[middleindex] = SequenceHelpers::reverseComplementInt2BitHiLoHalf(queryBackupLo[middleindex]);
            }

            //fix unused data

            const int unusedInts = N - SequenceHelpers::getEncodedNumInts2BitHiLo(querylength) / 2;
            if(unusedInts > 0){
                for(int iter = 0; iter < unusedInts; iter++){
                    #pragma unroll
                    for(int i = 0; i < N-1; ++i){
                        queryBackupHi[i] = queryBackupHi[i+1];
                        queryBackupLo[i] = queryBackupLo[i+1];
                    }
                }
            }

            const int unusedBitsInt = SDIV(querylength, 8 * sizeof(unsigned int)) * 8 * sizeof(unsigned int) - querylength;

            if(unusedBitsInt != 0){
                #pragma unroll
                for(int i = 0; i < N - 1; ++i){
                    queryBackupHi[i] = (queryBackupHi[i] << unusedBitsInt) | (queryBackupHi[i+1] >> (8 * sizeof(unsigned int) - unusedBitsInt));
                    queryBackupLo[i] = (queryBackupLo[i] << unusedBitsInt) | (queryBackupLo[i+1] >> (8 * sizeof(unsigned int) - unusedBitsInt));
                }
    
                queryBackupHi[N-1] <<= unusedBitsInt;
                queryBackupLo[N-1] <<= unusedBitsInt;
            }
        };

        for(int candidateIndex = threadIdx.x + blocksize * blockIdx.x; candidateIndex < n_candidates; candidateIndex += blocksize * gridDim.x){

            if(!(removeAmbiguousCandidates && candidateContainsN[candidateIndex])){

                const int anchorIndex = d_anchorIndicesOfCandidates[candidateIndex];  

                if(!(removeAmbiguousAnchors && anchorContainsN[anchorIndex])){

                    const int anchorbases = anchorSequencesLength[anchorIndex];
                    const int querybases = candidateSequencesLength[candidateIndex];

                    const unsigned int* anchorptr = anchorDataHiLoTransposed + std::size_t(anchorIndex);

                    #pragma unroll 
                    for(int i = 0; i < maxValidIntsPerSequence / 2; i++){
                        anchorBackupHi[i] = anchorptr[(i) * n_anchors];
                        anchorBackupLo[i] = anchorptr[(i + maxValidIntsPerSequence / 2) * n_anchors];
                    }

                    maskBitArray(anchorBackupHi, anchorBackupLo, anchorbases);

                    const unsigned int* candidateptr = candidateDataHiLoTransposed + std::size_t(candidateIndex);

                    //save query in reg

                    #pragma unroll 
                    for(int i = 0; i < maxValidIntsPerSequence / 2; i++){
                        queryBackupHi[i] = candidateptr[i * n_candidates];
                        queryBackupLo[i] = candidateptr[(i + maxValidIntsPerSequence / 2) * n_candidates];
                    }

                    maskBitArray(queryBackupHi, queryBackupLo, querybases);

                    //begin SHD algorithm

                    const int anchorints = SequenceHelpers::getEncodedNumInts2BitHiLo(anchorbases);
                    const int queryints = SequenceHelpers::getEncodedNumInts2BitHiLo(querybases);
                    const int totalbases = anchorbases + querybases;
                    const int minoverlap = max(min_overlap, int(float(anchorbases) * min_overlap_ratio));

                    int bestScore[2];
                    int bestShift[2];
                    int overlapsize[2];
                    int opnr[2];

                    #pragma unroll
                    for(int orientation = 0; orientation < 2; orientation++){
                        const bool isReverseComplement = orientation == 1;

                        if(isReverseComplement){
                            reverseComplementQuery(querybases, queryints);
                        }

                        bestScore[orientation] = totalbases;     // score is number of mismatches
                        bestShift[orientation] = -querybases;    // shift of query relative to anchor. shift < 0 if query begins before anchor

                        auto handle_shift = [&](int shift, int overlapsize,
                                                auto& shiftptr_hi, auto& shiftptr_lo,
                                                const auto& otherptr_hi, const auto& otherptr_lo){

                            //const int max_errors = int(float(overlapsize) * maxErrorRate);
                            const int max_errors_excl = min(int(float(overlapsize) * maxErrorRate),
                                                            bestScore[orientation] - totalbases + 2*overlapsize);

                            if(max_errors_excl > 0){

                                int score = hammingDistanceWithShift(shift != 0, overlapsize, max_errors_excl,
                                                    shiftptr_hi, shiftptr_lo,
                                                    otherptr_hi, otherptr_lo);

                                
                                // printf("%d, %d %d %d --- ", queryIndex, shift, overlapsize, score);

                                // printf("%d %d %d %d | %d %d %d %d --- ", 
                                //     shiftptr_hi[0], shiftptr_hi[1], shiftptr_hi[2], shiftptr_hi[3],
                                //     shiftptr_lo[0], shiftptr_lo[1], shiftptr_lo[2], shiftptr_lo[3]);

                                // printf("%d %d %d %d | %d %d %d %d\n", 
                                //     otherptr_hi[0], otherptr_hi[1], otherptr_hi[2], otherptr_hi[3],
                                //     otherptr_lo[0], otherptr_lo[1], otherptr_lo[2], otherptr_lo[3]);

                                score = (score < max_errors_excl ?
                                        score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                                        : std::numeric_limits<int>::max()); // too many errors, discard

                                if(score < bestScore[orientation]){
                                    bestScore[orientation] = score;
                                    bestShift[orientation] = shift;
                                }

                                return true;
                            }else{
                                //printf("%d, %d %d %d max_errors_excl\n", queryIndex, shift, overlapsize, max_errors_excl);

                                return false;
                            }
                        };

                        #pragma unroll 
                        for(int i = 0; i < maxValidIntsPerSequence / 2; i++){
                            mySequenceHi[i] = anchorBackupHi[i];
                            mySequenceLo[i] = anchorBackupLo[i];
                        }

                        for(int shift = 0; shift < anchorbases - minoverlap + 1; shift += 1) {
                            const int overlapsize = min(anchorbases - shift, querybases);

                            bool b = handle_shift(
                                shift, overlapsize,
                                mySequenceHi, mySequenceLo,
                                queryBackupHi, queryBackupLo
                            );
                            if(!b){
                                break;
                            }
                        }

                        //to compute negative shifts, shift query instead of anchor
                        #pragma unroll 
                        for(int i = 0; i < maxValidIntsPerSequence / 2; i++){
                            mySequenceHi[i] = queryBackupHi[i];
                            mySequenceLo[i] = queryBackupLo[i];
                        }

                        for(int shift = -1; shift >= -querybases + minoverlap; shift -= 1) {
                            const int overlapsize = min(anchorbases, querybases + shift);

                            bool b = handle_shift(
                                shift, overlapsize,
                                mySequenceHi, mySequenceLo,
                                anchorBackupHi, anchorBackupLo
                            );
                            if(!b){
                                break;
                            }
                        }

                        const int queryoverlapbegin_incl = max(-bestShift[orientation], 0);
                        const int queryoverlapend_excl = min(querybases, anchorbases - bestShift[orientation]);
                        overlapsize[orientation] = queryoverlapend_excl - queryoverlapbegin_incl;
                        opnr[orientation] = bestScore[orientation] - totalbases + 2*overlapsize[orientation];
                    }

                    const AlignmentOrientation flag = alignmentComparator(
                        overlapsize[0],
                        overlapsize[1],
                        opnr[0],
                        opnr[1],
                        bestShift[0] != -querybases,
                        bestShift[1] != -querybases,
                        anchorbases,
                        querybases
                    );

                    bestAlignmentFlags[candidateIndex] = flag;
                    alignment_overlaps[candidateIndex] = flag == AlignmentOrientation::Forward ? overlapsize[0] : overlapsize[1];
                    alignment_shifts[candidateIndex] = flag == AlignmentOrientation::Forward ? bestShift[0] : bestShift[1];
                    alignment_nOps[candidateIndex] = flag == AlignmentOrientation::Forward ? opnr[0] : opnr[1];
                    alignment_isValid[candidateIndex] = flag == AlignmentOrientation::Forward ? bestShift[0] != -querybases : bestShift[1] != -querybases;
                }else{
                    bestAlignmentFlags[candidateIndex] = AlignmentOrientation::None;
                    alignment_isValid[candidateIndex] = false;
                }
            }else{
                bestAlignmentFlags[candidateIndex] = AlignmentOrientation::None;
                alignment_isValid[candidateIndex] = false;
            }
        }
    }










    template<int blocksize, int maxValidIntsPerSequence>
    __global__
    void
    popcount_rightshifted_hamming_distance_reg_kernel(
                const unsigned int* __restrict__ anchorDataHiLoTransposed,
                const unsigned int* __restrict__ candidateDataHiLoTransposed,
                const int* __restrict__ anchorSequencesLength,
                const int* __restrict__ candidateSequencesLength,
                AlignmentOrientation* __restrict__ bestAlignmentFlags,
                int* __restrict__ alignment_overlaps,
                int* __restrict__ alignment_shifts,
                int* __restrict__ alignment_nOps,
                bool* __restrict__ alignment_isValid,
                const int* __restrict__ d_anchorIndicesOfCandidates,
                const int* __restrict__ numAnchorsPtr,
                const int* __restrict__ numCandidatesPtr,
                const bool* __restrict__ anchorContainsN,
                bool removeAmbiguousAnchors,
                const bool* __restrict__ candidateContainsN,
                bool removeAmbiguousCandidates,
                size_t encodedSequencePitchInInts2BitHiLo,
                int min_overlap,
                float maxErrorRate,
                float min_overlap_ratio,
                float estimatedNucleotideErrorRate){

        static_assert(maxValidIntsPerSequence % 2 == 0, ""); //2bithilo has even number of ints


        const int n_anchors = *numAnchorsPtr;
        const int n_candidates = *numCandidatesPtr;

        auto popcount = [](auto i){return __popc(i);};

        auto hammingdistanceHiLoReg = [&](
                            const auto& lhi,
                            const auto& llo,
                            const auto& rhi,
                            const auto& rlo,
                            int lhi_bitcount,
                            int rhi_bitcount,
                            int max_errors){

            constexpr int N = maxValidIntsPerSequence / 2;

            const int overlap_bitcount = std::min(lhi_bitcount, rhi_bitcount);

            if(overlap_bitcount == 0)
                return max_errors+1;

            const int partitions = SDIV(overlap_bitcount, (8 * sizeof(unsigned int)));
            const int remaining_bitcount = partitions * sizeof(unsigned int) * 8 - overlap_bitcount;

            int result = 0;

            #pragma unroll 
            for(int i = 0; i < N - 1; i++){
                if(i < partitions - 1 && result < max_errors){
                    const unsigned int hixor = lhi[i] ^ rhi[i];
                    const unsigned int loxor = llo[i] ^ rlo[i];
                    const unsigned int bits = hixor | loxor;
                    result += popcount(bits);
                }
            }

            if(result >= max_errors)
                return result;

            // i == partitions - 1

            #pragma unroll 
            for(int i = N-1; i >= 0; i--){
                if(partitions - 1 == i){
                    const unsigned int mask = remaining_bitcount == 0 ? 0xFFFFFFFF : 0xFFFFFFFF << (remaining_bitcount);
                    const unsigned int hixor = lhi[i] ^ rhi[i];
                    const unsigned int loxor = llo[i] ^ rlo[i];
                    const unsigned int bits = hixor | loxor;
                    result += popcount(bits & mask);
                }
            }

            return result;
        };

        auto maskBitArray = [](auto& uintarrayHi, auto& uintarrayLo, int keeplength){
            //only keep the first keeplength bits, set remaining bits to 0
            constexpr int N = maxValidIntsPerSequence / 2;

            const int unusedInts = N - SDIV(keeplength, 32);
            if(unusedInts > 0){
                #pragma unroll
                for(int i = 0; i < N; ++i){
                    if(i >= N-unusedInts){
                        uintarrayHi[i] = 0;
                        uintarrayLo[i] = 0;
                    }
                }
            }

            const int unusedBitsInt = SDIV(keeplength, 32) * 32 - keeplength;

            if(unusedBitsInt != 0){
                #pragma unroll
                for(int i = 0; i < N - 1; ++i){
                    if(i == N-unusedInts-1){
                        unsigned int mask = ~((1u << unusedBitsInt)-1);
                        uintarrayHi[i] &= mask;
                        uintarrayLo[i] &= mask;
                        break;
                    }
                }
            }
        };

        auto shiftBitArrayLeftBy1 = [](auto& uintarray){
            constexpr int shift = 1;
            static_assert(shift < 32, "");

            constexpr int N = maxValidIntsPerSequence / 2;    
            #pragma unroll
            for(int i = 0; i < N - 1; i += 1) {
                const unsigned int a = uintarray[i];
                const unsigned int b = uintarray[i+1];
    
                uintarray[i] = (a << shift) | (b >> (8 * sizeof(unsigned int) - shift));
            }
    
            uintarray[N-1] <<= shift;
        };

        auto hammingDistanceWithShift = [&](bool doShift, int overlapsize, int max_errors,
                                    auto& shiftptr_hi, auto& shiftptr_lo,
                                    const auto& otherptr_hi, const auto& otherptr_lo
                                    ){

            if(doShift){
                shiftBitArrayLeftBy1(shiftptr_hi);
                shiftBitArrayLeftBy1(shiftptr_lo);
            }

            const int score = hammingdistanceHiLoReg(shiftptr_hi,
                                                shiftptr_lo,
                                                otherptr_hi,
                                                otherptr_lo,
                                                overlapsize,
                                                overlapsize,
                                                max_errors);

            return score;
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


        unsigned int anchorBackupHi[maxValidIntsPerSequence / 2];
        unsigned int anchorBackupLo[maxValidIntsPerSequence / 2];
        unsigned int queryBackupHi[maxValidIntsPerSequence / 2];
        unsigned int queryBackupLo[maxValidIntsPerSequence / 2];
        unsigned int mySequenceHi[maxValidIntsPerSequence / 2];
        unsigned int mySequenceLo[maxValidIntsPerSequence / 2];

        auto reverseComplementQuery = [&](int querylength, int validInts){

            constexpr int N = maxValidIntsPerSequence / 2;

            #pragma unroll
            for(int i = 0; i < N/2; ++i){
                const unsigned int hifront = SequenceHelpers::reverseComplementInt2BitHiLoHalf(queryBackupHi[i]);
                const unsigned int hiback = SequenceHelpers::reverseComplementInt2BitHiLoHalf(queryBackupHi[N - 1 - i]);
                queryBackupHi[i] = hiback;
                queryBackupHi[N - 1 - i] = hifront;
    
                const unsigned int lofront = SequenceHelpers::reverseComplementInt2BitHiLoHalf(queryBackupLo[i]);
                const unsigned int loback = SequenceHelpers::reverseComplementInt2BitHiLoHalf(queryBackupLo[N - 1 - i]);
                queryBackupLo[i] = loback;
                queryBackupLo[N - 1 - i] = lofront;
            }

            if(N % 2 == 1){
                constexpr int middleindex = N/2;
                queryBackupHi[middleindex] = SequenceHelpers::reverseComplementInt2BitHiLoHalf(queryBackupHi[middleindex]);
                queryBackupLo[middleindex] = SequenceHelpers::reverseComplementInt2BitHiLoHalf(queryBackupLo[middleindex]);
            }

            //fix unused data

            const int unusedInts = N - SequenceHelpers::getEncodedNumInts2BitHiLo(querylength) / 2;
            if(unusedInts > 0){
                for(int iter = 0; iter < unusedInts; iter++){
                    #pragma unroll
                    for(int i = 0; i < N-1; ++i){
                        queryBackupHi[i] = queryBackupHi[i+1];
                        queryBackupLo[i] = queryBackupLo[i+1];
                    }
                }
            }

            const int unusedBitsInt = SDIV(querylength, 8 * sizeof(unsigned int)) * 8 * sizeof(unsigned int) - querylength;

            if(unusedBitsInt != 0){
                #pragma unroll
                for(int i = 0; i < N - 1; ++i){
                    queryBackupHi[i] = (queryBackupHi[i] << unusedBitsInt) | (queryBackupHi[i+1] >> (8 * sizeof(unsigned int) - unusedBitsInt));
                    queryBackupLo[i] = (queryBackupLo[i] << unusedBitsInt) | (queryBackupLo[i+1] >> (8 * sizeof(unsigned int) - unusedBitsInt));
                }
    
                queryBackupHi[N-1] <<= unusedBitsInt;
                queryBackupLo[N-1] <<= unusedBitsInt;
            }
        };

        for(int candidateIndex = threadIdx.x + blocksize * blockIdx.x; candidateIndex < n_candidates; candidateIndex += blocksize * gridDim.x){

            if(!(removeAmbiguousCandidates && candidateContainsN[candidateIndex])){

                const int anchorIndex = d_anchorIndicesOfCandidates[candidateIndex];  

                if(!(removeAmbiguousAnchors && anchorContainsN[anchorIndex])){

                    const int anchorbases = anchorSequencesLength[anchorIndex];
                    const int querybases = candidateSequencesLength[candidateIndex];

                    const unsigned int* anchorptr = anchorDataHiLoTransposed + std::size_t(anchorIndex);

                    #pragma unroll 
                    for(int i = 0; i < maxValidIntsPerSequence / 2; i++){
                        anchorBackupHi[i] = anchorptr[(i) * n_anchors];
                        anchorBackupLo[i] = anchorptr[(i + maxValidIntsPerSequence / 2) * n_anchors];
                    }

                    maskBitArray(anchorBackupHi, anchorBackupLo, anchorbases);

                    const unsigned int* candidateptr = candidateDataHiLoTransposed + std::size_t(candidateIndex);

                    //save query in reg

                    #pragma unroll 
                    for(int i = 0; i < maxValidIntsPerSequence / 2; i++){
                        queryBackupHi[i] = candidateptr[i * n_candidates];
                        queryBackupLo[i] = candidateptr[(i + maxValidIntsPerSequence / 2) * n_candidates];
                    }

                    maskBitArray(queryBackupHi, queryBackupLo, querybases);

                    //begin SHD algorithm

                    const int anchorints = SequenceHelpers::getEncodedNumInts2BitHiLo(anchorbases);
                    const int queryints = SequenceHelpers::getEncodedNumInts2BitHiLo(querybases);
                    const int totalbases = anchorbases + querybases;
                    const int minoverlap = max(min_overlap, int(float(anchorbases) * min_overlap_ratio));

                    int bestScore[2];
                    int bestShift[2];
                    int overlapsize[2];
                    int opnr[2];

                    #pragma unroll
                    for(int orientation = 0; orientation < 2; orientation++){
                        const bool isReverseComplement = orientation == 1;

                        if(isReverseComplement){
                            reverseComplementQuery(querybases, queryints);
                        }

                        bestScore[orientation] = totalbases;     // score is number of mismatches
                        bestShift[orientation] = -querybases;    // shift of query relative to anchor. shift < 0 if query begins before anchor

                        auto handle_shift = [&](int shift, int overlapsize,
                                                auto& shiftptr_hi, auto& shiftptr_lo,
                                                const auto& otherptr_hi, const auto& otherptr_lo){

                            //const int max_errors = int(float(overlapsize) * maxErrorRate);
                            const int max_errors_excl = min(int(float(overlapsize) * maxErrorRate),
                                                            bestScore[orientation] - totalbases + 2*overlapsize);

                            if(max_errors_excl > 0){

                                int score = hammingDistanceWithShift(shift != 0, overlapsize, max_errors_excl,
                                                    shiftptr_hi, shiftptr_lo,
                                                    otherptr_hi, otherptr_lo);

                                
                                // printf("%d, %d %d %d --- ", queryIndex, shift, overlapsize, score);

                                // printf("%d %d %d %d | %d %d %d %d --- ", 
                                //     shiftptr_hi[0], shiftptr_hi[1], shiftptr_hi[2], shiftptr_hi[3],
                                //     shiftptr_lo[0], shiftptr_lo[1], shiftptr_lo[2], shiftptr_lo[3]);

                                // printf("%d %d %d %d | %d %d %d %d\n", 
                                //     otherptr_hi[0], otherptr_hi[1], otherptr_hi[2], otherptr_hi[3],
                                //     otherptr_lo[0], otherptr_lo[1], otherptr_lo[2], otherptr_lo[3]);

                                score = (score < max_errors_excl ?
                                        score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                                        : std::numeric_limits<int>::max()); // too many errors, discard

                                if(score < bestScore[orientation]){
                                    bestScore[orientation] = score;
                                    bestShift[orientation] = shift;
                                }

                                return true;
                            }else{
                                //printf("%d, %d %d %d max_errors_excl\n", queryIndex, shift, overlapsize, max_errors_excl);

                                return false;
                            }
                        };

                        #pragma unroll 
                        for(int i = 0; i < maxValidIntsPerSequence / 2; i++){
                            mySequenceHi[i] = anchorBackupHi[i];
                            mySequenceLo[i] = anchorBackupLo[i];
                        }

                        for(int shift = 0; shift < anchorbases - minoverlap + 1; shift += 1) {
                            const int overlapsize = min(anchorbases - shift, querybases);

                            bool b = handle_shift(
                                shift, overlapsize,
                                mySequenceHi, mySequenceLo,
                                queryBackupHi, queryBackupLo
                            );
                            if(!b){
                                break;
                            }
                        }

                        const int queryoverlapbegin_incl = max(-bestShift[orientation], 0);
                        const int queryoverlapend_excl = min(querybases, anchorbases - bestShift[orientation]);
                        overlapsize[orientation] = queryoverlapend_excl - queryoverlapbegin_incl;
                        opnr[orientation] = bestScore[orientation] - totalbases + 2*overlapsize[orientation];
                    }

                    // if(candidateIndex == 8){
                    //     printf("(%d, %d, %d, %d) (%d, %d, %d, %d)", 
                    //         overlapsize[0], bestShift[0], opnr[0], bestShift[0] != -querybases,
                    //         overlapsize[1], bestShift[1], opnr[1], bestShift[1] != -querybases);
                    // }

                    const AlignmentOrientation flag = alignmentComparator(
                        overlapsize[0],
                        overlapsize[1],
                        opnr[0],
                        opnr[1],
                        bestShift[0] != -querybases,
                        bestShift[1] != -querybases,
                        anchorbases,
                        querybases
                    );

                    bestAlignmentFlags[candidateIndex] = flag;
                    alignment_overlaps[candidateIndex] = flag == AlignmentOrientation::Forward ? overlapsize[0] : overlapsize[1];
                    alignment_shifts[candidateIndex] = flag == AlignmentOrientation::Forward ? bestShift[0] : bestShift[1];
                    alignment_nOps[candidateIndex] = flag == AlignmentOrientation::Forward ? opnr[0] : opnr[1];
                    alignment_isValid[candidateIndex] = flag == AlignmentOrientation::Forward ? bestShift[0] != -querybases : bestShift[1] != -querybases;
                }else{
                    bestAlignmentFlags[candidateIndex] = AlignmentOrientation::None;
                    alignment_isValid[candidateIndex] = false;
                }
            }else{
                bestAlignmentFlags[candidateIndex] = AlignmentOrientation::None;
                alignment_isValid[candidateIndex] = false;
            }
        }
    }





    template<int BLOCKSIZE>
    __global__
    void cuda_filter_alignments_by_mismatchratio_kernel(
                AlignmentOrientation* __restrict__ bestAlignmentFlags,
                const int* __restrict__ nOps,
                const int* __restrict__ overlaps,
                const int* __restrict__ d_candidates_per_anchor_prefixsum,
                const int* __restrict__ d_numAnchors,
                const int* __restrict__ d_numCandidates,
                float mismatchratioBaseFactor,
                float goodAlignmentsCountThreshold){

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceInt::TempStorage intreduce;
            int broadcast[3];
        } temp_storage;

        const int n_anchors = *d_numAnchors;
        //const int n_candidates = *d_numCandidates;


        for(int anchorindex = blockIdx.x; anchorindex < n_anchors; anchorindex += gridDim.x) {

            const int candidatesForAnchor = d_candidates_per_anchor_prefixsum[anchorindex+1]
                                            - d_candidates_per_anchor_prefixsum[anchorindex];

            const int firstIndex = d_candidates_per_anchor_prefixsum[anchorindex];

            //printf("anchorindex %d\n", anchorindex);

            int counts[3]{0,0,0};

            //if(threadIdx.x == 0){
            //    printf("my_n_indices %d\n", my_n_indices);
            //}

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
                    //printf("count[%d] = %d\n", i, counts[i]);
                }
                //printf("mismatchratioBaseFactor %f, goodAlignmentsCountThreshold %f\n", mismatchratioBaseFactor, goodAlignmentsCountThreshold);
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







    //####################   KERNEL DISPATCH   ####################

    template<int maxValidIntsPerSequence>
    void call_popcount_shifted_hamming_distance_reg_kernel_async(
        void* d_tempstorage,
        size_t& tempstoragebytes,
        int* d_alignment_overlaps,
        int* d_alignment_shifts,
        int* d_alignment_nOps,
        bool* d_alignment_isValid,
        AlignmentOrientation* d_alignment_best_alignment_flags,
        const unsigned int* d_anchorSequencesData,
        const unsigned int* d_candidateSequencesData,
        const int* d_anchorSequencesLength,
        const int* d_candidateSequencesLength,
        const int* /*d_candidates_per_anchor_prefixsum*/,
        const int* /*d_candidates_per_anchor*/,
        const int* d_anchorIndicesOfCandidates,
        const int* d_numAnchors,
        const int* d_numCandidates,
        const bool* d_anchorContainsN,
        bool removeAmbiguousAnchors,
        const bool* d_candidateContainsN,
        bool removeAmbiguousCandidates,
        int maxNumAnchors,
        int maxNumCandidates,
        int maximumSequenceLength,
        int encodedSequencePitchInInts2Bit,
        int min_overlap,
        float maxErrorRate,
        float min_overlap_ratio,
        float estimatedNucleotideErrorRate,
        cudaStream_t stream){

        const int intsPerSequence2BitHiLo = SequenceHelpers::getEncodedNumInts2BitHiLo(maximumSequenceLength);
        
        
        const std::size_t d_candidateDataHiLoTransposedBytes = SDIV(sizeof(unsigned int) * intsPerSequence2BitHiLo * maxNumCandidates, 512) * 512;
        const std::size_t d_anchorDataHiLoTransposedBytes = SDIV(sizeof(unsigned int) * intsPerSequence2BitHiLo * maxNumAnchors, 512) * 512;
        
        {
            
            const std::size_t requiredTempBytes 
                = d_candidateDataHiLoTransposedBytes
                    + d_anchorDataHiLoTransposedBytes;
            
            if(d_tempstorage == 0){
                tempstoragebytes = requiredTempBytes;
                return;
            }else{
                assert(tempstoragebytes >= requiredTempBytes);
            }
            
        }
        
        //Alias temp storage 
        unsigned int* const d_anchorDataHiLoTransposed = (unsigned int*)d_tempstorage;
        unsigned int* const d_candidateDataHiLoTransposed 
            = (unsigned int*)(((char*)d_anchorDataHiLoTransposed) 
            + d_anchorDataHiLoTransposedBytes);
       

        callConversionKernel2BitTo2BitHiLoNT(
            d_candidateSequencesData,
            encodedSequencePitchInInts2Bit,
            d_candidateDataHiLoTransposed,
            intsPerSequence2BitHiLo,
            d_candidateSequencesLength,
            d_numCandidates,
            maxNumCandidates,
            stream
        );

        callConversionKernel2BitTo2BitHiLoNT(
            d_anchorSequencesData,
            encodedSequencePitchInInts2Bit,
            d_anchorDataHiLoTransposed,
            intsPerSequence2BitHiLo,
            d_anchorSequencesLength,
            d_numAnchors,
            maxNumAnchors,
            stream
        );
        

        constexpr int blocksize = 128;

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            popcount_shifted_hamming_distance_reg_kernel<blocksize, maxValidIntsPerSequence>,
            blocksize, 
            0
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

        dim3 block(blocksize, 1, 1);
        //const int numBlocks = SDIV(maxNumCandidates, blocksize);
        //dim3 grid(std::min(numBlocks, maxBlocks), 1, 1);
        dim3 grid(maxBlocks, 1, 1);

        popcount_shifted_hamming_distance_reg_kernel<blocksize, maxValidIntsPerSequence>
            <<<grid, block, 0, stream>>>(
                d_anchorDataHiLoTransposed,
                d_candidateDataHiLoTransposed,
                d_anchorSequencesLength,
                d_candidateSequencesLength,
                d_alignment_best_alignment_flags,
                d_alignment_overlaps,
                d_alignment_shifts,
                d_alignment_nOps,
                d_alignment_isValid,
                d_anchorIndicesOfCandidates,
                d_numAnchors,
                d_numCandidates,
                d_anchorContainsN,
                removeAmbiguousAnchors,
                d_candidateContainsN,
                removeAmbiguousCandidates,
                intsPerSequence2BitHiLo, 
                min_overlap,
                maxErrorRate,
                min_overlap_ratio,
                estimatedNucleotideErrorRate
        ); CUDACHECKASYNC;

    }

    template<int maxValidIntsPerSequence>
    void call_popcount_rightshifted_hamming_distance_reg_kernel_async(
        void* d_tempstorage,
        size_t& tempstoragebytes,
        int* d_alignment_overlaps,
        int* d_alignment_shifts,
        int* d_alignment_nOps,
        bool* d_alignment_isValid,
        AlignmentOrientation* d_alignment_best_alignment_flags,
        const unsigned int* d_anchorSequencesData,
        const unsigned int* d_candidateSequencesData,
        const int* d_anchorSequencesLength,
        const int* d_candidateSequencesLength,
        const int* /*d_candidates_per_anchor_prefixsum*/,
        const int* /*d_candidates_per_anchor*/,
        const int* d_anchorIndicesOfCandidates,
        const int* d_numAnchors,
        const int* d_numCandidates,
        const bool* d_anchorContainsN,
        bool removeAmbiguousAnchors,
        const bool* d_candidateContainsN,
        bool removeAmbiguousCandidates,
        int maxNumAnchors,
        int maxNumCandidates,
        int maximumSequenceLength,
        int encodedSequencePitchInInts2Bit,
        int min_overlap,
        float maxErrorRate,
        float min_overlap_ratio,
        float estimatedNucleotideErrorRate,
        cudaStream_t stream){

        const int intsPerSequence2BitHiLo = SequenceHelpers::getEncodedNumInts2BitHiLo(maximumSequenceLength);       
        
        const std::size_t d_candidateDataHiLoTransposedBytes = SDIV(sizeof(unsigned int) * intsPerSequence2BitHiLo * maxNumCandidates, 512) * 512;
        const std::size_t d_anchorDataHiLoTransposedBytes = SDIV(sizeof(unsigned int) * intsPerSequence2BitHiLo * maxNumAnchors, 512) * 512;
        
        {
            
            const std::size_t requiredTempBytes 
                = d_candidateDataHiLoTransposedBytes
                    + d_anchorDataHiLoTransposedBytes;
            
            if(d_tempstorage == 0){
                tempstoragebytes = requiredTempBytes;
                return;
            }else{
                assert(tempstoragebytes >= requiredTempBytes);
            }
            
        }
        
        //Alias temp storage 
        unsigned int* const d_anchorDataHiLoTransposed = (unsigned int*)d_tempstorage;
        unsigned int* const d_candidateDataHiLoTransposed 
            = (unsigned int*)(((char*)d_anchorDataHiLoTransposed) 
            + d_anchorDataHiLoTransposedBytes);
       

        callConversionKernel2BitTo2BitHiLoNT(
            d_candidateSequencesData,
            encodedSequencePitchInInts2Bit,
            d_candidateDataHiLoTransposed,
            intsPerSequence2BitHiLo,
            d_candidateSequencesLength,
            d_numCandidates,
            maxNumCandidates,
            stream
        );

        callConversionKernel2BitTo2BitHiLoNT(
            d_anchorSequencesData,
            encodedSequencePitchInInts2Bit,
            d_anchorDataHiLoTransposed,
            intsPerSequence2BitHiLo,
            d_anchorSequencesLength,
            d_numAnchors,
            maxNumAnchors,
            stream
        );

        constexpr int blocksize = 128;

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            popcount_rightshifted_hamming_distance_reg_kernel<blocksize, maxValidIntsPerSequence>,
            blocksize, 
            0
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;  

        dim3 block(blocksize, 1, 1);
        //const int numBlocks = SDIV(maxNumCandidates, blocksize);
        //dim3 grid(std::min(numBlocks, maxBlocks), 1, 1);
        dim3 grid(maxBlocks, 1, 1);

        popcount_rightshifted_hamming_distance_reg_kernel<blocksize, maxValidIntsPerSequence>
            <<<grid, block, 0, stream>>>(
                d_anchorDataHiLoTransposed,
                d_candidateDataHiLoTransposed,
                d_anchorSequencesLength,
                d_candidateSequencesLength,
                d_alignment_best_alignment_flags,
                d_alignment_overlaps,
                d_alignment_shifts,
                d_alignment_nOps,
                d_alignment_isValid,
                d_anchorIndicesOfCandidates,
                d_numAnchors,
                d_numCandidates,
                d_anchorContainsN,
                removeAmbiguousAnchors,
                d_candidateContainsN,
                removeAmbiguousCandidates,
                intsPerSequence2BitHiLo, 
                min_overlap,
                maxErrorRate,
                min_overlap_ratio,
                estimatedNucleotideErrorRate
        ); CUDACHECKASYNC;

    }


    void call_popcount_shifted_hamming_distance_smem_kernel_async(
            void* d_tempstorage,
            size_t& tempstoragebytes,
            int* d_alignment_overlaps,
            int* d_alignment_shifts,
            int* d_alignment_nOps,
            bool* d_alignment_isValid,
            AlignmentOrientation* d_alignment_best_alignment_flags,
            const unsigned int* d_anchorSequencesData,
            const unsigned int* d_candidateSequencesData,
            const int* d_anchorSequencesLength,
            const int* d_candidateSequencesLength,
            const int* d_candidates_per_anchor_prefixsum,
            const int* d_candidates_per_anchor,
            const int* /*d_anchorIndicesOfCandidates*/,
            const int* d_numAnchors,
            const int* d_numCandidates,
            const bool* d_anchorContainsN,
            bool removeAmbiguousAnchors,
            const bool* d_candidateContainsN,
            bool removeAmbiguousCandidates,
            int maxNumAnchors,
            int maxNumCandidates,
            int maximumSequenceLength,
            int encodedSequencePitchInInts2Bit,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio,
            float estimatedNucleotideErrorRate,
            cudaStream_t stream){
        
        constexpr int tilesize = 16;
        
        auto getTilesPerAnchor = [=] __device__ (int candidates_for_anchor){
            return SDIV(candidates_for_anchor, tilesize);
        };
        
        cub::TransformInputIterator<int,decltype(getTilesPerAnchor), const int*>
            d_tiles_per_anchor(d_candidates_per_anchor,
                            getTilesPerAnchor);

        const int intsPerSequence2BitHiLo = SequenceHelpers::getEncodedNumInts2BitHiLo(maximumSequenceLength);
        const int bytesPerSequence2BitHilo = intsPerSequence2BitHiLo * sizeof(unsigned int);
        
        const std::size_t d_candidateDataHiLoTransposedBytes = SDIV(sizeof(unsigned int) * intsPerSequence2BitHiLo * maxNumCandidates, 512) * 512;
        const std::size_t d_anchorDataHiLoBytes = SDIV(sizeof(unsigned int) * intsPerSequence2BitHiLo * maxNumAnchors, 512) * 512;
        const std::size_t d_tiles_per_anchor_prefixsumBytes = SDIV(sizeof(int) * (maxNumAnchors+1), 512) * 512;
        std::size_t cubBytes = 0;
        
        // CUDACHECK(cub::DeviceScan::InclusiveSum(
        //     nullptr,
        //     cubBytes,
        //     d_tiles_per_anchor,
        //     (int*) nullptr,
        //     maxNumAnchors,
        //     stream
        // ));
        
        {

            const std::size_t requiredTempBytes 
                = d_candidateDataHiLoTransposedBytes
                    + d_anchorDataHiLoBytes
                    + d_tiles_per_anchor_prefixsumBytes
                    + cubBytes;
            
            if(d_tempstorage == 0){
                tempstoragebytes = requiredTempBytes;
                return;
            }else{
                assert(tempstoragebytes >= requiredTempBytes);
            }
        
        }
        
        //Alias temp storage 
        unsigned int* const d_candidateDataHiLoTransposed = (unsigned int*)d_tempstorage;
        unsigned int* const d_anchorDataHiLo 
            = (unsigned int*)(((char*)d_candidateDataHiLoTransposed) 
                + d_candidateDataHiLoTransposedBytes);
        int* const d_tiles_per_anchor_prefixsum
            = (int*)(((char*)d_anchorDataHiLo) 
                + d_anchorDataHiLoBytes);

        callConversionKernel2BitTo2BitHiLoNT(
            d_candidateSequencesData,
            encodedSequencePitchInInts2Bit,
            d_candidateDataHiLoTransposed,
            intsPerSequence2BitHiLo,
            d_candidateSequencesLength,
            d_numCandidates,
            maxNumCandidates,
            stream
        );

        callConversionKernel2BitTo2BitHiLoNN(
            d_anchorSequencesData,
            encodedSequencePitchInInts2Bit,
            d_anchorDataHiLo,
            intsPerSequence2BitHiLo,
            d_anchorSequencesLength,
            d_numAnchors,
            maxNumAnchors,
            stream
        );

        //calculate d_tiles_per_anchor_prefixsum
        helpers::lambda_kernel<<<1, 256, 0, stream>>>([=]__device__(){
            using BlockScan = cub::BlockScan<int, 256>;

            __shared__ typename BlockScan::TempStorage temp_storage;

            const int numItems = *d_numAnchors;

            constexpr int ITEMS_PER_THREAD = 4;

            int aggregate = 0;

            const int iters = SDIV(numItems, 256 * ITEMS_PER_THREAD);

            const int threadoffset = ITEMS_PER_THREAD * threadIdx.x;

            if(threadIdx.x == 0){
                d_tiles_per_anchor_prefixsum[0] = 0;
            }

            for(int iter = 0; iter < iters; iter++){
                int thread_data[ITEMS_PER_THREAD];

                const int iteroffset = 256 * ITEMS_PER_THREAD * iter;

                #pragma unroll
                for(int k = 0; k < ITEMS_PER_THREAD; k++){
                    if(iteroffset + threadoffset + k < numItems){
                        thread_data[k] = d_tiles_per_anchor[iteroffset + threadoffset + k];
                    }else{
                        thread_data[k] = 0;
                    }
                }

                int block_aggregate = 0;
                BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, block_aggregate);

                #pragma unroll
                for(int k = 0; k < ITEMS_PER_THREAD; k++){
                    if(iteroffset + threadoffset + k < numItems){
                        d_tiles_per_anchor_prefixsum[1+iteroffset + threadoffset + k] = aggregate + thread_data[k];
                    }
                }

                aggregate += block_aggregate;

                __syncthreads();
            }
        }); CUDACHECKASYNC;


        constexpr int blocksize = 128;
        constexpr int tilesPerBlock = blocksize / tilesize;

        const std::size_t smem = sizeof(char) * (bytesPerSequence2BitHilo * tilesPerBlock + bytesPerSequence2BitHilo * blocksize * 2);

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            popcount_shifted_hamming_distance_smem_kernel<tilesize>,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

        dim3 block(blocksize, 1, 1);
        //dim3 grid(std::min(requiredBlocks, maxBlocks), 1, 1);
        dim3 grid(maxBlocks, 1, 1);

        popcount_shifted_hamming_distance_smem_kernel<tilesize><<<grid, block, smem, stream>>>(
            d_anchorDataHiLo, 
            d_candidateDataHiLoTransposed, 
            d_alignment_overlaps, 
            d_alignment_shifts, 
            d_alignment_nOps, 
            d_alignment_isValid, 
            d_alignment_best_alignment_flags, 
            d_anchorSequencesLength, 
            d_candidateSequencesLength, 
            d_candidates_per_anchor_prefixsum, 
            d_tiles_per_anchor_prefixsum, 
            d_numAnchors, 
            d_numCandidates, 
            d_anchorContainsN, 
            removeAmbiguousAnchors, 
            d_candidateContainsN, 
            removeAmbiguousCandidates, 
            intsPerSequence2BitHiLo, 
            min_overlap, 
            maxErrorRate, 
            min_overlap_ratio, 
            estimatedNucleotideErrorRate
        ); 
        CUDACHECKASYNC;
    }

    void call_popcount_rightshifted_hamming_distance_smem_kernel_async(
            void* d_tempstorage,
            size_t& tempstoragebytes,
            int* d_alignment_overlaps,
            int* d_alignment_shifts,
            int* d_alignment_nOps,
            bool* d_alignment_isValid,
            AlignmentOrientation* d_alignment_best_alignment_flags,
            const unsigned int* d_anchorSequencesData,
            const unsigned int* d_candidateSequencesData,
            const int* d_anchorSequencesLength,
            const int* d_candidateSequencesLength,
            const int* d_candidates_per_anchor_prefixsum,
            const int* d_candidates_per_anchor,
            const int* /*d_anchorIndicesOfCandidates*/,
            const int* d_numAnchors,
            const int* d_numCandidates,
            const bool* d_anchorContainsN,
            bool removeAmbiguousAnchors,
            const bool* d_candidateContainsN,
            bool removeAmbiguousCandidates,
            int maxNumAnchors,
            int maxNumCandidates,
            int maximumSequenceLength,
            int encodedSequencePitchInInts2Bit,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio,
            float estimatedNucleotideErrorRate,
            cudaStream_t stream){
        
        constexpr int tilesize = 16;
        
        auto getTilesPerAnchor = [=] __device__ (int candidates_for_anchor){
            return SDIV(candidates_for_anchor, tilesize);
        };
        
        cub::TransformInputIterator<int,decltype(getTilesPerAnchor), const int*>
            d_tiles_per_anchor(d_candidates_per_anchor,
                            getTilesPerAnchor);

        const int intsPerSequence2BitHiLo = SequenceHelpers::getEncodedNumInts2BitHiLo(maximumSequenceLength);
        const int bytesPerSequence2BitHilo = intsPerSequence2BitHiLo * sizeof(unsigned int);
        
        const std::size_t d_candidateDataHiLoTransposedBytes = SDIV(sizeof(unsigned int) * intsPerSequence2BitHiLo * maxNumCandidates, 512) * 512;
        const std::size_t d_anchorDataHiLoBytes = SDIV(sizeof(unsigned int) * intsPerSequence2BitHiLo * maxNumAnchors, 512) * 512;
        const std::size_t d_tiles_per_anchor_prefixsumBytes = SDIV(sizeof(int) * (maxNumAnchors+1), 512) * 512;
        std::size_t cubBytes = 0;
        
        // CUDACHECK(cub::DeviceScan::InclusiveSum(
        //     nullptr,
        //     cubBytes,
        //     d_tiles_per_anchor,
        //     (int*) nullptr,
        //     maxNumAnchors,
        //     stream
        // ));
        
        {

            const std::size_t requiredTempBytes 
                = d_candidateDataHiLoTransposedBytes
                    + d_anchorDataHiLoBytes
                    + d_tiles_per_anchor_prefixsumBytes
                    + cubBytes;
            
            if(d_tempstorage == 0){
                tempstoragebytes = requiredTempBytes;
                return;
            }else{
                assert(tempstoragebytes >= requiredTempBytes);
            }
        
        }
        
        //Alias temp storage 
        unsigned int* const d_candidateDataHiLoTransposed = (unsigned int*)d_tempstorage;
        unsigned int* const d_anchorDataHiLo 
            = (unsigned int*)(((char*)d_candidateDataHiLoTransposed) 
                + d_candidateDataHiLoTransposedBytes);
        int* const d_tiles_per_anchor_prefixsum
            = (int*)(((char*)d_anchorDataHiLo) 
                + d_anchorDataHiLoBytes);

        callConversionKernel2BitTo2BitHiLoNT(
            d_candidateSequencesData,
            encodedSequencePitchInInts2Bit,
            d_candidateDataHiLoTransposed,
            intsPerSequence2BitHiLo,
            d_candidateSequencesLength,
            d_numCandidates,
            maxNumCandidates,
            stream
        );

        callConversionKernel2BitTo2BitHiLoNN(
            d_anchorSequencesData,
            encodedSequencePitchInInts2Bit,
            d_anchorDataHiLo,
            intsPerSequence2BitHiLo,
            d_anchorSequencesLength,
            d_numAnchors,
            maxNumAnchors,
            stream
        );

        //calculate d_tiles_per_anchor_prefixsum
        helpers::lambda_kernel<<<1, 256, 0, stream>>>([=]__device__(){
            using BlockScan = cub::BlockScan<int, 256>;

            __shared__ typename BlockScan::TempStorage temp_storage;

            const int numItems = *d_numAnchors;

            constexpr int ITEMS_PER_THREAD = 4;

            int aggregate = 0;

            const int iters = SDIV(numItems, 256 * ITEMS_PER_THREAD);

            const int threadoffset = ITEMS_PER_THREAD * threadIdx.x;

            if(threadIdx.x == 0){
                d_tiles_per_anchor_prefixsum[0] = 0;
            }

            for(int iter = 0; iter < iters; iter++){
                int thread_data[ITEMS_PER_THREAD];

                const int iteroffset = 256 * ITEMS_PER_THREAD * iter;

                #pragma unroll
                for(int k = 0; k < ITEMS_PER_THREAD; k++){
                    if(iteroffset + threadoffset + k < numItems){
                        thread_data[k] = d_tiles_per_anchor[iteroffset + threadoffset + k];
                    }else{
                        thread_data[k] = 0;
                    }
                }

                int block_aggregate = 0;
                BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, block_aggregate);

                #pragma unroll
                for(int k = 0; k < ITEMS_PER_THREAD; k++){
                    if(iteroffset + threadoffset + k < numItems){
                        d_tiles_per_anchor_prefixsum[1+iteroffset + threadoffset + k] = aggregate + thread_data[k];
                    }
                }

                aggregate += block_aggregate;

                __syncthreads();
            } 
        }); CUDACHECKASYNC;


        constexpr int blocksize = 128;
        constexpr int tilesPerBlock = blocksize / tilesize;

        const std::size_t smem = sizeof(char) * (bytesPerSequence2BitHilo * tilesPerBlock + bytesPerSequence2BitHilo * blocksize * 2);

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            popcount_rightshifted_hamming_distance_smem_kernel<tilesize>,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

        dim3 block(blocksize, 1, 1);
        //dim3 grid(std::min(requiredBlocks, maxBlocks), 1, 1);
        dim3 grid(maxBlocks, 1, 1);

        popcount_rightshifted_hamming_distance_smem_kernel<tilesize><<<grid, block, smem, stream>>>( 
            d_anchorDataHiLo, 
            d_candidateDataHiLoTransposed, 
            d_alignment_overlaps, 
            d_alignment_shifts, 
            d_alignment_nOps, 
            d_alignment_isValid, 
            d_alignment_best_alignment_flags, 
            d_anchorSequencesLength, 
            d_candidateSequencesLength, 
            d_candidates_per_anchor_prefixsum, 
            d_tiles_per_anchor_prefixsum, 
            d_numAnchors, 
            d_numCandidates, 
            d_anchorContainsN, 
            removeAmbiguousAnchors, 
            d_candidateContainsN, 
            removeAmbiguousCandidates, 
            intsPerSequence2BitHiLo, 
            min_overlap, 
            maxErrorRate, 
            min_overlap_ratio, 
            estimatedNucleotideErrorRate
        ); 
        CUDACHECKASYNC;

    }


    void call_popcount_shifted_hamming_distance_kernel_async(
            void* d_tempstorage,
            size_t& tempstoragebytes,
            int* d_alignment_overlaps,
            int* d_alignment_shifts,
            int* d_alignment_nOps,
            bool* d_alignment_isValid,
            AlignmentOrientation* d_alignment_best_alignment_flags,
            const unsigned int* d_anchorSequencesData,
            const unsigned int* d_candidateSequencesData,
            const int* d_anchorSequencesLength,
            const int* d_candidateSequencesLength,
            const int* d_candidates_per_anchor_prefixsum,
            const int* d_candidates_per_anchor,
            const int* d_anchorIndicesOfCandidates,
            const int* d_numAnchors,
            const int* d_numCandidates,
            const bool* d_anchorContainsN,
            bool removeAmbiguousAnchors,
            const bool* d_candidateContainsN,
            bool removeAmbiguousCandidates,
            int maxNumAnchors,
            int maxNumCandidates,
            int maximumSequenceLength,
            int encodedSequencePitchInInts2Bit,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio,
            float estimatedNucleotideErrorRate,
            cudaStream_t stream){

            #define regKernel(intsPerSequence){ \
                call_popcount_shifted_hamming_distance_reg_kernel_async<intsPerSequence>( \
                    d_tempstorage, \
                    tempstoragebytes, \
                    d_alignment_overlaps, \
                    d_alignment_shifts, \
                    d_alignment_nOps, \
                    d_alignment_isValid, \
                    d_alignment_best_alignment_flags, \
                    d_anchorSequencesData, \
                    d_candidateSequencesData, \
                    d_anchorSequencesLength, \
                    d_candidateSequencesLength, \
                    d_candidates_per_anchor_prefixsum, \
                    d_candidates_per_anchor, \
                    d_anchorIndicesOfCandidates, \
                    d_numAnchors, \
                    d_numCandidates, \
                    d_anchorContainsN, \
                    removeAmbiguousAnchors, \
                    d_candidateContainsN, \
                    removeAmbiguousCandidates, \
                    maxNumAnchors, \
                    maxNumCandidates, \
                    maximumSequenceLength, \
                    encodedSequencePitchInInts2Bit, \
                    min_overlap, \
                    maxErrorRate, \
                    min_overlap_ratio, \
                    estimatedNucleotideErrorRate, \
                    stream \
                ); \
            };
            
            auto run = [&](){
                #if 1
                if(1 <= maximumSequenceLength && maximumSequenceLength <= 32){
                    
                    constexpr int maxValidIntsPerSequence = 2;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(33 <= maximumSequenceLength && maximumSequenceLength <= 64){
                    
                    constexpr int maxValidIntsPerSequence = 4;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(65 <= maximumSequenceLength && maximumSequenceLength <= 96){
                    
                    constexpr int maxValidIntsPerSequence = 6;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(97 <= maximumSequenceLength && maximumSequenceLength <= 128){
                    
                    constexpr int maxValidIntsPerSequence = 8;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(129 <= maximumSequenceLength && maximumSequenceLength <= 160){
                    
                    constexpr int maxValidIntsPerSequence = 10;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(161 <= maximumSequenceLength && maximumSequenceLength <= 192){
                    
                    constexpr int maxValidIntsPerSequence = 12;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(193 <= maximumSequenceLength && maximumSequenceLength <= 224){
                    
                    constexpr int maxValidIntsPerSequence = 14;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(225 <= maximumSequenceLength && maximumSequenceLength <= 256){
                    
                    constexpr int maxValidIntsPerSequence = 16;
                    regKernel(maxValidIntsPerSequence);
                    
                }else{
                    call_popcount_shifted_hamming_distance_smem_kernel_async(
                        d_tempstorage,
                        tempstoragebytes,
                        d_alignment_overlaps,
                        d_alignment_shifts,
                        d_alignment_nOps,
                        d_alignment_isValid,
                        d_alignment_best_alignment_flags,
                        d_anchorSequencesData,
                        d_candidateSequencesData,
                        d_anchorSequencesLength,
                        d_candidateSequencesLength,
                        d_candidates_per_anchor_prefixsum,
                        d_candidates_per_anchor,
                        d_anchorIndicesOfCandidates,
                        d_numAnchors,
                        d_numCandidates,
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
                        stream
                    );
                }

                #else 

                    call_popcount_shifted_hamming_distance_smem_kernel_async(
                        d_tempstorage,
                        tempstoragebytes,
                        d_alignment_overlaps,
                        d_alignment_shifts,
                        d_alignment_nOps,
                        d_alignment_isValid,
                        d_alignment_best_alignment_flags,
                        d_anchorSequencesData,
                        d_candidateSequencesData,
                        d_anchorSequencesLength,
                        d_candidateSequencesLength,
                        d_candidates_per_anchor_prefixsum,
                        d_candidates_per_anchor,
                        d_anchorIndicesOfCandidates,
                        d_numAnchors,
                        d_numCandidates,
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
                        stream
                    );
                #endif
            };
            
            if(d_tempstorage == nullptr){
                tempstoragebytes = 0;
                
                run();
                
                return;
            }

            
            run();

        #undef regKernel 
    }



    void call_popcount_rightshifted_hamming_distance_kernel_async(
            void* d_tempstorage,
            size_t& tempstoragebytes,
            int* d_alignment_overlaps,
            int* d_alignment_shifts,
            int* d_alignment_nOps,
            bool* d_alignment_isValid,
            AlignmentOrientation* d_alignment_best_alignment_flags,
            const unsigned int* d_anchorSequencesData,
            const unsigned int* d_candidateSequencesData,
            const int* d_anchorSequencesLength,
            const int* d_candidateSequencesLength,
            const int* d_candidates_per_anchor_prefixsum,
            const int* d_candidates_per_anchor,
            const int* d_anchorIndicesOfCandidates,
            const int* d_numAnchors,
            const int* d_numCandidates,
            const bool* d_anchorContainsN,
            bool removeAmbiguousAnchors,
            const bool* d_candidateContainsN,
            bool removeAmbiguousCandidates,
            int maxNumAnchors,
            int maxNumCandidates,
            int maximumSequenceLength,
            int encodedSequencePitchInInts2Bit,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio,
            float estimatedNucleotideErrorRate,
            cudaStream_t stream){

            #define regKernel(intsPerSequence){ \
                call_popcount_rightshifted_hamming_distance_reg_kernel_async<intsPerSequence>( \
                    d_tempstorage, \
                    tempstoragebytes, \
                    d_alignment_overlaps, \
                    d_alignment_shifts, \
                    d_alignment_nOps, \
                    d_alignment_isValid, \
                    d_alignment_best_alignment_flags, \
                    d_anchorSequencesData, \
                    d_candidateSequencesData, \
                    d_anchorSequencesLength, \
                    d_candidateSequencesLength, \
                    d_candidates_per_anchor_prefixsum, \
                    d_candidates_per_anchor, \
                    d_anchorIndicesOfCandidates, \
                    d_numAnchors, \
                    d_numCandidates, \
                    d_anchorContainsN, \
                    removeAmbiguousAnchors, \
                    d_candidateContainsN, \
                    removeAmbiguousCandidates, \
                    maxNumAnchors, \
                    maxNumCandidates, \
                    maximumSequenceLength, \
                    encodedSequencePitchInInts2Bit, \
                    min_overlap, \
                    maxErrorRate, \
                    min_overlap_ratio, \
                    estimatedNucleotideErrorRate, \
                    stream \
                ); \
            };
            
            auto run = [&](){
                #if 0
                call_popcount_rightshifted_hamming_distance_smem_kernel_async(
                    d_tempstorage,
                    tempstoragebytes,
                    d_alignment_overlaps,
                    d_alignment_shifts,
                    d_alignment_nOps,
                    d_alignment_isValid,
                    d_alignment_best_alignment_flags,
                    d_anchorSequencesData,
                    d_candidateSequencesData,
                    d_anchorSequencesLength,
                    d_candidateSequencesLength,
                    d_candidates_per_anchor_prefixsum,
                    d_candidates_per_anchor,
                    d_anchorIndicesOfCandidates,
                    d_numAnchors,
                    d_numCandidates,
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
                    stream
                );
                #else
                if(1 <= maximumSequenceLength && maximumSequenceLength <= 32){
                    
                    constexpr int maxValidIntsPerSequence = 2;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(33 <= maximumSequenceLength && maximumSequenceLength <= 64){
                    
                    constexpr int maxValidIntsPerSequence = 4;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(65 <= maximumSequenceLength && maximumSequenceLength <= 96){
                    
                    constexpr int maxValidIntsPerSequence = 6;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(97 <= maximumSequenceLength && maximumSequenceLength <= 128){
                    
                    constexpr int maxValidIntsPerSequence = 8;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(129 <= maximumSequenceLength && maximumSequenceLength <= 160){
                    
                    constexpr int maxValidIntsPerSequence = 10;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(161 <= maximumSequenceLength && maximumSequenceLength <= 192){
                    
                    constexpr int maxValidIntsPerSequence = 12;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(193 <= maximumSequenceLength && maximumSequenceLength <= 224){
                    
                    constexpr int maxValidIntsPerSequence = 14;
                    regKernel(maxValidIntsPerSequence);
                    
                }else if(225 <= maximumSequenceLength && maximumSequenceLength <= 256){
                    
                    constexpr int maxValidIntsPerSequence = 16;
                    regKernel(maxValidIntsPerSequence);
                    
                }else{
                    
                    call_popcount_rightshifted_hamming_distance_smem_kernel_async(
                        d_tempstorage,
                        tempstoragebytes,
                        d_alignment_overlaps,
                        d_alignment_shifts,
                        d_alignment_nOps,
                        d_alignment_isValid,
                        d_alignment_best_alignment_flags,
                        d_anchorSequencesData,
                        d_candidateSequencesData,
                        d_anchorSequencesLength,
                        d_candidateSequencesLength,
                        d_candidates_per_anchor_prefixsum,
                        d_candidates_per_anchor,
                        d_anchorIndicesOfCandidates,
                        d_numAnchors,
                        d_numCandidates,
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
                        stream
                    );
                }
                #endif
            };
            
            if(d_tempstorage == nullptr){
                tempstoragebytes = 0;
                
                run();
                
                return;
            }

            
            run();

        #undef regKernel 
    }


    void call_cuda_filter_alignments_by_mismatchratio_kernel_async(
        AlignmentOrientation* d_bestAlignmentFlags,
        const int* d_nOps,
        const int* d_overlaps,
        const int* d_candidates_per_anchor_prefixsum,
        const int* d_numAnchors,
        const int* d_numCandidates,
        int /*maxNumAnchors*/,
        int /*maxNumCandidates*/,
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
        //dim3 grid(std::min(maxNumAnchors, maxBlocks));
        dim3 grid(maxBlocks, 1, 1);

        cuda_filter_alignments_by_mismatchratio_kernel<blocksize><<<grid, block, smem, stream>>>( 
            d_bestAlignmentFlags, 
            d_nOps, 
            d_overlaps, 
            d_candidates_per_anchor_prefixsum, 
            d_numAnchors, 
            d_numCandidates, 
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
        const int* d_numAnchors,
        const int* d_numCandidates,
        int maxNumAnchors,
        int /*maxNumCandidates*/,
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

        helpers::lambda_kernel<<<4, 256, 0, stream>>>([=] __device__(){
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const int stride = blockDim.x * gridDim.x;

            const int numCandidates = *d_numCandidates;

            for(int i = tid; i < numCandidates; i += stride){
                d_indicesOfGoodCandidates[i] = -1;
            }

            for(int i = tid; i < maxNumAnchors; i += stride){
                d_numIndicesPerAnchor[i] = 0;
            }

            if(tid == 0){
                *d_totalNumIndices = 0;
            }
        }); CUDACHECKASYNC;

        dim3 block(blocksize, 1, 1);
        //dim3 grid(std::min(SDIV(maxNumCandidates, blocksize), maxBlocks));
        dim3 grid(maxBlocks, 1, 1);

        selectIndicesOfGoodCandidatesKernel<blocksize, tilesize><<<grid, block, 0, stream>>>(
            d_indicesOfGoodCandidates,
            d_numIndicesPerAnchor,
            d_totalNumIndices,
            d_alignmentFlags,
            d_candidates_per_anchor,
            d_candidates_per_anchor_prefixsum,
            d_anchorIndicesOfCandidates,
            d_numAnchors,
            d_numCandidates
        ); CUDACHECKASYNC;
    }


}
}
