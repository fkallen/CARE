//#define NDEBUG

#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
// #include <gpu/devicefunctionsforkernels.cuh>

// //#include <gpu/bestalignment.hpp>
#include <bestalignment.hpp>
#include <gpu/utility_kernels.cuh>
#include <gpu/cubcachingallocator.cuh>

// #include <msa.hpp>
#include <sequence.hpp>
// #include <correctionresultprocessing.hpp>

#include <shiftedhammingdistance_common.hpp>

// #include <hpc_helpers.cuh>
// #include <config.hpp>

// #include <cassert>


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
            const BestAlignment_t* __restrict__ d_alignmentFlags,
            const int* __restrict__ d_candidates_per_subject,
            const int* __restrict__ d_candidates_per_subject_prefixsum,
            const int* __restrict__ d_anchorIndicesOfCandidates,
            int numAnchors,
            int numCandidates){

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

            const int offset = d_candidates_per_subject_prefixsum[anchorIndex];
            int* const indicesPtr = d_indicesOfGoodCandidates + offset;
            int* const numIndicesPtr = d_numIndicesPerAnchor + anchorIndex;
            const BestAlignment_t* const myAlignmentFlagsPtr = d_alignmentFlags + offset;

            const int numCandidatesForAnchor = d_candidates_per_subject[anchorIndex];

            if(tile.thread_rank() == 0){
                counts[tileIdInBlock] = 0;
            }
            tile.sync();

            for(int localCandidateIndex = tile.thread_rank(); 
                    localCandidateIndex < numCandidatesForAnchor; 
                    localCandidateIndex += tile.size()){
                
                const BestAlignment_t alignmentflag = myAlignmentFlagsPtr[localCandidateIndex];

                if(alignmentflag != BestAlignment_t::None){
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
    popcount_shifted_hamming_distance_kernel(
                const unsigned int* __restrict__ subjectDataHiLo,
                const unsigned int* __restrict__ candidateDataHiLoTransposed,
                int* __restrict__ d_alignment_overlaps,
                int* __restrict__ d_alignment_shifts,
                int* __restrict__ d_alignment_nOps,
                bool* __restrict__ d_alignment_isValid,
                BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
                const int* __restrict__ subjectSequencesLength,
                const int* __restrict__ candidateSequencesLength,
                const int* __restrict__ candidates_per_subject_prefixsum,
                const int* __restrict__ tiles_per_subject_prefixsum,
                int n_subjects,
                int n_candidates,
                int encodedSequencePitchInInts2BitHiLo,
                int min_overlap,
                float maxErrorRate,
                float min_overlap_ratio,
                float estimatedNucleotideErrorRate){

        auto make_reverse_complement_inplace = [&](unsigned int* sequence, int sequencelength, auto indextrafo){
            reverseComplementInplace2BitHiLo((unsigned int*)sequence, sequencelength, indextrafo);
        };

        auto no_bank_conflict_index_tile = [&](int logical_index) -> int {
            return logical_index * tilesize;
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
            int subjectlength,
            int querylength)->BestAlignment_t{

            return choose_best_alignment(
                fwd_alignment_overlap,
                revc_alignment_overlap,
                fwd_alignment_nops,
                revc_alignment_nops,
                fwd_alignment_isvalid,
                revc_alignment_isvalid,
                subjectlength,
                querylength,
                min_overlap_ratio,
                min_overlap,
                estimatedNucleotideErrorRate * 4.0f
            );
        };

        // sizeof(char) * (max_sequence_bytes * num_tiles   // tiles share the subject
        //                    + max_sequence_bytes * num_threads // each thread works with its own candidate
        //                    + max_sequence_bytes * num_threads) // each thread needs memory to shift a sequence
        extern __shared__ unsigned int sharedmemory[];

        //set up shared memory pointers

        const int tiles = (blockDim.x * gridDim.x) / tilesize;
        const int globalTileId = (blockDim.x * blockIdx.x + threadIdx.x) / tilesize;
        const int localTileId = (threadIdx.x) / tilesize;
        const int tilesPerBlock = blockDim.x / tilesize;
        const int laneInTile = threadIdx.x % tilesize;
        const int requiredTiles = tiles_per_subject_prefixsum[n_subjects];

        unsigned int* const subjectBackupsBegin = sharedmemory; // per tile shared memory to store subject
        unsigned int* const queryBackupsBegin = subjectBackupsBegin + encodedSequencePitchInInts2BitHiLo * tilesPerBlock; // per thread shared memory to store query
        unsigned int* const mySequencesBegin = queryBackupsBegin + encodedSequencePitchInInts2BitHiLo * blockDim.x; // per thread shared memory to store shifted sequence

        unsigned int* const subjectBackup = subjectBackupsBegin + encodedSequencePitchInInts2BitHiLo * localTileId; // accesed via identity
        unsigned int* const queryBackup = queryBackupsBegin + threadIdx.x; // accesed via no_bank_conflict_index
        unsigned int* const mySequence = mySequencesBegin + threadIdx.x; // accesed via no_bank_conflict_index

        for(int logicalTileId = globalTileId; logicalTileId < requiredTiles ; logicalTileId += tiles){

            const int subjectIndex = thrust::distance(tiles_per_subject_prefixsum,
                                                    thrust::lower_bound(
                                                        thrust::seq,
                                                        tiles_per_subject_prefixsum,
                                                        tiles_per_subject_prefixsum + n_subjects + 1,
                                                        logicalTileId + 1))-1;

            const int candidatesBeforeThisSubject = candidates_per_subject_prefixsum[subjectIndex];
            const int maxCandidateIndex_excl = candidates_per_subject_prefixsum[subjectIndex+1];
            //const int tilesForThisSubject = tiles_per_subject_prefixsum[subjectIndex + 1] - tiles_per_subject_prefixsum[subjectIndex];
            const int tileForThisSubject = logicalTileId - tiles_per_subject_prefixsum[subjectIndex];
            const int candidateIndex = candidatesBeforeThisSubject + tileForThisSubject * tilesize + laneInTile;

            const int subjectbases = subjectSequencesLength[subjectIndex];
            const int subjectints = getEncodedNumInts2BitHiLo(subjectbases);
            const unsigned int* subjectptr = subjectDataHiLo + std::size_t(subjectIndex) * encodedSequencePitchInInts2BitHiLo;

            //save subject in shared memory (in parallel, per tile)
            for(int lane = laneInTile; lane < encodedSequencePitchInInts2BitHiLo; lane += tilesize) {
                subjectBackup[identity(lane)] = subjectptr[lane];
                //transposed
                //subjectBackup[identity(lane)] = ((unsigned int*)(subjectptr))[lane * n_subjects];
            }

            cg::tiled_partition<tilesize>(cg::this_thread_block()).sync();


            if(candidateIndex < maxCandidateIndex_excl){

                const int querybases = candidateSequencesLength[candidateIndex];
                const int queryints = getEncodedNumInts2BitHiLo(querybases);
                const int totalbases = subjectbases + querybases;
                const int minoverlap = max(min_overlap, int(float(subjectbases) * min_overlap_ratio));

                const unsigned int* candidateptr = candidateDataHiLoTransposed + std::size_t(candidateIndex);

                //save query in shared memory
                for(int i = 0; i < encodedSequencePitchInInts2BitHiLo; i += 1) {
                    //queryBackup[no_bank_conflict_index(i)] = ((unsigned int*)(candidateptr))[i];
                    //transposed
                    queryBackup[no_bank_conflict_index(i)] = candidateptr[i * n_candidates];
                }

                const unsigned int* const subjectBackup_hi = subjectBackup;
                const unsigned int* const subjectBackup_lo = subjectBackup + identity(subjectints/2);
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
                    bestShift[orientation] = -querybases;    // shift of query relative to subject. shift < 0 if query begins before subject

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

                    //initialize threadlocal smem array with subject
                    for(int i = 0; i < encodedSequencePitchInInts2BitHiLo; i += 1) {
                        mySequence[no_bank_conflict_index(i)] = subjectBackup[identity(i)];
                    }

                    unsigned int* mySequence_hi = mySequence;
                    unsigned int* mySequence_lo = mySequence + no_bank_conflict_index(subjectints / 2);

                    for(int shift = 0; shift < subjectbases - minoverlap + 1; shift += 1) {
                        const int overlapsize = min(subjectbases - shift, querybases);

                        bool b = handle_shift(shift, overlapsize,
                                        mySequence_hi, mySequence_lo, no_bank_conflict_index,
                                        subjectints,
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
                        const int overlapsize = min(subjectbases, querybases + shift);

                        bool b = handle_shift(shift, overlapsize,
                                        mySequence_hi, mySequence_lo, no_bank_conflict_index,
                                        queryints,
                                        subjectBackup_hi, subjectBackup_lo, identity);
                        if(!b){
                            break;
                        }
                    }

                    const int queryoverlapbegin_incl = max(-bestShift[orientation], 0);
                    const int queryoverlapend_excl = min(querybases, subjectbases - bestShift[orientation]);
                    overlapsize[orientation] = queryoverlapend_excl - queryoverlapbegin_incl;
                    opnr[orientation] = bestScore[orientation] - totalbases + 2*overlapsize[orientation];
                }

                const BestAlignment_t flag = alignmentComparator(
                    overlapsize[0],
                    overlapsize[1],
                    opnr[0],
                    opnr[1],
                    bestShift[0] != -querybases,
                    bestShift[1] != -querybases,
                    subjectbases,
                    querybases
                );

                d_alignment_best_alignment_flags[candidateIndex] = flag;
                d_alignment_overlaps[candidateIndex] = flag == BestAlignment_t::Forward ? overlapsize[0] : overlapsize[1];
                d_alignment_shifts[candidateIndex] = flag == BestAlignment_t::Forward ? bestShift[0] : bestShift[1];
                d_alignment_nOps[candidateIndex] = flag == BestAlignment_t::Forward ? opnr[0] : opnr[1];
                d_alignment_isValid[candidateIndex] = flag == BestAlignment_t::Forward ? bestShift[0] != -querybases : bestShift[1] != -querybases;
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
    popcount_shifted_hamming_distance_ctpitch_kernel(
                const unsigned int* __restrict__ subjectDataHiLoTransposed,
                const unsigned int* __restrict__ candidateDataHiLoTransposed,
                const int* __restrict__ subjectSequencesLength,
                const int* __restrict__ candidateSequencesLength,
                BestAlignment_t* __restrict__ bestAlignmentFlags,
                int* __restrict__ alignment_overlaps,
                int* __restrict__ alignment_shifts,
                int* __restrict__ alignment_nOps,
                bool* __restrict__ alignment_isValid,
                const int* __restrict__ d_anchorIndicesOfCandidates,
                int n_subjects,
                int n_candidates,
                size_t encodedSequencePitchInInts2BitHiLo,
                int min_overlap,
                float maxErrorRate,
                float min_overlap_ratio,
                float estimatedNucleotideErrorRate){

        static_assert(maxValidIntsPerSequence % 2 == 0, ""); //2bithilo has even number of ints

        auto identity = [](auto logical_index){
            return logical_index;
        };

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
            int subjectlength,
            int querylength)->BestAlignment_t{

            return choose_best_alignment(
                fwd_alignment_overlap,
                revc_alignment_overlap,
                fwd_alignment_nops,
                revc_alignment_nops,
                fwd_alignment_isvalid,
                revc_alignment_isvalid,
                subjectlength,
                querylength,
                min_overlap_ratio,
                min_overlap,
                estimatedNucleotideErrorRate * 4.0f
            );
        };


        unsigned int subjectBackupHi[maxValidIntsPerSequence / 2];
        unsigned int subjectBackupLo[maxValidIntsPerSequence / 2];
        unsigned int queryBackupHi[maxValidIntsPerSequence / 2];
        unsigned int queryBackupLo[maxValidIntsPerSequence / 2];
        unsigned int mySequenceHi[maxValidIntsPerSequence / 2];
        unsigned int mySequenceLo[maxValidIntsPerSequence / 2];

        auto reverseComplementQuery = [&](int querylength, int validInts){
            auto reverse_complement_int = [](auto n) {
                n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
                n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
                n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
                n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
                n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
                return ~n;
            };

            constexpr int N = maxValidIntsPerSequence / 2;

            #pragma unroll
            for(int i = 0; i < N/2; ++i){
                const unsigned int hifront = reverse_complement_int(queryBackupHi[i]);
                const unsigned int hiback = reverse_complement_int(queryBackupHi[N - 1 - i]);
                queryBackupHi[i] = hiback;
                queryBackupHi[N - 1 - i] = hifront;
    
                const unsigned int lofront = reverse_complement_int(queryBackupLo[i]);
                const unsigned int loback = reverse_complement_int(queryBackupLo[N - 1 - i]);
                queryBackupLo[i] = loback;
                queryBackupLo[N - 1 - i] = lofront;
            }

            if(N % 2 == 1){
                constexpr int middleindex = N/2;
                queryBackupHi[middleindex] = reverse_complement_int(queryBackupHi[middleindex]);
                queryBackupLo[middleindex] = reverse_complement_int(queryBackupLo[middleindex]);
            }

            //fix unused data

            const int unusedInts = N - getEncodedNumInts2BitHiLo(querylength) / 2;
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

            const int subjectIndex = d_anchorIndicesOfCandidates[candidateIndex];          
            const int subjectbases = subjectSequencesLength[subjectIndex];
            const int querybases = candidateSequencesLength[candidateIndex];

            const unsigned int* subjectptr = subjectDataHiLoTransposed + std::size_t(subjectIndex);

            #pragma unroll 
            for(int i = 0; i < maxValidIntsPerSequence / 2; i++){
                subjectBackupHi[i] = subjectptr[(i) * n_subjects];
                subjectBackupLo[i] = subjectptr[(i + maxValidIntsPerSequence / 2) * n_subjects];
            }

            const unsigned int* candidateptr = candidateDataHiLoTransposed + std::size_t(candidateIndex);

            //save query in reg

            #pragma unroll 
            for(int i = 0; i < maxValidIntsPerSequence / 2; i++){
                queryBackupHi[i] = candidateptr[i * n_candidates];
                queryBackupLo[i] = candidateptr[(i + maxValidIntsPerSequence / 2) * n_candidates];
            }

            //begin SHD algorithm

            const int subjectints = getEncodedNumInts2BitHiLo(subjectbases);
            const int queryints = getEncodedNumInts2BitHiLo(querybases);
            const int totalbases = subjectbases + querybases;
            const int minoverlap = max(min_overlap, int(float(subjectbases) * min_overlap_ratio));

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
                bestShift[orientation] = -querybases;    // shift of query relative to subject. shift < 0 if query begins before subject

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
                    mySequenceHi[i] = subjectBackupHi[i];
                    mySequenceLo[i] = subjectBackupLo[i];
                }

                for(int shift = 0; shift < subjectbases - minoverlap + 1; shift += 1) {
                    const int overlapsize = min(subjectbases - shift, querybases);

                    bool b = handle_shift(
                        shift, overlapsize,
                        mySequenceHi, mySequenceLo,
                        queryBackupHi, queryBackupLo
                    );
                    if(!b){
                        break;
                    }
                }

                //initialize threadlocal smem array with query
                #pragma unroll 
                for(int i = 0; i < maxValidIntsPerSequence / 2; i++){
                    mySequenceHi[i] = queryBackupHi[i];
                    mySequenceLo[i] = queryBackupLo[i];
                }

                for(int shift = -1; shift >= -querybases + minoverlap; shift -= 1) {
                    const int overlapsize = min(subjectbases, querybases + shift);

                    bool b = handle_shift(
                        shift, overlapsize,
                        mySequenceHi, mySequenceLo,
                        subjectBackupHi, subjectBackupLo
                    );
                    if(!b){
                        break;
                    }
                }

                const int queryoverlapbegin_incl = max(-bestShift[orientation], 0);
                const int queryoverlapend_excl = min(querybases, subjectbases - bestShift[orientation]);
                overlapsize[orientation] = queryoverlapend_excl - queryoverlapbegin_incl;
                opnr[orientation] = bestScore[orientation] - totalbases + 2*overlapsize[orientation];
            }

            const BestAlignment_t flag = alignmentComparator(
                overlapsize[0],
                overlapsize[1],
                opnr[0],
                opnr[1],
                bestShift[0] != -querybases,
                bestShift[1] != -querybases,
                subjectbases,
                querybases
            );

            bestAlignmentFlags[candidateIndex] = flag;
            alignment_overlaps[candidateIndex] = flag == BestAlignment_t::Forward ? overlapsize[0] : overlapsize[1];
            alignment_shifts[candidateIndex] = flag == BestAlignment_t::Forward ? bestShift[0] : bestShift[1];
            alignment_nOps[candidateIndex] = flag == BestAlignment_t::Forward ? opnr[0] : opnr[1];
            alignment_isValid[candidateIndex] = flag == BestAlignment_t::Forward ? bestShift[0] != -querybases : bestShift[1] != -querybases;
        }
    }














    __global__
    void cuda_find_best_alignment_kernel_exp(
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                const int* __restrict__ d_candidates_per_subject_prefixsum,
                int n_subjects,
                int n_queries,
                float min_overlap_ratio,
                int min_overlap,
                float estimatedErrorrate,
                read_number debugsubjectreadid){

        auto getSubjectLength = [&] (int subjectIndex){
            const int length = d_sequencePointers.subjectSequencesLength[subjectIndex];
            return length;
        };

        auto getCandidateLength = [&] (int resultIndex){
            const int length = d_sequencePointers.candidateSequencesLength[resultIndex];
            return length;
        };

        auto comp = [&] (int fwd_alignment_overlap,
                        int revc_alignment_overlap,
                        int fwd_alignment_nops,
                        int revc_alignment_nops,
                        bool fwd_alignment_isvalid,
                        bool revc_alignment_isvalid,
                        int subjectlength,
                        int querylength)->BestAlignment_t{

            return choose_best_alignment(fwd_alignment_overlap,
                        revc_alignment_overlap,
                        fwd_alignment_nops,
                        revc_alignment_nops,
                        fwd_alignment_isvalid,
                        revc_alignment_isvalid,
                        subjectlength,
                        querylength,
                        min_overlap_ratio,
                        min_overlap,
                        estimatedErrorrate * 4.0f);
        };

        int* const d_alignment_scores = d_alignmentresultpointers.scores;
        int* const d_alignment_overlaps = d_alignmentresultpointers.overlaps;
        int* const d_alignment_shifts = d_alignmentresultpointers.shifts;
        int* const d_alignment_nOps = d_alignmentresultpointers.nOps;
        bool* const d_alignment_isValid = d_alignmentresultpointers.isValid;
        BestAlignment_t* const d_alignment_best_alignment_flags = d_alignmentresultpointers.bestAlignmentFlags;

        for(unsigned resultIndex = threadIdx.x + blockDim.x * blockIdx.x; resultIndex < n_queries; resultIndex += gridDim.x * blockDim.x) {
            const unsigned fwdIndex = resultIndex;
            const unsigned revcIndex = resultIndex + n_queries;

            const int fwd_alignment_score = d_alignment_scores[fwdIndex];
            const int fwd_alignment_overlap = d_alignment_overlaps[fwdIndex];
            const int fwd_alignment_shift = d_alignment_shifts[fwdIndex];
            const int fwd_alignment_nops = d_alignment_nOps[fwdIndex];
            const bool fwd_alignment_isvalid = d_alignment_isValid[fwdIndex];

            const int revc_alignment_score = d_alignment_scores[revcIndex];
            const int revc_alignment_overlap = d_alignment_overlaps[revcIndex];
            const int revc_alignment_shift = d_alignment_shifts[revcIndex];
            const int revc_alignment_nops = d_alignment_nOps[revcIndex];
            const bool revc_alignment_isvalid = d_alignment_isValid[revcIndex];

            //assert(fwd_alignment_isvalid || fwd_alignment_shift == -101);
            //assert(revc_alignment_isvalid || revc_alignment_shift == -101);

            //const int querylength = d_candidate_sequences_lengths[resultIndex];
            const int querylength = getCandidateLength(resultIndex);

            //find subjectindex
            /*int subjectIndex = 0;
            for(; subjectIndex < n_subjects; subjectIndex++) {
                if(resultIndex < d_candidates_per_subject_prefixsum[subjectIndex+1])
                    break;
            }*/

            const int subjectIndex = thrust::distance(d_candidates_per_subject_prefixsum,
                                                    thrust::lower_bound(
                                                        thrust::seq,
                                                        d_candidates_per_subject_prefixsum,
                                                        d_candidates_per_subject_prefixsum + n_subjects + 1,
                                                        resultIndex + 1))-1;

            //const int subjectlength = d_subject_sequences_lengths[subjectIndex];
            const int subjectlength = getSubjectLength(subjectIndex);

            const BestAlignment_t flag = comp(fwd_alignment_overlap,
                        revc_alignment_overlap,
                        fwd_alignment_nops,
                        revc_alignment_nops,
                        fwd_alignment_isvalid,
                        revc_alignment_isvalid,
                        subjectlength,
                        querylength);

            d_alignment_best_alignment_flags[resultIndex] = flag;

            d_alignment_scores[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_score : revc_alignment_score;
            d_alignment_overlaps[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_overlap : revc_alignment_overlap;
            d_alignment_shifts[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_shift : revc_alignment_shift;
            d_alignment_nOps[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_nops : revc_alignment_nops;
            d_alignment_isValid[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_isvalid : revc_alignment_isvalid;
        }
    }


    template<int BLOCKSIZE>
    __global__
    void cuda_filter_alignments_by_mismatchratio_kernel(
                AlignmentResultPointers d_alignmentresultpointers,
                const int* __restrict__ d_candidates_per_subject_prefixsum,
                int n_subjects,
                int n_candidates,
                float mismatchratioBaseFactor,
                float goodAlignmentsCountThreshold){

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceInt::TempStorage intreduce;
            int broadcast[3];
        } temp_storage;


        for(int subjectindex = blockIdx.x; subjectindex < n_subjects; subjectindex += gridDim.x) {

            const int candidatesForSubject = d_candidates_per_subject_prefixsum[subjectindex+1]
                                            - d_candidates_per_subject_prefixsum[subjectindex];

            const int firstIndex = d_candidates_per_subject_prefixsum[subjectindex];

            //printf("subjectindex %d\n", subjectindex);

            int counts[3]{0,0,0};

            //if(threadIdx.x == 0){
            //    printf("my_n_indices %d\n", my_n_indices);
            //}

            for(int index = threadIdx.x; index < candidatesForSubject; index += blockDim.x) {

                const int candidate_index = firstIndex + index;
                if(d_alignmentresultpointers.bestAlignmentFlags[candidate_index] != BestAlignment_t::None) {

                    const int alignment_overlap = d_alignmentresultpointers.overlaps[candidate_index];
                    const int alignment_nops = d_alignmentresultpointers.nOps[candidate_index];

                    const float mismatchratio = float(alignment_nops) / alignment_overlap;

                    if(mismatchratio >= 4 * mismatchratioBaseFactor) {
                        d_alignmentresultpointers.bestAlignmentFlags[candidate_index] = BestAlignment_t::None;
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
                mismatchratioThreshold = -1.0f;                         //this will invalidate all alignments for subject
                //mismatchratioThreshold = 4 * mismatchratioBaseFactor; //use alignments from every bin
                //mismatchratioThreshold = 1.1f;
            }

            // Invalidate all alignments for subject with mismatchratio >= mismatchratioThreshold
            for(int index = threadIdx.x; index < candidatesForSubject; index += blockDim.x) {
                const int candidate_index = firstIndex + index;
                if(d_alignmentresultpointers.bestAlignmentFlags[candidate_index] != BestAlignment_t::None) {

                    const int alignment_overlap = d_alignmentresultpointers.overlaps[candidate_index];
                    const int alignment_nops = d_alignmentresultpointers.nOps[candidate_index];

                    const float mismatchratio = float(alignment_nops) / alignment_overlap;

                    const bool doRemove = mismatchratio >= mismatchratioThreshold;
                    if(doRemove){
                        d_alignmentresultpointers.bestAlignmentFlags[candidate_index] = BestAlignment_t::None;
                    }
                }
            }
        }
    }

    







    //####################   KERNEL DISPATCH   ####################


    void call_popcount_shifted_hamming_distance_kernel_async(
                int* d_alignment_overlaps,
                int* d_alignment_shifts,
                int* d_alignment_nOps,
                bool* d_alignment_isValid,
                BestAlignment_t* d_alignment_best_alignment_flags,
                const unsigned int* d_subjectSequencesData,
                const unsigned int* d_candidateSequencesData,
                const int* d_subjectSequencesLength,
                const int* d_candidateSequencesLength,
    			const int* d_candidates_per_subject_prefixsum,
                const int* h_candidates_per_subject,
                const int* d_candidates_per_subject,
                const int* d_anchorIndicesOfCandidates,
    			int n_subjects,
    			int n_queries,
                int maximumSequenceLength,
                int encodedSequencePitchInInts2Bit,
    			int min_overlap,
    			float maxErrorRate,
                float min_overlap_ratio,
                float estimatedNucleotideErrorRate,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

            const int intsPerSequence2BitHiLo = getEncodedNumInts2BitHiLo(maximumSequenceLength);
            const int bytesPerSequence2BitHilo = intsPerSequence2BitHiLo * sizeof(unsigned int);

            unsigned int* d_subjectDataHiLoTransposed = nullptr;
            unsigned int* d_candidateDataHiLoTransposed = nullptr;

            cubCachingAllocator.DeviceAllocate(
                (void**)&d_candidateDataHiLoTransposed, 
                sizeof(unsigned int) * intsPerSequence2BitHiLo * n_queries, 
                stream
            ); CUERR;

            callConversionKernel2BitTo2BitHiLoNT(
                d_candidateSequencesData,
                encodedSequencePitchInInts2Bit,
                d_candidateDataHiLoTransposed,
                intsPerSequence2BitHiLo,
                d_candidateSequencesLength,
                n_queries,
                stream,
                handle
            );
            
            if(maximumSequenceLength <= 256){
                cubCachingAllocator.DeviceAllocate(
                    (void**)&d_subjectDataHiLoTransposed, 
                    sizeof(unsigned int) * intsPerSequence2BitHiLo * n_subjects, 
                    stream
                ); CUERR;

                callConversionKernel2BitTo2BitHiLoNT(
                    d_subjectSequencesData,
                    encodedSequencePitchInInts2Bit,
                    d_subjectDataHiLoTransposed,
                    intsPerSequence2BitHiLo,
                    d_subjectSequencesLength,
                    n_subjects,
                    stream,
                    handle
                );
            }
            

            if(maximumSequenceLength <= 128){
               
                constexpr int maxValidIntsPerSequence = 8;

                constexpr int blocksize = 128;
                int max_blocks_per_device = 1;

                KernelLaunchConfig kernelLaunchConfig;
                kernelLaunchConfig.threads_per_block = blocksize;
                kernelLaunchConfig.smem = 0;

                auto iter = handle.kernelPropertiesMap.find(KernelId::PopcountSHDTiledPitch8);
                if(iter == handle.kernelPropertiesMap.end()) {

                    std::map<KernelLaunchConfig, KernelProperties> mymap;

                    KernelProperties kernelProperties;
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &kernelProperties.max_blocks_per_SM,
                        popcount_shifted_hamming_distance_ctpitch_kernel<blocksize, maxValidIntsPerSequence>,
                        kernelLaunchConfig.threads_per_block, 
                        kernelLaunchConfig.smem
                    ); CUERR;

                    mymap[kernelLaunchConfig] = kernelProperties;
                    max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

                    handle.kernelPropertiesMap[KernelId::PopcountSHDTiledPitch8] = std::move(mymap);
                }else{
                    std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
                    const KernelProperties& kernelProperties = map[kernelLaunchConfig];
                    max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
                }

                dim3 block(blocksize, 1, 1);
                const int numBlocks = SDIV(n_queries, blocksize);
                dim3 grid(std::min(numBlocks, max_blocks_per_device), 1, 1);

                popcount_shifted_hamming_distance_ctpitch_kernel<blocksize, maxValidIntsPerSequence>
                    <<<grid, block, 0, stream>>>(
                        d_subjectDataHiLoTransposed,
                        d_candidateDataHiLoTransposed,
                        d_subjectSequencesLength,
                        d_candidateSequencesLength,
                        d_alignment_best_alignment_flags,
                        d_alignment_overlaps,
                        d_alignment_shifts,
                        d_alignment_nOps,
                        d_alignment_isValid,
                        d_anchorIndicesOfCandidates,
                        n_subjects,
                        n_queries,
                        intsPerSequence2BitHiLo, 
                        min_overlap,
                        maxErrorRate,
                        min_overlap_ratio,
                        estimatedNucleotideErrorRate
                ); CUERR;

                cubCachingAllocator.DeviceFree(d_subjectDataHiLoTransposed);  CUERR;

            }else if(maximumSequenceLength <= 256){
               
                constexpr int maxValidIntsPerSequence = 16;

                constexpr int blocksize = 128;
                int max_blocks_per_device = 1;

                KernelLaunchConfig kernelLaunchConfig;
                kernelLaunchConfig.threads_per_block = blocksize;
                kernelLaunchConfig.smem = 0;

                auto iter = handle.kernelPropertiesMap.find(KernelId::PopcountSHDTiledPitch8);
                if(iter == handle.kernelPropertiesMap.end()) {

                    std::map<KernelLaunchConfig, KernelProperties> mymap;

                    KernelProperties kernelProperties;
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &kernelProperties.max_blocks_per_SM,
                        popcount_shifted_hamming_distance_ctpitch_kernel<blocksize, maxValidIntsPerSequence>,
                        kernelLaunchConfig.threads_per_block, 
                        kernelLaunchConfig.smem
                    ); CUERR;

                    mymap[kernelLaunchConfig] = kernelProperties;
                    max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

                    handle.kernelPropertiesMap[KernelId::PopcountSHDTiledPitch8] = std::move(mymap);
                }else{
                    std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
                    const KernelProperties& kernelProperties = map[kernelLaunchConfig];
                    max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
                }

                dim3 block(blocksize, 1, 1);
                const int numBlocks = SDIV(n_queries, blocksize);
                dim3 grid(std::min(numBlocks, max_blocks_per_device), 1, 1);

                

                popcount_shifted_hamming_distance_ctpitch_kernel<blocksize, maxValidIntsPerSequence>
                    <<<grid, block, 0, stream>>>(
                        d_subjectDataHiLoTransposed,
                        d_candidateDataHiLoTransposed,
                        d_subjectSequencesLength,
                        d_candidateSequencesLength,
                        d_alignment_best_alignment_flags,
                        d_alignment_overlaps,
                        d_alignment_shifts,
                        d_alignment_nOps,
                        d_alignment_isValid,
                        d_anchorIndicesOfCandidates,
                        n_subjects,
                        n_queries,
                        intsPerSequence2BitHiLo, 
                        min_overlap,
                        maxErrorRate,
                        min_overlap_ratio,
                        estimatedNucleotideErrorRate
                ); CUERR;

                cubCachingAllocator.DeviceFree(d_subjectDataHiLoTransposed);  CUERR;

            }else{

                unsigned int* d_subjectDataHiLo = nullptr;

                cubCachingAllocator.DeviceAllocate(
                    (void**)&d_subjectDataHiLo, 
                    sizeof(unsigned int) * intsPerSequence2BitHiLo * n_subjects, 
                    stream
                ); CUERR;

                callConversionKernel2BitTo2BitHiLoNN(
                    d_subjectSequencesData,
                    encodedSequencePitchInInts2Bit,
                    d_subjectDataHiLo,
                    intsPerSequence2BitHiLo,
                    d_subjectSequencesLength,
                    n_subjects,
                    stream,
                    handle
                );

                constexpr int tilesize = 16;

                int* d_tiles_per_subject_prefixsum;
                cubCachingAllocator.DeviceAllocate((void**)&d_tiles_per_subject_prefixsum, sizeof(int) * (n_subjects+1), stream);  CUERR;

                // calculate blocks per subject prefixsum
                auto getTilesPerSubject = [=] __device__ (int candidates_for_subject){
                    return SDIV(candidates_for_subject, tilesize);
                };
                cub::TransformInputIterator<int,decltype(getTilesPerSubject), const int*>
                    d_tiles_per_subject(d_candidates_per_subject,
                                getTilesPerSubject);

                void* tempstorage = nullptr;
                size_t tempstoragesize = 0;

                cub::DeviceScan::InclusiveSum(nullptr,
                            tempstoragesize,
                            d_tiles_per_subject,
                            d_tiles_per_subject_prefixsum+1,
                            n_subjects,
                            stream); CUERR;

                cubCachingAllocator.DeviceAllocate((void**)&tempstorage, tempstoragesize, stream);  CUERR;

                cub::DeviceScan::InclusiveSum(tempstorage,
                            tempstoragesize,
                            d_tiles_per_subject,
                            d_tiles_per_subject_prefixsum+1,
                            n_subjects,
                            stream); CUERR;

                cubCachingAllocator.DeviceFree(tempstorage);  CUERR;

                call_set_kernel_async(d_tiles_per_subject_prefixsum,
                                        0,
                                        0,
                                        stream);




                constexpr int blocksize = 128;
                constexpr int tilesPerBlock = blocksize / tilesize;

                //const int requiredTiles = h_tiles_per_subject_prefixsum[n_subjects];

                int requiredTiles = 0;
                for(int i = 0; i < n_subjects;i++){
                    requiredTiles += SDIV(h_candidates_per_subject[i], tilesize);
                }

                const int requiredBlocks = SDIV(requiredTiles, tilesPerBlock);

                //printf("n_subjects %d, n_queries %d\n", n_subjects, n_queries);


                const std::size_t smem = sizeof(char) * (bytesPerSequence2BitHilo * tilesPerBlock + bytesPerSequence2BitHilo * blocksize * 2);

                int max_blocks_per_device = 1;

                KernelLaunchConfig kernelLaunchConfig;
                kernelLaunchConfig.threads_per_block = blocksize;
                kernelLaunchConfig.smem = smem;

                auto iter = handle.kernelPropertiesMap.find(KernelId::PopcountSHDTiled);
                if(iter == handle.kernelPropertiesMap.end()) {

                    std::map<KernelLaunchConfig, KernelProperties> mymap;

                    #define getProp(blocksize, tilesize) { \
                            KernelLaunchConfig kernelLaunchConfig; \
                            kernelLaunchConfig.threads_per_block = (blocksize); \
                            kernelLaunchConfig.smem = sizeof(char) * (bytesPerSequence2BitHilo * tilesPerBlock + bytesPerSequence2BitHilo * blocksize * 2); \
                            KernelProperties kernelProperties; \
                            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                                popcount_shifted_hamming_distance_kernel<tilesize>, \
                                        kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
                            mymap[kernelLaunchConfig] = kernelProperties; \
                    }
                    getProp(1, tilesize);
                    getProp(32, tilesize);
                    getProp(64, tilesize);
                    getProp(96, tilesize);
                    getProp(128, tilesize);
                    getProp(160, tilesize);
                    getProp(192, tilesize);
                    getProp(224, tilesize);
                    getProp(256, tilesize);

                    const auto& kernelProperties = mymap[kernelLaunchConfig];
                    max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

                    handle.kernelPropertiesMap[KernelId::PopcountSHDTiled] = std::move(mymap);

                    #undef getProp
                }else{
                    std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
                    const KernelProperties& kernelProperties = map[kernelLaunchConfig];
                    max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
                }

                #define mycall popcount_shifted_hamming_distance_kernel<tilesize> \
                                                    <<<grid, block, smem, stream>>>( \
                                                    d_subjectDataHiLo, \
                                                    d_candidateDataHiLoTransposed, \
                                                    d_alignment_overlaps, \
                                                    d_alignment_shifts, \
                                                    d_alignment_nOps, \
                                                    d_alignment_isValid, \
                                                    d_alignment_best_alignment_flags, \
                                                    d_subjectSequencesLength, \
                                                    d_candidateSequencesLength, \
                                                    d_candidates_per_subject_prefixsum, \
                                                    d_tiles_per_subject_prefixsum, \
                                                    n_subjects, \
                                                    n_queries, \
                                                    intsPerSequence2BitHiLo, \
                                                    min_overlap, \
                                                    maxErrorRate, \
                                                    min_overlap_ratio, \
                                                    estimatedNucleotideErrorRate); CUERR;

                dim3 block(blocksize, 1, 1);
                dim3 grid(std::min(requiredBlocks, max_blocks_per_device), 1, 1);

                mycall;

                #undef mycall

                cubCachingAllocator.DeviceFree(d_tiles_per_subject_prefixsum);  CUERR;

                cubCachingAllocator.DeviceFree(d_subjectDataHiLo);  CUERR;
            }

            
            
            cubCachingAllocator.DeviceFree(d_candidateDataHiLoTransposed);  CUERR;

            // cudaDeviceSynchronize();
            // std::exit(0);
    }

    void call_cuda_find_best_alignment_kernel_async_exp(
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
    			const int* d_candidates_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
    			float min_overlap_ratio,
    			int min_overlap,
                float estimatedErrorrate,
    			cudaStream_t stream,
                KernelLaunchHandle& handle,
                read_number debugsubjectreadid){

    	const int blocksize = 128;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::FindBestAlignmentExp);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                            cuda_find_best_alignment_kernel_exp, \
                            kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
                mymap[kernelLaunchConfig] = kernelProperties; }

    		getProp(32);
    		getProp(64);
    		getProp(96);
    		getProp(128);
    		getProp(160);
    		getProp(192);
    		getProp(224);
    		getProp(256);

    		const auto& kernelProperties = mymap[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

    		handle.kernelPropertiesMap[KernelId::FindBestAlignmentExp] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize,1,1);
    	dim3 grid(std::min(max_blocks_per_device, SDIV(n_queries, blocksize)), 1, 1);

    	cuda_find_best_alignment_kernel_exp<<<grid, block, smem, stream>>>(
            		d_alignmentresultpointers,
                    d_sequencePointers,
            		d_candidates_per_subject_prefixsum,
            		n_subjects,
            		n_queries,
            		min_overlap_ratio,
            		min_overlap,
                    estimatedErrorrate,
                    debugsubjectreadid); CUERR;

    }


    void call_cuda_filter_alignments_by_mismatchratio_kernel_async(
    			AlignmentResultPointers d_alignmentresultpointers,
    			const int* d_candidates_per_subject_prefixsum,
    			int n_subjects,
    			int n_candidates,
    			float mismatchratioBaseFactor,
    			float goodAlignmentsCountThreshold,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

    	const int blocksize = 128;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::FilterAlignmentsByMismatchRatio);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
                    KernelLaunchConfig kernelLaunchConfig; \
                    kernelLaunchConfig.threads_per_block = (blocksize); \
                    kernelLaunchConfig.smem = 0; \
                    KernelProperties kernelProperties; \
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                                cuda_filter_alignments_by_mismatchratio_kernel<(blocksize)>, \
                                kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
                    mymap[kernelLaunchConfig] = kernelProperties; \
            }

    		getProp(32);
    		getProp(64);
    		getProp(96);
    		getProp(128);
    		getProp(160);
    		getProp(192);
    		getProp(224);
    		getProp(256);

    		const auto& kernelProperties = mymap[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

    		handle.kernelPropertiesMap[KernelId::FilterAlignmentsByMismatchRatio] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(max_blocks_per_device, n_subjects));

    	#define mycall(blocksize) cuda_filter_alignments_by_mismatchratio_kernel<(blocksize)> \
    	        <<<grid, block, smem, stream>>>( \
    		d_alignmentresultpointers, \
    		d_candidates_per_subject_prefixsum, \
    		n_subjects, \
    		n_candidates, \
    		mismatchratioBaseFactor, \
    		goodAlignmentsCountThreshold); CUERR;

    	switch(blocksize) {
    	case 32: mycall(32); break;
    	case 64: mycall(64); break;
    	case 96: mycall(96); break;
    	case 128: mycall(128); break;
    	case 160: mycall(160); break;
    	case 192: mycall(192); break;
    	case 224: mycall(224); break;
    	case 256: mycall(256); break;
    	default: mycall(256); break;
    	}

    		#undef mycall
    }


    void callSelectIndicesOfGoodCandidatesKernelAsync(
            int* d_indicesOfGoodCandidates,
            int* d_numIndicesPerAnchor,
            int* d_totalNumIndices,
            const BestAlignment_t* d_alignmentFlags,
            const int* d_candidates_per_subject,
            const int* d_candidates_per_subject_prefixsum,
            const int* d_anchorIndicesOfCandidates,
            int numAnchors,
            int numCandidates,
            cudaStream_t stream,
            KernelLaunchHandle& handle){

        constexpr int blocksize = 128;
        constexpr int tilesize = 32;

        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::SelectIndicesOfGoodCandidates);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    selectIndicesOfGoodCandidatesKernel<(blocksize), tilesize>, \
                                                                kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
                mymap[kernelLaunchConfig] = kernelProperties; \
            }

            getProp(32);
            getProp(64);
            getProp(96);
            getProp(128);
            getProp(160);
            getProp(192);
            getProp(224);
            getProp(256);

            const auto& kernelProperties = mymap[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

            handle.kernelPropertiesMap[KernelId::SelectIndicesOfGoodCandidates] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }

        cudaMemsetAsync(d_numIndicesPerAnchor, 0, numAnchors * sizeof(int), stream); CUERR;
        cudaMemsetAsync(d_totalNumIndices, 0, sizeof(int), stream); CUERR;

        dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(SDIV(numCandidates, blocksize), max_blocks_per_device));

        selectIndicesOfGoodCandidatesKernel<blocksize, tilesize><<<grid, block, 0, stream>>>(
            d_indicesOfGoodCandidates,
            d_numIndicesPerAnchor,
            d_totalNumIndices,
            d_alignmentFlags,
            d_candidates_per_subject,
            d_candidates_per_subject_prefixsum,
            d_anchorIndicesOfCandidates,
            numAnchors,
            numCandidates
        );
    }


}
}
