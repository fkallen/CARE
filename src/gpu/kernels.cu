#include <gpu/kernels.hpp>
#include <gpu/bestalignment.hpp>
#include <gpu/qualityscoreweights.hpp>

#include <msa.hpp>
#include <sequence.hpp>

#include <shiftedhammingdistance_common.hpp>

#include <hpc_helpers.cuh>
#include <config.hpp>

#include <cassert>


#ifdef __NVCC__
#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#endif

namespace care{
namespace gpu{

#ifdef __NVCC__

    KernelLaunchHandle make_kernel_launch_handle(int deviceId){
        KernelLaunchHandle handle;
        handle.deviceId = deviceId;
        cudaGetDeviceProperties(&handle.deviceProperties, deviceId); CUERR;
        return handle;
    }


    //####################   KERNELS   ####################


    __global__
    void
    cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel(
                int* __restrict__ alignment_scores,
                int* __restrict__ alignment_overlaps,
                int* __restrict__ alignment_shifts,
                int* __restrict__ alignment_nOps,
                bool* __restrict__ alignment_isValid,
                const char* __restrict__ subject_sequences_data,
                const char* __restrict__ candidate_sequences_data,
                const int* __restrict__ subject_sequences_lengths,
                const int* __restrict__ candidate_sequences_lengths,
                const int* __restrict__ candidates_per_subject_prefixsum,
                const int* __restrict__ tiles_per_subject_prefixsum,
                const int tilesize,
                int n_subjects,
                int n_candidates,
                size_t encodedsequencepitch,
                int max_sequence_bytes,
                int min_overlap,
                float maxErrorRate,
                float min_overlap_ratio){

        auto getNumBytes = [] (int sequencelength){
            return sizeof(unsigned int) * getEncodedNumInts2BitHiLo(sequencelength);
        };

        auto getSubjectPtr = [&] (int subjectIndex){
            const char* result = subject_sequences_data + std::size_t(subjectIndex) * encodedsequencepitch;
            return result;
        };

        auto getCandidatePtr = [&] (int candidateIndex){
            const char* result = candidate_sequences_data + std::size_t(candidateIndex) * encodedsequencepitch;
            return result;
        };

        auto getSubjectLength = [&] (int subjectIndex){
            const int length = subject_sequences_lengths[subjectIndex];
            return length;
        };

        auto getCandidateLength = [&] (int candidateIndex){
            const int length = candidate_sequences_lengths[candidateIndex];
            return length;
        };

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

        auto hammingDistanceWithShift = [&](int shift, int overlapsize, int max_errors,
                                    unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                    int shiftptr_size,
                                    const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                    auto transfunc2){

            const int shiftamount = shift == 0 ? 0 : 1;

            shiftBitArrayLeftBy(shiftptr_hi, shiftptr_size / 2, shiftamount, transfunc1);
            shiftBitArrayLeftBy(shiftptr_lo, shiftptr_size / 2, shiftamount, transfunc1);

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

        // sizeof(char) * (max_sequence_bytes * num_tiles   // tiles share the subject
        //                    + max_sequence_bytes * num_threads // each thread works with its own candidate
        //                    + max_sequence_bytes * num_threads) // each thread needs memory to shift a sequence
        extern __shared__ unsigned int sharedmemory[];

        //set up shared memory pointers

        const int max_sequence_ints = max_sequence_bytes / sizeof(unsigned int);

        const int tiles = (blockDim.x * gridDim.x) / tilesize;
        const int globalTileId = (blockDim.x * blockIdx.x + threadIdx.x) / tilesize;
        const int localTileId = (threadIdx.x) / tilesize;
        const int tilesPerBlock = blockDim.x / tilesize;
        const int laneInTile = threadIdx.x % tilesize;
        const int requiredTiles = tiles_per_subject_prefixsum[n_subjects];

        unsigned int* const subjectBackupsBegin = sharedmemory; // per tile shared memory to store subject
        unsigned int* const queryBackupsBegin = subjectBackupsBegin + max_sequence_ints * tilesPerBlock; // per thread shared memory to store query
        unsigned int* const mySequencesBegin = queryBackupsBegin + max_sequence_ints * blockDim.x; // per thread shared memory to store shifted sequence

        unsigned int* const subjectBackup = subjectBackupsBegin + max_sequence_ints * localTileId; // accesed via identity
        unsigned int* const queryBackup = queryBackupsBegin + threadIdx.x; // accesed via no_bank_conflict_index
        unsigned int* const mySequence = mySequencesBegin + threadIdx.x; // accesed via no_bank_conflict_index

        /*if(globalTileId == 0 && laneInTile == 0)
            printf("%d %d %d\n", candidates_per_subject_prefixsum[0], candidates_per_subject_prefixsum[1], candidates_per_subject_prefixsum[2]);*/

        for(int logicalTileId = globalTileId; logicalTileId < requiredTiles * 2; logicalTileId += tiles){
            const bool isReverseComplement = logicalTileId >= requiredTiles;
            const int forwardTileId = isReverseComplement ? logicalTileId - requiredTiles : logicalTileId;

            //find subjectindex
            int subjectIndex = 0;
            for(; subjectIndex < n_subjects; subjectIndex++) {
                if(forwardTileId < tiles_per_subject_prefixsum[subjectIndex+1])
                    break;
            }

            //assert(subjectIndex < n_subjects);


            const int candidatesBeforeThisSubject = candidates_per_subject_prefixsum[subjectIndex];
            const int maxCandidateIndex_excl = candidates_per_subject_prefixsum[subjectIndex+1];
            //const int tilesForThisSubject = tiles_per_subject_prefixsum[subjectIndex + 1] - tiles_per_subject_prefixsum[subjectIndex];
            const int tileForThisSubject = forwardTileId - tiles_per_subject_prefixsum[subjectIndex];
            const int queryIndex = candidatesBeforeThisSubject + tileForThisSubject * tilesize + laneInTile;
            const int resultIndex = isReverseComplement ? queryIndex + n_candidates : queryIndex;



            /*if(tileForThisSubject < 0){
                printf("block %d tile %d thread %d logicalTileId %d requiredTiles %d isReverseComplement %d forwardTileId %d candidatesBeforeThisSubject %d tileForThisSubject %d subjectIndex %d\n",
                        blockIdx.x, globalTileId, threadIdx.x, logicalTileId, requiredTiles, isReverseComplement, forwardTileId,
                        candidatesBeforeThisSubject, tileForThisSubject, subjectIndex);

            }
            assert(forwardTileId >= 0);
            assert(candidatesBeforeThisSubject >= 0);
            assert(tileForThisSubject >= 0);
            assert(queryIndex >= 0);
            assert(resultIndex >= 0);*/

            /*if(laneInTile == 0)
                printf("block %d tile %d thread %d globalTileId %d logicalTileId %d requiredTiles %d isReverseComplement %d forwardTileId %d candidatesBeforeThisSubject %d tileForThisSubject %d subjectIndex %d queryIndex %d resultIndex %d\n",
                    blockIdx.x, globalTileId, threadIdx.x, globalTileId, logicalTileId, requiredTiles, isReverseComplement, forwardTileId,
                    candidatesBeforeThisSubject, tileForThisSubject, subjectIndex, queryIndex, resultIndex);*/

            const int subjectbases = getSubjectLength(subjectIndex);
            const char* subjectptr = getSubjectPtr(subjectIndex);

            //save subject in shared memory (in parallel, per tile)
            for(int lane = laneInTile; lane < max_sequence_ints; lane += tilesize) {
                subjectBackup[identity(lane)] = ((unsigned int*)(subjectptr))[lane];
            }

            cg::tiled_partition(cg::this_thread_block(), tilesize).sync();


            if(queryIndex < maxCandidateIndex_excl){
                /*printf("block %d tile %d thread %d logicalTileId %d requiredTiles %d isReverseComplement %d forwardTileId %d candidatesBeforeThisSubject %d tileForThisSubject %d subjectIndex %d queryIndex %d resultIndex %d\n",
                        blockIdx.x, globalTileId, threadIdx.x, logicalTileId, requiredTiles, isReverseComplement, forwardTileId,
                        candidatesBeforeThisSubject, tileForThisSubject, subjectIndex, queryIndex, resultIndex);*/

                const int querybases = getCandidateLength(queryIndex);
                const char* candidateptr = getCandidatePtr(queryIndex);

                //save query in shared memory
                for(int i = 0; i < max_sequence_ints; i += 1) {
                    queryBackup[no_bank_conflict_index(i)] = ((unsigned int*)(candidateptr))[i];
                }

                //queryIndex != resultIndex -> reverse complement
                if(isReverseComplement) {
                    make_reverse_complement_inplace(queryBackup, querybases, no_bank_conflict_index);
                }

                //begin SHD algorithm

                const int subjectints = getNumBytes(subjectbases) / sizeof(unsigned int);
                const int queryints = getNumBytes(querybases) / sizeof(unsigned int);
                const int totalbases = subjectbases + querybases;
                const int minoverlap = max(min_overlap, int(float(subjectbases) * min_overlap_ratio));


                const unsigned int* const subjectBackup_hi = subjectBackup;
                const unsigned int* const subjectBackup_lo = subjectBackup + identity(subjectints/2);
                const unsigned int* const queryBackup_hi = queryBackup;
                const unsigned int* const queryBackup_lo = queryBackup + no_bank_conflict_index(queryints/2);

                int bestScore = totalbases;                 // score is number of mismatches
                int bestShift = -querybases;                 // shift of query relative to subject. shift < 0 if query begins before subject

                auto handle_shift = [&](int shift, int overlapsize,
                                            unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                            int shiftptr_size,
                                            const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                            auto transfunc2){

                    const int max_errors = int(float(overlapsize) * maxErrorRate);

                    int score = hammingDistanceWithShift(shift, overlapsize, max_errors,
                                        shiftptr_hi,shiftptr_lo, transfunc1,
                                        shiftptr_size,
                                        otherptr_hi, otherptr_lo, transfunc2);

                    score = (score < max_errors ?
                            score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                            : std::numeric_limits<int>::max()); // too many errors, discard

                    if(score < bestScore){
                        bestScore = score;
                        bestShift = shift;
                    }
                };

                //initialize threadlocal smem array with subject
                for(int i = 0; i < max_sequence_ints; i += 1) {
                    mySequence[no_bank_conflict_index(i)] = subjectBackup[identity(i)];
                }

                unsigned int* mySequence_hi = mySequence;
                unsigned int* mySequence_lo = mySequence + no_bank_conflict_index(subjectints / 2);

                for(int shift = 0; shift < subjectbases - minoverlap + 1; shift += 1) {
                    const int overlapsize = min(subjectbases - shift, querybases);

                    handle_shift(shift, overlapsize,
                                    mySequence_hi, mySequence_lo, no_bank_conflict_index,
                                    subjectints,
                                    queryBackup_hi, queryBackup_lo, no_bank_conflict_index);
                }

                //initialize threadlocal smem array with query
                for(int i = 0; i < max_sequence_ints; i += 1) {
                    mySequence[no_bank_conflict_index(i)] = queryBackup[no_bank_conflict_index(i)];
                }

                mySequence_hi = mySequence;
                mySequence_lo = mySequence + no_bank_conflict_index(queryints / 2);

                for(int shift = -1; shift >= -querybases + minoverlap; shift -= 1) {
                    const int overlapsize = min(subjectbases, querybases + shift);

                    handle_shift(shift, overlapsize,
                                    mySequence_hi, mySequence_lo, no_bank_conflict_index,
                                    queryints,
                                    subjectBackup_hi, subjectBackup_lo, identity);
                }

                const int queryoverlapbegin_incl = max(-bestShift, 0);
                const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
                const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
                const int opnr = bestScore - totalbases + 2*overlapsize;

                alignment_scores[resultIndex] = bestScore;
                alignment_overlaps[resultIndex] = overlapsize;
                alignment_shifts[resultIndex] = bestShift;
                alignment_nOps[resultIndex] = opnr;
                alignment_isValid[resultIndex] = (bestShift != -querybases);
            }
        }
    }

    __global__
    void cuda_find_best_alignment_kernel_exp(
                BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
                int* __restrict__ d_alignment_scores,
                int* __restrict__ d_alignment_overlaps,
                int* __restrict__ d_alignment_shifts,
                int* __restrict__ d_alignment_nOps,
                bool* __restrict__ d_alignment_isValid,
                const int* __restrict__ d_subject_sequences_lengths,
                const int* __restrict__ d_candidate_sequences_lengths,
                const int* __restrict__ d_candidates_per_subject_prefixsum,
                int n_subjects,
                int n_queries,
                float min_overlap_ratio,
                int min_overlap,
                float estimatedErrorrate){

        auto getSubjectLength = [&] (int subjectIndex){
            const int length = d_subject_sequences_lengths[subjectIndex];
            return length;
        };

        auto getCandidateLength = [&] (int resultIndex){
            const int length = d_candidate_sequences_lengths[resultIndex];
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
            int subjectIndex = 0;
            for(; subjectIndex < n_subjects; subjectIndex++) {
                if(resultIndex < d_candidates_per_subject_prefixsum[subjectIndex+1])
                    break;
            }

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
                BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
                const int* __restrict__ d_alignment_overlaps,
                const int* __restrict__ d_alignment_nOps,
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
                if(d_alignment_best_alignment_flags[candidate_index] != BestAlignment_t::None) {

                    const int alignment_overlap = d_alignment_overlaps[candidate_index];
                    const int alignment_nops = d_alignment_nOps[candidate_index];

                    const float mismatchratio = float(alignment_nops) / alignment_overlap;
                    if(mismatchratio >= 4 * mismatchratioBaseFactor) {
                        d_alignment_best_alignment_flags[candidate_index] = BestAlignment_t::None;
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
                if(d_alignment_best_alignment_flags[candidate_index] != BestAlignment_t::None) {

                    const int alignment_overlap = d_alignment_overlaps[candidate_index];
                    const int alignment_nops = d_alignment_nOps[candidate_index];

                    const float mismatchratio = float(alignment_nops) / alignment_overlap;

                    const bool remove = mismatchratio >= mismatchratioThreshold;
                    if(remove)
                        d_alignment_best_alignment_flags[candidate_index] = BestAlignment_t::None;

                }
            }
        }
    }

    template<int BLOCKSIZE>
    __global__
    void msa_init_kernel_exp(
                MSAColumnProperties* __restrict__ msa_column_properties,
                const int* __restrict__ alignment_shifts,
                const BestAlignment_t* __restrict__ alignment_best_alignment_flags,
                const int* __restrict__ subject_sequences_lengths,
                const int* __restrict__ candidate_sequences_lengths,
                const int* __restrict__ indices,
                const int* __restrict__ indices_per_subject,
                const int* __restrict__ indices_per_subject_prefixsum,
                int n_subjects){

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceInt::TempStorage reduce;
        } temp_storage;

        auto getSubjectLength = [&] __device__ (int subjectIndex){
            const int length = subject_sequences_lengths[subjectIndex];
            return length;
        };

        auto getCandidateLength = [&] __device__ (int subjectIndex, int localCandidateIndex){
            const int* const indices_for_this_subject = indices + indices_per_subject_prefixsum[subjectIndex];
            const int index = indices_for_this_subject[localCandidateIndex];
            const int length = candidate_sequences_lengths[index];
            return length;
        };

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
            MSAColumnProperties* const properties_ptr = msa_column_properties + subjectIndex;

            // We only want to consider the candidates with good alignments. the indices of those were determined in a previous step
            const int num_indices_for_this_subject = indices_per_subject[subjectIndex];

            if(num_indices_for_this_subject > 0){
                const int* const indices_for_this_subject = indices + indices_per_subject_prefixsum[subjectIndex];

                const int subjectLength = getSubjectLength(subjectIndex);
                int startindex = 0;
                int endindex = getSubjectLength(subjectIndex);

                for(int index = threadIdx.x; index < num_indices_for_this_subject; index += blockDim.x) {
                    const int queryIndex = indices_for_this_subject[index];

                    const int shift = alignment_shifts[queryIndex];
                    const BestAlignment_t flag = alignment_best_alignment_flags[queryIndex];
                    const int queryLength = getCandidateLength(subjectIndex, index);

                    assert(flag != BestAlignment_t::None);

                    const int queryEndsAt = queryLength + shift;
                    //printf("s %d QL %d: %d\n", subjectIndex, queryIndex, queryLength);
                    startindex = min(startindex, shift);
                    endindex = max(endindex, queryEndsAt);
                }

                startindex = BlockReduceInt(temp_storage.reduce).Reduce(startindex, cub::Min());
                __syncthreads();

                endindex = BlockReduceInt(temp_storage.reduce).Reduce(endindex, cub::Max());
                __syncthreads();

                if(threadIdx.x == 0) {
                    MSAColumnProperties my_columnproperties;

                    my_columnproperties.startindex = startindex;
                    my_columnproperties.endindex = endindex;
                    my_columnproperties.columnsToCheck = my_columnproperties.endindex - my_columnproperties.startindex;
                    my_columnproperties.subjectColumnsBegin_incl = max(-my_columnproperties.startindex, 0);
                    my_columnproperties.subjectColumnsEnd_excl = my_columnproperties.subjectColumnsBegin_incl + subjectLength;

                    *properties_ptr = my_columnproperties;
                }
            }else{
                //empty MSA
                if(threadIdx.x == 0) {
                    MSAColumnProperties my_columnproperties;

                    my_columnproperties.startindex = 0;
                    my_columnproperties.endindex = 0;
                    my_columnproperties.columnsToCheck = 0;
                    my_columnproperties.subjectColumnsBegin_incl = 0;
                    my_columnproperties.subjectColumnsEnd_excl = 0;

                    *properties_ptr = my_columnproperties;
                }
            }
        }
    }


    __global__
    void msa_add_sequences_kernel_exp(
                char* __restrict__ d_multiple_sequence_alignments,
                float* __restrict__ d_multiple_sequence_alignment_weights,
                const int* __restrict__ d_alignment_shifts,
                const BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
                const int* __restrict__ d_alignment_overlaps,
                const int* __restrict__ d_alignment_nOps,
                const char* __restrict__ d_subject_sequences_data,
                const char* __restrict__ d_candidate_sequences_data,
                const int* __restrict__ d_subject_sequences_lengths,
                const int* __restrict__ d_candidate_sequences_lengths,
                const char* __restrict__ d_subject_qualities,
                const char* __restrict__ d_candidate_qualities,
                const MSAColumnProperties*  __restrict__ d_msa_column_properties,
                const int* __restrict__ d_candidates_per_subject_prefixsum,
                const int* __restrict__ d_indices,
                const int* __restrict__ d_indices_per_subject,
                const int* __restrict__ d_indices_per_subject_prefixsum,
                int n_subjects,
                int n_queries,
                const int* __restrict__ d_num_indices,
                bool canUseQualityScores,
                float desiredAlignmentMaxErrorRate,
                int maximum_sequence_length,
                int max_sequence_bytes,
                size_t encoded_sequence_pitch,
                size_t quality_pitch,
                size_t msa_row_pitch,
                size_t msa_weights_row_pitch){

        auto get = [] (const char* data, int length, int index){
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
        };

        auto make_unpacked_reverse_complement_inplace = [] (std::uint8_t* sequence, int sequencelength){
            return reverseComplementStringInplace((char*)sequence, sequencelength);
        };

        auto getSubjectPtr = [&] (int subjectIndex){
            const char* result = d_subject_sequences_data + std::size_t(subjectIndex) * encoded_sequence_pitch;
            return result;
        };

        auto getCandidatePtr = [&] (int candidateIndex){
            const char* result = d_candidate_sequences_data + std::size_t(candidateIndex) * encoded_sequence_pitch;
            return result;
        };

        auto getSubjectQualityPtr = [&] (int subjectIndex){
            const char* result = d_subject_qualities + std::size_t(subjectIndex) * quality_pitch;
            return result;
        };

        auto getCandidateQualityPtr = [&] (int candidateIndex){
            const char* result = d_candidate_qualities + std::size_t(candidateIndex) * quality_pitch;
            return result;
        };

        auto getSubjectLength = [&] (int subjectIndex){
            const int length = d_subject_sequences_lengths[subjectIndex];
            return length;
        };

        auto getCandidateLength = [&] __device__ (int candidateIndex){
            const int length = d_candidate_sequences_lengths[candidateIndex];
            return length;
        };

        const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);
        const int n_indices = *d_num_indices;

        //copy each subject into the top row of its multiple sequence alignment
        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
            const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int candidates_before_this_subject = d_indices_per_subject_prefixsum[subjectIndex];

            const unsigned offset1 = msa_row_pitch * (subjectIndex + candidates_before_this_subject);
            const unsigned offset2 = msa_weights_row_pitch_floats * (subjectIndex + candidates_before_this_subject);

            char* const multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;
            float* const multiple_sequence_alignment_weight = d_multiple_sequence_alignment_weights + offset2;

            const int subjectLength = getSubjectLength(subjectIndex);
            const char* const subject = getSubjectPtr(subjectIndex);
            const char* const subjectQualityScore = getSubjectQualityPtr(subjectIndex);

            for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x) {
                multiple_sequence_alignment[subjectColumnsBegin_incl + i] = get(subject, subjectLength, i);
                multiple_sequence_alignment_weight[subjectColumnsBegin_incl + i] = canUseQualityScores ?
                                                                                   (float)d_qscore_to_weight[(unsigned char)subjectQualityScore[i]]
                                                                                   : 1.0f;
            }
        }

        // copy each query into the multiple sequence alignment of its subject
        for(unsigned index = blockIdx.x; index < n_indices; index += gridDim.x) {

            const int queryIndex = d_indices[index];

            const int shift = d_alignment_shifts[queryIndex];
            const BestAlignment_t flag = d_alignment_best_alignment_flags[queryIndex];

            //find subjectindex
            int subjectIndex = 0;
            for(; subjectIndex < n_subjects; subjectIndex++) {
                if(queryIndex < d_candidates_per_subject_prefixsum[subjectIndex+1])
                    break;
            }

            const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int localQueryIndex = index - d_indices_per_subject_prefixsum[subjectIndex];
            const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

            const int candidates_before_this_subject = d_indices_per_subject_prefixsum[subjectIndex];

            //printf("index %d, subjectindex %d, queryindex %d, shift %d, defaultcolumnoffset %d, candidates_before_this_subject %d, localQueryIndex %d\n", index, subjectIndex, queryIndex, shift, defaultcolumnoffset, candidates_before_this_subject, localQueryIndex);

            const unsigned offset1 = msa_row_pitch * (subjectIndex + candidates_before_this_subject);
            const unsigned offset2 = msa_weights_row_pitch_floats * (subjectIndex + candidates_before_this_subject);
            //const int rowOffset = (subjectIndex + candidates_before_this_subject);

            char* const multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;
            float* const multiple_sequence_alignment_weight = d_multiple_sequence_alignment_weights + offset2;

            const char* const query = getCandidatePtr(queryIndex);
            const int queryLength = getCandidateLength(queryIndex);
            const char* const queryQualityScore = getCandidateQualityPtr(index);

            const int query_alignment_overlap = d_alignment_overlaps[queryIndex];
            const int query_alignment_nops = d_alignment_nOps[queryIndex];

            const float defaultweight = 1.0f - sqrtf(query_alignment_nops
                        / (query_alignment_overlap * desiredAlignmentMaxErrorRate));

            assert(flag != BestAlignment_t::None);                 // indices should only be pointing to valid alignments

            //copy query into msa
            if(flag == BestAlignment_t::Forward) {
                const int row = 1 + localQueryIndex;
                for(int i = threadIdx.x; i < queryLength; i+= blockDim.x) {
                    const int globalIndex = defaultcolumnoffset + i;

                    multiple_sequence_alignment[row * msa_row_pitch + globalIndex] = get(query, queryLength, i);

                    multiple_sequence_alignment_weight[row * msa_weights_row_pitch_floats + globalIndex]
                            = canUseQualityScores ?
                              (float)d_qscore_to_weight[(unsigned char)queryQualityScore[i]] * defaultweight
                              : 1.0f;
                }
            }else{
                auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
                    constexpr std::uint8_t mask = 0x03;
                    return (~in & mask);
                };

                const int row = 1 + localQueryIndex;
                for(int i = threadIdx.x; i < queryLength; i+= blockDim.x){
                    const int reverseIndex = queryLength - 1 - i;
                    const int globalIndex = defaultcolumnoffset + i;
                    const char base = get(query, queryLength, reverseIndex);
                    const char revCompl = make_reverse_complement_byte(base);
                    const float weight = canUseQualityScores ? d_qscore_to_weight[(unsigned char)queryQualityScore[reverseIndex]] * defaultweight : 1.0f;
                    multiple_sequence_alignment[row * msa_row_pitch + globalIndex] = revCompl;
                    multiple_sequence_alignment_weight[row * msa_weights_row_pitch_floats + globalIndex] = weight;
                }
            }
        }
    }

    __global__
    void msa_add_sequences_kernel_implicit_global(
                int* d_counts,
                float* d_weights,
                int* __restrict__ d_coverage,
                float* __restrict__ d_origWeights,
                int* __restrict__ d_origCoverages,
    			const int* __restrict__ d_alignment_shifts,
    			const BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
                const int* __restrict__ d_alignment_overlaps,
    			const int* __restrict__ d_alignment_nOps,
                const char* __restrict__ d_subject_sequences_data,
                const char* __restrict__ d_candidate_sequences_data,
                const int* __restrict__ d_subject_sequences_lengths,
                const int* __restrict__ d_candidate_sequences_lengths,
                const char* __restrict__ d_subject_qualities,
				const char* __restrict__ d_candidate_qualities,
    			const MSAColumnProperties*  __restrict__ d_msa_column_properties,
    			const int* __restrict__ d_candidates_per_subject_prefixsum,
    			const int* __restrict__ d_indices,
    			const int* __restrict__ d_indices_per_subject,
    			const int* __restrict__ d_indices_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
    			const int* __restrict__ d_num_indices,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
    			int maximum_sequence_length,
    			int max_sequence_bytes,
                size_t encoded_sequence_pitch,
    			size_t quality_pitch,
    			size_t msa_row_pitch,
    			size_t msa_weights_row_pitch){

        auto get = [] (const char* data, int length, int index){
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
		};

		auto make_unpacked_reverse_complement_inplace = [] (std::uint8_t* sequence, int sequencelength){
			return reverseComplementStringInplace((char*)sequence, sequencelength);
		};

		auto getSubjectPtr = [&] (int subjectIndex){
			const char* result = d_subject_sequences_data + std::size_t(subjectIndex) * encoded_sequence_pitch;
			return result;
		};

		auto getCandidatePtr = [&] (int candidateIndex){
			const char* result = d_candidate_sequences_data + std::size_t(candidateIndex) * encoded_sequence_pitch;
			return result;
		};

		auto getSubjectQualityPtr = [&] (int subjectIndex){
			const char* result = d_subject_qualities + std::size_t(subjectIndex) * quality_pitch;
			return result;
		};

		auto getCandidateQualityPtr = [&] (int candidateIndex){
			const char* result = d_candidate_qualities + std::size_t(candidateIndex) * quality_pitch;
			return result;
		};

		auto getSubjectLength = [&] (int subjectIndex){
			const int length = d_subject_sequences_lengths[subjectIndex];
			return length;
		};

		auto getCandidateLength = [&] __device__ (int candidateIndex){
			const int length = d_candidate_sequences_lengths[candidateIndex];
			return length;
		};

    	const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);
    	const int n_indices = *d_num_indices;

        //add subjects
    	for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
    		const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = d_msa_column_properties[subjectIndex].subjectColumnsEnd_excl;
    		const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
    		const char* const subject = getSubjectPtr(subjectIndex);
    		const char* const subjectQualityScore = getSubjectQualityPtr(subjectIndex);
            const int shift = 0;

            int* const my_coverage = d_coverage + subjectIndex * msa_weights_row_pitch_floats;

            //printf("subject: ");
            for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x){
                const int globalIndex = subjectColumnsBegin_incl + shift + i;
                const char base = get(subject, subjectLength, i);
                //printf("%d ", int(base));
                const float weight = canUseQualityScores ? d_qscore_to_weight[(unsigned char)subjectQualityScore[i]] : 1.0f;
                const int ptrOffset = subjectIndex * 4 * msa_weights_row_pitch_floats + int(base) * msa_weights_row_pitch_floats;
                atomicAdd(d_counts + ptrOffset + globalIndex, 1);
                atomicAdd(d_weights + ptrOffset + globalIndex, weight);
                atomicAdd(my_coverage + globalIndex, 1);
            }
    	}
        //printf("\n");

        //add candidates
        for(unsigned index = blockIdx.x; index < n_indices; index += gridDim.x) {
            const int queryIndex = d_indices[index];

            const int shift = d_alignment_shifts[queryIndex];
            const BestAlignment_t flag = d_alignment_best_alignment_flags[queryIndex];

            //find subjectindex
            int subjectIndex = 0;
            for(; subjectIndex < n_subjects; subjectIndex++) {
                if(queryIndex < d_candidates_per_subject_prefixsum[subjectIndex+1])
                    break;
            }

            const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

            int* const my_coverage = d_coverage + subjectIndex * msa_weights_row_pitch_floats;

            const char* const query = getCandidatePtr(queryIndex);
    		const int queryLength = getCandidateLength(queryIndex);
    		const char* const queryQualityScore = getCandidateQualityPtr(index);

    		const int query_alignment_overlap = d_alignment_overlaps[queryIndex];
    		const int query_alignment_nops = d_alignment_nOps[queryIndex];

    		const float defaultweight = 1.0f - sqrtf(query_alignment_nops
    					/ (query_alignment_overlap * desiredAlignmentMaxErrorRate));

            assert(flag != BestAlignment_t::None);                 // indices should only be pointing to valid alignments
            //printf("candidate %d, shift %d default %d: ", index, shift, defaultcolumnoffset);
    		//copy query into msa
    		if(flag == BestAlignment_t::Forward) {
                for(int i = threadIdx.x; i < queryLength; i+= blockDim.x){
                    const int globalIndex = defaultcolumnoffset + i;
                    const char base = get(query, queryLength, i);
                    //printf("%d ", int(base));
                    const float weight = canUseQualityScores ? d_qscore_to_weight[(unsigned char)queryQualityScore[i]] * defaultweight : 1.0f;
                    const int ptrOffset = subjectIndex * 4 * msa_weights_row_pitch_floats + int(base) * msa_weights_row_pitch_floats;
    				atomicAdd(d_counts + ptrOffset + globalIndex, 1);
    				atomicAdd(d_weights + ptrOffset + globalIndex, weight);
                    atomicAdd(my_coverage + globalIndex, 1);
                }
    		}else{
                auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
                    constexpr std::uint8_t mask = 0x03;
                    return (~in & mask);
                };

                for(int i = threadIdx.x; i < queryLength; i+= blockDim.x){
                    const int reverseIndex = queryLength - 1 - i;
                    const int globalIndex = defaultcolumnoffset + i;
                    const char base = get(query, queryLength, reverseIndex);
                    const char revCompl = make_reverse_complement_byte(base);
                    //printf("%d ", int(revCompl));
                    const float weight = canUseQualityScores ? d_qscore_to_weight[(unsigned char)queryQualityScore[reverseIndex]] * defaultweight : 1.0f;
                    const int ptrOffset = subjectIndex * 4 * msa_weights_row_pitch_floats + int(revCompl) * msa_weights_row_pitch_floats;
    				atomicAdd(d_counts + ptrOffset + globalIndex, 1);
    				atomicAdd(d_weights + ptrOffset + globalIndex, weight);
                    atomicAdd(my_coverage + globalIndex, 1);
                }
            }
            //printf("\n");

    	}
    }




    __global__
    void msa_add_sequences_kernel_implicit_shared(
                int* d_counts,
                float* d_weights,
                int* __restrict__ d_coverage,
                float* __restrict__ d_origWeights,
                int* __restrict__ d_origCoverages,
    			const int* __restrict__ d_alignment_shifts,
    			const BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
                const int* __restrict__ d_alignment_overlaps,
    			const int* __restrict__ d_alignment_nOps,
                const char* __restrict__ d_subject_sequences_data,
                const char* __restrict__ d_candidate_sequences_data,
                const int* __restrict__ d_subject_sequences_lengths,
                const int* __restrict__ d_candidate_sequences_lengths,
                const char* __restrict__ d_subject_qualities,
				const char* __restrict__ d_candidate_qualities,
    			const MSAColumnProperties*  __restrict__ d_msa_column_properties,
    			const int* __restrict__ d_candidates_per_subject_prefixsum,
    			const int* __restrict__ d_indices,
    			const int* __restrict__ d_indices_per_subject,
    			const int* __restrict__ d_indices_per_subject_prefixsum,
                const int* __restrict__ blocks_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
    			const int* __restrict__ d_num_indices,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
    			int maximum_sequence_length,
    			int max_sequence_bytes,
                size_t encoded_sequence_pitch,
    			size_t quality_pitch,
    			size_t msa_row_pitch,
    			size_t msa_weights_row_pitch){

        // sizeof(float) * 4 * msa_weights_row_pitch_floats // weights
        //+ sizeof(int) * 4 * msa_weights_row_pitch_floats // counts
    	extern __shared__ float sharedmem[];

        auto get = [] (const char* data, int length, int index){
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
		};

		auto make_unpacked_reverse_complement_inplace = [] (std::uint8_t* sequence, int sequencelength){
			return reverseComplementStringInplace((char*)sequence, sequencelength);
		};

		auto getSubjectPtr = [&] (int subjectIndex){
			const char* result = d_subject_sequences_data + std::size_t(subjectIndex) * encoded_sequence_pitch;
			return result;
		};

		auto getCandidatePtr = [&] (int candidateIndex){
			const char* result = d_candidate_sequences_data + std::size_t(candidateIndex) * encoded_sequence_pitch;
			return result;
		};

		auto getSubjectQualityPtr = [&] (int subjectIndex){
			const char* result = d_subject_qualities + std::size_t(subjectIndex) * quality_pitch;
			return result;
		};

		auto getCandidateQualityPtr = [&] (int localCandidateIndex){
			const char* result = d_candidate_qualities + std::size_t(localCandidateIndex) * quality_pitch;
			return result;
		};

		auto getSubjectLength = [&] (int subjectIndex){
			const int length = d_subject_sequences_lengths[subjectIndex];
			return length;
		};

		auto getCandidateLength = [&] __device__ (int candidateIndex){
			//const int candidateIndex = d_indices[localCandidateIndex];
			const int length = d_candidate_sequences_lengths[candidateIndex];
			return length;
		};

        const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);
        const int smemsizefloats = 4 * msa_weights_row_pitch_floats + 4 * msa_weights_row_pitch_floats;

        float* const shared_weights = sharedmem;
        int* const shared_counts = (int*)(shared_weights + 4 * msa_weights_row_pitch_floats);

    	//const int requiredTiles = n_subjects;//blocks_per_subject_prefixsum[n_subjects];
        const int requiredTiles = blocks_per_subject_prefixsum[n_subjects];

        //const int num_indices = *d_num_indices;

    	for(int logicalBlockId = blockIdx.x; logicalBlockId < requiredTiles; logicalBlockId += gridDim.x){
            //clear shared memory
            for(int i = threadIdx.x; i < smemsizefloats; i += blockDim.x){
                sharedmem[i] = 0;
            }
            __syncthreads();

    		int subjectIndex = 0;
    		for(; subjectIndex < n_subjects; subjectIndex++) {
    			if(logicalBlockId < blocks_per_subject_prefixsum[subjectIndex+1])
                //if(logicalBlockId < subjectIndex+1)
    				break;
    		}

            const int blockForThisSubject = logicalBlockId - blocks_per_subject_prefixsum[subjectIndex];

            const int* const indices_for_this_subject = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
            //const int indicesBeforeThisSubject = d_indices_per_subject_prefixsum[subjectIndex];
            const int id = blockForThisSubject * blockDim.x + threadIdx.x;
            const int maxid_excl = d_indices_per_subject[subjectIndex];
            const int globalIndexlistIndex = d_indices_per_subject_prefixsum[subjectIndex] + id;


    		const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = d_msa_column_properties[subjectIndex].subjectColumnsEnd_excl;
            const int columnsToCheck = d_msa_column_properties[subjectIndex].columnsToCheck;

            int* const my_coverage = d_coverage + subjectIndex * msa_weights_row_pitch_floats;

            assert(columnsToCheck <= msa_weights_row_pitch_floats);

            //ensure that the subject is only inserted once, by the first block
            if(blockForThisSubject == 0){
        		const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
        		const char* const subject = getSubjectPtr(subjectIndex);
        		const char* const subjectQualityScore = getSubjectQualityPtr(subjectIndex);

                //printf("subject: ");
                for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x){
                    const int shift = 0;
                    const int globalIndex = subjectColumnsBegin_incl + shift + i;
                    const char base = get(subject, subjectLength, i);
                    //printf("%d ", int(base));
                    //if(subjectQualityScore[i] == '\0'){
                        assert(subjectQualityScore[i] != '\0');
                    //}

                    const float weight = canUseQualityScores ? d_qscore_to_weight[(unsigned char)subjectQualityScore[i]] : 1.0f;
                    const int ptrOffset = int(base) * msa_weights_row_pitch_floats;
                    atomicAdd(shared_counts + ptrOffset + globalIndex, 1);
                    atomicAdd(shared_weights + ptrOffset + globalIndex, weight);
                    atomicAdd(my_coverage + globalIndex, 1);
                }
            }
            //printf("\n");



            if(id < maxid_excl){
                const int queryIndex = indices_for_this_subject[id];
                const int shift = d_alignment_shifts[queryIndex];
                const BestAlignment_t flag = d_alignment_best_alignment_flags[queryIndex];
                const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

                const char* const query = getCandidatePtr(queryIndex);
        		const int queryLength = getCandidateLength(queryIndex);
        		const char* const queryQualityScore = getCandidateQualityPtr(globalIndexlistIndex);

        		const int query_alignment_overlap = d_alignment_overlaps[queryIndex];
        		const int query_alignment_nops = d_alignment_nOps[queryIndex];

        		const float defaultweight = 1.0f - sqrtf(query_alignment_nops
        					/ (query_alignment_overlap * desiredAlignmentMaxErrorRate));

                assert(flag != BestAlignment_t::None);                 // indices should only be pointing to valid alignments

                //printf("candidate %d, shift %d default %d: ", index, shift, defaultcolumnoffset);

                if(flag == BestAlignment_t::Forward) {
                    for(int i = 0; i < queryLength; i += 1){
                        const int globalIndex = defaultcolumnoffset + i;
                        const char base = get(query, queryLength, i);
                        //printf("%d ", int(base));
                        //if(queryQualityScore[i] == '\0'){
                            assert(queryQualityScore[i] != '\0');
                        //}

                        const float weight = canUseQualityScores ? d_qscore_to_weight[(unsigned char)queryQualityScore[i]] * defaultweight : 1.0f;
                        assert(weight != 0);
                        const int ptrOffset = int(base) * msa_weights_row_pitch_floats;
                        atomicAdd(shared_counts + ptrOffset + globalIndex, 1);
                        atomicAdd(shared_weights + ptrOffset + globalIndex, weight);
                        atomicAdd(my_coverage + globalIndex, 1);
                    }
        		}else{
                    auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
                        constexpr std::uint8_t mask = 0x03;
                        return (~in & mask);
                    };

                    for(int i = 0; i < queryLength; i += 1){
                        const int reverseIndex = queryLength - 1 - i;
                        const int globalIndex = defaultcolumnoffset + i;
                        const char base = get(query, queryLength, reverseIndex);
                        const char revCompl = make_reverse_complement_byte(base);
                        //printf("%d ", int(revCompl));

                        //if(queryQualityScore[reverseIndex] == '\0'){
                            assert(queryQualityScore[reverseIndex] != '\0');
                        //}

                        const float weight = canUseQualityScores ? d_qscore_to_weight[(unsigned char)queryQualityScore[reverseIndex]] * defaultweight : 1.0f;
                        assert(weight != 0);
                        const int ptrOffset = int(revCompl) * msa_weights_row_pitch_floats;
                        atomicAdd(shared_counts + ptrOffset + globalIndex, 1);
                        atomicAdd(shared_weights + ptrOffset + globalIndex, weight);
                        atomicAdd(my_coverage + globalIndex, 1);
                    }
                }
                //printf("\n");
            }

            __syncthreads();

            for(int index = threadIdx.x; index < columnsToCheck; index += blockDim.x){
                for(int k = 0; k < 4; k++){
                    const int* const srcCounts = shared_counts + k * msa_weights_row_pitch_floats + index;
                    int* const destCounts = d_counts + 4 * msa_weights_row_pitch_floats * subjectIndex + k * msa_weights_row_pitch_floats + index;
                    const float* const srcWeights = shared_weights + k * msa_weights_row_pitch_floats + index;
                    float* const destWeights = d_weights + 4 * msa_weights_row_pitch_floats * subjectIndex + k * msa_weights_row_pitch_floats + index;
                    atomicAdd(destCounts ,*srcCounts);
                    atomicAdd(destWeights, *srcWeights);
                }
            }

            __syncthreads();
        }

    }


    __global__
    void msa_find_consensus_kernel(
                            char* __restrict__ d_consensus,
                            float* __restrict__ d_support,
                            int* __restrict__ d_coverage,
                            float* __restrict__ d_origWeights,
                            int* __restrict__ d_origCoverages,
                            int* __restrict__ d_counts,
                            float* __restrict__ d_weights,
                            const char* __restrict__ d_multiple_sequence_alignments,
                            const float* __restrict__ d_multiple_sequence_alignment_weights,
                            const MSAColumnProperties* __restrict__ d_msa_column_properties,
                            const int* __restrict__ d_candidates_per_subject_prefixsum,
                            const int* __restrict__ d_indices_per_subject,
                            const int* __restrict__ d_indices_per_subject_prefixsum,
                            int n_subjects,
                            int n_queries,
                            const int* __restrict__ d_num_indices,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int msa_max_column_count,
                            int blocks_per_msa){

        constexpr char A_enc = 0x00;
        constexpr char C_enc = 0x01;
        constexpr char G_enc = 0x02;
        constexpr char T_enc = 0x03;

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        const int localBlockId = blockIdx.x % blocks_per_msa;
        //const int n_indices = *d_num_indices;

        //process multiple sequence alignment of each subject
        //for each column in msa, find consensus and support
        for(unsigned subjectIndex = blockIdx.x / blocks_per_msa; subjectIndex < n_subjects; subjectIndex += gridDim.x / blocks_per_msa){
            const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = d_msa_column_properties[subjectIndex].subjectColumnsEnd_excl;
            const int columnsToCheck = d_msa_column_properties[subjectIndex].columnsToCheck;

            //number of rows in multiple sequence alignment for subject[subjectIndex]
            const int msa_rows = 1 + d_indices_per_subject[subjectIndex];

            const unsigned offset1 = msa_pitch * (subjectIndex + d_indices_per_subject_prefixsum[subjectIndex]);
            const unsigned offset2 = msa_weights_pitch_floats * (subjectIndex + d_indices_per_subject_prefixsum[subjectIndex]);

            const char* const my_multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;
            const float* const my_multiple_sequence_alignment_weight = d_multiple_sequence_alignment_weights + offset2;

            char* const my_consensus = d_consensus + subjectIndex * msa_pitch;
            float* const my_support = d_support + subjectIndex * msa_weights_pitch_floats;
            int* const my_coverage = d_coverage + subjectIndex * msa_weights_pitch_floats;

            float* const my_orig_weights = d_origWeights + subjectIndex * msa_weights_pitch_floats;
            int* const my_orig_coverage = d_origCoverages + subjectIndex * msa_weights_pitch_floats;

            const int countsOffset = subjectIndex * 4 * msa_weights_pitch_floats;
            const int weightsOffset = subjectIndex * 4 * msa_weights_pitch_floats;

            int* const my_countsA = d_counts + countsOffset + 0 * msa_weights_pitch_floats;
            int* const my_countsC = d_counts + countsOffset + 1 * msa_weights_pitch_floats;
            int* const my_countsG = d_counts + countsOffset + 2 * msa_weights_pitch_floats;
            int* const my_countsT = d_counts + countsOffset + 3 * msa_weights_pitch_floats;
            float* const my_weightsA = d_weights + weightsOffset + 0 * msa_weights_pitch_floats;
            float* const my_weightsC = d_weights + weightsOffset + 1 * msa_weights_pitch_floats;
            float* const my_weightsG = d_weights + weightsOffset + 2 * msa_weights_pitch_floats;
            float* const my_weightsT = d_weights + weightsOffset + 3 * msa_weights_pitch_floats;

            for(int column = localBlockId * blockDim.x + threadIdx.x; column < columnsToCheck; column += blocks_per_msa * blockDim.x){

                float Aw = 0.0f;
                float Cw = 0.0f;
                float Gw = 0.0f;
                float Tw = 0.0f;

                int As = 0;
                int Cs = 0;
                int Gs = 0;
                int Ts = 0;

                int columnCoverage = 0;

                for(int row = 0; row < msa_rows; ++row){
                    const char base = my_multiple_sequence_alignment[row * msa_pitch + column];
                    const float weight = my_multiple_sequence_alignment_weight[row * msa_weights_pitch_floats + column];

                    //if(!(base == 'A' || base == 'C' || base == 'G' || base == 'T')){
                    //	assert(base == 'A' || base == 'C' || base == 'G' || base == 'T');
                    //}
    #if 0
                    Aw += (base == 'A' ? weight : 0);
                    Cw += (base == 'C' ? weight : 0);
                    Gw += (base == 'G' ? weight : 0);
                    Tw += (base == 'T' ? weight : 0);

                    As += (base == 'A');
                    Cs += (base == 'C');
                    Gs += (base == 'G');
                    Ts += (base == 'T');
    #else
                    Aw += (base == A_enc ? weight : 0);
                    Cw += (base == C_enc ? weight : 0);
                    Gw += (base == G_enc ? weight : 0);
                    Tw += (base == T_enc ? weight : 0);

                    As += (base == A_enc);
                    Cs += (base == C_enc);
                    Gs += (base == G_enc);
                    Ts += (base == T_enc);
    #endif
                    //columnCoverage += (base == 'A' || base == 'C' || base == 'G' || base == 'T');
                    columnCoverage += (base == A_enc || base == C_enc || base == G_enc || base == T_enc);//!(base & 0xFC);
                }

                my_countsA[column] = As;
                my_countsC[column] = Cs;
                my_countsG[column] = Gs;
                my_countsT[column] = Ts;
                my_weightsA[column] = Aw;
                my_weightsC[column] = Cw;
                my_weightsG[column] = Gw;
                my_weightsT[column] = Tw;


                if(columnCoverage <= 0){
                    //assert(columnCoverage > 0);
                }

                const float columnWeight = Aw + Cw + Gw + Tw;
                float consWeight = Aw;
    #if 0
                char cons = 'A';

                cons = Cw > consWeight ? 'C' : cons;
                consWeight = Cw > consWeight ? Cw : consWeight;

                cons = Gw > consWeight ? 'G' : cons;
                consWeight = Gw > consWeight ? Gw : consWeight;

                cons = Tw > consWeight ? 'T' : cons;
                consWeight = Tw > consWeight ? Tw : consWeight;
    #else
                char cons = 'A';

                cons = Cw > consWeight ? 'C' : cons;
                consWeight = Cw > consWeight ? Cw : consWeight;

                cons = Gw > consWeight ? 'G' : cons;
                consWeight = Gw > consWeight ? Gw : consWeight;

                cons = Tw > consWeight ? 'T' : cons;
                consWeight = Tw > consWeight ? Tw : consWeight;
    #endif
                /*char consByCount = 'A';
                int consByCountCount = As;
                consByCount = Cs > consByCountCount ? 'C' : cons;
                consByCountCount = Cs > consByCountCount ? Cs : consWeight;

                consByCount = Gs > consByCountCount ? 'G' : cons;
                consByCountCount = Gs > consByCountCount ? Gs : consWeight;

                consByCount = Ts > consByCountCount ? 'T' : cons;
                consByCountCount = Ts > consByCountCount ? Ts : consWeight;

                if(consByCount != cons){
                    printf("cons differ\n");
                }*/

                my_consensus[column] = cons;
                my_support[column] = consWeight / columnWeight;
                //printf("subject %d, column %d, support %f\n", subjectIndex, column, my_support[column]);
                my_coverage[column] = columnCoverage;

                if(subjectColumnsBegin_incl <= column && column < subjectColumnsEnd_excl){
                    const char subjectbase = my_multiple_sequence_alignment[column];
                    if(subjectbase == A_enc){
                        my_orig_weights[column] = Aw;
                        my_orig_coverage[column] = As;
                    }else if(subjectbase == C_enc){
                        my_orig_weights[column] = Cw;
                        my_orig_coverage[column] = Cs;
                    }else if(subjectbase == G_enc){
                        my_orig_weights[column] = Gw;
                        my_orig_coverage[column] = Gs;
                    }else if(subjectbase == T_enc){
                        my_orig_weights[column] = Tw;
                        my_orig_coverage[column] = Ts;
                    }
                }
            }
        }
    }


    __global__
    void msa_find_consensus_implicit_kernel(
                            int* __restrict__ d_counts,
                            float* __restrict__ d_weights,
                            char* __restrict__ d_consensus,
                            float* __restrict__ d_support,
                            int* __restrict__ d_coverage,
                            float* __restrict__ d_origWeights,
                            int* __restrict__ d_origCoverages,
                            const char* __restrict__ d_subject_sequences_data,
                            const MSAColumnProperties* __restrict__ d_msa_column_properties,
                            int n_subjects,
                            size_t encoded_sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int blocks_per_msa){

        constexpr char A_enc = 0x00;
        constexpr char C_enc = 0x01;
        constexpr char G_enc = 0x02;
        constexpr char T_enc = 0x03;

        auto get = [] (const char* data, int length, int index){
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
        };

        auto getSubjectPtr = [&] (int subjectIndex){
            const char* result = d_subject_sequences_data + std::size_t(subjectIndex) * encoded_sequence_pitch;
            return result;
        };

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        const int localBlockId = blockIdx.x % blocks_per_msa;
        //const int n_indices = *d_num_indices;

        //process multiple sequence alignment of each subject
        //for each column in msa, find consensus and support
        for(unsigned subjectIndex = blockIdx.x / blocks_per_msa; subjectIndex < n_subjects; subjectIndex += gridDim.x / blocks_per_msa){
            const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = d_msa_column_properties[subjectIndex].subjectColumnsEnd_excl;
            const int columnsToCheck = d_msa_column_properties[subjectIndex].columnsToCheck;

            assert(columnsToCheck <= msa_weights_pitch_floats);

            const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
            const char* const subject = getSubjectPtr(subjectIndex);

            char* const my_consensus = d_consensus + subjectIndex * msa_pitch;
            float* const my_support = d_support + subjectIndex * msa_weights_pitch_floats;

            float* const my_orig_weights = d_origWeights + subjectIndex * msa_weights_pitch_floats;
            int* const my_orig_coverage = d_origCoverages + subjectIndex * msa_weights_pitch_floats;

            const float* const my_weightsA = d_weights + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
            const float* const my_weightsC = d_weights + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
            const float* const my_weightsG = d_weights + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
            const float* const my_weightsT = d_weights + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

            for(int column = localBlockId * blockDim.x + threadIdx.x; column < columnsToCheck; column += blocks_per_msa * blockDim.x){
                const float wA = my_weightsA[column];
                const float wC = my_weightsC[column];
                const float wG = my_weightsG[column];
                const float wT = my_weightsT[column];
                char cons = 'A';
                float consWeight = wA;
                if(wC > consWeight){
                    cons = 'C';
                    consWeight = wC;
                }
                if(wG > consWeight){
                    cons = 'G';
                    consWeight = wG;
                }
                if(wT > consWeight){
                    cons = 'T';
                    consWeight = wT;
                }
                my_consensus[column] = cons;
                const float columnWeight = wA + wC + wG + wT;
                if(columnWeight == 0){
                    printf("s %d c %d\n", subjectIndex, column);
                    assert(columnWeight != 0);
                }

                my_support[column] = consWeight / columnWeight;


                if(subjectColumnsBegin_incl <= column && column < subjectColumnsEnd_excl){
                    const int* const my_countsA = d_counts + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                    const int* const my_countsC = d_counts + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                    const int* const my_countsG = d_counts + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                    const int* const my_countsT = d_counts + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                    const int localIndex = column - subjectColumnsBegin_incl;
                    const char subjectbase = get(subject, subjectLength, localIndex);

                    if(subjectbase == A_enc){
                        my_orig_weights[column] = wA;
                        my_orig_coverage[column] = my_countsA[column];
                    }else if(subjectbase == C_enc){
                        my_orig_weights[column] = wC;
                        my_orig_coverage[column] = my_countsC[column];;
                    }else if(subjectbase == G_enc){
                        my_orig_weights[column] = wG;
                        my_orig_coverage[column] = my_countsG[column];
                    }else if(subjectbase == T_enc){
                        my_orig_weights[column] = wT;
                        my_orig_coverage[column] = my_countsT[column];
                    }
                }
            }
        }
    }

    template<int BLOCKSIZE>
    __global__
    void msa_correct_subject_kernel(
                            const char* __restrict__ d_consensus,
                            const float* __restrict__ d_support,
                            const int* __restrict__ d_coverage,
                            const int* __restrict__ d_origCoverages,
                            const char* __restrict__ d_multiple_sequence_alignments,
                            const MSAColumnProperties* __restrict__ d_msa_column_properties,
                            const int* __restrict__ d_indices_per_subject_prefixsum,
                            bool* __restrict__ d_is_high_quality_subject,
                            char* __restrict__ d_corrected_subjects,
                            bool* __restrict__ d_subject_is_corrected,
                            int n_subjects,
                            int n_queries,
                            const int* __restrict__ d_num_indices,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            float estimatedErrorrate,
                            float avg_support_threshold,
                            float min_support_threshold,
                            float min_coverage_threshold,
                            int k_region){

        using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;
        using BlockReduceFloat = cub::BlockReduce<float, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceBool::TempStorage boolreduce;
            typename BlockReduceInt::TempStorage intreduce;
            typename BlockReduceFloat::TempStorage floatreduce;
        } temp_storage;

        __shared__ bool broadcastbuffer;

        auto isGoodAvgSupport = [&](float avgsupport){
            return avgsupport >= avg_support_threshold;
        };
        auto isGoodMinSupport = [&](float minsupport){
            return minsupport >= min_support_threshold;
        };
        auto isGoodMinCoverage = [&](float mincoverage){
            return mincoverage >= min_coverage_threshold;
        };

        constexpr char A_enc = 0x00;
        constexpr char C_enc = 0x01;
        constexpr char G_enc = 0x02;
        constexpr char T_enc = 0x03;

        auto to_nuc = [](char c){
            switch(c){
            case A_enc: return 'A';
            case C_enc: return 'C';
            case G_enc: return 'G';
            case T_enc: return 'T';
            default: return 'F';
            }
        };

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
        //const int n_indices = *d_num_indices;

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            const float* const my_support = d_support + msa_weights_pitch_floats * subjectIndex;
            const int* const my_coverage = d_coverage + msa_weights_pitch_floats * subjectIndex;
            const int* const my_orig_coverage = d_origCoverages + msa_weights_pitch_floats * subjectIndex;
            const char* const my_consensus = d_consensus + msa_pitch  * subjectIndex;
            char* const my_corrected_subject = d_corrected_subjects + subjectIndex * sequence_pitch;

            const MSAColumnProperties properties = d_msa_column_properties[subjectIndex];
            const int subjectColumnsBegin_incl = properties.subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = properties.subjectColumnsEnd_excl;

            float avg_support = 0;
            float min_support = 1.0f;
            //int max_coverage = 0;
            int min_coverage = std::numeric_limits<int>::max();

            for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                assert(i < properties.columnsToCheck);

                avg_support += my_support[i];
                min_support = min(my_support[i], min_support);
                //max_coverage = max(my_coverage[i], max_coverage);
                min_coverage = min(my_coverage[i], min_coverage);
            }

            avg_support = BlockReduceFloat(temp_storage.floatreduce).Sum(avg_support);
            __syncthreads();

            min_support = BlockReduceFloat(temp_storage.floatreduce).Reduce(min_support, cub::Min());
            __syncthreads();

            //max_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(max_coverage, cub::Max());

            min_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(min_coverage, cub::Min());
            __syncthreads();

            avg_support /= (subjectColumnsEnd_excl - subjectColumnsBegin_incl);

            bool isHQ = isGoodAvgSupport(avg_support) && isGoodMinSupport(min_support) && isGoodMinCoverage(min_coverage);

            if(threadIdx.x == 0){
                broadcastbuffer = isHQ;
                d_is_high_quality_subject[subjectIndex] = isHQ;
            }
            __syncthreads();

            isHQ = broadcastbuffer;

            if(isHQ){
                for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                    my_corrected_subject[i - subjectColumnsBegin_incl] = my_consensus[i];
                }
                if(threadIdx.x == 0){
                    d_subject_is_corrected[subjectIndex] = true;
                }
            }else{
                const unsigned offset1 = msa_pitch * (subjectIndex + d_indices_per_subject_prefixsum[subjectIndex]);
                const char* const my_multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;

                //copy orignal sequence, which is in first row of msa, to corrected sequences
                for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                    my_corrected_subject[i - subjectColumnsBegin_incl] = to_nuc(my_multiple_sequence_alignment[i]);
                }

                const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;

                bool foundAColumn = false;
                for(int i = threadIdx.x; i < subjectLength; i += BLOCKSIZE){
                    const int globalIndex = subjectColumnsBegin_incl + i;
#if 1
                    if(my_corrected_subject[i] != my_consensus[globalIndex]
                                && my_support[globalIndex] > 0.5f
                                && my_orig_coverage[globalIndex] <= min_coverage_threshold){
                        float avgsupportkregion = 0;
                        int c = 0;
                        bool kregioncoverageisgood = true;

                        for(int j = i - k_region/2; j <= i + k_region/2 && kregioncoverageisgood; j++){
                            if(j != i && j >= 0 && j < subjectLength){
                                avgsupportkregion += my_support[subjectColumnsBegin_incl + j];
                                kregioncoverageisgood &= (my_coverage[subjectColumnsBegin_incl + j] >= min_coverage_threshold);
                                //kregioncoverageisgood &= (my_coverage[subjectColumnsBegin_incl + j] >= 1);
                                c++;
                            }
                        }

                        avgsupportkregion /= c;

                        //if(i == 33 || i == 34){
                        //	printf("%d %f\n", i, avgsupportkregion);
                        //}
                        if(kregioncoverageisgood && avgsupportkregion >= 1.0f-estimatedErrorrate){
                            my_corrected_subject[i] = my_consensus[globalIndex];
                            foundAColumn = true;
                        }
                    }
#else
                    constexpr float support_theshold = 0.5;

                    if(my_support[globalIndex] >= support_theshold){
                        my_corrected_subject[i] = my_consensus[globalIndex];
                        foundAColumn = true;
                    }

#endif
                }

                //perform block wide or-reduction on foundAColumn
                foundAColumn = BlockReduceBool(temp_storage.boolreduce).Reduce(foundAColumn, [](bool a, bool b){return a || b;});
                __syncthreads();

                if(threadIdx.x == 0){
                    d_subject_is_corrected[subjectIndex] = foundAColumn;
                }
            }
        }
    }


    template<int BLOCKSIZE>
    __global__
    void msa_correct_subject_implicit_kernel(
                            const char* __restrict__ d_consensus,
                            const float* __restrict__ d_support,
                            const int* __restrict__ d_coverage,
                            const int* __restrict__ d_origCoverages,
                            const MSAColumnProperties* __restrict__ d_msa_column_properties,
                            const char* __restrict__ d_subject_sequences_data,
                            bool* __restrict__ d_is_high_quality_subject,
                            char* __restrict__ d_corrected_subjects,
                            bool* __restrict__ d_subject_is_corrected,
                            int n_subjects,
                            size_t encoded_sequence_pitch,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            float estimatedErrorrate,
                            float avg_support_threshold,
                            float min_support_threshold,
                            float min_coverage_threshold,
                            int k_region){

        using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;
        using BlockReduceFloat = cub::BlockReduce<float, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceBool::TempStorage boolreduce;
            typename BlockReduceInt::TempStorage intreduce;
            typename BlockReduceFloat::TempStorage floatreduce;
        } temp_storage;

        __shared__ bool broadcastbuffer;

        auto get = [] (const char* data, int length, int index){
			//return Sequence_t::get_as_nucleotide(data, length, index);
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
		};

		auto getSubjectPtr = [&] (int subjectIndex){
			const char* result = d_subject_sequences_data + std::size_t(subjectIndex) * encoded_sequence_pitch;
			return result;
		};

        auto isGoodAvgSupport = [&](float avgsupport){
            return avgsupport >= avg_support_threshold;
        };
        auto isGoodMinSupport = [&](float minsupport){
            return minsupport >= min_support_threshold;
        };
        auto isGoodMinCoverage = [&](float mincoverage){
            return mincoverage >= min_coverage_threshold;
        };

        constexpr char A_enc = 0x00;
        constexpr char C_enc = 0x01;
        constexpr char G_enc = 0x02;
        constexpr char T_enc = 0x03;

        auto to_nuc = [](char c){
            switch(c){
            case A_enc: return 'A';
            case C_enc: return 'C';
            case G_enc: return 'G';
            case T_enc: return 'T';
            default: return 'F';
            }
        };

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            const float* const my_support = d_support + msa_weights_pitch_floats * subjectIndex;
            const int* const my_coverage = d_coverage + msa_weights_pitch_floats * subjectIndex;
            const int* const my_orig_coverage = d_origCoverages + msa_weights_pitch_floats * subjectIndex;
            const char* const my_consensus = d_consensus + msa_pitch  * subjectIndex;
            char* const my_corrected_subject = d_corrected_subjects + subjectIndex * sequence_pitch;

            const MSAColumnProperties properties = d_msa_column_properties[subjectIndex];
            const int subjectColumnsBegin_incl = properties.subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = properties.subjectColumnsEnd_excl;

            float avg_support = 0;
            float min_support = 1.0f;
            //int max_coverage = 0;
            int min_coverage = std::numeric_limits<int>::max();

            for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                assert(i < properties.columnsToCheck);

                avg_support += my_support[i];
                min_support = min(my_support[i], min_support);
                //max_coverage = max(my_coverage[i], max_coverage);
                min_coverage = min(my_coverage[i], min_coverage);
            }

            avg_support = BlockReduceFloat(temp_storage.floatreduce).Sum(avg_support);
            __syncthreads();

            min_support = BlockReduceFloat(temp_storage.floatreduce).Reduce(min_support, cub::Min());
            __syncthreads();

            //max_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(max_coverage, cub::Max());

            min_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(min_coverage, cub::Min());
            __syncthreads();

            avg_support /= (subjectColumnsEnd_excl - subjectColumnsBegin_incl);

            bool isHQ = isGoodAvgSupport(avg_support) && isGoodMinSupport(min_support) && isGoodMinCoverage(min_coverage);

            if(threadIdx.x == 0){
                broadcastbuffer = isHQ;
                d_is_high_quality_subject[subjectIndex] = isHQ;
            }
            __syncthreads();

            isHQ = broadcastbuffer;

            if(isHQ){
                for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                    my_corrected_subject[i - subjectColumnsBegin_incl] = my_consensus[i];
                }
                if(threadIdx.x == 0){
                    d_subject_is_corrected[subjectIndex] = true;
                }
            }else{
                //decode orignal sequence and copy to corrected sequence
                const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                const char* const subject = getSubjectPtr(subjectIndex);
                for(int i = threadIdx.x; i < subjectLength; i += BLOCKSIZE){
                    my_corrected_subject[i] = to_nuc(get(subject, subjectLength, i));
                }

                bool foundAColumn = false;
                for(int i = threadIdx.x; i < subjectLength; i += BLOCKSIZE){
                    const int globalIndex = subjectColumnsBegin_incl + i;

                    if(my_corrected_subject[i] != my_consensus[globalIndex]
                                && my_support[globalIndex] > 0.5f
                                && my_orig_coverage[globalIndex] <= min_coverage_threshold){
                        float avgsupportkregion = 0;
                        int c = 0;
                        bool kregioncoverageisgood = true;

                        for(int j = i - k_region/2; j <= i + k_region/2 && kregioncoverageisgood; j++){
                            if(j != i && j >= 0 && j < subjectLength){
                                avgsupportkregion += my_support[subjectColumnsBegin_incl + j];
                                kregioncoverageisgood &= (my_coverage[subjectColumnsBegin_incl + j] >= min_coverage_threshold);
                                //kregioncoverageisgood &= (my_coverage[subjectColumnsBegin_incl + j] >= 1);
                                c++;
                            }
                        }

                        avgsupportkregion /= c;

                        if(kregioncoverageisgood && avgsupportkregion >= 1.0f-estimatedErrorrate){
                            my_corrected_subject[i] = my_consensus[globalIndex];
                            foundAColumn = true;
                        }
                    }
                }
                //perform block wide or-reduction on foundAColumn
                foundAColumn = BlockReduceBool(temp_storage.boolreduce).Reduce(foundAColumn, [](bool a, bool b){return a || b;});
                __syncthreads();

                if(threadIdx.x == 0){
                    d_subject_is_corrected[subjectIndex] = foundAColumn;
                }
            }
        }
    }


    template<int BLOCKSIZE>
    __global__
    void msa_correct_candidates_kernel_exp(
                const char* __restrict__ d_consensus,
                const float* __restrict__ d_support,
                const int* __restrict__ d_coverage,
                const int* __restrict__ d_origCoverages,
                const MSAColumnProperties* __restrict__ d_msa_column_properties,
                const int* __restrict__ d_candidate_sequences_lengths,
                const int* __restrict__ d_indices,
                const int* __restrict__ d_indices_per_subject,
                const int* __restrict__ d_indices_per_subject_prefixsum,
                const int* __restrict__ d_high_quality_subject_indices,
                const int* __restrict__ d_num_high_quality_subject_indices,
                const int* __restrict__ d_alignment_shifts,
                const BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
                int* __restrict__ d_num_corrected_candidates,
                char* __restrict__ d_corrected_candidates,
                int* __restrict__ d_indices_of_corrected_candidates,
                int n_subjects,
                int n_queries,
                const int* __restrict__ d_num_indices,
                size_t sequence_pitch,
                size_t msa_pitch,
                size_t msa_weights_pitch,
                float min_support_threshold,
                float min_coverage_threshold,
                int new_columns_to_correct){

        auto make_unpacked_reverse_complement_inplace = [] (std::uint8_t* sequence, int sequencelength){
            return reverseComplementStringInplace((char*)sequence, sequencelength);
        };

        auto getCandidateLength = [&] (int subjectIndex, int localCandidateIndex){
            const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
            const int index = my_indices[localCandidateIndex];
            const int length = d_candidate_sequences_lengths[index];
            return length;
        };

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
        const int num_high_quality_subject_indices = *d_num_high_quality_subject_indices;
        //const int n_indices = *d_num_indices;

        for(unsigned index = blockIdx.x; index < num_high_quality_subject_indices; index += gridDim.x) {
            const int subjectIndex = d_high_quality_subject_indices[index];
            const int my_num_candidates = d_indices_per_subject[subjectIndex];

            const float* const my_support = d_support + msa_weights_pitch_floats * subjectIndex;
            const int* const my_coverage = d_coverage + msa_weights_pitch_floats * subjectIndex;
            //const int* const my_orig_coverage = d_origCoverages + msa_weights_pitch_floats * subjectIndex;
            const char* const my_consensus = d_consensus + msa_pitch  * subjectIndex;
            const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
            char* const my_corrected_candidates = d_corrected_candidates + d_indices_per_subject_prefixsum[subjectIndex] * sequence_pitch;
            int* const my_indices_of_corrected_candidates = d_indices_of_corrected_candidates + d_indices_per_subject_prefixsum[subjectIndex];

            const MSAColumnProperties properties = d_msa_column_properties[subjectIndex];
            const int subjectColumnsBegin_incl = properties.subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = properties.subjectColumnsEnd_excl;

            int n_corrected_candidates = 0;

            for(int local_candidate_index = 0; local_candidate_index < my_num_candidates; ++local_candidate_index) {
                const int global_candidate_index = my_indices[local_candidate_index];
                const int shift = d_alignment_shifts[global_candidate_index];
                //const int candidate_length = d_candidate_sequences_lengths[global_candidate_index];
                const int candidate_length = getCandidateLength(subjectIndex, local_candidate_index);
                const BestAlignment_t bestAlignmentFlag = d_alignment_best_alignment_flags[global_candidate_index];
                const int queryColumnsBegin_incl = shift - properties.startindex;
                const int queryColumnsEnd_excl = queryColumnsBegin_incl + candidate_length;

                //check range condition and length condition
                if(subjectColumnsBegin_incl - new_columns_to_correct <= queryColumnsBegin_incl
                   && queryColumnsBegin_incl <= subjectColumnsBegin_incl + new_columns_to_correct
                   && queryColumnsEnd_excl <= subjectColumnsEnd_excl + new_columns_to_correct) {

                    float newColMinSupport = 1.0f;
                    int newColMinCov = std::numeric_limits<int>::max();
                    //check new columns left of subject
                    for(int columnindex = subjectColumnsBegin_incl - new_columns_to_correct;
                        columnindex < subjectColumnsBegin_incl;
                        columnindex++) {

                        assert(columnindex < properties.columnsToCheck);
                        if(queryColumnsBegin_incl <= columnindex) {
                            newColMinSupport = my_support[columnindex] < newColMinSupport ? my_support[columnindex] : newColMinSupport;
                            newColMinCov = my_coverage[columnindex] < newColMinCov ? my_coverage[columnindex] : newColMinCov;
                        }
                    }
                    //check new columns right of subject
                    for(int columnindex = subjectColumnsEnd_excl;
                        columnindex < subjectColumnsEnd_excl + new_columns_to_correct
                        && columnindex < properties.columnsToCheck;
                        columnindex++) {

                        newColMinSupport = my_support[columnindex] < newColMinSupport ? my_support[columnindex] : newColMinSupport;
                        newColMinCov = my_coverage[columnindex] < newColMinCov ? my_coverage[columnindex] : newColMinCov;
                    }

                    if(newColMinSupport >= min_support_threshold
                       && newColMinCov >= min_coverage_threshold) {

                        for(int i = queryColumnsBegin_incl + threadIdx.x; i < queryColumnsEnd_excl; i += BLOCKSIZE) {
                            my_corrected_candidates[n_corrected_candidates * sequence_pitch + (i - queryColumnsBegin_incl)] = my_consensus[i];
                        }

                        __syncthreads();                                         // need to wait until all threads have written my_corrected_candidates before calculating reverse complement

                        if(threadIdx.x == 0) {
                            //the forward strand will be returned -> make reverse complement again
                            if(bestAlignmentFlag == BestAlignment_t::ReverseComplement) {
                                make_unpacked_reverse_complement_inplace((std::uint8_t*)(my_corrected_candidates + n_corrected_candidates * sequence_pitch), candidate_length);
                            }
                            my_indices_of_corrected_candidates[n_corrected_candidates] = global_candidate_index;
                            //printf("subjectIndex %d global_candidate_index %d\n", subjectIndex, global_candidate_index);
                        }

                        ++n_corrected_candidates;
                    }
                }
            }

            //printf("%d %d\n", subjectIndex, n_corrected_candidates);

            if(threadIdx.x == 0) {
                d_num_corrected_candidates[subjectIndex] = n_corrected_candidates;
                //printf("%d %d\n", subjectIndex, n_corrected_candidates);
            }
        }
    }




    /*
        This kernel inspects a msa and identifies candidates which could originate
        from a different genome region than the subject.

        the output element shouldBeRemoved[i] indicates whether
        the candidate referred to by d_indices[i] should be removed from the msa
    */


    template<int BLOCKSIZE>
    __global__
    void msa_findCandidatesOfDifferentRegion_kernel(bool* __restrict__ d_shouldBeRemoved,
                                                    const char* __restrict__ d_subject_sequences_data,
                                                    const char* __restrict__ d_candidate_sequences_data,
                                                    const int* __restrict__ d_subject_sequences_lengths,
                                                    const int* __restrict__ d_candidate_sequences_lengths,
                                                    const int* __restrict__ d_candidates_per_subject_prefixsum,
                                                    const int* __restrict__ d_alignment_shifts,
                                                    int n_subjects,
                                                    int n_candidates,
                                                    int max_sequence_bytes,
                                                    size_t encodedsequencepitch,
                                                    const char* __restrict__ d_consensus,
                                                    const int* __restrict__ d_counts,
                                                    const float* __restrict__ d_weights,
                                                    const MSAColumnProperties* __restrict__ d_msa_column_properties,
                                                    size_t msa_pitch,
                                                    size_t msa_weights_pitch,
                                                    const int* __restrict__ d_indices,
                                                    const int* __restrict__ d_indices_per_subject,
                                                    const int* __restrict__ d_indices_per_subject_prefixsum,
                                                    int dataset_coverage){

        auto getNumBytes = [] (int sequencelength){
            return sizeof(unsigned int) * getEncodedNumInts2BitHiLo(sequencelength);
        };

        auto get = [] (const char* data, int length, int index){
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
    	};

        auto getSubjectPtr = [&] (int subjectIndex){
            const char* result = d_subject_sequences_data + std::size_t(subjectIndex) * encodedsequencepitch;
            return result;
        };

        auto getCandidatePtr = [&] (int candidateIndex){
            const char* result = d_candidate_sequences_data + std::size_t(candidateIndex) * encodedsequencepitch;
            return result;
        };

        auto getSubjectLength = [&] (int subjectIndex){
            const int length = d_subject_sequences_lengths[subjectIndex];
            return length;
        };

        auto getCandidateLength = [&] (int candidateIndex){
            const int length = d_candidate_sequences_lengths[candidateIndex];
            return length;
        };

        auto is_significant_count = [](int count, int dataset_coverage){
            if(int(dataset_coverage * 0.3f) <= count)
                return true;
            return false;
        };

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
        const char index_to_base[4]{'A','C','G','T'};

        using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
        using BlockReduceInt2 = cub::BlockReduce<int2, BLOCKSIZE>;

        __shared__ union{
            typename BlockReduceBool::TempStorage boolreduce;
            typename BlockReduceInt2::TempStorage int2reduce;
        } temp_storage;

        __shared__ bool broadcastbufferbool;
        __shared__ int broadcastbufferint4[4];

        extern __shared__ unsigned int sharedmemory[];

        for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){

            const int* myIndices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
            const int myNumIndices = d_indices_per_subject[subjectIndex];

            if(myNumIndices > 0){

                const char* subjectptr = getSubjectPtr(subjectIndex);
                const int subjectLength = getSubjectLength(subjectIndex);

                const char* myConsensus = d_consensus + subjectIndex * msa_pitch;

                const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
                const int subjectColumnsEnd_excl = d_msa_column_properties[subjectIndex].subjectColumnsEnd_excl;

                //check if subject and consensus differ at at least one position

                bool hasMismatchToConsensus = false;

                for(int pos = threadIdx.x; pos < subjectLength && !hasMismatchToConsensus; pos += blockDim.x){
                    const int column = subjectColumnsBegin_incl + pos;
                    const char consbase = myConsensus[column];
                    const char subjectbase = get(subjectptr, subjectLength, pos);

                    hasMismatchToConsensus |= (consbase != subjectbase);
                }

                hasMismatchToConsensus = BlockReduceBool(temp_storage.boolreduce).Reduce(hasMismatchToConsensus, [](auto l, auto r){return l || r;});

                if(threadIdx.x == 0){
                    broadcastbufferbool = hasMismatchToConsensus;
                }
                __syncthreads();

                hasMismatchToConsensus = broadcastbufferbool;

                //if subject and consensus differ at at least one position, check columns in msa

                if(hasMismatchToConsensus){
                    int col = std::numeric_limits<int>::max();
                    bool foundColumn = false;
                    char foundBase = 'F';
                    int foundBaseIndex = std::numeric_limits<int>::max();
                    int consindex = std::numeric_limits<int>::max();

                    const int* const myCountsA = d_counts + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                    const int* const myCountsC = d_counts + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                    const int* const myCountsG = d_counts + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                    const int* const myCountsT = d_counts + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                    for(int columnindex = subjectColumnsBegin_incl + threadIdx.x; columnindex < subjectColumnsEnd_excl && !foundColumn; columnindex += blockDim.x){
                        int counts[4];
                        counts[0] = myCountsA[columnindex];
                        counts[1] = myCountsC[columnindex];
                        counts[2] = myCountsG[columnindex];
                        counts[3] = myCountsT[columnindex];

                        const char consbase = myConsensus[columnindex];
                        consindex = -1;

                        switch(consbase){
                            case 'A': consindex = 0;break;
                            case 'C': consindex = 1;break;
                            case 'G': consindex = 2;break;
                            case 'T': consindex = 3;break;
                        }

                        //find out if there is a non-consensus base with significant coverage
                        int significantBaseIndex = -1;

                        #pragma unroll
                        for(int i = 0; i < 4; i++){
                            if(i != consindex){
                                const bool significant = is_significant_count(counts[i], dataset_coverage);
                                significantBaseIndex = significant ? i : significantBaseIndex;
                            }
                        }

                        if(significantBaseIndex != -1){
                            foundColumn = true;
                            col = columnindex;
                            foundBaseIndex = significantBaseIndex;
                        }
                    }

                    int2 packed{col, foundBaseIndex};
                    //find packed value with smallest col
                    packed = BlockReduceInt2(temp_storage.int2reduce).Reduce(packed, [](auto l, auto r){
                        if(l.x < r.x){
                            return l;
                        }else{
                            return r;
                        }
                    });

                    if(threadIdx.x == 0){
                        if(packed.x != std::numeric_limits<int>::max()){
                            broadcastbufferint4[0] = true;
                            broadcastbufferint4[1] = packed.x;
                            broadcastbufferint4[2] = index_to_base[packed.y];
                            broadcastbufferint4[3] = packed.y;
                        }else{
                            broadcastbufferint4[0] = false;
                        }
                    }

                    __syncthreads();

                    foundColumn = broadcastbufferint4[0];
                    col = broadcastbufferint4[1];
                    foundBase = broadcastbufferint4[2];
                    foundBaseIndex = broadcastbufferint4[3];

                    if(foundColumn){

                        //compare found base to original base
                        const char originalbase = get(subjectptr, subjectLength, col - subjectColumnsBegin_incl);

                        /*int counts[4];

                        counts[0] = myCountsA[col];
                        counts[1] = myCountsC[col];
                        counts[2] = myCountsG[col];
                        counts[3] = myCountsT[col];*/

                        auto discard_rows = [&](bool keepMatching){

                            //int seenCounts[4]{0,0,0,0};

                            const int indexoffset = d_indices_per_subject_prefixsum[subjectIndex];

                            for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                                const int candidateIndex = myIndices[k];
                                const char* const candidateptr = getCandidatePtr(candidateIndex);
                                const int candidateLength = getCandidateLength(candidateIndex);
                                const int shift = d_alignment_shifts[candidateIndex];

                                //check if row is affected by column col
                                const int row_begin_incl = subjectColumnsBegin_incl + shift;
                                const int row_end_excl = row_begin_incl + candidateLength;
                                const bool notAffected = (col < row_begin_incl || row_end_excl <= col);
                                const char base = notAffected ? 'F' : get(candidateptr, candidateLength, (col - row_begin_incl));

                                /*if(base == 'A') seenCounts[0]++;
                                if(base == 'C') seenCounts[1]++;
                                if(base == 'G') seenCounts[2]++;
                                if(base == 'T') seenCounts[3]++;*/

                                if(notAffected || (!(keepMatching ^ (base == foundBase)))){
                                    d_shouldBeRemoved[indexoffset + k] = false; //same region
                                }else{
                                    d_shouldBeRemoved[indexoffset + k] = true; //different region
                                }
                            }

                            /*if(originalbase == 'A') seenCounts[0]++;
                            if(originalbase == 'C') seenCounts[1]++;
                            if(originalbase == 'G') seenCounts[2]++;
                            if(originalbase == 'T') seenCounts[3]++;

                            #pragma unroll
                            for(int i = 0; i < 4; i++){
                                if(seenCounts[i] != counts[i]){
                                    assert(seenCounts[i] == counts[i]);
                                }
                            }*/
                        };



                        if(originalbase == foundBase){
                            //discard all candidates whose base in column col differs from foundBase
                            discard_rows(true);
                        }else{
                            //discard all candidates whose base in column col matches foundBase
                            discard_rows(false);
                        }

                    }else{
                        //did not find a significant columns

                        //remove no candidate
                        const int indexoffset = d_indices_per_subject_prefixsum[subjectIndex];

                        for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                            d_shouldBeRemoved[indexoffset + k] = false;
                        }
                    }

                }else{
                    //no mismatch between consensus and subject

                    //remove no candidate
                    const int indexoffset = d_indices_per_subject_prefixsum[subjectIndex];

                    for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                        d_shouldBeRemoved[indexoffset + k] = false;
                    }
                }
            }else{
                ; //nothing to do if there are no candidates in msa
            }
        }

    }









    //####################   KERNEL DISPATCH   ####################


    void call_cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel_async(
    			int* d_alignment_scores,
    			int* d_alignment_overlaps,
    			int* d_alignment_shifts,
    			int* d_alignment_nOps,
    			bool* d_alignment_isValid,
                const char* d_subject_sequences_data,
                const char* d_candidate_sequences_data,
                const int* d_subject_sequences_lengths,
                const int* d_candidate_sequences_lengths,
    			const int* d_candidates_per_subject_prefixsum,
                const int* h_tiles_per_subject_prefixsum,
                const int* d_tiles_per_subject_prefixsum,
                int tilesize,
    			int n_subjects,
    			int n_queries,
                size_t encodedsequencepitch,
    			int max_sequence_bytes,
    			int min_overlap,
    			float maxErrorRate,
    			float min_overlap_ratio,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

        	const int blocksize = 128;
            const int tilesPerBlock = blocksize / tilesize;
            const int requiredTiles = h_tiles_per_subject_prefixsum[n_subjects];
            const int requiredBlocks = SDIV(requiredTiles, tilesPerBlock);

            //printf("n_subjects %d, n_queries %d\n", n_subjects, n_queries);


        	const std::size_t smem = sizeof(char) * (max_sequence_bytes * tilesPerBlock + max_sequence_bytes * blocksize * 2);

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
                		kernelLaunchConfig.smem = sizeof(char) * (max_sequence_bytes * tilesPerBlock + max_sequence_bytes * blocksize * 2); \
                		KernelProperties kernelProperties; \
                		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                					cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel, \
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

            #define mycall(tilesize) cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel \
                                    	        <<<grid, block, smem, stream>>>( \
                                        		d_alignment_scores, \
                                        		d_alignment_overlaps, \
                                        		d_alignment_shifts, \
                                        		d_alignment_nOps, \
                                        		d_alignment_isValid, \
                                                d_subject_sequences_data, \
                                                d_candidate_sequences_data, \
                                                d_subject_sequences_lengths, \
                                                d_candidate_sequences_lengths, \
                                        		d_candidates_per_subject_prefixsum, \
                                                d_tiles_per_subject_prefixsum, \
                                                tilesize, \
                                        		n_subjects, \
                                        		n_queries, \
                                                encodedsequencepitch, \
                                        		max_sequence_bytes, \
                                        		min_overlap, \
                                        		maxErrorRate, \
                                        		min_overlap_ratio); CUERR;

        	dim3 block(blocksize, 1, 1);
        	dim3 grid(std::min(requiredBlocks, max_blocks_per_device), 1, 1);
            //dim3 grid(1,1,1);

        	mycall(tilesize);

    	    #undef mycall
    }


    void call_cuda_find_best_alignment_kernel_async_exp(
    			BestAlignment_t* d_alignment_best_alignment_flags,
    			int* d_alignment_scores,
    			int* d_alignment_overlaps,
    			int* d_alignment_shifts,
    			int* d_alignment_nOps,
    			bool* d_alignment_isValid,
                const int* d_subject_sequences_lengths,
                const int* d_candidate_sequences_lengths,
    			const int* d_candidates_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
    			float min_overlap_ratio,
    			int min_overlap,
                float estimatedErrorrate,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

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
            		d_alignment_best_alignment_flags,
            		d_alignment_scores,
            		d_alignment_overlaps,
            		d_alignment_shifts,
            		d_alignment_nOps,
            		d_alignment_isValid,
                    d_subject_sequences_lengths,
                    d_candidate_sequences_lengths,
            		d_candidates_per_subject_prefixsum,
            		n_subjects,
            		n_queries,
            		min_overlap_ratio,
            		min_overlap,
                    estimatedErrorrate); CUERR;

    }


    void call_cuda_filter_alignments_by_mismatchratio_kernel_async(
    			BestAlignment_t* d_alignment_best_alignment_flags,
    			const int* d_alignment_overlaps,
    			const int* d_alignment_nOps,
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
    		d_alignment_best_alignment_flags, \
    		d_alignment_overlaps, \
    		d_alignment_nOps, \
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


    void call_msa_init_kernel_async_exp(
    			MSAColumnProperties* d_msa_column_properties,
    			const int* d_alignment_shifts,
    			const BestAlignment_t* d_alignment_best_alignment_flags,
                const int* d_subject_sequences_lengths,
                const int* d_candidate_sequences_lengths,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			const int* d_indices_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

    	const int blocksize = 128;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAInitExp);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
    		KernelLaunchConfig kernelLaunchConfig; \
    		kernelLaunchConfig.threads_per_block = (blocksize); \
    		kernelLaunchConfig.smem = 0; \
    		KernelProperties kernelProperties; \
    		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
    					msa_init_kernel_exp<(blocksize)>, \
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

    		handle.kernelPropertiesMap[KernelId::MSAInitExp] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(max_blocks_per_device, n_subjects), 1, 1);

		#define mycall(blocksize) msa_init_kernel_exp<(blocksize)> \
                <<<grid, block, 0, stream>>>(d_msa_column_properties, \
                                               d_alignment_shifts, \
                                               d_alignment_best_alignment_flags, \
                                               d_subject_sequences_lengths, \
                                               d_candidate_sequences_lengths, \
                                               d_indices, \
                                               d_indices_per_subject, \
                                               d_indices_per_subject_prefixsum, \
                                               n_subjects); CUERR;

    	switch(blocksize) {
    	case 1: mycall(1); break;
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


    void call_msa_add_sequences_kernel_exp_async(
    			char* d_multiple_sequence_alignments,
    			float* d_multiple_sequence_alignment_weights,
    			const int* d_alignment_shifts,
    			const BestAlignment_t* d_alignment_best_alignment_flags,
    			const int* d_alignment_overlaps,
    			const int* d_alignment_nOps,
                const char* d_subject_sequences_data,
                const char* d_candidate_sequences_data,
                const int* d_subject_sequences_lengths,
                const int* d_candidate_sequences_lengths,
                const char* d_subject_qualities,
                const char* d_candidate_qualities,
    			const MSAColumnProperties*  d_msa_column_properties,
    			const int* d_candidates_per_subject_prefixsum,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			const int* d_indices_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
                const int* h_num_indices,
    			const int* d_num_indices,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
    			int maximum_sequence_length,
    			int max_sequence_bytes,
                size_t encoded_sequence_pitch,
    			size_t quality_pitch,
    			size_t msa_row_pitch,
    			size_t msa_weights_row_pitch,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){


    	const int blocksize = 128;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAAddSequences);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
    		KernelLaunchConfig kernelLaunchConfig; \
    		kernelLaunchConfig.threads_per_block = (blocksize); \
    		kernelLaunchConfig.smem = 0; \
    		KernelProperties kernelProperties; \
    		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
    					msa_add_sequences_kernel_exp, \
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
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;

    		handle.kernelPropertiesMap[KernelId::MSAAddSequences] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;
    		//std::cout << max_blocks_per_device << " = " << handle.deviceProperties.multiProcessorCount << " * " << kernelProperties.max_blocks_per_SM << std::endl;
    	}

    	dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(*h_num_indices, max_blocks_per_device), 1, 1);

    	msa_add_sequences_kernel_exp
                <<<grid, block, smem, stream>>>(d_multiple_sequence_alignments,
                                                d_multiple_sequence_alignment_weights,
                                                d_alignment_shifts,
                                                d_alignment_best_alignment_flags,
                                                d_alignment_overlaps,
                                                d_alignment_nOps,
                                                d_subject_sequences_data,
                                                d_candidate_sequences_data,
                                                d_subject_sequences_lengths,
                                                d_candidate_sequences_lengths,
                                                d_subject_qualities,
                                                d_candidate_qualities,
                                                d_msa_column_properties,
                                                d_candidates_per_subject_prefixsum,
                                                d_indices,
                                                d_indices_per_subject,
                                                d_indices_per_subject_prefixsum,
                                                n_subjects,
                                                n_queries,
                                                d_num_indices,
                                                canUseQualityScores,
                                                desiredAlignmentMaxErrorRate,
                                                maximum_sequence_length,
                                                max_sequence_bytes,
                                                encoded_sequence_pitch,
                                                quality_pitch,
                                                msa_row_pitch,
                                                msa_weights_row_pitch); CUERR;
    }


    void call_msa_add_sequences_kernel_implicit_global_async(
                int* d_counts,
                float* d_weights,
                int* d_coverage,
                float* d_origWeights,
                int* d_origCoverages,
    			const int* d_alignment_shifts,
    			const BestAlignment_t* d_alignment_best_alignment_flags,
                const int* d_alignment_overlaps,
    			const int* d_alignment_nOps,
                const char* d_subject_sequences_data,
                const char* d_candidate_sequences_data,
                const int* d_subject_sequences_lengths,
                const int* d_candidate_sequences_lengths,
                const char* d_subject_qualities,
				const char* d_candidate_qualities,
    			const MSAColumnProperties*  d_msa_column_properties,
    			const int* d_candidates_per_subject_prefixsum,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			const int* d_indices_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
                const int* h_num_indices,
    			const int* d_num_indices,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
    			int maximum_sequence_length,
    			int max_sequence_bytes,
                size_t encoded_sequence_pitch,
    			size_t quality_pitch,
    			size_t msa_row_pitch,
    			size_t msa_weights_row_pitch,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

    	const int blocksize = 128;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAAddSequencesImplicitGlobal);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
    		KernelLaunchConfig kernelLaunchConfig; \
    		kernelLaunchConfig.threads_per_block = (blocksize); \
    		kernelLaunchConfig.smem = 0; \
    		KernelProperties kernelProperties; \
    		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
    					msa_add_sequences_kernel_implicit_global, \
    					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
    		mymap[kernelLaunchConfig] = kernelProperties; \
    }

            getProp(1);
    		getProp(32);
    		getProp(64);
    		getProp(96);
    		getProp(128);
    		getProp(160);
    		getProp(192);
    		getProp(224);
    		getProp(256);

    		const auto& kernelProperties = mymap[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;

    		handle.kernelPropertiesMap[KernelId::MSAAddSequencesImplicitGlobal] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;
    		//std::cout << max_blocks_per_device << " = " << handle.deviceProperties.multiProcessorCount << " * " << kernelProperties.max_blocks_per_SM << std::endl;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(*h_num_indices, max_blocks_per_device), 1, 1);

    	msa_add_sequences_kernel_implicit_global<<<grid, block, smem, stream>>>(
                                        d_counts,
                                        d_weights,
                                        d_coverage,
                                        d_origWeights,
                                        d_origCoverages,
                                        d_alignment_shifts,
                                        d_alignment_best_alignment_flags,
                                        d_alignment_overlaps,
                            			d_alignment_nOps,
                                        d_subject_sequences_data,
                                        d_candidate_sequences_data,
                                        d_subject_sequences_lengths,
                                        d_candidate_sequences_lengths,
                                        d_subject_qualities,
                        				d_candidate_qualities,
                                        d_msa_column_properties,
                                        d_candidates_per_subject_prefixsum,
                                        d_indices,
                                        d_indices_per_subject,
                                        d_indices_per_subject_prefixsum,
                                        n_subjects,
                                        n_queries,
                                        d_num_indices,
                                        canUseQualityScores,
                                        desiredAlignmentMaxErrorRate,
                                        maximum_sequence_length,
                                        max_sequence_bytes,
                                        encoded_sequence_pitch,
                                        quality_pitch,
                                        msa_row_pitch,
                                        msa_weights_row_pitch); CUERR;
    }


    void call_msa_add_sequences_kernel_implicit_shared_async(
                int* d_counts,
                float* d_weights,
                int* d_coverage,
                float* d_origWeights,
                int* d_origCoverages,
    			const int* d_alignment_shifts,
    			const BestAlignment_t* d_alignment_best_alignment_flags,
                const int* d_alignment_overlaps,
    			const int* d_alignment_nOps,
                const char* d_subject_sequences_data,
                const char* d_candidate_sequences_data,
                const int* d_subject_sequences_lengths,
                const int* d_candidate_sequences_lengths,
                const char* d_subject_qualities,
				const char* d_candidate_qualities,
    			const MSAColumnProperties*  d_msa_column_properties,
    			const int* d_candidates_per_subject_prefixsum,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			const int* d_indices_per_subject_prefixsum,
                const int* d_blocks_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
                const int* h_num_indices,
    			const int* d_num_indices,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
    			int maximum_sequence_length,
    			int max_sequence_bytes,
                size_t encoded_sequence_pitch,
    			size_t quality_pitch,
    			size_t msa_row_pitch,
    			size_t msa_weights_row_pitch,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

    	const int blocksize = msa_add_sequences_kernel_implicit_shared_blocksize;
        const std::size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);

    	//const std::size_t smem = sizeof(char) * maximum_sequence_length + sizeof(float) * maximum_sequence_length;
        const std::size_t smem = sizeof(float) * 4 * msa_weights_row_pitch_floats // weights
                                + sizeof(int) * 4 * msa_weights_row_pitch_floats; // counts


    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAAddSequencesImplicitShared);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
    		KernelLaunchConfig kernelLaunchConfig; \
    		kernelLaunchConfig.threads_per_block = (blocksize); \
    		kernelLaunchConfig.smem = sizeof(float) * 4 * msa_weights_row_pitch_floats \
                                    + sizeof(int) * 4 * msa_weights_row_pitch_floats; \
    		KernelProperties kernelProperties; \
    		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
    					msa_add_sequences_kernel_implicit_shared, \
    					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
    		mymap[kernelLaunchConfig] = kernelProperties; \
    }

            getProp(1);
    		getProp(32);
    		getProp(64);
    		getProp(96);
    		getProp(128);
    		getProp(160);
    		getProp(192);
    		getProp(224);
    		getProp(256);

    		const auto& kernelProperties = mymap[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;

    		handle.kernelPropertiesMap[KernelId::MSAAddSequencesImplicitShared] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;
    		//std::cout << max_blocks_per_device << " = " << handle.deviceProperties.multiProcessorCount << " * " << kernelProperties.max_blocks_per_SM << std::endl;
    	}

    	dim3 block(blocksize, 1, 1);

        const int blocks = SDIV(*h_num_indices, blocksize);
    	dim3 grid(std::min(blocks, max_blocks_per_device), 1, 1);

    	msa_add_sequences_kernel_implicit_shared<<<grid, block, smem, stream>>>(
                                            d_counts,
                                            d_weights,
                                            d_coverage,
                                            d_origWeights,
                                            d_origCoverages,
                                            d_alignment_shifts,
                                            d_alignment_best_alignment_flags,
                                            d_alignment_overlaps,
                                			d_alignment_nOps,
                                            d_subject_sequences_data,
                                            d_candidate_sequences_data,
                                            d_subject_sequences_lengths,
                                            d_candidate_sequences_lengths,
                                            d_subject_qualities,
                            				d_candidate_qualities,
                                            d_msa_column_properties,
                                            d_candidates_per_subject_prefixsum,
                                            d_indices,
                                            d_indices_per_subject,
                                            d_indices_per_subject_prefixsum,
                                            d_blocks_per_subject_prefixsum,
                                            n_subjects,
                                            n_queries,
                                            d_num_indices,
                                            canUseQualityScores,
                                            desiredAlignmentMaxErrorRate,
                                            maximum_sequence_length,
                                            max_sequence_bytes,
                                            encoded_sequence_pitch,
                                            quality_pitch,
                                            msa_row_pitch,
                                            msa_weights_row_pitch); CUERR;
    }


    void call_msa_add_sequences_kernel_implicit_async(
                int* d_counts,
                float* d_weights,
                int* d_coverage,
                float* d_origWeights,
                int* d_origCoverages,
    			const int* d_alignment_shifts,
    			const BestAlignment_t* d_alignment_best_alignment_flags,
                const int* d_alignment_overlaps,
    			const int* d_alignment_nOps,
                const char* d_subject_sequences_data,
                const char* d_candidate_sequences_data,
                const int* d_subject_sequences_lengths,
                const int* d_candidate_sequences_lengths,
                const char* d_subject_qualities,
				const char* d_candidate_qualities,
    			const MSAColumnProperties*  d_msa_column_properties,
    			const int* d_candidates_per_subject_prefixsum,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			const int* d_indices_per_subject_prefixsum,
                const int* d_blocks_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
                const int* h_num_indices,
    			const int* d_num_indices,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
    			int maximum_sequence_length,
    			int max_sequence_bytes,
                size_t encoded_sequence_pitch,
    			size_t quality_pitch,
    			size_t msa_row_pitch,
    			size_t msa_weights_row_pitch,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

        //std::cout << n_subjects << " " << *h_num_indices << " " << n_queries << std::endl;

    #if 0
        call_msa_add_sequences_kernel_implicit_global_async(d_counts,
                                                            d_weights,
                                                            d_coverage,
                                                            d_origWeights,
                                                            d_origCoverages,
                                                            d_alignment_shifts,
                                                            d_alignment_best_alignment_flags,
                                                            d_alignment_overlaps,
                                                			d_alignment_nOps,
                                                            d_subject_sequences_data,
                                                            d_candidate_sequences_data,
                                                            d_subject_sequences_lengths,
                                                            d_candidate_sequences_lengths,
                                                            d_subject_qualities,
                                            				d_candidate_qualities,
                                                            d_msa_column_properties,
                                                            d_candidates_per_subject_prefixsum,
                                                            d_indices,
                                                            d_indices_per_subject,
                                                            d_indices_per_subject_prefixsum,
                                                            n_subjects,
                                                            n_queries,
                                                            h_num_indices,
                                                            d_num_indices,
                                                            canUseQualityScores,
                                                            desiredAlignmentMaxErrorRate,
                                                            maximum_sequence_length,
                                                            max_sequence_bytes,
                                                            encoded_sequence_pitch,
                                                            quality_pitch,
                                                            msa_row_pitch,
                                                            msa_weights_row_pitch,
                                                            stream,
                                                            handle); CUERR;
    #else


        call_msa_add_sequences_kernel_implicit_shared_async(d_counts,
                                                            d_weights,
                                                            d_coverage,
                                                            d_origWeights,
                                                            d_origCoverages,
                                                            d_alignment_shifts,
                                                            d_alignment_best_alignment_flags,
                                                            d_alignment_overlaps,
                                                			d_alignment_nOps,
                                                            d_subject_sequences_data,
                                                            d_candidate_sequences_data,
                                                            d_subject_sequences_lengths,
                                                            d_candidate_sequences_lengths,
                                                            d_subject_qualities,
                                            				d_candidate_qualities,
                                                            d_msa_column_properties,
                                                            d_candidates_per_subject_prefixsum,
                                                            d_indices,
                                                            d_indices_per_subject,
                                                            d_indices_per_subject_prefixsum,
                                                            d_blocks_per_subject_prefixsum,
                                                            n_subjects,
                                                            n_queries,
                                                            h_num_indices,
                                                            d_num_indices,
                                                            canUseQualityScores,
                                                            desiredAlignmentMaxErrorRate,
                                                            maximum_sequence_length,
                                                            max_sequence_bytes,
                                                            encoded_sequence_pitch,
                                                            quality_pitch,
                                                            msa_row_pitch,
                                                            msa_weights_row_pitch,
                                                            stream,
                                                            handle); CUERR;
    #endif
    }


    void call_msa_find_consensus_kernel_async(
                            char* d_consensus,
                            float* d_support,
                            int* d_coverage,
                            float* d_origWeights,
                            int* d_origCoverages,
                            int* d_counts,
                            float* d_weights,
                            const char* d_multiple_sequence_alignments,
                            const float* d_multiple_sequence_alignment_weights,
                            const MSAColumnProperties* d_msa_column_properties,
                            const int* d_candidates_per_subject_prefixsum,
                            const int* d_indices_per_subject,
                            const int* d_indices_per_subject_prefixsum,
                            int n_subjects,
                            int n_queries,
                            const int* d_num_indices,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int msa_max_column_count,
                            cudaStream_t stream,
                            KernelLaunchHandle& handle){


        const int blocksize = 128;
        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::MSAFindConsensus);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                                                                msa_find_consensus_kernel, \
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

            handle.kernelPropertiesMap[KernelId::MSAFindConsensus] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }


        const int blocks_per_msa = 2; //SDIV(msa_max_column_count, blocksize);

        dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(max_blocks_per_device, n_subjects * blocks_per_msa), 1, 1);

        msa_find_consensus_kernel<<<grid, block, 0, stream>>>(d_consensus,
                                                            d_support,
                                                            d_coverage,
                                                            d_origWeights,
                                                            d_origCoverages,
                                                            d_counts,
                                                            d_weights,
                                                            d_multiple_sequence_alignments,
                                                            d_multiple_sequence_alignment_weights,
                                                            d_msa_column_properties,
                                                            d_candidates_per_subject_prefixsum,
                                                            d_indices_per_subject,
                                                            d_indices_per_subject_prefixsum,
                                                            n_subjects,
                                                            n_queries,
                                                            d_num_indices,
                                                            msa_pitch,
                                                            msa_weights_pitch,
                                                            msa_max_column_count,
                                                            blocks_per_msa); CUERR;

    }


    void call_msa_find_consensus_implicit_kernel_async(
                            int* d_counts,
                            float* d_weights,
                            char* d_consensus,
                            float* d_support,
                            int* d_coverage,
                            float* d_origWeights,
                            int* d_origCoverages,
                            const char* d_subject_sequences_data,
                            const MSAColumnProperties* d_msa_column_properties,
                            int n_subjects,
                            size_t encoded_sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            cudaStream_t stream,
                            KernelLaunchHandle& handle){


        const int blocksize = 128;
        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::MSAFindConsensusImplicit);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                                                                msa_find_consensus_implicit_kernel, \
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

            handle.kernelPropertiesMap[KernelId::MSAFindConsensusImplicit] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }


        const int blocks_per_msa = 2; //SDIV(msa_max_column_count, blocksize);

        dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(max_blocks_per_device, n_subjects * blocks_per_msa), 1, 1);

        msa_find_consensus_implicit_kernel<<<grid, block, 0, stream>>>(
                                                            d_counts,
                                                            d_weights,
                                                            d_consensus,
                                                            d_support,
                                                            d_coverage,
                                                            d_origWeights,
                                                            d_origCoverages,
                                                            d_subject_sequences_data,
                                                            d_msa_column_properties,
                                                            n_subjects,
                                                            encoded_sequence_pitch,
                                                            msa_pitch,
                                                            msa_weights_pitch,
                                                            blocks_per_msa); CUERR;

    }



    void call_msa_correct_subject_kernel_async(
                            const char* d_consensus,
                            const float* d_support,
                            const int* d_coverage,
                            const int* d_origCoverages,
                            const char* d_multiple_sequence_alignments,
                            const MSAColumnProperties* d_msa_column_properties,
                            const int* d_indices_per_subject_prefixsum,
                            bool* d_is_high_quality_subject,
                            char* d_corrected_subjects,
							bool* d_subject_is_corrected,
                            int n_subjects,
                            int n_queries,
                            const int* d_num_indices,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            float estimatedErrorrate,
                            float avg_support_threshold,
                            float min_support_threshold,
                            float min_coverage_threshold,
                            int k_region,
                            int maximum_sequence_length,
                            cudaStream_t stream,
                            KernelLaunchHandle& handle){

        const int max_block_size = 256;
        const int blocksize = std::min(max_block_size, SDIV(maximum_sequence_length, 32) * 32);
        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::MSACorrectSubject);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                                                                msa_correct_subject_kernel<(blocksize)>, \
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

            handle.kernelPropertiesMap[KernelId::MSACorrectSubject] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }



        dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(n_subjects, max_blocks_per_device));

        #define mycall(blocksize) msa_correct_subject_kernel<(blocksize)> \
                                <<<grid, block, 0, stream>>>( \
                                    d_consensus, \
                                    d_support, \
                                    d_coverage, \
                                    d_origCoverages, \
                                    d_multiple_sequence_alignments, \
                                    d_msa_column_properties, \
                                    d_indices_per_subject_prefixsum, \
                                    d_is_high_quality_subject, \
                                    d_corrected_subjects, \
                                    d_subject_is_corrected, \
                                    n_subjects, \
                                    n_queries, \
                                    d_num_indices, \
                                    sequence_pitch, \
                                    msa_pitch, \
                                    msa_weights_pitch, \
                                    estimatedErrorrate, \
                                    avg_support_threshold, \
                                    min_support_threshold, \
                                    min_coverage_threshold, \
                                    k_region); CUERR;

        assert(blocksize > 0 && blocksize <= max_block_size);

        switch(blocksize){
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




    void call_msa_correct_subject_implicit_kernel_async(
                            const char* d_consensus,
                            const float* d_support,
                            const int* d_coverage,
                            const int* d_origCoverages,
                            const MSAColumnProperties* d_msa_column_properties,
                            const char* d_subject_sequences_data,
                            bool* d_is_high_quality_subject,
                            char* d_corrected_subjects,
                            bool* d_subject_is_corrected,
                            int n_subjects,
                            size_t encoded_sequence_pitch,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            float estimatedErrorrate,
                            float avg_support_threshold,
                            float min_support_threshold,
                            float min_coverage_threshold,
                            int k_region,
                            int maximum_sequence_length,
                            cudaStream_t stream,
                            KernelLaunchHandle& handle){

        const int max_block_size = 256;
        const int blocksize = std::min(max_block_size, SDIV(maximum_sequence_length, 32) * 32);
        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::MSACorrectSubjectImplicit);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                                                                msa_correct_subject_implicit_kernel<(blocksize)>, \
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

            handle.kernelPropertiesMap[KernelId::MSACorrectSubjectImplicit] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }



        dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(n_subjects, max_blocks_per_device));

        #define mycall(blocksize) msa_correct_subject_implicit_kernel<(blocksize)> \
                                <<<grid, block, 0, stream>>>( \
                                    d_consensus, \
                                    d_support, \
                                    d_coverage, \
                                    d_origCoverages, \
                                    d_msa_column_properties, \
                                    d_subject_sequences_data, \
                                    d_is_high_quality_subject, \
                                    d_corrected_subjects, \
                                    d_subject_is_corrected, \
                                    n_subjects, \
                                    encoded_sequence_pitch, \
                                    sequence_pitch, \
                                    msa_pitch, \
                                    msa_weights_pitch, \
                                    estimatedErrorrate, \
                                    avg_support_threshold, \
                                    min_support_threshold, \
                                    min_coverage_threshold, \
                                    k_region); CUERR;

        assert(blocksize > 0 && blocksize <= max_block_size);

        switch(blocksize){
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







    void call_msa_correct_candidates_kernel_async_exp(
    			const char* d_consensus,
    			const float* d_support,
    			const int* d_coverage,
    			const int* d_origCoverages,
    			const MSAColumnProperties* d_msa_column_properties,
                const int* d_candidate_sequences_lengths,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			const int* d_indices_per_subject_prefixsum,
    			const int* d_high_quality_subject_indices,
    			const int* d_num_high_quality_subject_indices,
    			const int* d_alignment_shifts,
    			const BestAlignment_t* d_alignment_best_alignment_flags,
    			int* d_num_corrected_candidates,
    			char* d_corrected_candidates,
    			int* d_indices_of_corrected_candidates,
    			int n_subjects,
    			int n_queries,
    			const int* d_num_indices,
    			size_t sequence_pitch,
    			size_t msa_pitch,
    			size_t msa_weights_pitch,
    			float min_support_threshold,
    			float min_coverage_threshold,
    			int new_columns_to_correct,
    			int maximum_sequence_length,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){


    	const int max_block_size = 256;
    	const int blocksize = std::min(max_block_size, SDIV(maximum_sequence_length, 32) * 32);
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSACorrectCandidates);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
    		KernelLaunchConfig kernelLaunchConfig; \
    		kernelLaunchConfig.threads_per_block = (blocksize); \
    		kernelLaunchConfig.smem = 0; \
    		KernelProperties kernelProperties; \
    		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
    					msa_correct_candidates_kernel_exp<(blocksize)>, \
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

    		handle.kernelPropertiesMap[KernelId::MSACorrectCandidates] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(max_blocks_per_device, n_subjects));

    		#define mycall(blocksize) msa_correct_candidates_kernel_exp<(blocksize)> \
    	        <<<grid, block, 0, stream>>>( \
    		d_consensus, \
    		d_support, \
    		d_coverage, \
    		d_origCoverages, \
    		d_msa_column_properties, \
            d_candidate_sequences_lengths, \
    		d_indices, \
    		d_indices_per_subject, \
    		d_indices_per_subject_prefixsum, \
    		d_high_quality_subject_indices, \
    		d_num_high_quality_subject_indices, \
    		d_alignment_shifts, \
    		d_alignment_best_alignment_flags, \
    		d_num_corrected_candidates, \
    		d_corrected_candidates, \
    		d_indices_of_corrected_candidates, \
    		n_subjects, \
    		n_queries, \
    		d_num_indices, \
    		sequence_pitch, \
    		msa_pitch, \
    		msa_weights_pitch, \
    		min_support_threshold, \
    		min_coverage_threshold, \
    		new_columns_to_correct); CUERR;

    	assert(blocksize > 0 && blocksize <= max_block_size);

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


    void call_msa_findCandidatesOfDifferentRegion_kernel(
                bool* d_shouldBeRemoved,
                const char* d_subject_sequences_data,
                const char* d_candidate_sequences_data,
                const int* d_subject_sequences_lengths,
                const int* d_candidate_sequences_lengths,
                const int* d_candidates_per_subject_prefixsum,
                const int* d_alignment_shifts,
                int n_subjects,
                int n_candidates,
                int max_sequence_bytes,
                size_t encodedsequencepitch,
                const char* d_consensus,
                const int* d_counts,
                const float* d_weights,
                const MSAColumnProperties* d_msa_column_properties,
                size_t msa_pitch,
                size_t msa_weights_pitch,
                const int* d_indices,
                const int* d_indices_per_subject,
                const int* d_indices_per_subject_prefixsum,
                int dataset_coverage,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){


    	const int max_block_size = 256;
    	const int blocksize = 256;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAFindCandidatesOfDifferentRegion);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
            		KernelLaunchConfig kernelLaunchConfig; \
            		kernelLaunchConfig.threads_per_block = (blocksize); \
            		kernelLaunchConfig.smem = 0; \
            		KernelProperties kernelProperties; \
            		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
            					msa_findCandidatesOfDifferentRegion_kernel<(blocksize)>, \
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

    		handle.kernelPropertiesMap[KernelId::MSAFindCandidatesOfDifferentRegion] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(max_blocks_per_device, n_subjects));

    		#define mycall(blocksize) msa_findCandidatesOfDifferentRegion_kernel<(blocksize)> \
    	        <<<grid, block, 0, stream>>>( \
                    d_shouldBeRemoved, \
                    d_subject_sequences_data, \
                    d_candidate_sequences_data, \
                    d_subject_sequences_lengths, \
                    d_candidate_sequences_lengths, \
                    d_candidates_per_subject_prefixsum, \
                    d_alignment_shifts, \
                    n_subjects, \
                    n_candidates, \
                    max_sequence_bytes, \
                    encodedsequencepitch, \
                    d_consensus, \
                    d_counts, \
                    d_weights, \
                    d_msa_column_properties, \
                    msa_pitch, \
                    msa_weights_pitch, \
                    d_indices, \
                    d_indices_per_subject, \
                    d_indices_per_subject_prefixsum, \
                    dataset_coverage); CUERR;

    	assert(blocksize > 0 && blocksize <= max_block_size);

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







































#if 0



template<int BLOCKSIZE, class GetCandidateLength>
__global__
void make_candidates_per_hq_subject_prefixsum_kernel(
			const MSAColumnProperties* __restrict__ d_msa_column_properties,
			const int* __restrict__ d_indices,
			const int* __restrict__ d_indices_per_subject,
			const int* __restrict__ d_indices_per_subject_prefixsum,
			const int* __restrict__ d_high_quality_subject_indices,
			const int* __restrict__ d_num_high_quality_subject_indices,
			const int* __restrict__ d_alignment_shifts,
            const int* __restrict__ d_candidate_sequences_lengths,
			int* __restrict__ d_candidates_per_hq_subject_prefixsum,
			int n_subjects,
			int n_queries,
			const int* __restrict__ d_num_indices,
			float min_support_threshold,
			float min_coverage_threshold,
			int new_columns_to_correct){

	constexpr int chunksize = 4;

	using BlockScan = cub::BlockScan<int, BLOCKSIZE>;

	__shared__ typename BlockScan::TempStorage temp_storage;

    auto getCandidateLength = [&] (int subjectIndex, int localCandidateIndex){
        const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
        const int index = my_indices[localCandidateIndex];
        const int length = d_candidate_sequences_lengths[index];
        return length;
    };

	const int num_high_quality_subject_indices = *d_num_high_quality_subject_indices;
	const int chunks = SDIV(num_high_quality_subject_indices, chunksize);

	const int loop_end = SDIV(chunks, BLOCKSIZE) * BLOCKSIZE;

	int previous_aggregate = 0;
	for(int index = chunksize * threadIdx.x; index < loop_end; index += chunksize * BLOCKSIZE){
		int my_num_candidates[chunksize];

		#pragma unroll
		for(int i = 0; i < chunksize; i++){
			if(index + i < num_high_quality_subject_indices){
				const int subjectIndex = d_high_quality_subject_indices[index + i];
				my_num_candidates[i] = d_indices_per_subject[subjectIndex];
			}else{
				my_num_candidates[i] = 0;
			}
		}


		int aggregate = 0;
		BlockScan(temp_storage).ExclusiveSum(my_num_candidates, my_num_candidates, aggregate);

		#pragma unroll
		for(int i = 0; i < chunksize; i++){
			if(index + i < num_high_quality_subject_indices){
				d_candidates_per_hq_subject_prefixsum[index] = my_num_candidates[i] + previous_aggregate;
			}
		}

		previous_aggregate = aggregate;
	}

	if(threadIdx.x == 0)
		d_candidates_per_hq_subject_prefixsum[num_high_quality_subject_indices] = previous_aggregate;





	/*for(unsigned index = blockIdx.x; index < num_high_quality_subject_indices; index += gridDim.x) {
		const int subjectIndex = d_high_quality_subject_indices[index];
		const int my_num_candidates = d_indices_per_subject[subjectIndex];

		const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];

		const MSAColumnProperties properties = d_msa_column_properties[subjectIndex];
		const int subjectColumnsBegin_incl = properties.subjectColumnsBegin_incl;
		const int subjectColumnsEnd_excl = properties.subjectColumnsEnd_excl;

		for(int local_candidate_index = 0; local_candidate_index < my_num_candidates; ++local_candidate_index) {
			const int global_candidate_index = my_indices[local_candidate_index];
			const int shift = d_alignment_shifts[global_candidate_index];
			const int candidate_length = getCandidateLength(subjectIndex, local_candidate_index);
			const int queryColumnsBegin_incl = shift - properties.startindex;
			const int queryColumnsEnd_excl = queryColumnsBegin_incl + candidate_length;

			//check range condition and length condition
			if(subjectColumnsBegin_incl - new_columns_to_correct <= queryColumnsBegin_incl
			   && queryColumnsBegin_incl <= subjectColumnsBegin_incl + new_columns_to_correct
			   && queryColumnsEnd_excl <= subjectColumnsEnd_excl + new_columns_to_correct) {

				d_candidate_available_for_correction[] = 1;
			}else{
				d_candidate_available_for_correction[] = 0;
			}
		}
	}*/

#if 0
	const int num_candidates_of_hq_subjects = candidates_per_hq_subject_prefixsum[n_subjects];
	for(int index = threadIdx.x + blockDim.x * blockIdx.x; index < num_candidates_of_hq_subjects; index += blockDim.x * gridDim.x){

		int subjectIndex = 0;
		for(; subjectIndex < n_subjects; subjectIndex++) {
			if(index < candidates_per_hq_subject_prefixsum[subjectIndex+1])
				break;
		}

		const int my_num_candidates = candidates_per_hq_subject_prefixsum[subjectIndex+1] - candidates_per_hq_subject_prefixsum[subjectIndex];
		const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
		const int local_candidate_index = index - candidates_per_hq_subject_prefixsum[subjectIndex];

		const MSAColumnProperties properties = d_msa_column_properties[subjectIndex];
		const int subjectColumnsBegin_incl = properties.subjectColumnsBegin_incl;
		const int subjectColumnsEnd_excl = properties.subjectColumnsEnd_excl;

		const int global_candidate_index = my_indices[local_candidate_index];
		const int shift = d_alignment_shifts[global_candidate_index];
		const int candidate_length = getCandidateLength(subjectIndex, local_candidate_index);
		const int queryColumnsBegin_incl = shift - properties.startindex;
		const int queryColumnsEnd_excl = queryColumnsBegin_incl + candidate_length;

		//check range condition and length condition
		if(subjectColumnsBegin_incl - new_columns_to_correct <= queryColumnsBegin_incl
			&& queryColumnsBegin_incl <= subjectColumnsBegin_incl + new_columns_to_correct
			&& queryColumnsEnd_excl <= subjectColumnsEnd_excl + new_columns_to_correct) {

			d_candidate_available_for_correction[index] = 1;
		}else{
			d_candidate_available_for_correction[index] = 0;
		}
	}
#endif
}

struct candidates_per_hq_subject_transformop{
    const int* d_high_quality_subject_indices = nullptr;
    const int* d_indices_per_subject = nullptr;
    __host__ __device__
    candidates_per_hq_subject_transformop(const int* hqindices, const int* indices_per_subject)
        : d_high_quality_subject_indices(hqindices), d_indices_per_subject(indices_per_subject){}

    __host__ __device__
    int operator()(int index) const{
        const int subjectIndex = d_high_quality_subject_indices[index];
		return d_indices_per_subject[subjectIndex];
    }
};
/*
void make_candidates_per_hq_subject_prefixsum(void* d_temp_storage,
											  size_t& temp_storage_bytes,
											  const int* d_indices_per_subject,
											  const int* d_high_quality_subject_indices,
											  int num_indices,
											  int* prefixsum,
											  cudaStream_t stream){
	candidates_per_hq_subject_transformop transformOp(d_high_quality_subject_indices, d_indices_per_subject);
    using CountIt = cub::CountingInputIterator<int>;
    CountIt countingIter(0);
	cub::TransformInputIterator<int, candidates_per_hq_subject_transformop, CountIt> transformIter(countingIter, transformOp);
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, transformIter, prefixsum+1, num_indices, stream);
}*/

#endif

#endif

}
}
