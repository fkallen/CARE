#include <gpu/kernels.hpp>
//#include <gpu/bestalignment.hpp>
#include <bestalignment.hpp>
#include <gpu/utility_kernels.cuh>
#include <gpu/cubcachingallocator.cuh>

#include <msa.hpp>
#include <sequence.hpp>

#include <shiftedhammingdistance_common.hpp>

#include <hpc_helpers.cuh>
#include <config.hpp>

#include <cassert>


#include <cub/cub.cuh>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <thrust/binary_search.h>


namespace care{
namespace gpu{

    KernelLaunchHandle make_kernel_launch_handle(int deviceId){
        KernelLaunchHandle handle;
        handle.deviceId = deviceId;
        cudaGetDeviceProperties(&handle.deviceProperties, deviceId); CUERR;
        return handle;
    }



    //####################   DEVICE FUNCTIONS #############

    __device__
    inline
    float getQualityWeight(char qualitychar){
        constexpr int ascii_base = 33;
        constexpr float min_weight = 0.001f;

        const int q(qualitychar);
        const float errorprob = exp10f(-(q-ascii_base)/10.0f);

        return max(min_weight, 1.0f - errorprob);
    }


    //####################   KERNELS   ####################


    template<int tilesize>
    __global__
    void
    cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel(
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                const int* __restrict__ candidates_per_subject_prefixsum,
                const int* __restrict__ tiles_per_subject_prefixsum,
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

        /*auto getSubjectPtr = [&] (int subjectIndex){
            const char* result = subject_sequences_data + std::size_t(subjectIndex) * encodedsequencepitch;
            return result;
        };

        auto getCandidatePtr = [&] (int candidateIndex){
            const char* result = candidate_sequences_data + std::size_t(candidateIndex) * encodedsequencepitch;
            return result;
        };*/

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

        for(int logicalTileId = globalTileId; logicalTileId < requiredTiles * 2; logicalTileId += tiles){
            const bool isReverseComplement = logicalTileId >= requiredTiles;
            const int forwardTileId = isReverseComplement ? logicalTileId - requiredTiles : logicalTileId;

            const int subjectIndex = thrust::distance(tiles_per_subject_prefixsum,
                                                    thrust::lower_bound(
                                                        thrust::seq,
                                                        tiles_per_subject_prefixsum,
                                                        tiles_per_subject_prefixsum + n_subjects + 1,
                                                        forwardTileId + 1))-1;

            const int candidatesBeforeThisSubject = candidates_per_subject_prefixsum[subjectIndex];
            const int maxCandidateIndex_excl = candidates_per_subject_prefixsum[subjectIndex+1];
            //const int tilesForThisSubject = tiles_per_subject_prefixsum[subjectIndex + 1] - tiles_per_subject_prefixsum[subjectIndex];
            const int tileForThisSubject = forwardTileId - tiles_per_subject_prefixsum[subjectIndex];
            const int queryIndex = candidatesBeforeThisSubject + tileForThisSubject * tilesize + laneInTile;
            const int resultIndex = isReverseComplement ? queryIndex + n_candidates : queryIndex;

            const int subjectbases = d_sequencePointers.subjectSequencesLength[subjectIndex];
            const char* subjectptr = d_sequencePointers.subjectSequencesData + std::size_t(subjectIndex) * encodedsequencepitch;
            //transposed
            //const char* subjectptr =  (const char*)((unsigned int*)(subject_sequences_data) + std::size_t(subjectIndex));

            //save subject in shared memory (in parallel, per tile)
            for(int lane = laneInTile; lane < max_sequence_ints; lane += tilesize) {
                subjectBackup[identity(lane)] = ((unsigned int*)(subjectptr))[lane];
                //transposed
                //subjectBackup[identity(lane)] = ((unsigned int*)(subjectptr))[lane * n_subjects];
            }

            cg::tiled_partition<tilesize>(cg::this_thread_block()).sync();


            if(queryIndex < maxCandidateIndex_excl){

                const int querybases = d_sequencePointers.candidateSequencesLength[queryIndex];
                //const char* candidateptr = candidate_sequences_data + std::size_t(queryIndex) * encodedsequencepitch;
                //transposed
                const char* candidateptr = (const char*)((unsigned int*)(d_sequencePointers.candidateSequencesDataTransposed) + std::size_t(queryIndex));

                //save query in shared memory
                for(int i = 0; i < max_sequence_ints; i += 1) {
                    //queryBackup[no_bank_conflict_index(i)] = ((unsigned int*)(candidateptr))[i];
                    //transposed
                    queryBackup[no_bank_conflict_index(i)] = ((unsigned int*)(candidateptr))[i * n_candidates];
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

                int* const alignment_scores = d_alignmentresultpointers.scores;
                int* const alignment_overlaps = d_alignmentresultpointers.overlaps;
                int* const alignment_shifts = d_alignmentresultpointers.shifts;
                int* const alignment_nOps = d_alignmentresultpointers.nOps;
                bool* const alignment_isValid = d_alignmentresultpointers.isValid;

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
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                const int* __restrict__ d_candidates_per_subject_prefixsum,
                int n_subjects,
                int n_queries,
                float min_overlap_ratio,
                int min_overlap,
                float estimatedErrorrate){

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

    template<int BLOCKSIZE>
    __global__
    void msa_init_kernel_exp(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                const int* __restrict__ indices,
                const int* __restrict__ indices_per_subject,
                const int* __restrict__ indices_per_subject_prefixsum,
                int n_subjects){

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceInt::TempStorage reduce;
        } temp_storage;

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
            MSAColumnProperties* const properties_ptr = d_msapointers.msaColumnProperties + subjectIndex;

            // We only want to consider the candidates with good alignments. the indices of those were determined in a previous step
            const int num_indices_for_this_subject = indices_per_subject[subjectIndex];

            if(num_indices_for_this_subject > 0){
                const int* const indices_for_this_subject = indices + indices_per_subject_prefixsum[subjectIndex];

                const int subjectLength = d_sequencePointers.subjectSequencesLength[subjectIndex];
                int startindex = 0;
                int endindex = d_sequencePointers.subjectSequencesLength[subjectIndex];

                for(int index = threadIdx.x; index < num_indices_for_this_subject; index += blockDim.x) {
                    const int queryIndex = indices_for_this_subject[index];

                    const int shift = d_alignmentresultpointers.shifts[queryIndex];
                    const BestAlignment_t flag = d_alignmentresultpointers.bestAlignmentFlags[queryIndex];
                    const int queryLength = d_sequencePointers.candidateSequencesLength[queryIndex];

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

                    my_columnproperties.subjectColumnsBegin_incl = max(-startindex, 0);
                    my_columnproperties.subjectColumnsEnd_excl = my_columnproperties.subjectColumnsBegin_incl + subjectLength;
                    my_columnproperties.firstColumn_incl = 0;
                    my_columnproperties.lastColumn_excl = endindex - startindex;

                    *properties_ptr = my_columnproperties;
                }
            }/*else{
                //empty MSA
                if(threadIdx.x == 0) {
                    MSAColumnProperties my_columnproperties;

                    my_columnproperties.subjectColumnsBegin_incl = 0;
                    my_columnproperties.subjectColumnsEnd_excl = 0;
                    my_columnproperties.firstColumn_incl = 0;
                    my_columnproperties.lastColumn_excl = 0;

                    *properties_ptr = my_columnproperties;
                }
            }*/
        }
    }


    __global__
    void msa_update_properties_kernel(
                MSAPointers d_msapointers,
                const int* __restrict__ d_indices_per_subject,
                size_t msa_weights_pitch,
                int n_subjects){

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
            MSAColumnProperties* const properties_ptr = d_msapointers.msaColumnProperties + subjectIndex;
            const int firstColumn_incl = properties_ptr->firstColumn_incl;
            const int lastColumn_excl = properties_ptr->lastColumn_excl;

            // We only want to consider the candidates with good alignments. the indices of those were determined in a previous step
            const int num_indices_for_this_subject = d_indices_per_subject[subjectIndex];

            if(num_indices_for_this_subject > 0){
                const int* const my_coverage = d_msapointers.coverage + subjectIndex * msa_weights_pitch_floats;

                for(int column = threadIdx.x; firstColumn_incl <= column && column < lastColumn_excl-1; column += blockDim.x){
                    assert(my_coverage[column] >= 0);

                    if(my_coverage[column] == 0 && my_coverage[column+1] > 0){
                        properties_ptr->firstColumn_incl = column+1;
                    }

                    if(my_coverage[column] > 0 && my_coverage[column+1] == 0){
                        properties_ptr->lastColumn_excl = column+1;
                    }
                }

            }else{
                //clear MSA
                if(threadIdx.x == 0) {
                    MSAColumnProperties my_columnproperties;

                    my_columnproperties.subjectColumnsBegin_incl = 0;
                    my_columnproperties.subjectColumnsEnd_excl = 0;
                    my_columnproperties.firstColumn_incl = 0;
                    my_columnproperties.lastColumn_excl = 0;

                    *properties_ptr = my_columnproperties;
                }
            }
        }
    }


    __global__
    void msa_add_sequences_kernel_implicit_global(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                ReadQualitiesPointers d_qualityPointers,
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
    			size_t msa_weights_row_pitch,
                bool debug){

        auto get = [] (const char* data, int length, int index){
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
		};

		auto make_unpacked_reverse_complement_inplace = [] (std::uint8_t* sequence, int sequencelength){
			return reverseComplementStringInplace((char*)sequence, sequencelength);
		};

		auto getSubjectPtr = [&] (int subjectIndex){
			const char* result = d_sequencePointers.subjectSequencesData + std::size_t(subjectIndex) * encoded_sequence_pitch;
			return result;
		};

		auto getCandidatePtr = [&] (int candidateIndex){
			const char* result = d_sequencePointers.candidateSequencesData + std::size_t(candidateIndex) * encoded_sequence_pitch;
			return result;
		};

		auto getSubjectQualityPtr = [&] (int subjectIndex){
			const char* result = d_qualityPointers.subjectQualities + std::size_t(subjectIndex) * quality_pitch;
			return result;
		};

		auto getCandidateQualityPtr = [&] (int candidateIndex){
			const char* result = d_qualityPointers.candidateQualities + std::size_t(candidateIndex) * quality_pitch;
			return result;
		};

		auto getSubjectLength = [&] (int subjectIndex){
			const int length = d_sequencePointers.subjectSequencesLength[subjectIndex];
			return length;
		};

		auto getCandidateLength = [&] __device__ (int candidateIndex){
			const int length = d_sequencePointers.candidateSequencesLength[candidateIndex];
			return length;
		};

    	const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);
    	const int n_indices = *d_num_indices;

        //add subjects
    	for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
            if(d_indices_per_subject[subjectIndex] > 0){
        		const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
        		const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
        		const char* const subject = getSubjectPtr(subjectIndex);
        		const char* const subjectQualityScore = getSubjectQualityPtr(subjectIndex);
                const int shift = 0;

                int* const my_coverage = d_msapointers.coverage + subjectIndex * msa_weights_row_pitch_floats;

                //printf("subject: ");
                for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x){
                    const int globalIndex = subjectColumnsBegin_incl + shift + i;
                    const char base = get(subject, subjectLength, i);
                    //printf("%d ", int(base));
                    const float weight = canUseQualityScores ? getQualityWeight(subjectQualityScore[i]) : 1.0f;
                    const int ptrOffset = subjectIndex * 4 * msa_weights_row_pitch_floats + int(base) * msa_weights_row_pitch_floats;
                    atomicAdd(d_msapointers.counts + ptrOffset + globalIndex, 1);
                    atomicAdd(d_msapointers.weights + ptrOffset + globalIndex, weight);
                    atomicAdd(my_coverage + globalIndex, 1);
                }
            }
    	}
        //printf("\n");

        //add candidates
        for(unsigned index = blockIdx.x; index < n_indices; index += gridDim.x) {
            const int queryIndex = d_indices[index];

            const int shift = d_alignmentresultpointers.shifts[queryIndex];
            const BestAlignment_t flag = d_alignmentresultpointers.bestAlignmentFlags[queryIndex];

            //find subjectindex
            int subjectIndex = 0;
            for(; subjectIndex < n_subjects; subjectIndex++) {
                if(queryIndex < d_candidates_per_subject_prefixsum[subjectIndex+1])
                    break;
            }

            if(d_indices_per_subject[subjectIndex] > 0){

                const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

                int* const my_coverage = d_msapointers.coverage + subjectIndex * msa_weights_row_pitch_floats;

                const char* const query = getCandidatePtr(queryIndex);
        		const int queryLength = getCandidateLength(queryIndex);
        		const char* const queryQualityScore = getCandidateQualityPtr(index);

        		const int query_alignment_overlap = d_alignmentresultpointers.overlaps[queryIndex];
        		const int query_alignment_nops = d_alignmentresultpointers.nOps[queryIndex];

        		const float overlapweight = 1.0f - sqrtf(query_alignment_nops
        					/ (query_alignment_overlap * desiredAlignmentMaxErrorRate));

                assert(overlapweight <= 1.0f);
                assert(overlapweight >= 0.0f);

                assert(flag != BestAlignment_t::None);                 // indices should only be pointing to valid alignments
                //printf("candidate %d, shift %d default %d: ", index, shift, defaultcolumnoffset);
        		//copy query into msa
        		if(flag == BestAlignment_t::Forward) {
                    for(int i = threadIdx.x; i < queryLength; i+= blockDim.x){
                        const int globalIndex = defaultcolumnoffset + i;
                        const char base = get(query, queryLength, i);
                        //printf("%d ", int(base));
                        const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[i]) * overlapweight : overlapweight;
                        const int ptrOffset = subjectIndex * 4 * msa_weights_row_pitch_floats + int(base) * msa_weights_row_pitch_floats;
                        atomicAdd(d_msapointers.counts + ptrOffset + globalIndex, 1);
                        atomicAdd(d_msapointers.weights + ptrOffset + globalIndex, weight);
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
                        const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[reverseIndex]) * overlapweight : overlapweight;
                        const int ptrOffset = subjectIndex * 4 * msa_weights_row_pitch_floats + int(revCompl) * msa_weights_row_pitch_floats;
                        atomicAdd(d_msapointers.counts + ptrOffset + globalIndex, 1);
                        atomicAdd(d_msapointers.weights + ptrOffset + globalIndex, weight);
                        atomicAdd(my_coverage + globalIndex, 1);
                    }
                }
            }

    	}
    }



    __global__
    void msa_add_sequences_kernel_implicit_shared(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                ReadQualitiesPointers d_qualityPointers,
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
    			size_t msa_weights_row_pitch,
                bool debug){
//#define transposequal

        // sizeof(float) * 4 * msa_weights_row_pitch_floats // weights
        //+ sizeof(int) * 4 * msa_weights_row_pitch_floats // counts
    	extern __shared__ float sharedmem[];

        if(debug && blockIdx.x == 0 && threadIdx.x == 0) printf("implicit_shared\n");

        auto get = [] (const char* data, int length, int index, auto trafo){
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, trafo);
		};

		auto make_unpacked_reverse_complement_inplace = [] (std::uint8_t* sequence, int sequencelength){
			return reverseComplementStringInplace((char*)sequence, sequencelength);
		};

		auto getSubjectPtr = [&] (int subjectIndex){
			const char* result = d_sequencePointers.subjectSequencesData + std::size_t(subjectIndex) * encoded_sequence_pitch;
			return result;
		};

		auto getCandidatePtr = [&] (int candidateIndex){
			const char* result = d_sequencePointers.candidateSequencesData + std::size_t(candidateIndex) * encoded_sequence_pitch;
			return result;
		};

		auto getSubjectQualityPtr = [&] (int subjectIndex){
			const char* result = d_qualityPointers.subjectQualities + std::size_t(subjectIndex) * quality_pitch;
			return result;
		};
#ifndef transposequal
		auto getCandidateQualityPtr = [&] (int candidateIndex){
			const char* result = d_qualityPointers.candidateQualities + std::size_t(candidateIndex) * quality_pitch;
			return result;
		};
#else
        auto getCandidateQualityPtr = [&] (int candidateIndex){
            const char* result = d_qualityPointers.candidateQualitiesTransposed + std::size_t(candidateIndex);
            return result;
        };
#endif

		auto getSubjectLength = [&] (int subjectIndex){
			const int length = d_sequencePointers.subjectSequencesLength[subjectIndex];
			return length;
		};

		auto getCandidateLength = [&] __device__ (int candidateIndex){
			//const int candidateIndex = d_indices[localCandidateIndex];
			const int length = d_sequencePointers.candidateSequencesLength[candidateIndex];
			return length;
		};

        const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);
        const int smemsizefloats = 4 * msa_weights_row_pitch_floats + 4 * msa_weights_row_pitch_floats;

        float* const shared_weights = sharedmem;
        int* const shared_counts = (int*)(shared_weights + 4 * msa_weights_row_pitch_floats);

    	//const int requiredTiles = n_subjects;//blocks_per_subject_prefixsum[n_subjects];
        const int requiredTiles = blocks_per_subject_prefixsum[n_subjects];
#ifdef transposequal
        const int num_indices = *d_num_indices;
#endif
    	for(int logicalBlockId = blockIdx.x; logicalBlockId < requiredTiles; logicalBlockId += gridDim.x){
            //clear shared memory
            for(int i = threadIdx.x; i < smemsizefloats; i += blockDim.x){
                sharedmem[i] = 0;
            }
            __syncthreads();

            const int subjectIndex = thrust::distance(blocks_per_subject_prefixsum,
                                                    thrust::lower_bound(
                                                        thrust::seq,
                                                        blocks_per_subject_prefixsum,
                                                        blocks_per_subject_prefixsum + n_subjects + 1,
                                                        logicalBlockId + 1))-1;

            if(d_indices_per_subject[subjectIndex] > 0){

                const int blockForThisSubject = logicalBlockId - blocks_per_subject_prefixsum[subjectIndex];

                const int* const indices_for_this_subject = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
                const int id = blockForThisSubject * blockDim.x + threadIdx.x;
                const int maxid_excl = d_indices_per_subject[subjectIndex];
                const int globalIndexlistIndex = d_indices_per_subject_prefixsum[subjectIndex] + id;

        		const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                const int columnsToCheck = d_msapointers.msaColumnProperties[subjectIndex].lastColumn_excl;

                int* const my_coverage = d_msapointers.coverage + subjectIndex * msa_weights_row_pitch_floats;

                //if(size_t(columnsToCheck) > msa_weights_row_pitch_floats){
                //    printf("columnsToCheck %d, msa_weights_row_pitch_floats %lu\n", columnsToCheck, msa_weights_row_pitch_floats);
                    assert(columnsToCheck <= msa_weights_row_pitch_floats);
                //}


                //ensure that the subject is only inserted once, by the first block
                if(blockForThisSubject == 0){
            		const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
            		const char* const subject = getSubjectPtr(subjectIndex);
            		const char* const subjectQualityScore = getSubjectQualityPtr(subjectIndex);

                    for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x){
                        const int shift = 0;
                        const int globalIndex = subjectColumnsBegin_incl + shift + i;
                        const char base = get(subject, subjectLength, i, [](auto i){return i;});

                        const float weight = canUseQualityScores ? getQualityWeight(subjectQualityScore[i]) : 1.0f;
                        const int ptrOffset = int(base) * msa_weights_row_pitch_floats;
                        atomicAdd(shared_counts + ptrOffset + globalIndex, 1);
                        atomicAdd(shared_weights + ptrOffset + globalIndex, weight);
                        atomicAdd(my_coverage + globalIndex, 1);
                    }

                }

                if(id < maxid_excl){
                    const int queryIndex = indices_for_this_subject[id];
                    const int shift = d_alignmentresultpointers.shifts[queryIndex];
                    const BestAlignment_t flag = d_alignmentresultpointers.bestAlignmentFlags[queryIndex];
                    const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

                    const char* const query = getCandidatePtr(queryIndex);
            		const int queryLength = getCandidateLength(queryIndex);
            		const char* const queryQualityScore = getCandidateQualityPtr(globalIndexlistIndex);

            		const int query_alignment_overlap = d_alignmentresultpointers.overlaps[queryIndex];
            		const int query_alignment_nops = d_alignmentresultpointers.nOps[queryIndex];

            		const float overlapweight = 1.0f - sqrtf(query_alignment_nops
            					/ (query_alignment_overlap * desiredAlignmentMaxErrorRate));
                    assert(overlapweight <= 1.0f);
                    assert(overlapweight >= 0.0f);

                    assert(flag != BestAlignment_t::None); // indices should only be pointing to valid alignments

                    if(flag == BestAlignment_t::Forward) {
                        for(int i = 0; i < queryLength; i += 1){
                            const int globalIndex = defaultcolumnoffset + i;
                            const char base = get(query, queryLength, i, [](auto i){return i;});
                            //printf("%d ", int(base));
                            //if(queryQualityScore[i] == '\0'){
                                //assert(queryQualityScore[i] != '\0');
                            //}

#ifndef transposequal
                            const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[i]) * overlapweight : overlapweight;
#else
                            const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[i * num_indices]) * overlapweight : overlapweight;
#endif
                            assert(weight != 0);
                            const int ptrOffset = int(base) * msa_weights_row_pitch_floats;
                            atomicAdd(shared_counts + ptrOffset + globalIndex, 1);
                            atomicAdd(shared_weights + ptrOffset + globalIndex, weight);
                            atomicAdd(my_coverage + globalIndex, 1);
                            // if(debug && (globalIndex == 23 || globalIndex == 35 || globalIndex == 42)){
                            //     printf("globalIndex %d, index %d, queryIndex %d, encodedBase %d, weight %.10f\n", globalIndex, id, queryIndex, base, weight);
                            // }
                        }
            		}else{
                        auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
                            constexpr std::uint8_t mask = 0x03;
                            return (~in & mask);
                        };

                        for(int i = 0; i < queryLength; i += 1){
                            const int reverseIndex = queryLength - 1 - i;
                            const int globalIndex = defaultcolumnoffset + i;
                            const char base = get(query, queryLength, reverseIndex, [](auto i){return i;});
                            const char revCompl = make_reverse_complement_byte(base);
                            //printf("%d ", int(revCompl));

                            //if(queryQualityScore[reverseIndex] == '\0'){
                                //assert(queryQualityScore[reverseIndex] != '\0');
                            //}
#ifndef transposequal
                            const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[reverseIndex]) * overlapweight : overlapweight;
#else
                            const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[reverseIndex*num_indices]) * overlapweight : overlapweight;
#endif

                            assert(weight != 0);
                            const int ptrOffset = int(revCompl) * msa_weights_row_pitch_floats;
                            atomicAdd(shared_counts + ptrOffset + globalIndex, 1);
                            atomicAdd(shared_weights + ptrOffset + globalIndex, weight);
                            atomicAdd(my_coverage + globalIndex, 1);

                            // if(debug && (globalIndex == 23 || globalIndex == 35 || globalIndex == 42)){
                            //     printf("globalIndex %d, index %d, queryIndex %d, reverseIndex %d, encodedBase %d, weight %.10f\n", globalIndex, id, queryIndex, reverseIndex, base, weight);
                            // }
                        }
                    }
                    //printf("\n");
                }

                __syncthreads();

                for(int index = threadIdx.x; index < columnsToCheck; index += blockDim.x){
                    for(int k = 0; k < 4; k++){
                        const int* const srcCounts = shared_counts + k * msa_weights_row_pitch_floats + index;
                        int* const destCounts = d_msapointers.counts + 4 * msa_weights_row_pitch_floats * subjectIndex + k * msa_weights_row_pitch_floats + index;
                        const float* const srcWeights = shared_weights + k * msa_weights_row_pitch_floats + index;
                        float* const destWeights = d_msapointers.weights + 4 * msa_weights_row_pitch_floats * subjectIndex + k * msa_weights_row_pitch_floats + index;
                        atomicAdd(destCounts ,*srcCounts);
                        atomicAdd(destWeights, *srcWeights);

                        // if(debug && (index == 23 || index == 35 || index == 42)){
                        //     printf("globalIndex %d, %.10f\n", index, *srcWeights);
                        // }
                    }
                }

                __syncthreads();
            }
        }
#ifdef transposequal
#undef transposequal
#endif

    }


    __global__
    void msa_add_sequences_kernel_implicit_shared_testwithsubjectselection(
        MSAPointers d_msapointers,
        AlignmentResultPointers d_alignmentresultpointers,
        ReadSequencesPointers d_sequencePointers,
        ReadQualitiesPointers d_qualityPointers,
        const int* __restrict__ d_candidates_per_subject_prefixsum,
        const int* __restrict__ d_active_candidate_indices,
        const int* __restrict__ d_active_candidate_indices_per_subject,
        const int* __restrict__ d_active_candidate_indices_per_subject_prefixsum,
        const int* __restrict__ d_active_subject_indices,
        const int* __restrict__ blocksPerActiveSubjectPrefixsum,
        int n_subjects,
        int n_queries,
        const int* __restrict__ d_num_active_candidate_indices,
        const int* __restrict__ d_num_active_subject_indices,
        bool canUseQualityScores,
        float desiredAlignmentMaxErrorRate,
        int maximum_sequence_length,
        int max_sequence_bytes,
        size_t encoded_sequence_pitch,
        size_t quality_pitch,
        size_t msa_row_pitch,
        size_t msa_weights_row_pitch,
        bool debug){
        //#define transposequal

        // sizeof(float) * 4 * msa_weights_row_pitch_floats // weights
        //+ sizeof(int) * 4 * msa_weights_row_pitch_floats // counts
        extern __shared__ float sharedmem[];

        if(debug && blockIdx.x == 0 && threadIdx.x == 0) printf("implicit_shared\n");

        auto get = [] (const char* data, int length, int index){
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
        };

        auto make_unpacked_reverse_complement_inplace = [] (std::uint8_t* sequence, int sequencelength){
            return reverseComplementStringInplace((char*)sequence, sequencelength);
        };

        auto getSubjectPtr = [&] (int subjectIndex){
            const char* result = d_sequencePointers.subjectSequencesData + std::size_t(subjectIndex) * encoded_sequence_pitch;
            return result;
        };

        auto getCandidatePtr = [&] (int candidateIndex){
            const char* result = d_sequencePointers.candidateSequencesData + std::size_t(candidateIndex) * encoded_sequence_pitch;
            return result;
        };

        auto getSubjectQualityPtr = [&] (int subjectIndex){
            const char* result = d_qualityPointers.subjectQualities + std::size_t(subjectIndex) * quality_pitch;
            return result;
        };
        #ifndef transposequal
        auto getCandidateQualityPtr = [&] (int localCandidateIndex){
            const char* result = d_qualityPointers.candidateQualities + std::size_t(localCandidateIndex) * quality_pitch;
            return result;
        };
        #else
        auto getCandidateQualityPtr = [&] (int candidateIndex){
            const char* result = d_qualityPointers.candidateQualitiesTransposed + std::size_t(candidateIndex);
            return result;
        };
        #endif

        auto getSubjectLength = [&] (int subjectIndex){
            const int length = d_sequencePointers.subjectSequencesLength[subjectIndex];
            return length;
        };

        auto getCandidateLength = [&] __device__ (int candidateIndex){
            //const int candidateIndex = d_active_candidate_indices[localCandidateIndex];
            const int length = d_sequencePointers.candidateSequencesLength[candidateIndex];
            return length;
        };

        const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);
        const int smemsizefloats = 4 * msa_weights_row_pitch_floats + 4 * msa_weights_row_pitch_floats;

        float* const shared_weights = sharedmem;
        int* const shared_counts = (int*)(shared_weights + 4 * msa_weights_row_pitch_floats);

        const int num_active_subject_indices = *d_num_active_subject_indices;

        //const int requiredTiles = n_subjects;//blocksPerActiveSubjectPrefixsum[n_subjects];
        const int requiredTiles = blocksPerActiveSubjectPrefixsum[num_active_subject_indices];
        #ifdef transposequal
        const int num_indices = *d_num_active_candidate_indices;
        #endif
        for(int logicalBlockId = blockIdx.x; logicalBlockId < requiredTiles; logicalBlockId += gridDim.x){
            //clear shared memory
            for(int i = threadIdx.x; i < smemsizefloats; i += blockDim.x){
                sharedmem[i] = 0;
            }
            __syncthreads();

            int subjectindicesIndex = 0;
            for(; subjectindicesIndex < num_active_subject_indices; subjectindicesIndex++) {
                if(logicalBlockId < blocksPerActiveSubjectPrefixsum[subjectindicesIndex+1])
                    break;
            }

            const int subjectIndex = d_active_subject_indices[subjectindicesIndex];

            const int blockForThisSubject = logicalBlockId - blocksPerActiveSubjectPrefixsum[subjectIndex];

            const int* const indices_for_this_subject = d_active_candidate_indices + d_active_candidate_indices_per_subject_prefixsum[subjectIndex];
            //const int indicesBeforeThisSubject = d_active_candidate_indices_per_subject_prefixsum[subjectIndex];
            const int id = blockForThisSubject * blockDim.x + threadIdx.x;
            const int maxid_excl = d_active_candidate_indices_per_subject[subjectIndex];
            const int globalIndexlistIndex = d_active_candidate_indices_per_subject_prefixsum[subjectIndex] + id;


            const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
            const int columnsToCheck = d_msapointers.msaColumnProperties[subjectIndex].lastColumn_excl;

            int* const my_coverage = d_msapointers.coverage + subjectIndex * msa_weights_row_pitch_floats;

            //if(size_t(columnsToCheck) > msa_weights_row_pitch_floats){
            //    printf("columnsToCheck %d, msa_weights_row_pitch_floats %lu\n", columnsToCheck, msa_weights_row_pitch_floats);
            assert(columnsToCheck <= msa_weights_row_pitch_floats);
            //}


            //ensure that the subject is only inserted once, by the first block
            if(blockForThisSubject == 0){
                const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                const char* const subject = getSubjectPtr(subjectIndex);
                const char* const subjectQualityScore = getSubjectQualityPtr(subjectIndex);

                for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x){
                    const int shift = 0;
                    const int globalIndex = subjectColumnsBegin_incl + shift + i;
                    const char base = get(subject, subjectLength, i);

                    const float weight = canUseQualityScores ? getQualityWeight(subjectQualityScore[i]) : 1.0f;
                    const int ptrOffset = int(base) * msa_weights_row_pitch_floats;
                    atomicAdd(shared_counts + ptrOffset + globalIndex, 1);
                    atomicAdd(shared_weights + ptrOffset + globalIndex, weight);
                    atomicAdd(my_coverage + globalIndex, 1);

                }

            }


            if(id < maxid_excl){
                const int queryIndex = indices_for_this_subject[id];
                const int shift = d_alignmentresultpointers.shifts[queryIndex];
                const BestAlignment_t flag = d_alignmentresultpointers.bestAlignmentFlags[queryIndex];
                const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

                const char* const query = getCandidatePtr(queryIndex);
                const int queryLength = getCandidateLength(queryIndex);
                const char* const queryQualityScore = getCandidateQualityPtr(globalIndexlistIndex);

                const int query_alignment_overlap = d_alignmentresultpointers.overlaps[queryIndex];
                const int query_alignment_nops = d_alignmentresultpointers.nOps[queryIndex];

                const float defaultweight = 1.0f - sqrtf(query_alignment_nops
                / (query_alignment_overlap * desiredAlignmentMaxErrorRate));

                assert(flag != BestAlignment_t::None); // indices should only be pointing to valid alignments

                if(flag == BestAlignment_t::Forward) {
                    for(int i = 0; i < queryLength; i += 1){
                        const int globalIndex = defaultcolumnoffset + i;
                        const char base = get(query, queryLength, i);

                        #ifndef transposequal
                        const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[i]) * defaultweight : defaultweight;
                        #else
                        const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[i * num_indices]) * defaultweight : defaultweight;
                        #endif
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

                        #ifndef transposequal
                        const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[reverseIndex]) * defaultweight : defaultweight;
                        #else
                        const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[reverseIndex*num_indices]) * defaultweight : defaultweight;
                        #endif

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
                    int* const destCounts = d_msapointers.counts + 4 * msa_weights_row_pitch_floats * subjectIndex + k * msa_weights_row_pitch_floats + index;
                    const float* const srcWeights = shared_weights + k * msa_weights_row_pitch_floats + index;
                    float* const destWeights = d_msapointers.weights + 4 * msa_weights_row_pitch_floats * subjectIndex + k * msa_weights_row_pitch_floats + index;
                    atomicAdd(destCounts ,*srcCounts);
                    atomicAdd(destWeights, *srcWeights);
                }
            }

            __syncthreads();
        }
        #ifdef transposequal
        #undef transposequal
        #endif

        }



    __global__
    void msa_add_sequences_implicit_singlecol_kernel(
                                    MSAPointers d_msapointers,
                                    AlignmentResultPointers d_alignmentresultpointers,
                                    ReadSequencesPointers d_sequencePointers,
                                    ReadQualitiesPointers d_qualityPointers,
                                    const int* __restrict__ d_candidates_per_subject_prefixsum,
                                    const int* __restrict__ d_indices,
                                    const int* __restrict__ d_indices_per_subject,
                                    const int* __restrict__ d_indices_per_subject_prefixsum,
                                    int n_subjects,
                                    int n_queries,
                                    const int* __restrict__ d_columns_per_subject_prefixsum,
                                    bool canUseQualityScores,
                                    float desiredAlignmentMaxErrorRate,
                                    int maximum_sequence_length,
                                    int max_sequence_bytes,
                                    size_t encoded_sequence_pitch,
                                    size_t quality_pitch,
                                    size_t msa_weights_pitch,
                                    const read_number* d_subject_read_ids,
                                    bool debug){

        constexpr char baseA_enc = 0x00;
        constexpr char baseC_enc = 0x01;
        constexpr char baseG_enc = 0x02;
        constexpr char baseT_enc = 0x03;

        constexpr read_number debugid = 207;

        //if(debug && blockIdx.x == 0 && threadIdx.x == 0) printf("singlecol\n");

        auto get = [] (const char* data, int length, int index){
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
        };


        const int numTotalColumns = d_columns_per_subject_prefixsum[n_subjects];

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        for(int columnid = blockIdx.x * blockDim.x + threadIdx.x; columnid < numTotalColumns; columnid += blockDim.x * gridDim.x){
            int subjectIndex = 0;

            for(; subjectIndex < n_subjects; subjectIndex++) {
                if(columnid < d_columns_per_subject_prefixsum[subjectIndex+1])
                    break;
            }

            if(d_indices_per_subject[subjectIndex] > 0){

                const read_number subjectReadId = d_subject_read_ids[subjectIndex];

                const int columnsBeforeThisSubject = d_columns_per_subject_prefixsum[subjectIndex];
                const int columnForThisSubject = columnid - columnsBeforeThisSubject;

                //if(debug && subjectReadId == debugid){
                //    printf("block %d thread %d : column %d / %d\n", blockIdx.x, threadIdx.x, columnForThisSubject, d_msa_column_properties[subjectIndex].lastColumn_excl);
                //}

                int countsMatrix[4];
                float weightsMatrix[4];
                int coverage;

                #pragma unroll
                for(int i = 0; i < 4; i++){
                    countsMatrix[i] = 0;
                    weightsMatrix[i] = 0;
                }

                coverage = 0;

                const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;

                auto countSequence = [&](const char* sequenceptr, const char* qualptr, int sequenceLength, int shift, float weightFactor, bool forward, bool print, int index, int queryIndex){
                    const int positionInMSABegin = subjectColumnsBegin_incl + shift;
                    const int positionInMSAEnd = subjectColumnsBegin_incl + shift + sequenceLength;

                    if(forward){

                        if(positionInMSABegin <= columnForThisSubject && columnForThisSubject < positionInMSAEnd){
                            const int baseposition = columnForThisSubject - positionInMSABegin;
                            const char encodedBase = get(sequenceptr, sequenceLength, baseposition);

                            countsMatrix[0] += (encodedBase == baseA_enc);
                            countsMatrix[1] += (encodedBase == baseC_enc);
                            countsMatrix[2] += (encodedBase == baseG_enc);
                            countsMatrix[3] += (encodedBase == baseT_enc);

                            const float weight = canUseQualityScores ? getQualityWeight(qualptr[baseposition]) * weightFactor : 1.0f;

                            //if(debug && print){
                            //    printf("%f ", weight);
                            //}
                            weightsMatrix[0] += (encodedBase == baseA_enc) * weight;
                            weightsMatrix[1] += (encodedBase == baseC_enc) * weight;
                            weightsMatrix[2] += (encodedBase == baseG_enc) * weight;
                            weightsMatrix[3] += (encodedBase == baseT_enc) * weight;


                            coverage++;

                            if(debug && subjectReadId == debugid){
                                //printf("globalIndex %d, index %d, queryIndex %d, encodedBase %d, weight %.10f\n", columnForThisSubject, index, queryIndex, encodedBase, weight);
                            }
                        }
                    }else{
                        auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
                            constexpr std::uint8_t mask = 0x03;
                            return (~in & mask);
                        };

                        if(positionInMSABegin <= columnForThisSubject && columnForThisSubject < positionInMSAEnd){
                            const int reverseIndex = sequenceLength - 1 - (columnForThisSubject - positionInMSABegin);
                            const char base = get(sequenceptr, sequenceLength, reverseIndex);
                            const char revCompl = make_reverse_complement_byte(base);

                            countsMatrix[0] += (revCompl == baseA_enc);
                            countsMatrix[1] += (revCompl == baseC_enc);
                            countsMatrix[2] += (revCompl == baseG_enc);
                            countsMatrix[3] += (revCompl == baseT_enc);

                            const float weight = canUseQualityScores ? getQualityWeight(qualptr[reverseIndex]) * weightFactor : weightFactor;

                            weightsMatrix[0] += (revCompl == baseA_enc) * weight;
                            weightsMatrix[1] += (revCompl == baseC_enc) * weight;
                            weightsMatrix[2] += (revCompl == baseG_enc) * weight;
                            weightsMatrix[3] += (revCompl == baseT_enc) * weight;

                            coverage++;

                            if(debug && subjectReadId == debugid){
                                //printf("globalIndex %d, index %d, queryIndex %d, reverseIndex %d, encodedBase %d, weight %.10f\n", columnForThisSubject, index, queryIndex, reverseIndex, base, weight);
                            }
                        }
                    }

                    assert(countsMatrix[0] + countsMatrix[1] + countsMatrix[2] + countsMatrix[3] == coverage);
                };

                //count bases of subject in chunk
                const char* const subjectptr = &d_sequencePointers.subjectSequencesData[encoded_sequence_pitch * subjectIndex];
                const char* const subjectqualptr = &d_qualityPointers.subjectQualities[quality_pitch * subjectIndex];
                const int subjectLength = d_sequencePointers.subjectSequencesLength[subjectIndex];

                countSequence(subjectptr, subjectqualptr, subjectLength, 0, 1.0f, true, true,-1,-1);

                //count bases of candidates in chunk
                const int numIndicesForThisSubject = d_indices_per_subject[subjectIndex];
                const int* const indicesForThisSubject = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
                const char* const qualitiesOfCandidates = d_qualityPointers.candidateQualities + (quality_pitch * d_indices_per_subject_prefixsum[subjectIndex]);

                for(int indexToIndices = 0; indexToIndices < numIndicesForThisSubject; indexToIndices++){
                    const int candidateIndex = indicesForThisSubject[indexToIndices];
                    const char* const candidatePtr = &d_sequencePointers.candidateSequencesData[encoded_sequence_pitch * candidateIndex];
                    const char* const candidatequalPtr = &qualitiesOfCandidates[quality_pitch * indexToIndices];
                    const int candidateLength = d_sequencePointers.candidateSequencesLength[candidateIndex];
                    const int shift = d_alignmentresultpointers.shifts[candidateIndex];
                    const bool forward = d_alignmentresultpointers.bestAlignmentFlags[candidateIndex] == BestAlignment_t::Forward;
                    const int query_alignment_overlap = d_alignmentresultpointers.overlaps[candidateIndex];
                    const int query_alignment_nops = d_alignmentresultpointers.nOps[candidateIndex];

                    const float weightFactor = 1.0f - sqrtf(query_alignment_nops
                                / (query_alignment_overlap * desiredAlignmentMaxErrorRate));

                    //if(debug) printf("candidate %d\n", candidateIndex);
                    countSequence(candidatePtr, candidatequalPtr, candidateLength, shift, weightFactor, forward, false, indexToIndices, candidateIndex);
                }

                //save counts to global memory
                int* const myCountsA = d_msapointers.counts + subjectIndex * 4 * msa_weights_pitch_floats + 0 * msa_weights_pitch_floats;
                int* const myCountsC = d_msapointers.counts + subjectIndex * 4 * msa_weights_pitch_floats + 1 * msa_weights_pitch_floats;
                int* const myCountsG = d_msapointers.counts + subjectIndex * 4 * msa_weights_pitch_floats + 2 * msa_weights_pitch_floats;
                int* const myCountsT = d_msapointers.counts + subjectIndex * 4 * msa_weights_pitch_floats + 3 * msa_weights_pitch_floats;
                float* const myWeightsA = d_msapointers.weights + subjectIndex * 4 * msa_weights_pitch_floats + 0 * msa_weights_pitch_floats;
                float* const myWeightsC = d_msapointers.weights + subjectIndex * 4 * msa_weights_pitch_floats + 1 * msa_weights_pitch_floats;
                float* const myWeightsG = d_msapointers.weights + subjectIndex * 4 * msa_weights_pitch_floats + 2 * msa_weights_pitch_floats;
                float* const myWeightsT = d_msapointers.weights + subjectIndex * 4 * msa_weights_pitch_floats + 3 * msa_weights_pitch_floats;
                int* const myCoverage = d_msapointers.coverage + subjectIndex * msa_weights_pitch_floats;

                myCountsA[columnForThisSubject] = countsMatrix[0];
                myCountsC[columnForThisSubject] = countsMatrix[1];
                myCountsG[columnForThisSubject] = countsMatrix[2];
                myCountsT[columnForThisSubject] = countsMatrix[3];
                myWeightsA[columnForThisSubject] = weightsMatrix[0];
                myWeightsC[columnForThisSubject] = weightsMatrix[1];
                myWeightsG[columnForThisSubject] = weightsMatrix[2];
                myWeightsT[columnForThisSubject] = weightsMatrix[3];
                myCoverage[columnForThisSubject] = coverage;

                /*if(debug && subjectReadId == debugid){
                    printf("globalIndex %d A , %.10f\n", columnForThisSubject, weightsMatrix[0]);
                    printf("globalIndex %d C , %.10f\n", columnForThisSubject, weightsMatrix[1]);
                    printf("globalIndex %d G , %.10f\n", columnForThisSubject, weightsMatrix[2]);
                    printf("globalIndex %d T , %.10f\n", columnForThisSubject, weightsMatrix[3]);
                }*/
            }
        }
    }

    template<int BLOCKSIZE>
    __global__
    void msa_find_consensus_implicit_kernel(
                            MSAPointers d_msapointers,
                            ReadSequencesPointers d_sequencePointers,
                            const int* __restrict__ d_indices_per_subject,
                            int n_subjects,
                            size_t encoded_sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch){

        constexpr char A_enc = 0x00;
        constexpr char C_enc = 0x01;
        constexpr char G_enc = 0x02;
        constexpr char T_enc = 0x03;

        constexpr int blocks_per_msa = 1;

        using BlockReduceFloat = cub::BlockReduce<float, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceFloat::TempStorage floatreduce;
        } temp_storage;

        __shared__ float avgCountPerWeight[4];

        auto get = [] (const char* data, int length, int index){
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
        };

        auto getSubjectPtr = [&] (int subjectIndex){
            const char* result = d_sequencePointers.subjectSequencesData + std::size_t(subjectIndex) * encoded_sequence_pitch;
            return result;
        };

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        const int localBlockId = blockIdx.x % blocks_per_msa;
        //const int n_indices = *d_num_indices;

        //process multiple sequence alignment of each subject
        //for each column in msa, find consensus and support
        for(unsigned subjectIndex = blockIdx.x / blocks_per_msa; subjectIndex < n_subjects; subjectIndex += gridDim.x / blocks_per_msa){
            if(d_indices_per_subject[subjectIndex] > 0){
                const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                const int firstColumn_incl = d_msapointers.msaColumnProperties[subjectIndex].firstColumn_incl;
                const int lastColumn_excl = d_msapointers.msaColumnProperties[subjectIndex].lastColumn_excl;

                assert(lastColumn_excl <= msa_weights_pitch_floats);

                const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                const char* const subject = getSubjectPtr(subjectIndex);

                char* const my_consensus = d_msapointers.consensus + subjectIndex * msa_pitch;
                float* const my_support = d_msapointers.support + subjectIndex * msa_weights_pitch_floats;

                float* const my_orig_weights = d_msapointers.origWeights + subjectIndex * msa_weights_pitch_floats;
                int* const my_orig_coverage = d_msapointers.origCoverages + subjectIndex * msa_weights_pitch_floats;

                const int* const myCountsA = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                const int* const myCountsC = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                const int* const myCountsG = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                const int* const myCountsT = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                const float* const my_weightsA = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                const float* const my_weightsC = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                const float* const my_weightsG = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                const float* const my_weightsT = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                //calculate average count per weight
                float myaverageCountPerWeightA = 0.0f;
                float myaverageCountPerWeightG = 0.0f;
                float myaverageCountPerWeightC = 0.0f;
                float myaverageCountPerWeightT = 0.0f;

                for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                    assert(i < lastColumn_excl);

                    const int ca = myCountsA[i];
                    const int cc = myCountsC[i];
                    const int cg = myCountsG[i];
                    const int ct = myCountsT[i];
                    const float wa = my_weightsA[i];
                    const float wc = my_weightsC[i];
                    const float wg = my_weightsG[i];
                    const float wt = my_weightsT[i];

                    myaverageCountPerWeightA += ca / wa;
                    myaverageCountPerWeightC += cc / wc;
                    myaverageCountPerWeightG += cg / wg;
                    myaverageCountPerWeightT += ct / wt;
                }

                myaverageCountPerWeightA = BlockReduceFloat(temp_storage.floatreduce).Sum(myaverageCountPerWeightA);
                __syncthreads();
                myaverageCountPerWeightC = BlockReduceFloat(temp_storage.floatreduce).Sum(myaverageCountPerWeightC);
                __syncthreads();
                myaverageCountPerWeightG = BlockReduceFloat(temp_storage.floatreduce).Sum(myaverageCountPerWeightG);
                __syncthreads();
                myaverageCountPerWeightT = BlockReduceFloat(temp_storage.floatreduce).Sum(myaverageCountPerWeightT);

                if(threadIdx.x == 0){
                    avgCountPerWeight[0] = myaverageCountPerWeightA / (subjectColumnsEnd_excl - subjectColumnsBegin_incl);
                    avgCountPerWeight[1] = myaverageCountPerWeightC / (subjectColumnsEnd_excl - subjectColumnsBegin_incl);
                    avgCountPerWeight[2] = myaverageCountPerWeightG / (subjectColumnsEnd_excl - subjectColumnsBegin_incl);
                    avgCountPerWeight[3] = myaverageCountPerWeightT / (subjectColumnsEnd_excl - subjectColumnsBegin_incl);
                }
                __syncthreads();

                for(int column = localBlockId * blockDim.x + threadIdx.x; firstColumn_incl <= column && column < lastColumn_excl; column += blocks_per_msa * BLOCKSIZE){
                    const int ca = myCountsA[column];
                    const int cc = myCountsC[column];
                    const int cg = myCountsG[column];
                    const int ct = myCountsT[column];
                    const float wa = my_weightsA[column];
                    const float wc = my_weightsC[column];
                    const float wg = my_weightsG[column];
                    const float wt = my_weightsT[column];

                    char cons = 'F';
                    float consWeight = 0.0f;
                    //float consWeightPerCount = 0.0f;
                    //float weightPerCountSum = 0.0f;
                    //if(ca != 0){
                    if(wa > consWeight){
                        cons = 'A';
                        consWeight = wa;
                        //consWeightPerCount = wa / ca;
                        //weightPerCountSum += wa / ca;
                    }
                    //if(cc != 0 && wc / cc > consWeightPerCount){
                    if(wc > consWeight){
                        cons = 'C';
                        consWeight = wc;
                        //consWeightPerCount = wc / cc;
                        //weightPerCountSum += wc / cc;
                    }
                    //if(cg != 0 && wg / cg > consWeightPerCount){
                    if(wg > consWeight){
                        cons = 'G';
                        consWeight = wg;
                        //consWeightPerCount = wg / cg;
                        //weightPerCountSum += wg / cg;
                    }
                    //if(ct != 0 && wt / ct > consWeightPerCount){
                    if(wt > consWeight){
                        cons = 'T';
                        consWeight = wt;
                        //consWeightPerCount = wt / ct;
                        //weightPerCountSum += wt / ct;
                    }
                    my_consensus[column] = cons;
                    const float columnWeight = wa + wc + wg + wt;
                    if(columnWeight == 0){
                        printf("s %d c %d\n", subjectIndex, column);
                        assert(columnWeight != 0);
                    }
                    //assert(weightPerCountSum != 0);
                    my_support[column] = consWeight / columnWeight;
                    //my_support[column] = consWeightPerCount / weightPerCountSum;


                    if(subjectColumnsBegin_incl <= column && column < subjectColumnsEnd_excl){

                        const int localIndex = column - subjectColumnsBegin_incl;
                        const char subjectbase = get(subject, subjectLength, localIndex);

                        if(subjectbase == A_enc){
                            my_orig_weights[column] = wa;
                            my_orig_coverage[column] = myCountsA[column];
                            //printf("%c", 'A');
                            //printf("%d %d %d %c\n", column, localIndex, my_orig_coverage[column], 'A');
                        }else if(subjectbase == C_enc){
                            my_orig_weights[column] = wc;
                            my_orig_coverage[column] = myCountsC[column];
                            //printf("%c", 'C');
                            //printf("%d %d %d %c\n", column, localIndex, my_orig_coverage[column], 'C');
                        }else if(subjectbase == G_enc){
                            my_orig_weights[column] = wg;
                            my_orig_coverage[column] = myCountsG[column];
                            //printf("%c", 'G');
                            //printf("%d %d %d %c\n", column, localIndex, my_orig_coverage[column], 'G');
                        }else if(subjectbase == T_enc){
                            my_orig_weights[column] = wt;
                            my_orig_coverage[column] = myCountsT[column];
                            //printf("%c", 'T');
                            //printf("%d %d %d %c\n", column, localIndex, my_orig_coverage[column], 'T');
                        }
                    }
                }
                //printf("\n");
            }
        }
    }


    template<int BLOCKSIZE>
    __global__
    void msa_correct_subject_implicit_kernel(
                            MSAPointers msapointers,
                            AlignmentResultPointers alignmentresultpointers,
                            ReadSequencesPointers d_sequencePointers,
                            CorrectionResultPointers d_correctionResultPointers,
                            const int* __restrict__ d_indices,
                            const int* __restrict__ d_indices_per_subject,
                            const int* __restrict__ d_indices_per_subject_prefixsum,
                            int n_subjects,
                            size_t encoded_sequence_pitch,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int maximumSequenceLength,
                            float estimatedErrorrate,
                            float desiredAlignmentMaxErrorRate,
                            float avg_support_threshold,
                            float min_support_threshold,
                            float min_coverage_threshold,
                            float max_coverage_threshold,
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

        __shared__ int numUncorrectedPositions;
        __shared__ int uncorrectedPositions[BLOCKSIZE];
        __shared__ float avgCountPerWeight[4];

        auto get = [] (const char* data, int length, int index){
            //return Sequence_t::get_as_nucleotide(data, length, index);
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
        };

        auto getSubjectPtr = [&] (int subjectIndex){
            const char* result = d_sequencePointers.subjectSequencesData + std::size_t(subjectIndex) * encoded_sequence_pitch;
            return result;
        };

        auto getCandidatePtr = [&] (int candidateIndex){
            const char* result = d_sequencePointers.candidateSequencesData + std::size_t(candidateIndex) * encoded_sequence_pitch;
            return result;
        };

        auto getCandidateLength = [&](int candidateIndex){
            return d_sequencePointers.candidateSequencesLength[candidateIndex];
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
            const int myNumIndices = d_indices_per_subject[subjectIndex];
            if(myNumIndices > 0){

                const float* const my_support = msapointers.support + msa_weights_pitch_floats * subjectIndex;
                const int* const my_coverage = msapointers.coverage + msa_weights_pitch_floats * subjectIndex;
                const int* const my_orig_coverage = msapointers.origCoverages + msa_weights_pitch_floats * subjectIndex;
                const char* const my_consensus = msapointers.consensus + msa_pitch  * subjectIndex;
                char* const my_corrected_subject = d_correctionResultPointers.correctedSubjects + subjectIndex * sequence_pitch;

                const int subjectColumnsBegin_incl = msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                const int subjectColumnsEnd_excl = msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                const int lastColumn_excl = msapointers.msaColumnProperties[subjectIndex].lastColumn_excl;

                float avg_support = 0;
                float min_support = 1.0f;
                //int max_coverage = 0;
                int min_coverage = std::numeric_limits<int>::max();

                for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                    assert(i < lastColumn_excl);

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
                //bool isHQ = true;

                if(threadIdx.x == 0){
                    broadcastbuffer = isHQ;
                    d_correctionResultPointers.isHighQualitySubject[subjectIndex] = isHQ;
                    //printf("%f %f %d %d\n", avg_support, min_support, min_coverage, isHQ);
                }
                __syncthreads();

                isHQ = broadcastbuffer;

                if(isHQ){
                    for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                        //assert(my_consensus[i] == 'A' || my_consensus[i] == 'C' || my_consensus[i] == 'G' || my_consensus[i] == 'T');
                        my_corrected_subject[i - subjectColumnsBegin_incl] = my_consensus[i];
                    }
                    if(threadIdx.x == 0){
                        d_correctionResultPointers.subjectIsCorrected[subjectIndex] = true;
                    }
                }else{

                    const int* const myCountsA = msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                    const int* const myCountsC = msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                    const int* const myCountsG = msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                    const int* const myCountsT = msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                    const float* const myWeightsA = msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                    const float* const myWeightsC = msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                    const float* const myWeightsG = msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                    const float* const myWeightsT = msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;


                    //calculate average count per weight
                    float myaverageCountPerWeightA = 0.0f;
                    float myaverageCountPerWeightG = 0.0f;
                    float myaverageCountPerWeightC = 0.0f;
                    float myaverageCountPerWeightT = 0.0f;

                    for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                        assert(i < lastColumn_excl);

                        const int ca = myCountsA[i];
                        const int cc = myCountsC[i];
                        const int cg = myCountsG[i];
                        const int ct = myCountsT[i];
                        const float wa = myWeightsA[i];
                        const float wc = myWeightsC[i];
                        const float wg = myWeightsG[i];
                        const float wt = myWeightsT[i];

                        myaverageCountPerWeightA += ca / wa;
                        myaverageCountPerWeightC += cc / wc;
                        myaverageCountPerWeightG += cg / wg;
                        myaverageCountPerWeightT += ct / wt;
                    }

                    myaverageCountPerWeightA = BlockReduceFloat(temp_storage.floatreduce).Sum(myaverageCountPerWeightA);
                    __syncthreads();
                    myaverageCountPerWeightC = BlockReduceFloat(temp_storage.floatreduce).Sum(myaverageCountPerWeightC);
                    __syncthreads();
                    myaverageCountPerWeightG = BlockReduceFloat(temp_storage.floatreduce).Sum(myaverageCountPerWeightG);
                    __syncthreads();
                    myaverageCountPerWeightT = BlockReduceFloat(temp_storage.floatreduce).Sum(myaverageCountPerWeightT);

                    if(threadIdx.x == 0){
                        avgCountPerWeight[0] = myaverageCountPerWeightA / (subjectColumnsEnd_excl - subjectColumnsBegin_incl);
                        avgCountPerWeight[1] = myaverageCountPerWeightC / (subjectColumnsEnd_excl - subjectColumnsBegin_incl);
                        avgCountPerWeight[2] = myaverageCountPerWeightG / (subjectColumnsEnd_excl - subjectColumnsBegin_incl);
                        avgCountPerWeight[3] = myaverageCountPerWeightT / (subjectColumnsEnd_excl - subjectColumnsBegin_incl);
                    }
                    __syncthreads();


                    //decode orignal sequence and copy to corrected sequence
                    const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                    const char* const subject = getSubjectPtr(subjectIndex);
                    for(int i = threadIdx.x; i < subjectLength; i += BLOCKSIZE){
                        my_corrected_subject[i] = to_nuc(get(subject, subjectLength, i));
                    }

                    bool foundAColumn = false;
                    int* globalUncorrectedPostitionsPtr = d_correctionResultPointers.uncorrected_positions_per_subject + subjectIndex * maximumSequenceLength;
                    int* const globalNumUncorrectedPositionsPtr = d_correctionResultPointers.num_uncorrected_positions_per_subject + subjectIndex;

                    //round up to next multiple of BLOCKSIZE;
                    const int loopIters = SDIV(subjectLength, BLOCKSIZE) * BLOCKSIZE;
                    for(int loopIter = 0; loopIter < loopIters; loopIter++){
                        if(threadIdx.x == 0){
                            numUncorrectedPositions = 0;
                        }
                        __syncthreads();

                        const int i = threadIdx.x + loopIter * BLOCKSIZE;

                        if(i < subjectLength){
                            const int globalIndex = subjectColumnsBegin_incl + i;
                            const int origCoverage = my_orig_coverage[globalIndex];
                            const char origBase = my_corrected_subject[i];

                            if(origBase != my_consensus[globalIndex]
                                        && my_support[globalIndex] > 0.5f
                                        //&& my_orig_coverage[globalIndex] <= ceil(min_coverage_threshold * 0.5f)+1
                                        //&& origCoverage <= 7//ceil(estimatedErrorrate * my_coverage[globalIndex])
                                        //&& my_orig_coverage[globalIndex] <= 1
                                    ){
                                /*printf("%f %d, %d <= %f\n",
                                        estimatedErrorrate,
                                        my_coverage[globalIndex],
                                        my_orig_coverage[globalIndex],
                                        ceil(estimatedErrorrate * my_coverage[globalIndex]));*/

                                bool canCorrect = true;
                                // if(origCoverage > 2){
                                //     canCorrect = false;
                                // }

                                int numCandidatesWithOrigBaseAndGoodWeight = 0;
                                int numCandidatesWithoutOrigBaseAndGoodWeight = 0;
                                int baseCountsOfHighQualityOverlaps[4]{0};
                                float overlapWeightPerBaseOfHighQualityOverlaps[4]{0};

                                float bestoverlapweightofmatchingbase = -1.0f;
                                float bestoverlapweightofconsensusbase = -1.0f;
                                int numFoundCandidates = 0;

                                if(canCorrect && origCoverage > 1){
                                    const int* myIndices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
                                    char origBase = my_corrected_subject[i];
                                    //iterate over candidates


                                    int origsToFind = origCoverage;

                                    assert(origCoverage <= myNumIndices);

                                    for(int candidatenr = 0; candidatenr < myNumIndices/* && numFoundCandidates < origCoverage*/; candidatenr++){
                                        const int arrayindex = myIndices[candidatenr];

                                        const char* candidateptr = getCandidatePtr(arrayindex);
                                        const int candidateLength = getCandidateLength(arrayindex);
                                        const int candidateShift = alignmentresultpointers.shifts[arrayindex];
                                        const int candidateBasePosition = globalIndex - (subjectColumnsBegin_incl + candidateShift);
                                        if(candidateBasePosition >= 0 && candidateBasePosition < candidateLength){
                                            char candidateBaseEnc = 0xFF;
                                            if(alignmentresultpointers.bestAlignmentFlags[arrayindex] == BestAlignment_t::ReverseComplement){
                                                candidateBaseEnc = get(candidateptr, candidateLength, candidateLength - candidateBasePosition-1);
                                                candidateBaseEnc = (~candidateBaseEnc) & 0x03;
                                            }else{
                                                candidateBaseEnc = get(candidateptr, candidateLength, candidateBasePosition);
                                            }
                                            const char candidateBase = to_nuc(candidateBaseEnc);

                                            const int nOps = alignmentresultpointers.nOps[arrayindex];
                                            const int overlapsize = alignmentresultpointers.overlaps[arrayindex];
                                            const float overlapweight = 1.0f - sqrtf(nOps / (overlapsize * desiredAlignmentMaxErrorRate));
                                            assert(overlapweight <= 1.0f);
                                            assert(overlapweight >= 0.0f);

                                            constexpr float goodOverlapThreshold = 0.70f;

                                            if(origBase == candidateBase){

                                                numFoundCandidates++;
                                                 if(overlapweight >= goodOverlapThreshold){
                                                     numCandidatesWithOrigBaseAndGoodWeight++;

                                                     baseCountsOfHighQualityOverlaps[candidateBaseEnc]++;
                                                     overlapWeightPerBaseOfHighQualityOverlaps[candidateBaseEnc] += overlapweight;
                                                 }else{
                                                     ; //nothing
                                                 }

                                                 bestoverlapweightofmatchingbase = max(bestoverlapweightofmatchingbase, overlapweight);

                                            }else{
                                                if(candidateBase == my_consensus[globalIndex]){
                                                    bestoverlapweightofconsensusbase = max(bestoverlapweightofconsensusbase, overlapweight);
                                                }

                                                if(overlapweight >= goodOverlapThreshold){
                                                    numCandidatesWithoutOrigBaseAndGoodWeight++;

                                                    baseCountsOfHighQualityOverlaps[candidateBaseEnc]++;
                                                    overlapWeightPerBaseOfHighQualityOverlaps[candidateBaseEnc] += overlapweight;
                                                }else{
                                                    ; //nothing
                                                }
                                            }
                                        }
                                    }
                                    assert(numFoundCandidates+1 == origCoverage);
                                }

                                //assert(canCorrect || origCoverage > 2);

                                // if(numFoundCandidates != 0){
                                //     printf("%f %f myNumIndices %d numFoundorigCandidates %d, do correct %d\n",
                                //             bestoverlapweightofconsensusbase, bestoverlapweightofmatchingbase,
                                //             myNumIndices, numFoundCandidates, numCandidatesWithOrigBaseAndGoodWeight == 0);
                                // }



                                if(numCandidatesWithOrigBaseAndGoodWeight == 0){
                                //if(false){



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

                                    if(kregioncoverageisgood && avgsupportkregion >= 1.0f-4*estimatedErrorrate){


                                        float maxweightpercount[2]{0,0};
                                        char cons[2]{'F','F'};

                                        auto sortmaxima = [&](){
                                            auto swap = [](auto& a, auto& b){auto tmp = a; a = b; b = tmp;};

                                            if(maxweightpercount[1] > maxweightpercount[0]){
                                                swap(maxweightpercount[1], maxweightpercount[0]);
                                                swap(cons[1], cons[0]);
                                            }
                                        };

                                        const int ca = myCountsA[i];
                                        const int cc = myCountsC[i];
                                        const int cg = myCountsG[i];
                                        const int ct = myCountsT[i];
                                        const float wa = myWeightsA[i];
                                        const float wc = myWeightsC[i];
                                        const float wg = myWeightsG[i];
                                        const float wt = myWeightsT[i];

                                        if(ca > 0 && wa / ca > maxweightpercount[1]){
                                            maxweightpercount[1] =  wa / ca;
                                            cons[1] = 'A';
                                        }

                                        sortmaxima();

                                        if(cc > 0 && wc / cc > maxweightpercount[1]){
                                            maxweightpercount[1] =  wc / cc;
                                            cons[1] = 'C';
                                        }

                                        sortmaxima();

                                        if(cg > 0 && wg / cg > maxweightpercount[1]){
                                            maxweightpercount[1] =  wg / cg;
                                            cons[1] = 'G';
                                        }

                                        sortmaxima();

                                        if(ct > 0 && wt / ct > maxweightpercount[1]){
                                            maxweightpercount[1] =  wt / ct;
                                            cons[1] = 'T';
                                        }

                                        sortmaxima();

                                        const int validratios = (maxweightpercount[0] > 0) + (maxweightpercount[1] > 0);


                                        assert(validratios > 0);

                                        // if(validratios == 1){
                                        //     my_corrected_subject[i] = my_consensus[globalIndex];
                                        //
                                        //     //printf("%c %f, %c %f. correct to %c. normal cons %c\n", cons[0], maxweightpercount[0], cons[1], maxweightpercount[1], my_consensus[globalIndex], my_consensus[globalIndex]);
                                        //
                                        // }else{
                                        //     assert(validratios == 2);
                                        //
                                        //     constexpr float threshold = 3.0f;
                                        //
                                        //     if(maxweightpercount[0] > maxweightpercount[1] && maxweightpercount[0] / maxweightpercount[1] >= threshold){
                                        //         my_corrected_subject[i] = cons[0];
                                        //         //printf("%c %f, %c %f. correct to %c. normal cons %c\n",
                                        //         //        cons[0], maxweightpercount[0], cons[1], maxweightpercount[1], cons[0], my_consensus[globalIndex]);
                                        //     }else if(maxweightpercount[1] > maxweightpercount[0] && maxweightpercount[1] / maxweightpercount[0] >= threshold){
                                        //         my_corrected_subject[i] = cons[1];
                                        //         //printf("%c %f, %c %f. correct to %c. normal cons %c\n",
                                        //         //        cons[0], maxweightpercount[0], cons[1], maxweightpercount[1],cons[1], my_consensus[globalIndex]);
                                        //     }else{
                                        //         my_corrected_subject[i] = my_consensus[globalIndex];
                                        //         //printf("%c %f, %c %f. correct to %c. normal cons %c\n",
                                        //         //        cons[0], maxweightpercount[0], cons[1], maxweightpercount[1], my_consensus[globalIndex], my_consensus[globalIndex]);
                                        //     }
                                        // }

                                        my_corrected_subject[i] = my_consensus[globalIndex];
                                        foundAColumn = true;
                                    }else{
                                        const int smemindex = atomicAdd(&numUncorrectedPositions, 1);
                                        uncorrectedPositions[smemindex] = i;
                                    }
                                    // my_corrected_subject[i] = my_consensus[globalIndex];
                                    // foundAColumn = true;
                                }else{
                                    //determine base to correct to by comparing weights and counts

                                    // auto swap = [](auto& a, auto& b){auto tmp = a; a = b; b = tmp;};
                                    //
                                    // float maxweights[2]{0,0}; //maximum at [0], second largest at [1]
                                    // float countsofweights[2]{0,0}; //maximum at [0], second largest at [1]
                                    // float avgcounts[2]{0,0};
                                    // char cons[2]{'F','F'};
                                    // if(myWeightsA[globalIndex] > myWeightsC[globalIndex]){
                                    //     maxweights[0] = myWeightsA[globalIndex];
                                    //     countsofweights[0] = myCountsA[globalIndex];
                                    //     cons[0] = 'A';
                                    //     avgcounts[0] = avgCountPerWeight[0];
                                    //     maxweights[1] = myWeightsC[globalIndex];
                                    //     countsofweights[1] = myCountsC[globalIndex];
                                    //     cons[1] = 'C';
                                    //     avgcounts[1] = avgCountPerWeight[1];
                                    // }else{
                                    //     maxweights[1] = myWeightsA[globalIndex];
                                    //     countsofweights[1] = myCountsA[globalIndex];
                                    //     cons[1] = 'A';
                                    //     avgcounts[1] = avgCountPerWeight[0];
                                    //     maxweights[0] = myWeightsC[globalIndex];
                                    //     countsofweights[0] = myCountsC[globalIndex];
                                    //     cons[0] = 'C';
                                    //     avgcounts[1] = avgCountPerWeight[1];
                                    // }
                                    //
                                    // if(myWeightsG[globalIndex] > maxweights[1]){
                                    //     maxweights[1] = myWeightsG[globalIndex];
                                    //     countsofweights[1] = myCountsG[globalIndex];
                                    //     cons[1] = 'G';
                                    //     avgcounts[1] = avgCountPerWeight[2];
                                    // }
                                    //
                                    // if(maxweights[1] > maxweights[0]){
                                    //     swap(maxweights[1], maxweights[0]);
                                    //     swap(countsofweights[1], countsofweights[0]);
                                    //     swap(cons[1], cons[0]);
                                    //     swap(avgcounts[1], avgcounts[0]);
                                    // }
                                    //
                                    // if(myWeightsT[globalIndex] > maxweights[1]){
                                    //     maxweights[1] = myWeightsT[globalIndex];
                                    //     countsofweights[1] = myCountsT[globalIndex];
                                    //     cons[1] = 'T';
                                    //     avgcounts[1] = avgCountPerWeight[3];
                                    // }
                                    //
                                    // if(maxweights[1] > maxweights[0]){
                                    //     swap(maxweights[1], maxweights[0]);
                                    //     swap(countsofweights[1], countsofweights[0]);
                                    //     swap(cons[1], cons[0]);
                                    //     swap(avgcounts[1], avgcounts[0]);
                                    // }
                                    //
                                    // auto getNewBase = [&](){
                                    //     constexpr float threshold = 1.5f;
                                    //     const float r0 = maxweights[0] / countsofweights[0];
                                    //     const float r1 = maxweights[1] / countsofweights[1];
                                    //     //
                                    //     // if((r0 > r1 && r0 / r1 > threshold) || (r1 > r0 && r1 / r0 > threshold)){
                                    //     //     return cons[1];
                                    //     // }else{
                                    //     //     return cons[0];
                                    //     // }
                                    //
                                    //     // if(r0 > r1){
                                    //     //     return cons[0];
                                    //     // }else{
                                    //     //     return cons[1];
                                    //     // }
                                    //
                                    //     if(r0 / avgcounts[0] < threshold || avgcounts[0] / r0 < threshold){
                                    //         if(r1 / avgcounts[1] < threshold || avgcounts[1] / r1 < threshold){
                                    //             return cons[0];
                                    //         }else{
                                    //             if(r0 > r1){
                                    //                 return cons[0];
                                    //             }else{
                                    //                 return cons[1];
                                    //             }
                                    //         }
                                    //     }else{
                                    //         if(r1 / avgcounts[1] < threshold || avgcounts[1] / r1 < threshold){
                                    //             if(r0 > r1){
                                    //                 return cons[0];
                                    //             }else{
                                    //                 return cons[1];
                                    //             }
                                    //         }else{
                                    //             return cons[0];
                                    //         }
                                    //     }
                                    // };

                                    // my_corrected_subject[i] = getNewBase();
                                    // foundAColumn = true;

                                    // int numCandidatesWithOrigBaseAndGoodWeight = 0;
                                    // int numCandidatesWithoutOrigBaseAndGoodWeight = 0;
                                    // int baseCountsOfHighQualityOverlaps[4]{0};
                                    // float overlapWeightPerBaseOfHighQualityOverlaps[4]{0};
                                    //
                                    // float agg = -1.0f;
                                    // char cons = 'F';
                                    //
                                    // auto makeagg = [](int count, float overlapweight){
                                    //     if(count > 0){
                                    //         return overlapweight;
                                    //     }else{
                                    //         return 0.0f;
                                    //     }
                                    // };
                                    //
                                    // auto update = [&](int i, char c){
                                    //     const float newagg = makeagg(baseCountsOfHighQualityOverlaps[i], overlapWeightPerBaseOfHighQualityOverlaps[i]);
                                    //     if(newagg > agg){
                                    //         agg = newagg;
                                    //         cons = c;
                                    //     }
                                    // };
                                    //
                                    // update(0, 'A');
                                    // update(1, 'C');
                                    // update(2, 'G');
                                    // update(3, 'T');
                                    //
                                    // my_corrected_subject[i] = cons;
                                    // foundAColumn = true;

                                    const int smemindex = atomicAdd(&numUncorrectedPositions, 1);
                                    uncorrectedPositions[smemindex] = i;
                                }
                            }
                        }

                        __syncthreads();

                        if(threadIdx.x == 0){
                            *globalNumUncorrectedPositionsPtr += numUncorrectedPositions;
                        }

                        for(int k = threadIdx.x; k < numUncorrectedPositions; k++){
                            globalUncorrectedPostitionsPtr[k] = uncorrectedPositions[k];
                        }
                        globalUncorrectedPostitionsPtr += numUncorrectedPositions;

                        if(loopIter < loopIters - 1){
                            __syncthreads();
                        }
                    }

                    //perform block wide or-reduction on foundAColumn
                    foundAColumn = BlockReduceBool(temp_storage.boolreduce).Reduce(foundAColumn, [](bool a, bool b){return a || b;});
                    __syncthreads();

                    if(threadIdx.x == 0){
                        d_correctionResultPointers.subjectIsCorrected[subjectIndex] = foundAColumn;
                        // for(int k = 0; k < *globalNumUncorrectedPositionsPtr; k++){
                        //     const int count = d_correctionResultPointers.uncorrected_positions_per_subject[subjectIndex * maximumSequenceLength + k];
                        //     //printf("%d ", count);
                        // }
                        //printf("\n");
                    }
                }
            }
        }
    }

    __device__ __forceinline__
    bool checkIfCandidateShouldBeCorrected(const MSAPointers& d_msapointers,
                        const AlignmentResultPointers& d_alignmentresultpointers,
                        const ReadSequencesPointers& d_sequencePointers,
                        const CorrectionResultPointers& d_correctionResultPointers,
                        const int* __restrict__ d_indices,
                        const int* __restrict__ d_indices_per_subject_prefixsum,
                        size_t msa_weights_pitch_floats,
                        float min_support_threshold,
                        float min_coverage_threshold,
                        int new_columns_to_correct,
                        int subjectIndex,
                        int local_candidate_index){

        const float* const my_support = d_msapointers.support + msa_weights_pitch_floats * subjectIndex;
        const int* const my_coverage = d_msapointers.coverage + msa_weights_pitch_floats * subjectIndex;

        const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];

        const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
        const int lastColumn_excl = d_msapointers.msaColumnProperties[subjectIndex].lastColumn_excl;

        const int global_candidate_index = my_indices[local_candidate_index];

        const int shift = d_alignmentresultpointers.shifts[global_candidate_index];
        const int candidate_length = d_sequencePointers.candidateSequencesLength[global_candidate_index];
        const int queryColumnsBegin_incl = subjectColumnsBegin_incl + shift;
        const int queryColumnsEnd_excl = subjectColumnsBegin_incl + shift + candidate_length;

        if(subjectColumnsBegin_incl - new_columns_to_correct <= queryColumnsBegin_incl
           && queryColumnsBegin_incl <= subjectColumnsBegin_incl + new_columns_to_correct
           && queryColumnsEnd_excl <= subjectColumnsEnd_excl + new_columns_to_correct) {

            float newColMinSupport = 1.0f;
            int newColMinCov = std::numeric_limits<int>::max();
            //check new columns left of subject
            for(int columnindex = subjectColumnsBegin_incl - new_columns_to_correct;
                columnindex < subjectColumnsBegin_incl;
                columnindex++) {

                assert(columnindex < lastColumn_excl);
                if(queryColumnsBegin_incl <= columnindex) {
                    newColMinSupport = my_support[columnindex] < newColMinSupport ? my_support[columnindex] : newColMinSupport;
                    newColMinCov = my_coverage[columnindex] < newColMinCov ? my_coverage[columnindex] : newColMinCov;
                }
            }
            //check new columns right of subject
            for(int columnindex = subjectColumnsEnd_excl;
                columnindex < subjectColumnsEnd_excl + new_columns_to_correct
                && columnindex < lastColumn_excl;
                columnindex++) {

                newColMinSupport = my_support[columnindex] < newColMinSupport ? my_support[columnindex] : newColMinSupport;
                newColMinCov = my_coverage[columnindex] < newColMinCov ? my_coverage[columnindex] : newColMinCov;
            }

            bool result = newColMinSupport >= min_support_threshold
                            && newColMinCov >= min_coverage_threshold;

            return result;
        }else{
            return false;
        }

    }


    template<int BLOCKSIZE>
    __global__
    void msa_correct_candidates_kernel(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                CorrectionResultPointers d_correctionResultPointers,
                const int* __restrict__ d_indices,
                const int* __restrict__ d_indices_per_subject,
                const int* __restrict__ d_indices_per_subject_prefixsum,
                int n_subjects,
                int n_queries,
                const int* __restrict__ d_num_indices,
                size_t encoded_sequence_pitch,
                size_t sequence_pitch,
                size_t msa_pitch,
                size_t msa_weights_pitch,
                float min_support_threshold,
                float min_coverage_threshold,
                int new_columns_to_correct){

        auto make_unpacked_reverse_complement_inplace = [] (std::uint8_t* sequence, int sequencelength){
            return reverseComplementStringInplace((char*)sequence, sequencelength);
        };

        auto get = [] (const char* data, int length, int index){
            //return Sequence_t::get_as_nucleotide(data, length, index);
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
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

        auto getCandidatePtr = [&] (int candidateIndex){
            const char* result = d_sequencePointers.candidateSequencesData + std::size_t(candidateIndex) * encoded_sequence_pitch;
            return result;
        };

        __shared__ int numberOfCorrectedCandidatesForThisSubject;

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
        const int num_high_quality_subject_indices = *d_correctionResultPointers.numHighQualitySubjectIndices;
        //const int n_indices = *d_num_indices;

        for(unsigned index = blockIdx.x; index < num_high_quality_subject_indices; index += gridDim.x) {

            if(threadIdx.x == 0){
                numberOfCorrectedCandidatesForThisSubject = 0;
            }
            __syncthreads();

            const int subjectIndex = d_correctionResultPointers.highQualitySubjectIndices[index];
            const int my_num_candidates = d_indices_per_subject[subjectIndex];

            const char* const my_consensus = d_msapointers.consensus + msa_pitch  * subjectIndex;
            const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
            char* const my_corrected_candidates = d_correctionResultPointers.correctedCandidates + d_indices_per_subject_prefixsum[subjectIndex] * sequence_pitch;
            int* const my_indices_of_corrected_candidates = d_correctionResultPointers.indicesOfCorrectedCandidates + d_indices_per_subject_prefixsum[subjectIndex];

            for(int local_candidate_index = threadIdx.x; local_candidate_index < my_num_candidates; local_candidate_index += blockDim.x) {

                const bool canHandleCandidate = checkIfCandidateShouldBeCorrected(
                                                        d_msapointers,
                                                        d_alignmentresultpointers,
                                                        d_sequencePointers,
                                                        d_correctionResultPointers,
                                                        d_indices,
                                                        d_indices_per_subject_prefixsum,
                                                        msa_weights_pitch_floats,
                                                        min_support_threshold,
                                                        min_coverage_threshold,
                                                        new_columns_to_correct,
                                                        subjectIndex,
                                                        local_candidate_index);

                if(canHandleCandidate) {

                    const int destinationindex = atomicAdd(&numberOfCorrectedCandidatesForThisSubject, 1);

                    const int global_candidate_index = my_indices[local_candidate_index];

                    const int shift = d_alignmentresultpointers.shifts[global_candidate_index];
                    const int candidate_length = d_sequencePointers.candidateSequencesLength[global_candidate_index];

                    const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                    //const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                    //const int lastColumn_excl = d_msapointers.msaColumnProperties[subjectIndex].lastColumn_excl;
                    const int queryColumnsBegin_incl = subjectColumnsBegin_incl + shift;
                    const int queryColumnsEnd_excl = subjectColumnsBegin_incl + shift + candidate_length;
                    const int copyposbegin = queryColumnsBegin_incl; //max(queryColumnsBegin_incl, subjectColumnsBegin_incl);
                    const int copyposend = queryColumnsEnd_excl; //min(queryColumnsEnd_excl, subjectColumnsEnd_excl);

                    for(int i = copyposbegin; i < copyposend; i += 1) {
                        my_corrected_candidates[destinationindex * sequence_pitch + (i - queryColumnsBegin_incl)] = my_consensus[i];
                    }

                    // const char* const candidate = getCandidatePtr(global_candidate_index);
                    // for(int i = threadIdx.x; i < copyposbegin - queryColumnsBegin_incl; i += BLOCKSIZE){
                    //     my_corrected_candidates[destinationindex * sequence_pitch + i] = to_nuc(get(candidate, candidate_length, i));
                    // }
                    //
                    // for(int i = copyposend - queryColumnsBegin_incl + threadIdx.x; i < candidate_length; i += BLOCKSIZE){
                    //     my_corrected_candidates[destinationindex * sequence_pitch + i] = to_nuc(get(candidate, candidate_length, i));
                    // }

                    const BestAlignment_t bestAlignmentFlag = d_alignmentresultpointers.bestAlignmentFlags[global_candidate_index];

                        //the forward strand will be returned -> make reverse complement again
                    if(bestAlignmentFlag == BestAlignment_t::ReverseComplement) {
                        make_unpacked_reverse_complement_inplace((std::uint8_t*)(my_corrected_candidates + destinationindex * sequence_pitch), candidate_length);
                    }
                    my_indices_of_corrected_candidates[destinationindex] = global_candidate_index;
                        //printf("subjectIndex %d global_candidate_index %d\n", subjectIndex, global_candidate_index);
                }
            }

            __syncthreads();
            if(threadIdx.x == 0) {
                d_correctionResultPointers.numCorrectedCandidates[subjectIndex] = numberOfCorrectedCandidatesForThisSubject;
            }
        }
    }






    template<int BLOCKSIZE>
    __global__
    void msa_correct_candidates_kernel_new(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                CorrectionResultPointers d_correctionResultPointers,
                const int* __restrict__ d_indices,
                const int* __restrict__ d_indices_per_subject,
                const int* __restrict__ d_indices_per_subject_prefixsum,
                const int* __restrict__ d_candidates_per_hq_subject_prefixsum, // inclusive, with leading zero
                //int* __restrict__ globalCommBuffer, // at least n_subjects elements, must be zero'd
                int n_subjects,
                int n_queries,
                const int* __restrict__ d_num_indices,
                size_t encoded_sequence_pitch,
                size_t sequence_pitch,
                size_t msa_pitch,
                size_t msa_weights_pitch,
                float min_support_threshold,
                float min_coverage_threshold,
                int new_columns_to_correct){

        auto make_unpacked_reverse_complement_inplace = [] (std::uint8_t* sequence, int sequencelength){
            return reverseComplementStringInplace((char*)sequence, sequencelength);
        };

        auto get = [] (const char* data, int length, int index){
            //return Sequence_t::get_as_nucleotide(data, length, index);
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
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

        auto getCandidatePtr = [&] (int candidateIndex){
            const char* result = d_sequencePointers.candidateSequencesData + std::size_t(candidateIndex) * encoded_sequence_pitch;
            return result;
        };

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

        // __shared__ int numCandidatesForSubjectInThisBlockShared[BLOCKSIZE];
        // __shared__ int numCorrectedCandidatesForSubjectInThisBlockShared[BLOCKSIZE];
        // __shared__ int histogram[BLOCKSIZE];
        // __shared__ int hqsubjectIndices[BLOCKSIZE];
        // __shared__ int broadcastbuffer;
        // __shared__ union{
        //     typename BlockReduceInt::TempStorage intreduce;
        // } temp_storage;

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
        const int num_high_quality_subject_indices = *d_correctionResultPointers.numHighQualitySubjectIndices;
        const int num_candidates_of_hq_subjects = d_candidates_per_hq_subject_prefixsum[num_high_quality_subject_indices];

        //round up to next multiple of BLOCKSIZE;
        const int loopEnd = SDIV(num_candidates_of_hq_subjects, BLOCKSIZE) * BLOCKSIZE;

        for(int candidateHQid = threadIdx.x + blockIdx.x * blockDim.x;
                candidateHQid < loopEnd;
                candidateHQid += blockDim.x * gridDim.x){

            //__syncthreads();

            const int hqsubjectIndex = candidateHQid >= num_candidates_of_hq_subjects
                                        ?   std::numeric_limits<int>::max()
                                        :   thrust::distance(d_candidates_per_hq_subject_prefixsum,
                                                thrust::lower_bound(
                                                    thrust::seq,
                                                    d_candidates_per_hq_subject_prefixsum,
                                                    d_candidates_per_hq_subject_prefixsum + num_high_quality_subject_indices + 1,
                                                    candidateHQid + 1))-1;

            // if(candidateHQid < num_candidates_of_hq_subjects){
            //     hqsubjectIndices[threadIdx.x] = hqsubjectIndex;
            //     histogram[threadIdx.x] = 0;
            //     numCandidatesForSubjectInThisBlockShared[threadIdx.x] = 0;
            // }
            // __syncthreads();
            //
            // const int smallestHqsubjectIndexInBlock = hqsubjectIndices[0];
            //
            // //count histogram
            // if(candidateHQid < num_candidates_of_hq_subjects){
            //     atomicAdd(histogram + (hqsubjectIndex - smallestHqsubjectIndexInBlock), 1);
            // }
            //
            // int discontinuity = (threadIdx.x == 0);
            // if(threadIdx.x > 0 && candidateHQid < num_candidates_of_hq_subjects){
            //     discontinuity = (hqsubjectIndices[threadIdx.x] != hqsubjectIndices[threadIdx.x-1]);
            // }
            //
            // int numberOfUniquehqsubjectindices = BlockReduceInt(temp_storage.intreduce).Reduce(discontinuity, cub::Sum{});
            // if(threadIdx.x == 0){
            //     broadcastbuffer = numberOfUniquehqsubjectindices;
            // }
            // __syncthreads();
            // numberOfUniquehqsubjectindices = broadcastbuffer;
            //
            // if(threadIdx.x < numberOfUniquehqsubjectindices){
            //     const int localcount = histogram[threadIdx.x];
            //     const int hqsubindex = smallestHqsubjectIndexInBlock + threadIdx.x;
            //     const int subjectIndex = d_correctionResultPointers.highQualitySubjectIndices[hqsubindex];
            //
            //     assert(subjectIndex < n_subjects);
            //
            //     numCandidatesForSubjectInThisBlockShared[threadIdx.x] = atomicAdd(globalCommBuffer + subjectIndex, localcount);
            // }
            //
            // __syncthreads();

            if(candidateHQid < num_candidates_of_hq_subjects){

                const int subjectIndex = d_correctionResultPointers.highQualitySubjectIndices[hqsubjectIndex];
                const int local_candidate_index = candidateHQid - d_candidates_per_hq_subject_prefixsum[hqsubjectIndex];

                const bool canHandleCandidate = checkIfCandidateShouldBeCorrected(
                                                        d_msapointers,
                                                        d_alignmentresultpointers,
                                                        d_sequencePointers,
                                                        d_correctionResultPointers,
                                                        d_indices,
                                                        d_indices_per_subject_prefixsum,
                                                        msa_weights_pitch_floats,
                                                        min_support_threshold,
                                                        min_coverage_threshold,
                                                        new_columns_to_correct,
                                                        subjectIndex,
                                                        local_candidate_index);

                if(canHandleCandidate) {

                    //assert((hqsubjectIndex - smallestHqsubjectIndexInBlock) < numberOfUniquehqsubjectindices);

                    //const int destinationindex = atomicAdd(numCandidatesForSubjectInThisBlockShared + (hqsubjectIndex - smallestHqsubjectIndexInBlock), 1);
                    //atomicAdd(numCorrectedCandidatesForSubjectInThisBlockShared + (hqsubjectIndex - smallestHqsubjectIndexInBlock), 1);

                    const int destinationindex = atomicAdd(d_correctionResultPointers.numCorrectedCandidates + subjectIndex, 1);

                    const char* const my_consensus = d_msapointers.consensus + msa_pitch  * subjectIndex;
                    const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
                    char* const my_corrected_candidates = d_correctionResultPointers.correctedCandidates + d_indices_per_subject_prefixsum[subjectIndex] * sequence_pitch;
                    int* const my_indices_of_corrected_candidates = d_correctionResultPointers.indicesOfCorrectedCandidates + d_indices_per_subject_prefixsum[subjectIndex];

                    const int global_candidate_index = my_indices[local_candidate_index];
                    const int candidate_length = d_sequencePointers.candidateSequencesLength[global_candidate_index];
                    const int shift = d_alignmentresultpointers.shifts[global_candidate_index];

                    const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                    //const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                    //const int lastColumn_excl = d_msapointers.msaColumnProperties[subjectIndex].lastColumn_excl;
                    const int queryColumnsBegin_incl = subjectColumnsBegin_incl + shift;
                    const int queryColumnsEnd_excl = subjectColumnsBegin_incl + shift + candidate_length;

                    const int copyposbegin = queryColumnsBegin_incl; //max(queryColumnsBegin_incl, subjectColumnsBegin_incl);
                    const int copyposend = queryColumnsEnd_excl; //min(queryColumnsEnd_excl, subjectColumnsEnd_excl);

                    for(int i = copyposbegin; i < copyposend; i += 1) {
                        my_corrected_candidates[destinationindex * sequence_pitch + (i - queryColumnsBegin_incl)] = my_consensus[i];
                    }

                    const BestAlignment_t bestAlignmentFlag = d_alignmentresultpointers.bestAlignmentFlags[global_candidate_index];

                    //the forward strand will be returned -> make reverse complement again
                    if(bestAlignmentFlag == BestAlignment_t::ReverseComplement) {
                        make_unpacked_reverse_complement_inplace((std::uint8_t*)(my_corrected_candidates + destinationindex * sequence_pitch), candidate_length);
                    }

                    my_indices_of_corrected_candidates[destinationindex] = global_candidate_index;
                    //printf("subjectIndex %d global_candidate_index %d\n", subjectIndex, global_candidate_index);
                }

            }
        }
    }


#if 0
    __global__
    void selectCandidatesToCorrect(
                bool* __restrict__ candidateCanBeCorrected,
                int* __restrict__ candidateIndices
                int* __restrict__ subjectIndices,
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                CorrectionResultPointers d_correctionResultPointers,
                const int* __restrict__ d_indices,
                const int* __restrict__ d_indices_per_subject,
                const int* __restrict__ d_indices_per_subject_prefixsum,
                int n_subjects,
                int n_queries,
                const int* __restrict__ d_num_indices,
                size_t msa_weights_pitch,
                float min_support_threshold,
                float min_coverage_threshold,
                int new_columns_to_correct){

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
        const int numIndices = *d_num_indices;

        for(int index = threadIdx.x + blockIdx.x * blockDim.x;
                index < numIndices;
                index += blockDim.x * gridDim.x){

            const int subjectIndex = thrust::distance(d_indices_per_subject_prefixsum,
                                                    thrust::lower_bound(
                                                        thrust::seq,
                                                        d_indices_per_subject_prefixsum,
                                                        d_indices_per_subject_prefixsum + n_subjects + 1,
                                                        index + 1))-1;

            if(d_correctionResultPointers.isHighQualitySubject[subjectIndex]){
                const int local_candidate_index = index - d_indices_per_subject_prefixsum[subjectIndex];
                const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];

                const bool canHandleCandidate = checkIfCandidateShouldBeCorrected(
                                                        d_msapointers,
                                                        d_alignmentresultpointers,
                                                        d_sequencePointers,
                                                        d_correctionResultPointers,
                                                        d_indices,
                                                        d_indices_per_subject_prefixsum,
                                                        msa_weights_pitch_floats,
                                                        min_support_threshold,
                                                        min_coverage_threshold,
                                                        new_columns_to_correct,
                                                        subjectIndex,
                                                        local_candidate_index);

                candidateCanBeCorrected[index] = canHandleCandidate;
                candidateIndices[index] = my_indices[local_candidate_index];
                subjectIndices[index] = subjectIndex;
            }
        }
    }

    selectCandidatesToCorrect(
                bool* __restrict__ candidateCanBeCorrected,
                int* __restrict__ candidateIndices
                int* __restrict__ subjectIndices,
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                CorrectionResultPointers d_correctionResultPointers,
                const int* __restrict__ d_indices,
                const int* __restrict__ d_indices_per_subject,
                const int* __restrict__ d_indices_per_subject_prefixsum,
                int n_subjects,
                int n_queries,
                const int* __restrict__ d_num_indices,
                size_t msa_weights_pitch,
                float min_support_threshold,
                float min_coverage_threshold,
                int new_columns_to_correct);

    DeviceAllocate(d_tempids);
    DeviceAllocate(d_tempids_per_subject);
    DeviceAllocate(d_tempids_per_subject_prefixsum);
    DeviceAllocate(d_tempnumids);
    DeviceAllocate(candidateIndicesToCorrect);
    DeviceAllocate(subjectIndicesToCorrect);

    cub::DeviceSelect::Flagged(dataArrays.d_cub_temp_storage.get(),
                cubTempSize,
                cub::CountingInputIterator<int>(0),
                candidateCanBeCorrected,
                d_tempids,
                d_tempnumids,
                *h_num_indices,
                streams[primary_stream_index]); CUERR;

    call_compact_kernel_async(candidateIndicesToCorrect,
                            candidateIndices,
                            d_tempids,
                            dataArrays.h_num_indices[0],
                            d_tempnumids
                            streams[primary_stream_index]);

    call_compact_kernel_async(subjectIndicesToCorrect,
                            subjectIndices,
                            d_tempids,
                            dataArrays.h_num_indices[0],
                            d_tempnumids
                            streams[primary_stream_index]);

    cub::DeviceHistogram::HistogramRange(dataArrays.d_cub_temp_storage.get(),
                cubTempSize,
                d_tempids,
                d_tempids_per_subject,
                dataArrays.n_subjects+1,
                dataArrays.d_candidates_per_subject_prefixsum.get(),
                dataArrays.n_queries,
                streams[primary_stream_index]); CUERR;

    //make indices per subject prefixsum
    call_set_kernel_async(d_tempids_per_subject_prefixsum,
                            0,
                            0,
                            streams[primary_stream_index]);

    cub::DeviceScan::InclusiveSum(dataArrays.d_cub_temp_storage.get(),
                cubTempSize,
                d_tempids_per_subject,
                d_tempids_per_subject_prefixsum+1,
                dataArrays.n_subjects,
                streams[primary_stream_index]); CUERR;

    template<int BLOCKSIZE>
    __global__
    void msa_correct_candidates_kernel_new2(
                const int* __restrict__ candidateIndicesToCorrect,
                const int* __restrict__ subjectIndicesToCorrect,
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                CorrectionResultPointers d_correctionResultPointers,
                const int* __restrict__ d_indices,
                const int* __restrict__ d_indices_per_subject,
                const int* __restrict__ d_indices_per_subject_prefixsum,
                const int* __restrict__ d_candidates_per_hq_subject_prefixsum, // inclusive, with leading zero
                //int* __restrict__ globalCommBuffer, // at least n_subjects elements, must be zero'd
                int n_subjects,
                int n_queries,
                const int* __restrict__ d_num_indices,
                size_t encoded_sequence_pitch,
                size_t sequence_pitch,
                size_t msa_pitch,
                size_t msa_weights_pitch,
                float min_support_threshold,
                float min_coverage_threshold,
                int new_columns_to_correct){

        auto make_unpacked_reverse_complement_inplace = [] (std::uint8_t* sequence, int sequencelength){
            return reverseComplementStringInplace((char*)sequence, sequencelength);
        };


        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
        const int num_candidates_of_hq_subjects = d_candidates_per_hq_subject_prefixsum[num_high_quality_subject_indices];

        //round up to next multiple of BLOCKSIZE;
        const int loopEnd = SDIV(num_candidates_of_hq_subjects, BLOCKSIZE) * BLOCKSIZE;

        for(int candidateHQid = threadIdx.x + blockIdx.x * blockDim.x;
                candidateHQid < loopEnd;
                candidateHQid += blockDim.x * gridDim.x){

            if(candidateHQid < num_candidates_of_hq_subjects){

                const int global_candidate_index = candidateIndicesToCorrect[candidateHQid];
                const int subjectIndex = subjectIndicesToCorrect[candidateHQid];

                const int subjectIndex = d_correctionResultPointers.highQualitySubjectIndices[hqsubjectIndex];
                const int local_candidate_index = candidateHQid - d_candidates_per_hq_subject_prefixsum[hqsubjectIndex];

                if(canHandleCandidate) {

                    //assert((hqsubjectIndex - smallestHqsubjectIndexInBlock) < numberOfUniquehqsubjectindices);

                    //const int destinationindex = atomicAdd(numCandidatesForSubjectInThisBlockShared + (hqsubjectIndex - smallestHqsubjectIndexInBlock), 1);
                    //atomicAdd(numCorrectedCandidatesForSubjectInThisBlockShared + (hqsubjectIndex - smallestHqsubjectIndexInBlock), 1);

                    const int destinationindex = atomicAdd(d_correctionResultPointers.numCorrectedCandidates + subjectIndex, 1);

                    const char* const my_consensus = d_msapointers.consensus + msa_pitch  * subjectIndex;
                    const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
                    char* const my_corrected_candidates = d_correctionResultPointers.correctedCandidates + d_indices_per_subject_prefixsum[subjectIndex] * sequence_pitch;
                    int* const my_indices_of_corrected_candidates = d_correctionResultPointers.indicesOfCorrectedCandidates + d_indices_per_subject_prefixsum[subjectIndex];

                    const int global_candidate_index = my_indices[local_candidate_index];
                    const int candidate_length = d_sequencePointers.candidateSequencesLength[global_candidate_index];
                    const int shift = d_alignmentresultpointers.shifts[global_candidate_index];

                    const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                    //const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                    //const int lastColumn_excl = d_msapointers.msaColumnProperties[subjectIndex].lastColumn_excl;
                    const int queryColumnsBegin_incl = subjectColumnsBegin_incl + shift;
                    const int queryColumnsEnd_excl = subjectColumnsBegin_incl + shift + candidate_length;

                    const int copyposbegin = queryColumnsBegin_incl; //max(queryColumnsBegin_incl, subjectColumnsBegin_incl);
                    const int copyposend = queryColumnsEnd_excl; //min(queryColumnsEnd_excl, subjectColumnsEnd_excl);

                    for(int i = copyposbegin; i < copyposend; i += 1) {
                        my_corrected_candidates[destinationindex * sequence_pitch + (i - queryColumnsBegin_incl)] = my_consensus[i];
                    }

                    const BestAlignment_t bestAlignmentFlag = d_alignmentresultpointers.bestAlignmentFlags[global_candidate_index];

                    //the forward strand will be returned -> make reverse complement again
                    if(bestAlignmentFlag == BestAlignment_t::ReverseComplement) {
                        make_unpacked_reverse_complement_inplace((std::uint8_t*)(my_corrected_candidates + destinationindex * sequence_pitch), candidate_length);
                    }

                    my_indices_of_corrected_candidates[destinationindex] = global_candidate_index;
                    //printf("subjectIndex %d global_candidate_index %d\n", subjectIndex, global_candidate_index);
                }

            }
        }
    }
#endif

    /*
        This kernel inspects a msa and identifies candidates which could originate
        from a different genome region than the subject.

        the output element shouldBeRemoved[i] indicates whether
        the candidate referred to by d_indices[i] should be removed from the msa
    */


    template<int BLOCKSIZE>
    __global__
    void msa_findCandidatesOfDifferentRegion_kernel(
                        MSAPointers d_msapointers,
                        AlignmentResultPointers d_alignmentresultpointers,
                        ReadSequencesPointers d_sequencePointers,
                        bool* __restrict__ d_shouldBeKept,
                        const int* __restrict__ d_candidates_per_subject_prefixsum,
                        int n_subjects,
                        int n_candidates,
                        int max_sequence_bytes,
                        size_t encodedsequencepitch,
                        size_t msa_pitch,
                        size_t msa_weights_pitch,
                        const int* __restrict__ d_indices,
                        const int* __restrict__ d_indices_per_subject,
                        const int* __restrict__ d_indices_per_subject_prefixsum,
                        float desiredAlignmentMaxErrorRate,
                        int dataset_coverage,
                        const unsigned int* d_readids,
                        bool debug = false){

        auto getNumBytes = [] (int sequencelength){
            return sizeof(unsigned int) * getEncodedNumInts2BitHiLo(sequencelength);
        };

        auto get = [] (const char* data, int length, int index){
            return getEncodedNuc2BitHiLo((const unsigned int*)data, length, index, [](auto i){return i;});
    	};

        auto getSubjectPtr = [&] (int subjectIndex){
            const char* result = d_sequencePointers.subjectSequencesData + std::size_t(subjectIndex) * encodedsequencepitch;
            return result;
        };

        auto getCandidatePtr = [&] (int candidateIndex){
            const char* result = d_sequencePointers.candidateSequencesData + std::size_t(candidateIndex) * encodedsequencepitch;
            return result;
        };

        auto getSubjectLength = [&] (int subjectIndex){
            const int length = d_sequencePointers.subjectSequencesLength[subjectIndex];
            return length;
        };

        auto getCandidateLength = [&] (int candidateIndex){
            const int length = d_sequencePointers.candidateSequencesLength[candidateIndex];
            return length;
        };

        auto is_significant_count = [](int count, int coverage){
            /*if(ceil(estimatedErrorrate * coverage)*2 <= count ){
                return true;
            }
            return false;*/
            if(int(coverage * 0.3f) <= count)
                return true;
            return false;

        };

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
        //const char index_to_base[4]{'A','C','G','T'};

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

            if(debug && threadIdx.x == 0){
                //printf("myNumIndices %d\n", myNumIndices);
            }

            if(myNumIndices > 0){

                const char* subjectptr = getSubjectPtr(subjectIndex);
                const int subjectLength = getSubjectLength(subjectIndex);

                const char* myConsensus = d_msapointers.consensus + subjectIndex * msa_pitch;

                const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;

                //check if subject and consensus differ at at least one position

                bool hasMismatchToConsensus = false;

                for(int pos = threadIdx.x; pos < subjectLength && !hasMismatchToConsensus; pos += blockDim.x){
                    const int column = subjectColumnsBegin_incl + pos;
                    const char consbase = myConsensus[column];
                    const char subjectbase = to_nuc(get(subjectptr, subjectLength, pos));

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

                    const int* const myCountsA = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                    const int* const myCountsC = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                    const int* const myCountsG = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                    const int* const myCountsT = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                    const float* const myWeightsA = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                    const float* const myWeightsC = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                    const float* const myWeightsG = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                    const float* const myWeightsT = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                    const float* const mySupport = d_msapointers.support + subjectIndex * msa_weights_pitch_floats;

                    for(int columnindex = subjectColumnsBegin_incl + threadIdx.x; columnindex < subjectColumnsEnd_excl && !foundColumn; columnindex += blockDim.x){
                        int counts[4];
                        counts[0] = myCountsA[columnindex];
                        counts[1] = myCountsC[columnindex];
                        counts[2] = myCountsG[columnindex];
                        counts[3] = myCountsT[columnindex];

                        float weights[4];
                        weights[0] = myWeightsA[columnindex];
                        weights[1] = myWeightsC[columnindex];
                        weights[2] = myWeightsG[columnindex];
                        weights[3] = myWeightsT[columnindex];

                        const float support = mySupport[columnindex];

                        const char consbase = myConsensus[columnindex];
                        consindex = -1;

                        switch(consbase){
                            case 'A': consindex = 0;break;
                            case 'C': consindex = 1;break;
                            case 'G': consindex = 2;break;
                            case 'T': consindex = 3;break;
                        }

                        // char consensusByCount = 'A';
                        // int maxCount = counts[0];
                        // if(counts[1] > maxCount){
                        //     consensusByCount = 'C';
                        //     maxCount = counts[1];
                        // }
                        // if(counts[2] > maxCount){
                        //     consensusByCount = 'G';
                        //     maxCount = counts[2];
                        // }
                        // if(counts[3] > maxCount){
                        //     consensusByCount = 'T';
                        //     maxCount = counts[3];
                        // }
                        //
                        // if(consbase != consensusByCount){
                        //     printf("bycounts %c %.6f %.6f %.6f %.6f,\nbyweight %c %.6f %.6f %.6f %.6f\n\n",
                        //             consensusByCount, float(counts[0]), float(counts[1]), float(counts[2]), float(counts[3]),
                        //             consbase, weights[0], weights[1], weights[2], weights[3]);
                        // }

                        //find out if there is a non-consensus base with significant coverage
                        int significantBaseIndex = -1;

                        #pragma unroll
                        for(int i = 0; i < 4; i++){
                            if(i != consindex){
                                //const bool significant = is_significant_count(counts[i], dataset_coverage);
                                //const int columnCoverage = counts[0] + counts[1] +counts[2] + counts[3];

                                const bool significant = is_significant_count(counts[i], dataset_coverage);

                                //const bool significant = weights[i] / support >= 0.5f;

                                significantBaseIndex = significant ? i : significantBaseIndex;
                            }
                        }

                        if(significantBaseIndex != -1){
                            foundColumn = true;
                            col = columnindex;
                            foundBaseIndex = significantBaseIndex;

                            // if(debug){
                            //     printf("found col %d, baseIndex %d\n", col, foundBaseIndex);
                            // }
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
                            broadcastbufferint4[0] = 1;
                            broadcastbufferint4[1] = packed.x;
                            broadcastbufferint4[2] = to_nuc(packed.y);
                            broadcastbufferint4[3] = packed.y;
                        }else{
                            broadcastbufferint4[0] = 0;
                        }
                    }

                    __syncthreads();

                    foundColumn = (1 == broadcastbufferint4[0]);
                    col = broadcastbufferint4[1];
                    foundBase = broadcastbufferint4[2];
                    foundBaseIndex = broadcastbufferint4[3];

                    // if(debug && threadIdx.x == 0 /*&& d_readids[subjectIndex] == 207*/){
                    //     printf("reduced: found a column: %d, found col %d, found base %c, baseIndex %d\n", foundColumn, col, foundBase, foundBaseIndex);
                    // }

                    if(foundColumn){

                        //compare found base to original base
                        const char originalbase = to_nuc(get(subjectptr, subjectLength, col - subjectColumnsBegin_incl));

                        /*int counts[4];

                        counts[0] = myCountsA[col];
                        counts[1] = myCountsC[col];
                        counts[2] = myCountsG[col];
                        counts[3] = myCountsT[col];*/

                        auto discard_rows = [&](bool keepMatching){

                            const int indexoffset = d_indices_per_subject_prefixsum[subjectIndex];

                            for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                                const int candidateIndex = myIndices[k];
                                const char* const candidateptr = getCandidatePtr(candidateIndex);
                                const int candidateLength = getCandidateLength(candidateIndex);
                                const int shift = d_alignmentresultpointers.shifts[candidateIndex];
                                const BestAlignment_t alignmentFlag = d_alignmentresultpointers.bestAlignmentFlags[candidateIndex];

                                //check if row is affected by column col
                                const int row_begin_incl = subjectColumnsBegin_incl + shift;
                                const int row_end_excl = row_begin_incl + candidateLength;
                                const bool notAffected = (col < row_begin_incl || row_end_excl <= col);
                                char base = 'F';
                                if(!notAffected){
                                    if(alignmentFlag == BestAlignment_t::Forward){
                                        base = to_nuc(get(candidateptr, candidateLength, (col - row_begin_incl)));
                                    }else{
                                        assert(alignmentFlag == BestAlignment_t::ReverseComplement); //all candidates of MSA must not have alignmentflag None
                                        const char forwardbaseEncoded = get(candidateptr, candidateLength, row_end_excl-1 - col);
                                        base = to_nuc((~forwardbaseEncoded & 0x03));
                                    }
                                }

                                if(notAffected || (!(keepMatching ^ (base == foundBase)))){
                                    d_shouldBeKept[indexoffset + k] = true; //same region
                                }else{
                                    d_shouldBeKept[indexoffset + k] = false; //different region
                                }
                            }
#if 1
                            //check that no candidate which should be removed has very good alignment.
                            //if there is such a candidate, none of the candidates will be removed.
                            bool veryGoodAlignment = false;
                            for(int k = threadIdx.x; k < myNumIndices && !veryGoodAlignment; k += blockDim.x){
                                if(!d_shouldBeKept[indexoffset + k]){
                                    const int candidateIndex = myIndices[k];
                                    const int nOps = d_alignmentresultpointers.nOps[candidateIndex];
                                    const int overlapsize = d_alignmentresultpointers.overlaps[candidateIndex];
                                    const float overlapweight = 1.0f - sqrtf(nOps / (overlapsize * desiredAlignmentMaxErrorRate));
                                    assert(overlapweight <= 1.0f);
                                    assert(overlapweight >= 0.0f);

                                    if(overlapweight >= 0.9f){
                                        veryGoodAlignment = true;
                                    }
                                }
                            }

                            veryGoodAlignment = BlockReduceBool(temp_storage.boolreduce).Reduce(veryGoodAlignment, [](auto l, auto r){return l || r;});

                            if(threadIdx.x == 0){
                                broadcastbufferbool = veryGoodAlignment;
                            }
                            __syncthreads();

                            veryGoodAlignment = broadcastbufferbool;

                            if(veryGoodAlignment){
                                for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                                    d_shouldBeKept[indexoffset + k] = true;
                                }
                            }
#endif


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
                            d_shouldBeKept[indexoffset + k] = true;
                        }
                    }

                }else{
                    //no mismatch between consensus and subject

                    //remove no candidate
                    const int indexoffset = d_indices_per_subject_prefixsum[subjectIndex];

                    for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                        d_shouldBeKept[indexoffset + k] = true;
                    }
                }
            }else{
                ; //nothing to do if there are no candidates in msa
            }
        }

    }









    //####################   KERNEL DISPATCH   ####################


    void call_cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel_async(
    			AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
    			const int* d_candidates_per_subject_prefixsum,
                const int* h_candidates_per_subject,
                const int* d_candidates_per_subject,
    			int n_subjects,
    			int n_queries,
                size_t encodedsequencepitch,
    			int max_sequence_bytes,
    			int min_overlap,
    			float maxErrorRate,
    			float min_overlap_ratio,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

            constexpr int tilesize = 32;

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




        	const int blocksize = 128;
            const int tilesPerBlock = blocksize / tilesize;

            //const int requiredTiles = h_tiles_per_subject_prefixsum[n_subjects];

            int requiredTiles = 0;
            for(int i = 0; i < n_subjects;i++){
                requiredTiles += SDIV(h_candidates_per_subject[i], tilesize);
            }

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
                					cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel<tilesize>, \
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

            #define mycall cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel<tilesize> \
                                    	        <<<grid, block, smem, stream>>>( \
                                        		d_alignmentresultpointers, \
                                                d_sequencePointers, \
                                        		d_candidates_per_subject_prefixsum, \
                                                d_tiles_per_subject_prefixsum, \
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

        	mycall;

    	    #undef mycall

            cubCachingAllocator.DeviceFree(d_tiles_per_subject_prefixsum);  CUERR;
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
            		d_alignmentresultpointers,
                    d_sequencePointers,
            		d_candidates_per_subject_prefixsum,
            		n_subjects,
            		n_queries,
            		min_overlap_ratio,
            		min_overlap,
                    estimatedErrorrate); CUERR;

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


    void call_msa_init_kernel_async_exp(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
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
                <<<grid, block, 0, stream>>>(d_msapointers, \
                                               d_alignmentresultpointers, \
                                               d_sequencePointers, \
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

    void call_msa_update_properties_kernel_async(
                    MSAPointers d_msapointers,
                    const int* d_indices_per_subject,
                    int n_subjects,
                    size_t msa_weights_pitch,
                    cudaStream_t stream,
                    KernelLaunchHandle& handle){

    	const int blocksize = 128;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAUpdateProperties);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    		KernelLaunchConfig kernelLaunchConfig;
    		kernelLaunchConfig.threads_per_block = (blocksize);
    		kernelLaunchConfig.smem = 0;
    		KernelProperties kernelProperties;
    		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM,
            					msa_update_properties_kernel,
            					kernelLaunchConfig.threads_per_block,
                                kernelLaunchConfig.smem); CUERR;
    		mymap[kernelLaunchConfig] = kernelProperties;

    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

    		handle.kernelPropertiesMap[KernelId::MSAUpdateProperties] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(max_blocks_per_device, n_subjects), 1, 1);

        msa_update_properties_kernel<<<grid, block, 0, stream>>>(d_msapointers,
                                                                d_indices_per_subject,
                                                                msa_weights_pitch,
                                                                n_subjects); CUERR;




    }


    void call_msa_add_sequences_kernel_implicit_global_async(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                ReadQualitiesPointers d_qualityPointers,
    			const int* d_candidates_per_subject_prefixsum,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			const int* d_indices_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
                const int* h_num_indices,
    			const int* d_num_indices,
                float expectedAffectedIndicesFraction,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
    			int maximum_sequence_length,
    			int max_sequence_bytes,
                size_t encoded_sequence_pitch,
    			size_t quality_pitch,
    			size_t msa_row_pitch,
    			size_t msa_weights_row_pitch,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle,
                bool debug){

        // set counts, weights, and coverages to zero for subjects with valid indices
        generic_kernel<<<n_subjects, 128, 0, stream>>>([=] __device__ (){
            for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
                if(d_indices_per_subject[subjectIndex] > 0){
                    const size_t msa_weights_pitch_floats = msa_weights_row_pitch / sizeof(float);

                    int* const mycounts = d_msapointers.counts + msa_weights_pitch_floats * 4 * subjectIndex;
                    float* const myweights = d_msapointers.weights + msa_weights_pitch_floats * 4 * subjectIndex;
                    int* const mycoverages = d_msapointers.coverage + msa_weights_pitch_floats * subjectIndex;

                    for(int column = threadIdx.x; column < msa_weights_pitch_floats * 4; column += blockDim.x){
                        mycounts[column] = 0;
                        myweights[column] = 0;
                    }

                    for(int column = threadIdx.x; column < msa_weights_pitch_floats; column += blockDim.x){
                        mycoverages[column] = 0;
                    }
                }
            }
        }); CUERR;

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
        dim3 grid(std::min(std::max(1, int(*h_num_indices * expectedAffectedIndicesFraction)), max_blocks_per_device), 1, 1);
        //dim3 grid(std::min(n_queries, max_blocks_per_device), 1, 1);

    	msa_add_sequences_kernel_implicit_global<<<grid, block, smem, stream>>>(
                                        d_msapointers,
                                        d_alignmentresultpointers,
                                        d_sequencePointers,
                                        d_qualityPointers,
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
                                        msa_weights_row_pitch,
                                        debug); CUERR;
    }


    void call_msa_add_sequences_kernel_implicit_shared_async(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                ReadQualitiesPointers d_qualityPointers,
    			const int* d_candidates_per_subject_prefixsum,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			const int* d_indices_per_subject_prefixsum,
                //const int* d_blocks_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
                const int* h_num_indices,
    			const int* d_num_indices,
                float expectedAffectedIndicesFraction,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
    			int maximum_sequence_length,
    			int max_sequence_bytes,
                size_t encoded_sequence_pitch,
    			size_t quality_pitch,
    			size_t msa_row_pitch,
    			size_t msa_weights_row_pitch,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle,
                bool debug){

        // set counts, weights, and coverages to zero for subjects with valid indices
        generic_kernel<<<n_subjects, 128, 0, stream>>>([=] __device__ (){
            for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
                if(d_indices_per_subject[subjectIndex] > 0){
                    const size_t msa_weights_pitch_floats = msa_weights_row_pitch / sizeof(float);

                    int* const mycounts = d_msapointers.counts + msa_weights_pitch_floats * 4 * subjectIndex;
                    float* const myweights = d_msapointers.weights + msa_weights_pitch_floats * 4 * subjectIndex;
                    int* const mycoverages = d_msapointers.coverage + msa_weights_pitch_floats * subjectIndex;

                    for(int column = threadIdx.x; column < msa_weights_pitch_floats * 4; column += blockDim.x){
                        mycounts[column] = 0;
                        myweights[column] = 0;
                    }

                    for(int column = threadIdx.x; column < msa_weights_pitch_floats; column += blockDim.x){
                        mycoverages[column] = 0;
                    }
                }
            }
        }); CUERR;









//std::cerr << "n_subjects: " << n_subjects << ", n_queries: " << n_queries << ", *h_num_indices: " << *h_num_indices << '\n';
    	const int blocksize = 128;
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


        int* d_blocksPerSubjectPrefixSum;
        cubCachingAllocator.DeviceAllocate((void**)&d_blocksPerSubjectPrefixSum, sizeof(int) * (n_subjects+1), stream);  CUERR;

        // calculate blocks per subject prefixsum
        auto getBlocksPerSubject = [=] __device__ (int indices_for_subject){
            return SDIV(indices_for_subject, blocksize);
        };
        cub::TransformInputIterator<int,decltype(getBlocksPerSubject), const int*>
            d_blocksPerSubject(d_indices_per_subject,
                          getBlocksPerSubject);

        void* tempstorage = nullptr;
        size_t tempstoragesize = 0;

        cub::DeviceScan::InclusiveSum(nullptr,
                    tempstoragesize,
                    d_blocksPerSubject,
                    d_blocksPerSubjectPrefixSum+1,
                    n_subjects,
                    stream); CUERR;

        cubCachingAllocator.DeviceAllocate((void**)&tempstorage, tempstoragesize, stream);  CUERR;

        cub::DeviceScan::InclusiveSum(tempstorage,
                    tempstoragesize,
                    d_blocksPerSubject,
                    d_blocksPerSubjectPrefixSum+1,
                    n_subjects,
                    stream); CUERR;

        cubCachingAllocator.DeviceFree(tempstorage);  CUERR;

        call_set_kernel_async(d_blocksPerSubjectPrefixSum,
                                0,
                                0,
                                stream);


    	dim3 block(blocksize, 1, 1);

        const int blocks = SDIV(std::max(1, int(*h_num_indices * expectedAffectedIndicesFraction)), blocksize);
        //const int blocks = SDIV(n_queries, blocksize);
    	dim3 grid(std::min(blocks, max_blocks_per_device), 1, 1);

        /*if(debug){
            block.x = 1;
            grid.x = 1;
        }*/

    	msa_add_sequences_kernel_implicit_shared<<<grid, block, smem, stream>>>(
                                            d_msapointers,
                                            d_alignmentresultpointers,
                                            d_sequencePointers,
                                            d_qualityPointers,
                                            d_candidates_per_subject_prefixsum,
                                            d_indices,
                                            d_indices_per_subject,
                                            d_indices_per_subject_prefixsum,
                                            d_blocksPerSubjectPrefixSum,
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
                                            msa_weights_row_pitch,
                                            debug); CUERR;

        cubCachingAllocator.DeviceFree(d_blocksPerSubjectPrefixSum); CUERR;
    }



    void call_msa_add_sequences_kernel_implicit_shared_testwithsubjectselection_async(
        MSAPointers d_msapointers,
        AlignmentResultPointers d_alignmentresultpointers,
        ReadSequencesPointers d_sequencePointers,
        ReadQualitiesPointers d_qualityPointers,
        const int* d_candidates_per_subject_prefixsum,
        const int* d_active_candidate_indices,
        const int* d_active_candidate_indices_per_subject,
        const int* d_active_candidate_indices_per_subject_prefixsum,
        const int* d_active_subject_indices,
        int n_subjects,
        int n_queries,
        const int* d_num_active_candidate_indices,
        const int* h_num_active_candidate_indices,
        const int* d_num_active_subject_indices,
        const int* h_num_active_subject_indices,
        bool canUseQualityScores,
        float desiredAlignmentMaxErrorRate,
        int maximum_sequence_length,
        int max_sequence_bytes,
        size_t encoded_sequence_pitch,
        size_t quality_pitch,
        size_t msa_row_pitch,
        size_t msa_weights_row_pitch,
        cudaStream_t stream,
        KernelLaunchHandle& handle,
        bool debug){

        // set counts, weights, and coverages to zero for subjects with valid indices
        generic_kernel<<<n_subjects, 128, 0, stream>>>([=] __device__ (){
            const int numActiveSubjects = *d_num_active_subject_indices;

            for(int subjectindicesIndex = blockIdx.x; subjectindicesIndex < numActiveSubjects; subjectindicesIndex += gridDim.x){
                const int subjectIndex = d_active_subject_indices[subjectindicesIndex];

                const size_t msa_weights_pitch_floats = msa_weights_row_pitch / sizeof(float);

                int* const mycounts = d_msapointers.counts + msa_weights_pitch_floats * 4 * subjectIndex;
                float* const myweights = d_msapointers.weights + msa_weights_pitch_floats * 4 * subjectIndex;
                int* const mycoverages = d_msapointers.coverage + msa_weights_pitch_floats * subjectIndex;

                for(int column = threadIdx.x; column < msa_weights_pitch_floats * 4; column += blockDim.x){
                    mycounts[column] = 0;
                    myweights[column] = 0;
                }

                for(int column = threadIdx.x; column < msa_weights_pitch_floats; column += blockDim.x){
                    mycoverages[column] = 0;
                }
            }
        }); CUERR;









        //std::cerr << "n_subjects: " << n_subjects << ", n_queries: " << n_queries << ", *h_num_indices: " << *h_num_indices << '\n';
        const int blocksize = 128;
        const std::size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);

        //const std::size_t smem = sizeof(char) * maximum_sequence_length + sizeof(float) * maximum_sequence_length;
        const std::size_t smem = sizeof(float) * 4 * msa_weights_row_pitch_floats // weights
        + sizeof(int) * 4 * msa_weights_row_pitch_floats; // counts


        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::MSAAddSequencesImplicitSharedTest);
        if(iter == handle.kernelPropertiesMap.end()) {

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
            KernelLaunchConfig kernelLaunchConfig; \
            kernelLaunchConfig.threads_per_block = (blocksize); \
            kernelLaunchConfig.smem = sizeof(float) * 4 * msa_weights_row_pitch_floats \
            + sizeof(int) * 4 * msa_weights_row_pitch_floats; \
            KernelProperties kernelProperties; \
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
            msa_add_sequences_kernel_implicit_shared_testwithsubjectselection, \
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

            handle.kernelPropertiesMap[KernelId::MSAAddSequencesImplicitSharedTest] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;
            //std::cout << max_blocks_per_device << " = " << handle.deviceProperties.multiProcessorCount << " * " << kernelProperties.max_blocks_per_SM << std::endl;
        }


        int* d_blocksPerActiveSubjectPrefixSum;
        cubCachingAllocator.DeviceAllocate((void**)&d_blocksPerActiveSubjectPrefixSum, sizeof(int) * (n_subjects+1), stream);  CUERR;

        // calculate blocks per subject prefixsum

        auto getBlocksPerActiveSubject = [=] __device__ (int subjectindicesIndex){
            const int subjectIndex = d_active_subject_indices[subjectindicesIndex];
            return SDIV(d_active_candidate_indices_per_subject[subjectIndex], blocksize);
        };
        cub::CountingInputIterator<int> countingIter(0);
        cub::TransformInputIterator<int,decltype(getBlocksPerActiveSubject), cub::CountingInputIterator<int>>
                d_blocksPerActiveSubject(countingIter, getBlocksPerActiveSubject);

        void* tempstorage = nullptr;
        size_t tempstoragesize = 0;

        cub::DeviceScan::InclusiveSum(nullptr,
                                      tempstoragesize,
                                      d_blocksPerActiveSubject,
                                      d_blocksPerActiveSubjectPrefixSum+1,
                                      n_subjects,
                                      stream); CUERR;

        cubCachingAllocator.DeviceAllocate((void**)&tempstorage, tempstoragesize, stream);  CUERR;

        cub::DeviceScan::InclusiveSum(tempstorage,
                                    tempstoragesize,
                                    d_blocksPerActiveSubject,
                                    d_blocksPerActiveSubjectPrefixSum+1,
                                    n_subjects,
                                    stream); CUERR;

        cubCachingAllocator.DeviceFree(tempstorage);  CUERR;

        call_set_kernel_async(d_blocksPerActiveSubjectPrefixSum,
                                0,
                                0,
                                stream);


        dim3 block(blocksize, 1, 1);

        const int blocks = SDIV(*h_num_active_candidate_indices, blocksize);
        dim3 grid(std::min(blocks, max_blocks_per_device), 1, 1);

        msa_add_sequences_kernel_implicit_shared_testwithsubjectselection<<<grid, block, smem, stream>>>(
            d_msapointers,
            d_alignmentresultpointers,
            d_sequencePointers,
            d_qualityPointers,
            d_candidates_per_subject_prefixsum,
            d_active_candidate_indices,
            d_active_candidate_indices_per_subject,
            d_active_candidate_indices_per_subject_prefixsum,
            d_active_subject_indices,
            d_blocksPerActiveSubjectPrefixSum,
            n_subjects,
            n_queries,
            d_num_active_candidate_indices,
            d_num_active_subject_indices,
            canUseQualityScores,
            desiredAlignmentMaxErrorRate,
            maximum_sequence_length,
            max_sequence_bytes,
            encoded_sequence_pitch,
            quality_pitch,
            msa_row_pitch,
            msa_weights_row_pitch,
            debug); CUERR;

        cubCachingAllocator.DeviceFree(d_blocksPerActiveSubjectPrefixSum); CUERR;
    }



    void call_msa_add_sequences_implicit_singlecol_kernel_async(
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            ReadQualitiesPointers d_qualityPointers,
            const int* d_candidates_per_subject_prefixsum,
            const int* d_indices,
            const int* d_indices_per_subject,
            const int* d_indices_per_subject_prefixsum,
            int n_subjects,
            int n_queries,
            bool canUseQualityScores,
            float desiredAlignmentMaxErrorRate,
            int maximum_sequence_length,
            int max_sequence_bytes,
            size_t encoded_sequence_pitch,
            size_t quality_pitch,
            size_t msa_weights_pitch,
            cudaStream_t stream,
            KernelLaunchHandle& handle,
            const read_number* d_subject_read_ids,
            bool debug){

        // set counts, weights, and coverages to zero for subjects with valid indices
        generic_kernel<<<n_subjects, 128, 0, stream>>>([=] __device__ (){
            for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
                if(d_indices_per_subject[subjectIndex] > 0){
                    const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

                    int* const mycounts = d_msapointers.counts + msa_weights_pitch_floats * 4 * subjectIndex;
                    float* const myweights = d_msapointers.weights + msa_weights_pitch_floats * 4 * subjectIndex;
                    int* const mycoverages = d_msapointers.coverage + msa_weights_pitch_floats * subjectIndex;

                    for(int column = threadIdx.x; column < msa_weights_pitch_floats * 4; column += blockDim.x){
                        mycounts[column] = 0;
                        myweights[column] = 0;
                    }

                    for(int column = threadIdx.x; column < msa_weights_pitch_floats; column += blockDim.x){
                        mycoverages[column] = 0;
                    }
                }
            }
        }); CUERR;

        constexpr int blocksize = 128;

        int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = 0;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAAddSequencesImplicitSinglecol);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
            		KernelLaunchConfig kernelLaunchConfig; \
            		kernelLaunchConfig.threads_per_block = (blocksize); \
            		kernelLaunchConfig.smem = 0; \
            		KernelProperties kernelProperties; \
            		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
            					msa_add_sequences_implicit_singlecol_kernel, \
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

    		handle.kernelPropertiesMap[KernelId::MSAAddSequencesImplicitSinglecol] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;
    		//std::cout << max_blocks_per_device << " = " << handle.deviceProperties.multiProcessorCount << " * " << kernelProperties.max_blocks_per_SM << std::endl;
    	}

        auto getColumnsPerSubject = [] __device__ (const MSAColumnProperties& columnProperties){
            return columnProperties.lastColumn_excl;
        };
        cub::TransformInputIterator<int,decltype(getColumnsPerSubject), const MSAColumnProperties*>
            d_columns_per_subject(d_msapointers.msaColumnProperties, getColumnsPerSubject);

        int* d_columns_per_subject_prefixsum;
        cubCachingAllocator.DeviceAllocate((void**)&d_columns_per_subject_prefixsum,
                                            sizeof(int) * (n_subjects + 1),
                                            stream); CUERR;

        void* tempstorage = nullptr;
        size_t tempstoragesize = 0;
        cub::DeviceScan::InclusiveSum(tempstorage,
                    tempstoragesize,
                    d_columns_per_subject,
                    d_columns_per_subject_prefixsum+1,
                    n_subjects,
                    stream); CUERR;

        cubCachingAllocator.DeviceAllocate((void**)&tempstorage,
                                            tempstoragesize,
                                            stream); CUERR;

        cub::DeviceScan::InclusiveSum(tempstorage,
                    tempstoragesize,
                    d_columns_per_subject,
                    d_columns_per_subject_prefixsum+1,
                    n_subjects,
                    stream); CUERR;

        cubCachingAllocator.DeviceFree(tempstorage); CUERR;

        call_set_kernel_async(d_columns_per_subject_prefixsum,
                                0,
                                0,
                                stream); CUERR;

        const int totalNumColumnsUpperBound = n_subjects * msa_weights_pitch / sizeof(float);

        dim3 block(blocksize,1,1);

        int gridsize = std::min(max_blocks_per_device, SDIV(totalNumColumnsUpperBound, blocksize));
        dim3 grid(gridsize,1,1);

        /*if(debug){
            block.x = 1;
            grid.x = 1;
        }*/

        msa_add_sequences_implicit_singlecol_kernel<<<grid, block, 0, stream>>>(
                    d_msapointers,
                    d_alignmentresultpointers,
                    d_sequencePointers,
                    d_qualityPointers,
                    d_candidates_per_subject_prefixsum,
                    d_indices,
                    d_indices_per_subject,
                    d_indices_per_subject_prefixsum,
                    n_subjects,
                    n_queries,
                    d_columns_per_subject_prefixsum,
                    canUseQualityScores,
                    desiredAlignmentMaxErrorRate,
                    maximum_sequence_length,
                    max_sequence_bytes,
                    encoded_sequence_pitch,
                    quality_pitch,
                    msa_weights_pitch,
                    d_subject_read_ids,
                    debug); CUERR;

        cubCachingAllocator.DeviceFree(d_columns_per_subject_prefixsum); CUERR;
    }


    void call_msa_add_sequences_kernel_implicit_async(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                ReadQualitiesPointers d_qualityPointers,
    			const int* d_candidates_per_subject_prefixsum,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			const int* d_indices_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
                const int* h_num_indices,
    			const int* d_num_indices,
                float expectedAffectedIndicesFraction,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
    			int maximum_sequence_length,
    			int max_sequence_bytes,
                size_t encoded_sequence_pitch,
    			size_t quality_pitch,
    			size_t msa_row_pitch,
    			size_t msa_weights_row_pitch,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle,
                bool debug){

        //std::cout << n_subjects << " " << *h_num_indices << " " << n_queries << std::endl;

    #if 0
        call_msa_add_sequences_kernel_implicit_global_async(d_msapointers,
                                                            d_alignmentresultpointers,
                                                            d_sequencePointers,
                                                            d_qualityPointers,
                                                            d_candidates_per_subject_prefixsum,
                                                            d_indices,
                                                            d_indices_per_subject,
                                                            d_indices_per_subject_prefixsum,
                                                            n_subjects,
                                                            n_queries,
                                                            h_num_indices,
                                                            d_num_indices,
                                                            expectedAffectedIndicesFraction,
                                                            canUseQualityScores,
                                                            desiredAlignmentMaxErrorRate,
                                                            maximum_sequence_length,
                                                            max_sequence_bytes,
                                                            encoded_sequence_pitch,
                                                            quality_pitch,
                                                            msa_row_pitch,
                                                            msa_weights_row_pitch,
                                                            stream,
                                                            handle,
                                                            debug); CUERR;
    #else


        call_msa_add_sequences_kernel_implicit_shared_async(d_msapointers,
                                                            d_alignmentresultpointers,
                                                            d_sequencePointers,
                                                            d_qualityPointers,
                                                            d_candidates_per_subject_prefixsum,
                                                            d_indices,
                                                            d_indices_per_subject,
                                                            d_indices_per_subject_prefixsum,
                                                            n_subjects,
                                                            n_queries,
                                                            h_num_indices,
                                                            d_num_indices,
                                                            expectedAffectedIndicesFraction,
                                                            canUseQualityScores,
                                                            desiredAlignmentMaxErrorRate,
                                                            maximum_sequence_length,
                                                            max_sequence_bytes,
                                                            encoded_sequence_pitch,
                                                            quality_pitch,
                                                            msa_row_pitch,
                                                            msa_weights_row_pitch,
                                                            stream,
                                                            handle,
                                                            debug); CUERR;
    #endif
    }


    void call_msa_find_consensus_implicit_kernel_async(
                            MSAPointers d_msapointers,
                            ReadSequencesPointers d_sequencePointers,
                            const int* d_indices_per_subject,
                            int n_subjects,
                            size_t encoded_sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            cudaStream_t stream,
                            KernelLaunchHandle& handle){


        generic_kernel<<<n_subjects, 128, 0, stream>>>([=] __device__ (){
            for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
                if(d_indices_per_subject[subjectIndex] > 0){
                    const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

                    float* const mysupport = d_msapointers.support + msa_weights_pitch_floats * subjectIndex;
                    float* const myorigweights = d_msapointers.origWeights + msa_weights_pitch_floats * subjectIndex;
                    int* const myorigcoverages = d_msapointers.origCoverages + msa_weights_pitch_floats * subjectIndex;
                    char* const myconsensus = d_msapointers.consensus + msa_pitch * subjectIndex;

                    for(int column = threadIdx.x; column < msa_weights_pitch_floats; column += blockDim.x){
                        mysupport[column] = 0;
                        myorigweights[column] = 0;
                        myorigcoverages[column] = 0;
                    }

                    for(int column = threadIdx.x; column < msa_pitch; column += blockDim.x){
                        myconsensus[column] = 0;
                    }
                }
            }
        }); CUERR;

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
                                                                msa_find_consensus_implicit_kernel<blocksize>, \
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

        #define launch(blocksize) \
            dim3 block((blocksize), 1, 1); \
            dim3 grid(std::min(max_blocks_per_device, n_subjects), 1, 1); \
            msa_find_consensus_implicit_kernel<(blocksize)><<<grid, block, 0, stream>>>( \
                                                                d_msapointers, \
                                                                d_sequencePointers, \
                                                                d_indices_per_subject, \
                                                                n_subjects, \
                                                                encoded_sequence_pitch, \
                                                                msa_pitch, \
                                                                msa_weights_pitch); CUERR;

        launch(128);

        #undef launch

    }


    void call_msa_correct_subject_implicit_kernel_async(
                            MSAPointers d_msapointers,
                            AlignmentResultPointers d_alignmentresultpointers,
                            ReadSequencesPointers d_sequencePointers,
                            CorrectionResultPointers d_correctionResultPointers,
                            const int* d_indices,
                            const int* d_indices_per_subject,
                            const int* d_indices_per_subject_prefixsum,
                            int n_subjects,
                            size_t encoded_sequence_pitch,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int maximumSequenceLength,
                            float estimatedErrorrate,
                            float desiredAlignmentMaxErrorRate,
                            float avg_support_threshold,
                            float min_support_threshold,
                            float min_coverage_threshold,
                            float max_coverage_threshold,
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
                                    d_msapointers, \
                                    d_alignmentresultpointers, \
                                    d_sequencePointers, \
                                    d_correctionResultPointers, \
                                    d_indices, \
                                    d_indices_per_subject, \
                                    d_indices_per_subject_prefixsum, \
                                    n_subjects, \
                                    encoded_sequence_pitch, \
                                    sequence_pitch, \
                                    msa_pitch, \
                                    msa_weights_pitch, \
                                    maximumSequenceLength, \
                                    estimatedErrorrate, \
                                    desiredAlignmentMaxErrorRate, \
                                    avg_support_threshold, \
                                    min_support_threshold, \
                                    min_coverage_threshold, \
                                    max_coverage_threshold, \
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







    void call_msa_correct_candidates_kernel_async(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                CorrectionResultPointers d_correctionResultPointers,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			const int* d_indices_per_subject_prefixsum,
    			int n_subjects,
    			int n_queries,
    			const int* d_num_indices,
                size_t encoded_sequence_pitch,
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
    	const int blocksize = 64;// std::min(max_block_size, SDIV(maximum_sequence_length, 32) * 32);
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
    					msa_correct_candidates_kernel<(blocksize)>, \
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

    		#define mycall(blocksize) msa_correct_candidates_kernel<(blocksize)> \
    	        <<<grid, block, 0, stream>>>( \
            d_msapointers, \
            d_alignmentresultpointers, \
            d_sequencePointers, \
            d_correctionResultPointers, \
    		d_indices, \
    		d_indices_per_subject, \
    		d_indices_per_subject_prefixsum, \
    		n_subjects, \
    		n_queries, \
    		d_num_indices, \
            encoded_sequence_pitch, \
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



    void call_msa_correct_candidates_kernel_async_experimental(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                CorrectionResultPointers d_correctionResultPointers,
                CorrectionResultPointers h_correctionResultPointers,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			const int* d_indices_per_subject_prefixsum,
                const int* h_indices_per_subject,
    			int n_subjects,
    			int n_queries,
    			const int* d_num_indices,
                size_t encoded_sequence_pitch,
    			size_t sequence_pitch,
    			size_t msa_pitch,
    			size_t msa_weights_pitch,
    			float min_support_threshold,
    			float min_coverage_threshold,
    			int new_columns_to_correct,
    			int maximum_sequence_length,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

        //constexpr int tilesize = 32;
        const int max_block_size = 256;
        const int blocksize = 64;// std::min(max_block_size, SDIV(maximum_sequence_length, 32) * 32);

        const int* d_highQualitySubjectIndices =  d_correctionResultPointers.highQualitySubjectIndices;

        auto getCandidatesPerHQAnchor = [=] __device__(int hqIndex){
            const int subjectIndex = d_highQualitySubjectIndices[hqIndex];
            return d_indices_per_subject[subjectIndex];
        };

        // auto getTilesPerHQAnchor = [=] __device__ (int hqIndex){
        //     const int numCandidatesOfAnchor = getCandidatesPerHQAnchor(hqIndex);
        //     return SDIV(numCandidatesOfAnchor, tilesize);
        // };

        using CperHQA_t = decltype(getCandidatesPerHQAnchor);
        //using TperHQA_t = decltype(getTilesPerHQAnchor);
        using CountIt = cub::CountingInputIterator<int>;

        void* tempstorage = nullptr;
        size_t tempstoragesize = 0;

        const int numHQSubjects = *(h_correctionResultPointers.numHighQualitySubjectIndices);


        //make prefixsum of number of candidates per high quality subject
        int* d_candidatesPerHQAnchorPrefixSum = nullptr;
        cubCachingAllocator.DeviceAllocate((void**)&d_candidatesPerHQAnchorPrefixSum, sizeof(int) * (numHQSubjects+1), stream);  CUERR;

        cub::TransformInputIterator<int, CperHQA_t, CountIt> transformIter(CountIt{0}, getCandidatesPerHQAnchor);

        cub::DeviceScan::InclusiveSum(nullptr, tempstoragesize, transformIter, d_candidatesPerHQAnchorPrefixSum+1, numHQSubjects, stream);
        cubCachingAllocator.DeviceAllocate((void**)&tempstorage, tempstoragesize, stream);  CUERR;
        cub::DeviceScan::InclusiveSum(tempstorage, tempstoragesize, transformIter, d_candidatesPerHQAnchorPrefixSum+1, numHQSubjects, stream);
        cubCachingAllocator.DeviceFree(tempstorage);  CUERR;

        call_set_kernel_async(d_candidatesPerHQAnchorPrefixSum, 0, 0, stream);

        //make tiles per anchor prefixsum
        // int* d_tilesPerHQAnchorPrefixSum;
        // cubCachingAllocator.DeviceAllocate((void**)&d_tilesPerHQAnchorPrefixSum, sizeof(int) * (numHQSubjects+1), stream);  CUERR;
        //
        // cub::TransformInputIterator<int, TperHQA_t, CountIt> transformIter2(CountIt{0}, getTilesPerHQAnchor);
        //
        // cub::DeviceScan::InclusiveSum(nullptr, tempstoragesize, transformIter2, d_tilesPerHQAnchorPrefixSum+1, numHQSubjects, stream);
        // cubCachingAllocator.DeviceAllocate((void**)&tempstorage, tempstoragesize, stream);  CUERR;
        // cub::DeviceScan::InclusiveSum(tempstorage, tempstoragesize, transformIter2, d_tilesPerHQAnchorPrefixSum+1, numHQSubjects, stream);
        // cubCachingAllocator.DeviceFree(tempstorage);  CUERR;
        //
        // call_set_kernel_async(d_tilesPerHQAnchorPrefixSum, 0, 0, stream);
        //
        // const int blocksize = 128;
        // const int tilesPerBlock = blocksize / tilesize;
        //
        // //const int requiredTiles = h_tiles_per_subject_prefixsum[n_subjects];
        //
        // int requiredTiles = 0;
        // for(int i = 0; i < numHQSubjects; i++){
        //     const int subjectIndex = h_correctionResultPointers.highQualitySubjectIndices[i];
        //     const int numCandidatesOfAnchor = h_indices_per_subject[subjectIndex];
        //     requiredTiles += SDIV(numCandidatesOfAnchor, tilesize);
        // }
        //
        // const int requiredBlocks = SDIV(requiredTiles, tilesPerBlock);


        // int* d_blocksPerHQAnchorPrefixSum;
        // cubCachingAllocator.DeviceAllocate((void**)&d_blocksPerHQAnchorPrefixSum, sizeof(int) * (n_subjects+1), stream);  CUERR;
        //
        // // calculate blocks per subject prefixsum
        // auto getBlocksPerHQAnchor = [=] __device__ (int hqIndex){
        //     const int numCandidatesOfAnchor = getCandidatesPerHQAnchor(hqIndex);
        //     return SDIV(numCandidatesOfAnchor, blocksize);
        // };
        // cub::TransformInputIterator<int,decltype(getBlocksPerHQAnchor), CountIt>
        //     d_blocksPerHQAnchor(CountIt{0}, getBlocksPerHQAnchor);
        //
        // cub::DeviceScan::InclusiveSum(nullptr,
        //             tempstoragesize,
        //             d_blocksPerHQAnchor,
        //             d_blocksPerHQAnchorPrefixSum+1,
        //             numHQSubjects,
        //             stream); CUERR;
        //
        // cubCachingAllocator.DeviceAllocate((void**)&tempstorage, tempstoragesize, stream);  CUERR;
        //
        // cub::DeviceScan::InclusiveSum(tempstorage,
        //             tempstoragesize,
        //             d_blocksPerHQAnchor,
        //             d_blocksPerHQAnchorPrefixSum+1,
        //             numHQSubjects,
        //             stream); CUERR;
        //
        // cubCachingAllocator.DeviceFree(tempstorage);  CUERR;
        //
        // call_set_kernel_async(d_blocksPerHQAnchorPrefixSum,
        //                         0,
        //                         0,
        //                         stream);


    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSACorrectCandidatesExperimental);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
    		KernelLaunchConfig kernelLaunchConfig; \
    		kernelLaunchConfig.threads_per_block = (blocksize); \
    		kernelLaunchConfig.smem = 0; \
    		KernelProperties kernelProperties; \
    		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
    					msa_correct_candidates_kernel_new<(blocksize)>, \
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

    		handle.kernelPropertiesMap[KernelId::MSACorrectCandidatesExperimental] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(max_blocks_per_device, n_subjects));

    		#define mycall(blocksize) msa_correct_candidates_kernel_new<(blocksize)> \
    	        <<<grid, block, 0, stream>>>( \
            d_msapointers, \
            d_alignmentresultpointers, \
            d_sequencePointers, \
            d_correctionResultPointers, \
    		d_indices, \
    		d_indices_per_subject, \
    		d_indices_per_subject_prefixsum, \
            d_candidatesPerHQAnchorPrefixSum, \
    		n_subjects, \
    		n_queries, \
    		d_num_indices, \
            encoded_sequence_pitch, \
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

        //cubCachingAllocator.DeviceFree(d_blocksPerHQAnchorPrefixSum);  CUERR;
        //cubCachingAllocator.DeviceFree(d_tilesPerHQAnchorPrefixSum);  CUERR;
        cubCachingAllocator.DeviceFree(d_candidatesPerHQAnchorPrefixSum);  CUERR;
    }


    void call_msa_findCandidatesOfDifferentRegion_kernel_async(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                bool* d_shouldBeKept,
                const int* d_candidates_per_subject_prefixsum,
                int n_subjects,
                int n_candidates,
                int max_sequence_bytes,
                size_t encodedsequencepitch,
                size_t msa_pitch,
                size_t msa_weights_pitch,
                const int* d_indices,
                const int* d_indices_per_subject,
                const int* d_indices_per_subject_prefixsum,
                float desiredAlignmentMaxErrorRate,
                int dataset_coverage,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle,
                const unsigned int* d_readids,
                bool debug){


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
                    d_msapointers, \
                    d_alignmentresultpointers, \
                    d_sequencePointers, \
                    d_shouldBeKept, \
                    d_candidates_per_subject_prefixsum, \
                    n_subjects, \
                    n_candidates, \
                    max_sequence_bytes, \
                    encodedsequencepitch, \
                    msa_pitch, \
                    msa_weights_pitch, \
                    d_indices, \
                    d_indices_per_subject, \
                    d_indices_per_subject_prefixsum, \
                    desiredAlignmentMaxErrorRate, \
                    dataset_coverage, \
                    d_readids, \
                    debug); CUERR;

    	assert(blocksize > 0 && blocksize <= max_block_size);

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



}
}
