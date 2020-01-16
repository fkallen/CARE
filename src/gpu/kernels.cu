//#define NDEBUG

#include <gpu/kernels.hpp>
#include <gpu/devicefunctionsforkernels.cuh>

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








    template<int tilesize>
    __global__
    void
    cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_new_kernel(
                const unsigned int* subjectDataHiLo,
                const unsigned int* candidateDataHiLoTransposed,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                const int* __restrict__ candidates_per_subject_prefixsum,
                const int* __restrict__ tiles_per_subject_prefixsum,
                int n_subjects,
                int n_candidates,
                int maximumSequenceLength,
                int min_overlap,
                float maxErrorRate,
                float min_overlap_ratio){

        const int maximumNumberOfIntsPerSequence = getEncodedNumInts2BitHiLo(maximumSequenceLength);

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

        // sizeof(char) * (max_sequence_bytes * num_tiles   // tiles share the subject
        //                    + max_sequence_bytes * num_threads // each thread works with its own candidate
        //                    + max_sequence_bytes * num_threads) // each thread needs memory to shift a sequence
        extern __shared__ unsigned int sharedmemory[];

        //set up shared memory pointers

        const int max_sequence_ints = getEncodedNumInts2BitHiLo(maximumSequenceLength);

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

            const unsigned int* subjectptr = subjectDataHiLo + std::size_t(subjectIndex) * maximumNumberOfIntsPerSequence;
            //transposed
            //const char* subjectptr =  (const char*)((unsigned int*)(subject_sequences_data) + std::size_t(subjectIndex));

            //save subject in shared memory (in parallel, per tile)
            for(int lane = laneInTile; lane < max_sequence_ints; lane += tilesize) {
                subjectBackup[identity(lane)] = subjectptr[lane];
                //transposed
                //subjectBackup[identity(lane)] = ((unsigned int*)(subjectptr))[lane * n_subjects];
            }

            cg::tiled_partition<tilesize>(cg::this_thread_block()).sync();


            if(queryIndex < maxCandidateIndex_excl){

                const int querybases = d_sequencePointers.candidateSequencesLength[queryIndex];

                const unsigned int* candidateptr = candidateDataHiLoTransposed + std::size_t(queryIndex);

                //save query in shared memory
                for(int i = 0; i < max_sequence_ints; i += 1) {
                    //queryBackup[no_bank_conflict_index(i)] = ((unsigned int*)(candidateptr))[i];
                    //transposed
                    queryBackup[no_bank_conflict_index(i)] = candidateptr[i * n_candidates];
                }

                //queryIndex != resultIndex -> reverse complement
                if(isReverseComplement) {
                    make_reverse_complement_inplace(queryBackup, querybases, no_bank_conflict_index);
                }

                //begin SHD algorithm

                const int subjectints = getEncodedNumInts2BitHiLo(subjectbases);
                const int queryints = getEncodedNumInts2BitHiLo(querybases);
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

                    //const int max_errors = int(float(overlapsize) * maxErrorRate);
                    const int max_errors_excl = min(int(float(overlapsize) * maxErrorRate),
                                                    bestScore - totalbases + 2*overlapsize);

                    if(max_errors_excl > 0){

                        int score = hammingDistanceWithShift(shift != 0, overlapsize, max_errors_excl,
                                            shiftptr_hi,shiftptr_lo, transfunc1,
                                            shiftptr_size,
                                            otherptr_hi, otherptr_lo, transfunc2);

                        score = (score < max_errors_excl ?
                                score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                                : std::numeric_limits<int>::max()); // too many errors, discard

                        if(score < bestScore){
                            bestScore = score;
                            bestShift = shift;
                        }

                        return true;
                    }else{
                        return false;
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

                    bool b = handle_shift(shift, overlapsize,
                                    mySequence_hi, mySequence_lo, no_bank_conflict_index,
                                    subjectints,
                                    queryBackup_hi, queryBackup_lo, no_bank_conflict_index);
                    if(!b){
                        break;
                    }
                }

                //initialize threadlocal smem array with query
                for(int i = 0; i < max_sequence_ints; i += 1) {
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
            return getEncodedNuc2Bit((const unsigned int*)data, length, index, [](auto i){return i;});
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

        auto saveUncorrectedPositionInSmem = [&](int pos){
            const int smemindex = atomicAdd(&numUncorrectedPositions, 1);
            uncorrectedPositions[smemindex] = pos;
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
                    d_correctionResultPointers.isHighQualitySubject[subjectIndex].hq(isHQ);
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
                            const char consensusBase = my_consensus[globalIndex];

                            float maxOverlapWeightOrigBase = 0.0f;
                            float maxOverlapWeightConsensusBase = 0.0f;
                            int origBaseCount = 1;
                            int consensusBaseCount = 0;

                            bool goodOrigOverlapExists = false;

                            const int* myIndices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];

                            for(int candidatenr = 0; candidatenr < myNumIndices; candidatenr++){
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
                                    const float overlapweight = calculateOverlapWeight(subjectLength, nOps, overlapsize);
                                    assert(overlapweight <= 1.0f);
                                    assert(overlapweight >= 0.0f);

                                    constexpr float goodOverlapThreshold = 0.90f;

                                    if(origBase == candidateBase){
                                        maxOverlapWeightOrigBase = max(maxOverlapWeightOrigBase, overlapweight);
                                        origBaseCount++;

                                        if(overlapweight >= goodOverlapThreshold){
                                            goodOrigOverlapExists = true;
                                        }
                                    }else{
                                        if(consensusBase == candidateBase){
                                            maxOverlapWeightConsensusBase = max(maxOverlapWeightConsensusBase, overlapweight);
                                            consensusBaseCount++;
                                        }
                                    }
                                }
                            }

                            if(my_support[globalIndex] > 0.5f){

                                constexpr float maxOverlapWeightLowerBound = 0.15f;

                                bool allowCorrectionToConsensus = false;

                                //if(maxOverlapWeightOrigBase < maxOverlapWeightConsensusBase){
                                    allowCorrectionToConsensus = true;
                                //}

                                // if(maxOverlapWeightOrigBase == 0 && maxOverlapWeightConsensusBase == 0){
                                //     //correct to orig;
                                //     allowCorrectionToConsensus = false;
                                // }else if(maxOverlapWeightConsensusBase < maxOverlapWeightLowerBound){
                                //     //correct to orig
                                //     allowCorrectionToConsensus = false;
                                // }else if(maxOverlapWeightOrigBase < maxOverlapWeightLowerBound){
                                //     //correct to consensus
                                //     allowCorrectionToConsensus = true;
                                //     if(origBaseCount < 4){
                                //         allowCorrectionToConsensus = true;
                                //     }
                                // }else if(maxOverlapWeightConsensusBase < maxOverlapWeightOrigBase - 0.2f){
                                //     //maybe correct to orig
                                //     allowCorrectionToConsensus = false;
                                // }else if(maxOverlapWeightConsensusBase  - 0.2f > maxOverlapWeightOrigBase){
                                //     //maybe correct to consensus
                                //     if(origBaseCount < 4){
                                //         allowCorrectionToConsensus = true;
                                //     }
                                // }

                                if(!goodOrigOverlapExists && allowCorrectionToConsensus){

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

                                    if(kregioncoverageisgood && avgsupportkregion >= 1.0f-4*estimatedErrorrate / 2.0f){


                                        // constexpr float maxOverlapWeightLowerBound = 0.25f;
                                        //
                                        // bool correctToConsensus = false;//maxOverlapWeightOrigBase < maxOverlapWeightLowerBound;
                                        // // correctToConsensus |= maxOverlapWeightConsensusBase >= maxOverlapWeightOrigBase;
                                        // // correctToConsensus &= !goodOrigOverlapExists;
                                        // if(!goodOrigOverlapExists && (origBase != consensusBase && my_support[globalIndex] > 0.5f)){
                                        //     correctToConsensus = true;
                                        // }

                                        // if(maxOverlapWeightOrigBase == 0 && maxOverlapWeightConsensusBase == 0){
                                        //     //correct to orig;
                                        // }else if(maxOverlapWeightConsensusBase < maxOverlapWeightLowerBound){
                                        //     //correct to orig
                                        // }else if(maxOverlapWeightOrigBase < maxOverlapWeightLowerBound){
                                        //     //correct to consensus
                                        //     my_corrected_subject[i] = consensusBase;
                                        // }else if(maxOverlapWeightConsensusBase < maxOverlapWeightOrigBase){
                                        //     //maybe correct to orig
                                        // }else if(maxOverlapWeightConsensusBase >= maxOverlapWeightOrigBase){
                                        //     //maybe correct to consensus
                                        //     my_corrected_subject[i] = consensusBase;
                                        // }

                                        //if(correctToConsensus){
                                            my_corrected_subject[i] = consensusBase;
                                            foundAColumn = true;
                                        // }else{
                                        //     saveUncorrectedPositionInSmem(i);
                                        // }
                                    }else{
                                        saveUncorrectedPositionInSmem(i);
                                    }
                                }
                            }else{
                                saveUncorrectedPositionInSmem(i);
                            }
                        }

                        __syncthreads();

                        if(threadIdx.x == 0){
                            *globalNumUncorrectedPositionsPtr += numUncorrectedPositions;
                        }

                        for(int k = threadIdx.x; k < numUncorrectedPositions; k += BLOCKSIZE){
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
                        d_correctionResultPointers.subjectIsCorrected[subjectIndex] = true;//foundAColumn;
                    }
                }
            }
        }
    }





    template<int BLOCKSIZE>
    __global__
    void msa_correct_subject_implicit_kernel2(
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

        __shared__ int broadcastbuffer;

        __shared__ int numUncorrectedPositions;
        __shared__ int uncorrectedPositions[BLOCKSIZE];
        __shared__ float avgCountPerWeight[4];

        auto get = [] (const char* data, int length, int index){
            //return Sequence_t::get_as_nucleotide(data, length, index);
            return getEncodedNuc2Bit((const unsigned int*)data, length, index, [](auto i){return i;});
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


                const float avg_support_threshold = 1.0f-1.0f*estimatedErrorrate;
        		const float min_support_threshold = 1.0f-3.0f*estimatedErrorrate;

                if(threadIdx.x == 0){
                    d_correctionResultPointers.subjectIsCorrected[subjectIndex] = true; //canBeCorrected;

                    const bool canBeCorrectedByConsensus = isGoodAvgSupport(avg_support) && isGoodMinSupport(min_support) && isGoodMinCoverage(min_coverage);
                    int flag = 0;

                    if(canBeCorrectedByConsensus){
                        int smallestErrorrateThatWouldMakeHQ = 100;

                        const int estimatedErrorratePercent = ceil(estimatedErrorrate * 100.0f);
                        for(int percent = estimatedErrorratePercent; percent >= 0; percent--){
                            float factor = percent / 100.0f;
                            if(avg_support >= 1.0f - 1.0f * factor && min_support >= 1.0f - 3.0f * factor){
                                smallestErrorrateThatWouldMakeHQ = percent;
                            }
                        }

                        const bool isHQ = isGoodMinCoverage(min_coverage)
                                            && smallestErrorrateThatWouldMakeHQ <= estimatedErrorratePercent * 0.5f;

                        //broadcastbuffer = isHQ;
                        d_correctionResultPointers.isHighQualitySubject[subjectIndex].hq(isHQ);

                        flag = isHQ ? 2 : 1;
                    }

                    broadcastbuffer = flag;
                }
                __syncthreads();

                // for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                //     //assert(my_consensus[i] == 'A' || my_consensus[i] == 'C' || my_consensus[i] == 'G' || my_consensus[i] == 'T');
                //     if(my_support[i] > 0.90f && my_orig_coverage[i] <= 2){
                //         my_corrected_subject[i - subjectColumnsBegin_incl] = my_consensus[i];
                //     }else{
                //         const char* subject = getSubjectPtr(subjectIndex);
                //         const char encodedBase = get(subject, subjectColumnsEnd_excl- subjectColumnsBegin_incl, i - subjectColumnsBegin_incl);
                //         const char base = to_nuc(encodedBase);
                //         my_corrected_subject[i - subjectColumnsBegin_incl] = base;
                //     }
                // }

                const int flag = broadcastbuffer;

                if(flag > 0){
                    for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                        my_corrected_subject[i - subjectColumnsBegin_incl] = my_consensus[i];
                    }
                }else{
                    //correct only positions with high support.
                    for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                        //assert(my_consensus[i] == 'A' || my_consensus[i] == 'C' || my_consensus[i] == 'G' || my_consensus[i] == 'T');
                        if(my_support[i] > 0.90f && my_orig_coverage[i] <= 2){
                            my_corrected_subject[i - subjectColumnsBegin_incl] = my_consensus[i];
                        }else{
                            const char* subject = getSubjectPtr(subjectIndex);
                            const char encodedBase = get(subject, subjectColumnsEnd_excl- subjectColumnsBegin_incl, i - subjectColumnsBegin_incl);
                            const char base = to_nuc(encodedBase);
                            my_corrected_subject[i - subjectColumnsBegin_incl] = base;
                        }
                    }
                }
            }else{
                if(threadIdx.x == 0){
                    d_correctionResultPointers.isHighQualitySubject[subjectIndex].hq(false);
                    d_correctionResultPointers.subjectIsCorrected[subjectIndex] = false;
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

            //return result;
            return true;
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
            return getEncodedNuc2Bit((const unsigned int*)data, length, index, [](auto i){return i;});
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
            return getEncodedNuc2Bit((const unsigned int*)data, length, index, [](auto i){return i;});
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

                    const float* const my_support = d_msapointers.support + msa_weights_pitch_floats * subjectIndex;
                    const char* candidate = d_sequencePointers.candidateSequencesData + std::size_t(global_candidate_index) * encoded_sequence_pitch;

                    // for(int i = copyposbegin; i < copyposend; i += 1) {
                    //     //assert(my_consensus[i] == 'A' || my_consensus[i] == 'C' || my_consensus[i] == 'G' || my_consensus[i] == 'T');
                    //     if(my_support[i] > 0.90f){
                    //         my_corrected_candidates[destinationindex * sequence_pitch + (i - queryColumnsBegin_incl)] = my_consensus[i];
                    //     }else{
                    //         const char encodedBase = get(candidate, queryColumnsEnd_excl- queryColumnsBegin_incl, i - queryColumnsBegin_incl);
                    //         const char base = to_nuc(encodedBase);
                    //         my_corrected_candidates[destinationindex * sequence_pitch + (i - queryColumnsBegin_incl)] = base;
                    //     }
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

    







    //####################   KERNEL DISPATCH   ####################


    void call_cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel_async(
    			AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
    			const int* d_candidates_per_subject_prefixsum,
                const int* h_candidates_per_subject,
                const int* d_candidates_per_subject,
    			int n_subjects,
    			int n_queries,
    			int maximumSequenceLength,
    			int min_overlap,
    			float maxErrorRate,
    			float min_overlap_ratio,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

            const int intsPerSequence2Bit = getEncodedNumInts2Bit(maximumSequenceLength);
            const int intsPerSequence2BitHiLo = getEncodedNumInts2BitHiLo(maximumSequenceLength);
            const int bytesPerSequence2BitHilo = intsPerSequence2BitHiLo * sizeof(unsigned int);

            unsigned int* d_subjectDataHiLo = nullptr;
            unsigned int* d_candidateDataHiLoTransposed = nullptr;

            cubCachingAllocator.DeviceAllocate(
                (void**)&d_candidateDataHiLoTransposed, 
                sizeof(unsigned int) * intsPerSequence2BitHiLo * n_queries, 
                stream
            ); CUERR;
#if 0
            unsigned int* d_candidateDataTransposed = nullptr;

            cubCachingAllocator.DeviceAllocate(
                (void**)&d_candidateDataTransposed, 
                sizeof(unsigned int) * intsPerSequence2Bit * n_queries, 
                stream
            ); CUERR;

            call_transpose_kernel(d_candidateDataTransposed,
                (const unsigned int*)d_sequencePointers.candidateSequencesData,
                n_queries,
                intsPerSequence2Bit,
                intsPerSequence2Bit,
                stream
            );

            callConversionKernel2BitTo2BitHiLoTT(
                d_candidateDataTransposed,
                intsPerSequence2Bit,
                d_candidateDataHiLoTransposed,
                intsPerSequence2BitHiLo,
                d_sequencePointers.candidateSequencesLength,
                n_queries,
                stream,
                handle
            );

            cubCachingAllocator.DeviceFree(d_candidateDataTransposed); CUERR;
#else 

            callConversionKernel2BitTo2BitHiLoNT(
                (const unsigned int*)d_sequencePointers.candidateSequencesData,
                intsPerSequence2Bit,
                d_candidateDataHiLoTransposed,
                intsPerSequence2BitHiLo,
                d_sequencePointers.candidateSequencesLength,
                n_queries,
                stream,
                handle
            );

#endif 


            

            cubCachingAllocator.DeviceAllocate(
                (void**)&d_subjectDataHiLo, 
                sizeof(unsigned int) * intsPerSequence2BitHiLo * n_subjects, 
                stream
            ); CUERR;

            callConversionKernel2BitTo2BitHiLoNN(
                (const unsigned int*)d_sequencePointers.subjectSequencesData,
                intsPerSequence2Bit,
                d_subjectDataHiLo,
                intsPerSequence2BitHiLo,
                d_sequencePointers.subjectSequencesLength,
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




        	const int blocksize = 128;
            const int tilesPerBlock = blocksize / tilesize;

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
                            cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_new_kernel<tilesize>, \
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

            #define mycall cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_new_kernel<tilesize> \
                                                <<<grid, block, smem, stream>>>( \
                                                d_subjectDataHiLo, \
                                                d_candidateDataHiLoTransposed, \
                                        		d_alignmentresultpointers, \
                                                d_sequencePointers, \
                                        		d_candidates_per_subject_prefixsum, \
                                                d_tiles_per_subject_prefixsum, \
                                        		n_subjects, \
                                        		n_queries, \
                                        		maximumSequenceLength, \
                                        		min_overlap, \
                                        		maxErrorRate, \
                                        		min_overlap_ratio); CUERR;

        	dim3 block(blocksize, 1, 1);
        	dim3 grid(std::min(requiredBlocks, max_blocks_per_device), 1, 1);
            //dim3 grid(1,1,1);

        	mycall;

    	    #undef mycall

            cubCachingAllocator.DeviceFree(d_tiles_per_subject_prefixsum);  CUERR;
            cubCachingAllocator.DeviceFree(d_subjectDataHiLo);  CUERR;
            cubCachingAllocator.DeviceFree(d_candidateDataHiLoTransposed);  CUERR;
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
                                                                msa_correct_subject_implicit_kernel2<(blocksize)>, \
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

        cudaMemsetAsync(d_correctionResultPointers.isHighQualitySubject, 0, n_subjects * sizeof(AnchorHighQualityFlag), stream); CUERR;

        dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(n_subjects, max_blocks_per_device));

        #define mycall(blocksize) msa_correct_subject_implicit_kernel2<(blocksize)> \
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

        // set number of corrected candidates per subject to 0
        cudaMemsetAsync(d_correctionResultPointers.numCorrectedCandidates, 0, sizeof(int) * n_subjects, stream); CUERR;
        //call_fill_kernel_async(d_correctionResultPointers.numCorrectedCandidates, n_subjects, 0, stream);

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
