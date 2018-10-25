#ifndef BEST_ALIGNMENT_HPP
#define BEST_ALIGNMENT_HPP

#include "hpc_helpers.cuh"

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif

enum class BestAlignment_t {Forward, ReverseComplement, None};

namespace care{
namespace cpu{

    template<class Alignment>
    BestAlignment_t choose_best_alignment(const Alignment& fwdAlignment,
                                        const Alignment& revcmplAlignment,
                                        int subjectlength,
                                        int querylength,
                                        double min_overlap_ratio,
                                        int min_overlap,
                                        double maxErrorRate){
        const int overlap = fwdAlignment.get_overlap();
        const int revcomploverlap = revcmplAlignment.get_overlap();
        const int fwdMismatches = fwdAlignment.get_nOps();
        const int revcmplMismatches = revcmplAlignment.get_nOps();

        BestAlignment_t retval = BestAlignment_t::None;

        const int minimumOverlap = int(subjectlength * min_overlap_ratio) > min_overlap
                        ? int(subjectlength * min_overlap_ratio) : min_overlap;

        if(fwdAlignment.get_isValid() && overlap >= minimumOverlap){
            if(revcmplAlignment.get_isValid() && revcomploverlap >= minimumOverlap){
                const double ratio = (double)fwdMismatches / overlap;
                const double revcomplratio = (double)revcmplMismatches / revcomploverlap;

                if(ratio < revcomplratio){
                    if(ratio < maxErrorRate){
                        retval = BestAlignment_t::Forward;
                    }
                }else if(revcomplratio < ratio){
                    if(revcomplratio < maxErrorRate){
                        retval = BestAlignment_t::ReverseComplement;
                    }
                }else{
                    if(ratio < maxErrorRate){
                        // both have same mismatch ratio, choose longest overlap
                        if(overlap > revcomploverlap){
                            retval = BestAlignment_t::Forward;
                        }else{
                            retval = BestAlignment_t::ReverseComplement;
                        }
                    }
                }
            }else{
                if((double)fwdMismatches / overlap < maxErrorRate){
                    retval = BestAlignment_t::Forward;
                }
            }
        }else{
            if(revcmplAlignment.get_isValid() && revcomploverlap >= minimumOverlap){
                if((double)revcmplMismatches / revcomploverlap < maxErrorRate){
                    retval = BestAlignment_t::ReverseComplement;
                }
            }
        }

        return retval;
    }
}
}

/*
    Find alignment with lowest mismatch ratio.
    If both have same ratio choose alignment with longer overlap.
*/
template<class Alignment>
HOSTDEVICEQUALIFIER
BestAlignment_t choose_best_alignment(const Alignment& fwdAlignment,
                                    const Alignment& revcmplAlignment,
                                    int subjectlength,
                                    int querylength,
                                    double min_overlap_ratio,
                                    int min_overlap,
                                    double maxErrorRate){
    const int overlap = fwdAlignment.get_overlap();
    const int revcomploverlap = revcmplAlignment.get_overlap();
    const int fwdMismatches = fwdAlignment.get_nOps();
    const int revcmplMismatches = revcmplAlignment.get_nOps();

    BestAlignment_t retval = BestAlignment_t::None;

    const int minimumOverlap = int(subjectlength * min_overlap_ratio) > min_overlap
                    ? int(subjectlength * min_overlap_ratio) : min_overlap;

    if(fwdAlignment.get_isValid() && overlap >= minimumOverlap){
        if(revcmplAlignment.get_isValid() && revcomploverlap >= minimumOverlap){
            const double ratio = (double)fwdMismatches / overlap;
            const double revcomplratio = (double)revcmplMismatches / revcomploverlap;

            if(ratio < revcomplratio){
                if(ratio < maxErrorRate){
                    retval = BestAlignment_t::Forward;
                }
            }else if(revcomplratio < ratio){
                if(revcomplratio < maxErrorRate){
                    retval = BestAlignment_t::ReverseComplement;
                }
            }else{
                if(ratio < maxErrorRate){
                    // both have same mismatch ratio, choose longest overlap
                    if(overlap > revcomploverlap){
                        retval = BestAlignment_t::Forward;
                    }else{
                        retval = BestAlignment_t::ReverseComplement;
                    }
                }
            }
        }else{
            if((double)fwdMismatches / overlap < maxErrorRate){
                retval = BestAlignment_t::Forward;
            }
        }
    }else{
        if(revcmplAlignment.get_isValid() && revcomploverlap >= minimumOverlap){
            if((double)revcmplMismatches / revcomploverlap < maxErrorRate){
                retval = BestAlignment_t::ReverseComplement;
            }
        }
    }

    return retval;
}



#ifdef __NVCC__

/*
    2*N results. compare alignment[i] with alignment[i + N].
    store the best alignment at alignment[i] and the flag at flags[i]
*/
template<class Alignment, class AlignmentComp>
__global__
void cuda_find_best_alignment_kernel(Alignment* results,
                              BestAlignment_t* bestAlignmentFlags,
                              const int* subjectlengths,
                              const int* querylengths,
                              const int* NqueriesPrefixSum,
                              int Nsubjects,
                              AlignmentComp comp){

    const int N = NqueriesPrefixSum[Nsubjects];

    for(unsigned resultIndex = threadIdx.x + blockDim.x * blockIdx.x; resultIndex < N; resultIndex += gridDim.x * blockDim.x){
        const Alignment fwd = results[resultIndex];
        const Alignment revcompl = results[resultIndex + N];
        const int querybases = querylengths[resultIndex];
        //find subjectindex
        int subjectIndex = 0;
        for(; subjectIndex < Nsubjects; subjectIndex++){
            if(resultIndex < NqueriesPrefixSum[subjectIndex+1])
                break;
        }
        const int subjectbases = subjectlengths[subjectIndex];


        const BestAlignment_t flag = comp(fwd, revcompl, subjectbases, querybases);
        bestAlignmentFlags[resultIndex] = flag;
        results[resultIndex] = flag == BestAlignment_t::Forward ? fwd : revcompl;
    }
}

template<class Alignment, class AlignmentComp>
void call_cuda_find_best_alignment_kernel_async(Alignment* d_results,
                              BestAlignment_t* d_bestAlignmentFlags,
                              const int* d_subjectlengths,
                              const int* d_querylengths,
                              const int* d_NqueriesPrefixSum,
                              int Nsubjects,
                              AlignmentComp d_comp,
                              int Nqueries,
                              cudaStream_t stream){

    dim3 block(128,1,1);
    dim3 grid(SDIV(Nqueries, block.x), 1, 1);

    cuda_find_best_alignment_kernel<<<grid, block, 0, stream>>>(d_results,
                                                    d_bestAlignmentFlags,
                                                    d_subjectlengths,
                                                    d_querylengths,
                                                    d_NqueriesPrefixSum,
                                                    Nsubjects,
                                                    d_comp); CUERR;

}

template<class Alignment, class AlignmentComp>
void call_cuda_find_best_alignment_kernel(Alignment* d_results,
                              BestAlignment_t* d_bestAlignmentFlags,
                              const int* d_subjectlengths,
                              const int* d_querylengths,
                              const int* d_NqueriesPrefixSum,
                              int Nsubjects,
                              AlignmentComp d_comp,
                              int Nqueries,
                              cudaStream_t stream){

    call_cuda_find_best_alignment_kernel_async(d_results,
                                                d_bestAlignmentFlags,
                                                d_subjectlengths,
                                                d_querylengths,
                                                d_NqueriesPrefixSum,
                                                Nsubjects,
                                                d_comp,
                                                Nqueries,
                                                stream);

    cudaStreamSynchronize(stream);

}

template<class Accessor, class UnpackedRevcomplInplace>
__global__
void
cuda_unpack_sequences_with_good_alignments_kernel(char* unpacked_sequences,
                              const BestAlignment_t* bestAlignmentFlags,
                              const char* queriesdata,
                              const int* querylengths,
                              const int n_queries,
                              int max_sequence_length,
                              size_t sequencepitch,
                              Accessor get,
                              UnpackedRevcomplInplace make_reverse_complement_inplace){

    for(unsigned index = threadIdx.x + blockDim.x * blockIdx.x; index < n_queries; index += blockDim.x * gridDim.x){
        BestAlignment_t flag = bestAlignmentFlags[index];
        const int length = querylengths[index];

        const char* const query = queriesdata + index * sequencepitch;
        char* const unpacked_query = unpacked_sequences + index * max_sequence_length;

        for(int i = 0; i < length; i++){
            unpacked_query[i] = get(query, length, i);
        }

        if(flag == BestAlignment_t::ReverseComplement){
            make_reverse_complement_inplace((unsigned char*)unpacked_query, length);
        }
    }
}

template<class Accessor, class UnpackedRevcomplInplace>
void call_cuda_unpack_sequences_with_good_alignments_kernel_async(
                              char* d_unpacked_sequences,
                              const BestAlignment_t* d_bestAlignmentFlags,
                              const char* d_queriesdata,
                              const int* d_querylengths,
                              const int n_queries,
                              int max_sequence_length,
                              size_t sequencepitch,
                              Accessor get,
                              UnpackedRevcomplInplace make_reverse_complement_inplace,
                              cudaStream_t stream){

    dim3 block(128,1,1);
    dim3 grid(SDIV(n_queries, block.x), 1, 1);

    cuda_unpack_sequences_with_good_alignments_kernel<<<grid, block, 0, stream>>>(d_unpacked_sequences,
                                                    d_bestAlignmentFlags,
                                                    d_queriesdata,
                                                    d_querylengths,
                                                    n_queries,
                                                    max_sequence_length,
                                                    sequencepitch,
                                                    get,
                                                    make_reverse_complement_inplace); CUERR;
}



template<class Alignment, class AlignmentComp, class Accessor, class UnpackedRevcomplInplace>
__global__
void cuda_find_best_alignment_and_unpack_good_sequences_kernel(
                            Alignment* results,
                            BestAlignment_t* bestAlignmentFlags,
                            char* unpacked_sequences,
                            const int* subjectlengths,
                            const int* querylengths,
                            const int* NqueriesPrefixSum,
                            const int n_subjects,
                            const char* queriesdata,
                            int max_sequence_length,
                            size_t sequencepitch,
                            AlignmentComp comp,
                            Accessor get,
                            UnpackedRevcomplInplace make_reverse_complement_inplace){

    const int n_queries = NqueriesPrefixSum[n_subjects];

    for(unsigned resultIndex = threadIdx.x + blockDim.x * blockIdx.x; resultIndex < n_queries; resultIndex += gridDim.x * blockDim.x){
        const Alignment fwd = results[resultIndex];
        const Alignment revcompl = results[resultIndex + n_queries];
        const int queryLength = querylengths[resultIndex];

        //find subjectindex
        int subjectIndex = 0;
        for(; subjectIndex < n_subjects; subjectIndex++){
            if(resultIndex < NqueriesPrefixSum[subjectIndex+1])
                break;
        }
        const int subjectLength = subjectlengths[subjectIndex];

        const BestAlignment_t flag = comp(fwd, revcompl, subjectLength, queryLength);
        bestAlignmentFlags[resultIndex] = flag;
        results[resultIndex] = flag == BestAlignment_t::Forward ? fwd : revcompl;

        const char* const query = queriesdata + resultIndex * sequencepitch;
        char* const unpacked_query = unpacked_sequences + resultIndex * max_sequence_length;

        for(int i = 0; i < queryLength; i++){
            unpacked_query[i] = get(query, queryLength, i);
        }

        if(flag == BestAlignment_t::ReverseComplement){
            make_reverse_complement_inplace((unsigned char*)unpacked_query, queryLength);
        }
    }
}

template<class Alignment, class AlignmentComp, class Accessor, class UnpackedRevcomplInplace>
void call_cuda_find_best_alignment_and_unpack_good_sequences_kernel_async(
                                Alignment* d_results,
                                BestAlignment_t* d_bestAlignmentFlags,
                                char* d_unpacked_sequences,
                                const int* d_subjectlengths,
                                const int* d_querylengths,
                                const int* d_NqueriesPrefixSum,
                                const int n_subjects,
                                const int n_queries,
                                const char* d_queriesdata,
                                int max_sequence_length,
                                size_t sequencepitch,
                                AlignmentComp d_comp,
                                Accessor d_get,
                                UnpackedRevcomplInplace d_make_reverse_complement_inplace,
                                cudaStream_t stream){

    dim3 block(128,1,1);
    dim3 grid(SDIV(n_queries, block.x), 1, 1);

    cuda_find_best_alignment_and_unpack_good_sequences_kernel<<<grid, block, 0, stream>>>(d_results,
                                                    d_bestAlignmentFlags,
                                                    d_unpacked_sequences,
                                                    d_subjectlengths,
                                                    d_querylengths,
                                                    d_NqueriesPrefixSum,
                                                    n_subjects,
                                                    d_queriesdata,
                                                    max_sequence_length,
                                                    sequencepitch,
                                                    d_comp,
                                                    d_get,
                                                    d_make_reverse_complement_inplace); CUERR;

}



template<class Alignment, class AlignmentComp, class Accessor, class UnpackedRevcomplInplace>
__global__
void cuda_find_best_alignment_and_unpack_good_sequences_kernel(
                            Alignment* results,
                            BestAlignment_t* bestAlignmentFlags,
                            char* unpacked_subjects,
                            char* unpacked_queries,
                            const int* subjectlengths,
                            const int* querylengths,
                            const int* NqueriesPrefixSum,
                            const int n_subjects,
                            const char* subjectsdata,
                            const char* queriesdata,
                            int max_sequence_length,
                            size_t sequencepitch,
                            AlignmentComp comp,
                            Accessor get,
                            UnpackedRevcomplInplace make_reverse_complement_inplace){

    const int n_queries = NqueriesPrefixSum[n_subjects];

    for(unsigned resultIndex = threadIdx.x + blockDim.x * blockIdx.x; resultIndex < n_queries; resultIndex += gridDim.x * blockDim.x){
        const Alignment fwd = results[resultIndex];
        const Alignment revcompl = results[resultIndex + n_queries];
        const int queryLength = querylengths[resultIndex];

        //find subjectindex
        int subjectIndex = 0;
        for(; subjectIndex < n_subjects; subjectIndex++){
            if(resultIndex < NqueriesPrefixSum[subjectIndex+1])
                break;
        }
        const int subjectLength = subjectlengths[subjectIndex];

        const BestAlignment_t flag = comp(fwd, revcompl, subjectLength, queryLength);
        bestAlignmentFlags[resultIndex] = flag;
        results[resultIndex] = flag == BestAlignment_t::Forward ? fwd : revcompl;

        const char* const subject = subjectsdata + resultIndex * sequencepitch;
        const char* const query = queriesdata + resultIndex * sequencepitch;
        char* const unpacked_subject = unpacked_subjects + resultIndex * max_sequence_length;
        char* const unpacked_query = unpacked_queries + resultIndex * max_sequence_length;

        for(int i = 0; i < queryLength; i++){
            unpacked_query[i] = get(query, queryLength, i);
        }

        if(flag == BestAlignment_t::ReverseComplement){
            make_reverse_complement_inplace((unsigned char*)unpacked_query, queryLength);
        }

        if(resultIndex < n_subjects){
            for(int i = 0; i < subjectLength; i++){
                unpacked_subject[i] = get(subject, subjectLength, i);
            }
        }
    }

}

template<class Alignment, class AlignmentComp, class Accessor, class UnpackedRevcomplInplace>
void call_cuda_find_best_alignment_and_unpack_good_sequences_kernel_async(
                                Alignment* d_results,
                                BestAlignment_t* d_bestAlignmentFlags,
                                char* d_unpacked_subjects,
                                char* d_unpacked_queries,
                                const int* d_subjectlengths,
                                const int* d_querylengths,
                                const int* d_NqueriesPrefixSum,
                                const int n_subjects,
                                const int n_queries,
                                const char* d_subjectsdata,
                                const char* d_queriesdata,
                                int max_sequence_length,
                                size_t sequencepitch,
                                AlignmentComp d_comp,
                                Accessor d_get,
                                UnpackedRevcomplInplace d_make_reverse_complement_inplace,
                                cudaStream_t stream){

    dim3 block(128,1,1);
    dim3 grid(SDIV(n_queries, block.x), 1, 1);

    cuda_find_best_alignment_and_unpack_good_sequences_kernel<<<grid, block, 0, stream>>>(d_results,
                                                    d_bestAlignmentFlags,
                                                    d_unpacked_subjects,
                                                    d_unpacked_queries,
                                                    d_subjectlengths,
                                                    d_querylengths,
                                                    d_NqueriesPrefixSum,
                                                    n_subjects,
                                                    d_subjectsdata,
                                                    d_queriesdata,
                                                    max_sequence_length,
                                                    sequencepitch,
                                                    d_comp,
                                                    d_get,
                                                    d_make_reverse_complement_inplace); CUERR;

}










#if 0
template<int BLOCKSIZE, class Alignment>
__global__
void cuda_filter_alignments_by_mismatchratio_kernel(
                                    BestAlignment_t* d_alignment_best_alignment_flags,
                                    const Alignment* d_alignments;
                                    const int* d_indices,
                                    const int* d_indices_per_subject,
                                    const int* d_indices_per_subject_prefixsum,
                                    int n_subjects,
                                    int n_candidates,
                                    const int* d_num_indices,
                                    double binsize,
                                    int min_remaining_candidates_per_subject){

    using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

    __shared__ union {
        typename BlockReduceInt::TempStorage intreduce;
        int broadcast[3];
    } temp_storage;

    for(int subjectindex = blockDim.x; subjectindex < n_subjects; subjectindex += gridDim.x){

        const int my_n_indices = d_indices_per_subject[subjectindex];
        const int* my_indices = d_indices + d_indices_per_subject_prefixsum[subjectindex];

        int counts[3]{0,0,0};

        for(int index = threadIdx.x; index < my_n_indices; index += blockDim.x){
            const int candidate_index = my_indices[index];
            const int alignment_overlap = d_alignments[candidate_index].overlap;
            const int alignment_nops = d_alignments[candidate_index].nOps;

            const double mismatchratio = double(alignment_nops) / alignment_overlap;

            assert(mismatchratio < 4 * binsize);

            #pragma unroll
            for(int i = 2; i <= 4; i++){
                counts[i-2] += (mismatchratio < i * binsize);
            }
        }

        //accumulate counts over block
        #pragma unroll
        for(int i = 0; i < 3; i++){
            counts[i] = BlockReduceInt(temp_storage.intreduce).Sum(counts[i]);
            __syncthreads();
        }

        //broadcast accumulated counts to block
        #pragma unroll
        for(int i = 0; i < 3; i++){
            if(threadIdx.x == 0){
                temp_storage.broadcast[i] = counts[i];
            }
            __syncthreads();
            counts[i] = temp_storage.broadcast[i];
        }

        double mismatchratioThreshold = 0;
        if (counts[0] >= min_remaining_candidates_per_subject) {
            mismatchratioThreshold = 2 * binsize;
        } else if (counts[1] >= min_remaining_candidates_per_subject) {
            mismatchratioThreshold = 3 * binsize;
        } else if (counts[2] >= min_remaining_candidates_per_subject) {
            mismatchratioThreshold = 4 * binsize;
        } else {
            mismatchratioThreshold = -1; //this will invalidate all alignments for subject
        }

        // Invalidate all alignments for subject with mismatchratio >= mismatchratioThreshold
        for(int index = threadIdx.x; index < my_n_indices; index += blockDim.x){
            const int candidate_index = my_indices[index];
            const int alignment_overlap = d_alignment_overlaps[candidate_index];
            const int alignment_nops = d_alignment_nOps[candidate_index];

            const double mismatchratio = double(alignment_nops) / alignment_overlap;

            const bool remove = mismatchratio >= mismatchratioThreshold;
            if(remove)
                d_alignment_best_alignment_flags[candidate_index] = BestAlignment_t::None;
        }
    }
}


template<class Alignment>
void call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                                    BestAlignment_t* d_alignment_best_alignment_flags,
                                    const Alignment* d_alignments,
                                    const int* d_indices,
                                    const int* d_indices_per_subject,
                                    const int* d_indices_per_subject_prefixsum,
                                    int n_subjects,
                                    int n_candidates,
                                    const int* d_num_indices,
                                    double binsize,
                                    int min_remaining_candidates_per_subject,
                                    cudaStream_t stream){

    const int blocksize = 128;

    dim3 block(blocksize, 1, 1);
    dim3 grid(n_subjects);

    #define mycall(blocksize) cuda_filter_alignments_by_mismatchratio_kernel<(blocksize)> \
                            <<<grid, block, 0, stream>>>( \
                                d_alignment_best_alignment_flags, \
                                d_alignment_overlaps, \
                                d_alignment_nOps, \
                                d_indices, \
                                d_indices_per_subject, \
                                d_indices_per_subject_prefixsum, \
                                n_subjects, \
                                n_candidates, \
                                d_num_indices, \
                                binsize, \
                                min_remaining_candidates_per_subject); CUERR;

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
#endif

#endif




#endif
