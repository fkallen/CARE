#ifndef BEST_ALIGNMENT_HPP
#define BEST_ALIGNMENT_HPP

#include "hpc_helpers.cuh"

enum class BestAlignment_t {Forward, ReverseComplement, None};

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




#endif




#endif
