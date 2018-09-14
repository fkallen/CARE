#ifndef CARE_GPU_BEST_ALIGNMENT_HPP
#define CARE_GPU_BEST_ALIGNMENT_HPP

#include "../hpc_helpers.cuh"

namespace care{
namespace gpu{

    enum class BestAlignment_t {Forward, ReverseComplement, None};

    template<int dummy = 0>
    HOSTDEVICEQUALIFIER
    BestAlignment_t choose_best_alignment(int fwd_alignment_overlap,
                                        int revc_alignment_overlap,
                                        int fwd_alignment_nops,
                                        int revc_alignment_nops,
                                        bool fwd_alignment_isvalid,
                                        bool revc_alignment_isvalid,
                                        int subjectlength,
                                        int querylength,
                                        double min_overlap_ratio,
                                        int min_overlap,
                                        double maxErrorRate){
        const int overlap = fwd_alignment_overlap;
        const int revcomploverlap = revc_alignment_overlap;
        const int fwdMismatches = fwd_alignment_nops;
        const int revcmplMismatches = revc_alignment_nops;

        BestAlignment_t retval = BestAlignment_t::None;

        const int minimumOverlap = int(subjectlength * min_overlap_ratio) > min_overlap
                        ? int(subjectlength * min_overlap_ratio) : min_overlap;

        if(fwd_alignment_isvalid && fwd_alignment_overlap >= minimumOverlap){
            if(revc_alignment_isvalid && revc_alignment_overlap >= minimumOverlap){
                const double ratio = (double)fwd_alignment_nops / fwd_alignment_overlap;
                const double revcomplratio = (double)revc_alignment_nops / revc_alignment_overlap;

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
                        if(fwd_alignment_overlap > revc_alignment_overlap){
                            retval = BestAlignment_t::Forward;
                        }else{
                            retval = BestAlignment_t::ReverseComplement;
                        }
                    }
                }
            }else{
                if((double)fwd_alignment_nops / fwd_alignment_overlap < maxErrorRate){
                    retval = BestAlignment_t::Forward;
                }
            }
        }else{
            if(revc_alignment_isvalid && revc_alignment_overlap >= minimumOverlap){
                if((double)revc_alignment_nops / revc_alignment_overlap < maxErrorRate){
                    retval = BestAlignment_t::ReverseComplement;
                }
            }
        }

        return retval;
    }



}
}


#endif
