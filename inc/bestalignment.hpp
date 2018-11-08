#ifndef CARE_CPU_BEST_ALIGNMENT_HPP
#define CARE_CPU_BEST_ALIGNMENT_HPP

namespace care{
namespace cpu{

    enum class BestAlignment_t {Forward, ReverseComplement, None};

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


#endif
