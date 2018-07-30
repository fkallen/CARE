#include "../inc/featureextractor.hpp"

#include <iomanip>
#include <limits>
#include <algorithm>

namespace care{

    std::ostream& operator<<(std::ostream& os, const Feature& f){
        auto maybezero = [](double d){
            return d < 1e-10 ? 0.0 : d;
        };
        //os << std::setprecision(5) << maybezero(f.A_weight_normalized) << '\t';
        //os << std::setprecision(5) << maybezero(f.C_weight_normalized) << '\t';
        //os << std::setprecision(5) << maybezero(f.G_weight_normalized) << '\t';
        //os << std::setprecision(5) << maybezero(f.T_weight_normalized) << '\t';
        os << std::setprecision(3) << maybezero(f.support) << '\t';
        os << std::setprecision(3) << maybezero(f.col_support) << '\t';
        os << f.original_base_coverage << '\t';
        os << f.col_coverage << '\t';
        os << f.alignment_coverage << '\t';
        os << f.dataset_coverage << '\t';
        os << f.position_in_read << '\t';
        //os << f.k << '\t';
        //os << f.original_base;

        return os;
    }

    std::vector<MSAFeature> extractFeatures(const pileup::PileupImage& pileup, const std::string& sequence,
                                    int k, double support_threshold,
                                    int dataset_coverage){
        auto isValidIndex = [&](int i){
            //return pileup.columnProperties.subjectColumnsBegin_incl <= i
            //    && i < pileup.columnProperties.subjectColumnsEnd_excl;
            return 0 <= i && i < pileup.columnProperties.columnsToCheck;
        };

        std::vector<MSAFeature> result;

        const int alignment_coverage = *std::max_element(pileup.h_coverage.begin(), pileup.h_coverage.end());

        for(int i = pileup.columnProperties.subjectColumnsBegin_incl;
            i < pileup.columnProperties.subjectColumnsEnd_excl;
            i++){

            const int localindex = i - pileup.columnProperties.subjectColumnsBegin_incl;

            if(pileup.h_support[i] >= support_threshold && pileup.h_consensus[i] != sequence[localindex]){
            //if(pileup.h_consensus[i] != sequence[localindex]){
                MSAFeature f;
                f.position = localindex;
                f.features.reserve(k+1);

                for(int j = -k/2; j <= k/2; j++){
                    const int featIndex = i+j;
                    if(isValidIndex(featIndex)){

                        double weightsum = pileup.h_Aweights[featIndex] + pileup.h_Cweights[featIndex]
                                         + pileup.h_Gweights[featIndex] + pileup.h_Tweights[featIndex];
                        f.features.push_back({  pileup.h_Aweights[featIndex] / weightsum,
                                            pileup.h_Cweights[featIndex] / weightsum,
                                            pileup.h_Gweights[featIndex] / weightsum,
                                            pileup.h_Tweights[featIndex] / weightsum,
                                            pileup.h_support[i],
                                            pileup.h_support[featIndex],
                                            pileup.h_origCoverage[i],
                                            pileup.h_coverage[featIndex],
                                            alignment_coverage,
                                            dataset_coverage,
                                            localindex,
                                            k,
                                            sequence[localindex]});
                    }
                }

                result.emplace_back(f);
            }
        }

        return result;
    }
}
