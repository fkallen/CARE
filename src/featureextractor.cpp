#include "../inc/featureextractor.hpp"

#include <iomanip>
#include <limits>

namespace care{

    std::ostream& operator<<(std::ostream& os, const Feature& f){
        auto maybezero = [](double d){
            return d < 1e-10 ? 0.0 : d;
        };
        os << std::setprecision(5) << maybezero(f.A_weight_normalized) << '\t'
        << std::setprecision(5) << maybezero(f.C_weight_normalized) << '\t'
        << std::setprecision(5) << maybezero(f.G_weight_normalized) << '\t'
        << std::setprecision(5) << maybezero(f.T_weight_normalized) << '\t'
        << std::setprecision(5) << maybezero(f.support) << '\t'
        << f.original_base_coverage << '\t'
        << f.coverage << '\t'
        << f.dataset_coverage << '\t'
        << f.position_in_read << '\t'
        << f.k << '\t'
        << f.original_base;

        return os;
    }

    std::vector<MSAFeature> extractFeatures(const pileup::PileupImage& pileup, const std::string& sequence,
                                    int k, double support_threshold,
                                    int dataset_coverage){
        auto isValidIndex = [&](int i){
            return pileup.columnProperties.subjectColumnsBegin_incl <= i
                && i < pileup.columnProperties.subjectColumnsEnd_excl;
        };

        std::vector<MSAFeature> result;

        for(int i = pileup.columnProperties.subjectColumnsBegin_incl;
            i < pileup.columnProperties.subjectColumnsEnd_excl;
            i++){

            const int localindex = i - pileup.columnProperties.subjectColumnsBegin_incl;

            if(pileup.h_support[i] >= support_threshold && pileup.h_consensus[i] != sequence[localindex]){
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
                                            pileup.h_origCoverage[i],
                                            pileup.h_coverage[featIndex],
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
