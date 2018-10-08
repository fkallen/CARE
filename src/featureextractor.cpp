#include "../inc/featureextractor.hpp"
#include "../inc/multiple_sequence_alignment.hpp"

#include <iomanip>
#include <limits>
#include <algorithm>
#include <numeric>

namespace care{

    std::ostream& operator<<(std::ostream& os, const MSAFeature& f){
        auto maybezero = [](double d){
            return d < 1e-10 ? 0.0 : d;
        };

        os << std::setprecision(3) << maybezero(f.position_support) << '\t';
        os << f.position_coverage << '\t';
        os << f.alignment_coverage << '\t';
        os << f.dataset_coverage << '\t';

        os << std::setprecision(3) << maybezero(f.min_support) << '\t';
        os << std::setprecision(3) << maybezero(f.min_coverage) << '\t';
        os << std::setprecision(3) << maybezero(f.max_support) << '\t';
        os << std::setprecision(3) << maybezero(f.max_coverage) << '\t';
        os << std::setprecision(3) << maybezero(f.mean_support) << '\t';
        os << std::setprecision(3) << maybezero(f.mean_coverage) << '\t';
        os << std::setprecision(3) << maybezero(f.median_support) << '\t';
        os << std::setprecision(3) << maybezero(f.median_coverage) << '\t';

        return os;
    }



    std::vector<MSAFeature> extractFeatures(const pileup::PileupImage& pileup, const std::string& sequence,
                                    int k, double support_threshold,
                                    int dataset_coverage){
        auto isValidIndex = [&](int i){
            return 0 <= i && i < pileup.columnProperties.columnsToCheck;
        };

        auto median = [](auto begin, auto end){
            std::size_t n = std::distance(begin, end);
            std::sort(begin, end);

    		if(n % 2 == 0){
    			return (*(begin + n / 2 - 1) + *(begin + n / 2) / 2);
    		}else{
    			return *(begin + n / 2 - 1);
    		}
        };

        std::vector<MSAFeature> result;

        const int alignment_coverage = *std::max_element(pileup.h_coverage.begin(), pileup.h_coverage.end());

        for(int i = pileup.columnProperties.subjectColumnsBegin_incl;
            i < pileup.columnProperties.subjectColumnsEnd_excl;
            i++){

            const int localindex = i - pileup.columnProperties.subjectColumnsBegin_incl;

            if(pileup.h_support[i] >= support_threshold && pileup.h_consensus[i] != sequence[localindex]){

                int begin = i-k/2;
                int end = i+k/2;

                for(int j = 0; j < k; j++){
                    if(!isValidIndex(begin))
                        begin++;
                    if(!isValidIndex(end))
                        end--;
                }
                end++;

                MSAFeature f;
                f.position = localindex;
                f.position_support = pileup.h_support[i]; // support of the center of k-region (at read position "position")
                f.position_coverage = pileup.h_origCoverage[i]; // coverage of the center of k-region (at read position "position")
                f.alignment_coverage = alignment_coverage; // number of sequences in MSA. equivalent to the max possible value of coverage.
                f.dataset_coverage = dataset_coverage;

                f.min_support = *std::min_element(pileup.h_support.begin() + begin, pileup.h_support.begin() + end);
                f.min_coverage = *std::min_element(pileup.h_coverage.begin() + begin, pileup.h_coverage.begin() + end);
                f.max_support = *std::max_element(pileup.h_support.begin() + begin, pileup.h_support.begin() + end);
                f.max_coverage = *std::max_element(pileup.h_coverage.begin() + begin, pileup.h_coverage.begin() + end);
                f.mean_support = std::accumulate(pileup.h_support.begin() + begin, pileup.h_support.begin() + end, 0.0) / (end - begin);
                f.mean_coverage = std::accumulate(pileup.h_coverage.begin() + begin, pileup.h_coverage.begin() + end, 0.0) / (end - begin);

                std::array<double, 33> arr;

                std::copy(pileup.h_support.begin() + begin, pileup.h_support.begin() + end, arr.begin());
                f.median_support = median(arr.begin(), arr.begin() + (end-begin));

                std::copy(pileup.h_coverage.begin() + begin, pileup.h_coverage.begin() + end, arr.begin());
                f.median_coverage = median(arr.begin(), arr.begin() + (end-begin));

                result.emplace_back(f);
            }
        }

        return result;
    }

    std::vector<MSAFeature> extractFeatures(const char* consensusptr,
                                            const float* supportptr,
                                            const int* coverageptr,
                                            const int* origcoverageptr,
                                            int columnsToCheck,
                                            int subjectColumnsBegin_incl,
                                            int subjectColumnsEnd_excl,
                                            const std::string& sequence,
                                            int k, double
                                            support_threshold,
                                            int dataset_coverage){

        auto isValidIndex = [&](int i){
            return 0 <= i && i < columnsToCheck;
        };

        auto median = [](auto begin, auto end){
            std::size_t n = std::distance(begin, end);
            std::sort(begin, end);

    		if(n % 2 == 0){
    			return (*(begin + n / 2 - 1) + *(begin + n / 2) / 2);
    		}else{
    			return *(begin + n / 2 - 1);
    		}
        };

        std::vector<MSAFeature> result;

        const int alignment_coverage = *std::max_element(coverageptr, coverageptr + columnsToCheck);

        for(int i = subjectColumnsBegin_incl; i < subjectColumnsEnd_excl; i++){

            const int localindex = i - subjectColumnsBegin_incl;

            if(supportptr[i] >= support_threshold && consensusptr[i] != sequence[localindex]){

                int begin = i-k/2;
                int end = i+k/2;

                for(int j = 0; j < k; j++){
                    if(!isValidIndex(begin))
                        begin++;
                    if(!isValidIndex(end))
                        end--;
                }
                end++;

                MSAFeature f;
                f.position = localindex;
                f.position_support = supportptr[i]; // support of the center of k-region (at read position "position")
                f.position_coverage = origcoverageptr[i]; // coverage of the center of k-region (at read position "position")
                f.alignment_coverage = alignment_coverage; // number of sequences in MSA. equivalent to the max possible value of coverage.
                f.dataset_coverage = dataset_coverage;

                f.min_support = *std::min_element(supportptr + begin, supportptr + end);
                f.min_coverage = *std::min_element(coverageptr + begin, coverageptr + end);
                f.max_support = *std::max_element(supportptr + begin, supportptr + end);
                f.max_coverage = *std::max_element(coverageptr + begin, coverageptr + end);
                f.mean_support = std::accumulate(supportptr + begin, supportptr + end, 0.0) / (end - begin);
                f.mean_coverage = std::accumulate(coverageptr + begin, coverageptr + end, 0.0) / (end - begin);

                std::array<double, 33> arr;

                std::copy(supportptr + begin, supportptr + end, arr.begin());
                f.median_support = median(arr.begin(), arr.begin() + (end-begin));

                std::copy(coverageptr + begin, coverageptr + end, arr.begin());
                f.median_coverage = median(arr.begin(), arr.begin() + (end-begin));

                result.emplace_back(f);
            }
        }

        return result;
    }
}
