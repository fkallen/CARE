#include "../include/featureextractor.hpp"

#include <iomanip>
#include <limits>
#include <algorithm>
#include <numeric>

namespace care{

    std::ostream& operator<<(std::ostream& os, const MSAFeature& f){
        auto maybezero = [](double d){
            return d < 1e-10 ? 0.0 : d;
        };

        //os << f.position << '\t';
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

        auto almost_equal = [](auto a, auto b){
            constexpr double prec = 1e-6;
            return std::abs(a-b) < prec;
        };

        std::vector<MSAFeature> result;

        const int alignment_coverage = *std::max_element(coverageptr, coverageptr + columnsToCheck);

        for(int i = subjectColumnsBegin_incl; i < subjectColumnsEnd_excl; i++){

            const int localindex = i - subjectColumnsBegin_incl;

            //if(supportptr[i] >= 0.5 && consensusptr[i] != sequence[localindex]){
            //if(supportptr[i] >= support_threshold && consensusptr[i] != sequence[localindex]){
            //if(!almost_equal(supportptr[i], 1.0)){
            //if(true){
            //if(origcoverageptr[i] <= 5 && consensusptr[i] != sequence[localindex]){
            if(origcoverageptr[i] > 5 && consensusptr[i] != sequence[localindex]){

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

    std::ostream& operator<<(std::ostream& os, const MSAFeature2& f){
        os << f.position << '\t';
        os << f.base << '\t';
        os << f.alignment_coverage << '\t';
        os << f.dataset_coverage << '\t';

        for(const auto& base : f.partialsequence){
            os << base << '\t';
        }

        for(const auto& col : f.weights){
            for(const auto& c : col){
                os << std::setprecision(3) << c << '\t';
            }
        }

        return os;
    }

    std::vector<MSAFeature2> extractFeatures2(
                                            const char* multiple_sequence_alignment,
                                            const float* multiple_sequence_alignment_weights,
                                            int nRows,
                                            int nColumns,
                                            bool canUseWeights,
                                            const char* consensusptr,
                                            const float* supportptr,
                                            const int* coverageptr,
                                            const int* origcoverageptr,
                                            int subjectColumnsBegin_incl,
                                            int subjectColumnsEnd_excl,
                                            const std::string& sequence,
                                            int dataset_coverage){

        constexpr int k = 16;

        auto isValidColumnIndexInMSA = [&](int i){
            return 0 <= i && i < nColumns;
        };

        auto isValidColumnIndexInMSASequence = [&](int i){
            //std::cout << subjectColumnsBegin_incl << " <= " << i  << " && " << i  << " < " << subjectColumnsEnd_excl
            //<< " : " << (subjectColumnsBegin_incl <= i && i < subjectColumnsEnd_excl) << std::endl;
            return subjectColumnsBegin_incl <= i && i < subjectColumnsEnd_excl;
        };

        std::vector<MSAFeature2> result;

        const int alignment_coverage = nRows;

        for(int i = subjectColumnsBegin_incl; i < subjectColumnsEnd_excl; i++){

            const int localindex = i - subjectColumnsBegin_incl;

            if(supportptr[i] >= 0.5 && consensusptr[i] != sequence[localindex]){

                MSAFeature2 f;
                f.position = localindex;
                f.base = sequence[localindex];
                f.alignment_coverage = alignment_coverage;
                f.dataset_coverage = dataset_coverage;

                for(int column = i-k/2, columncount = 0; column <= i+k/2; ++column, ++columncount){
                    auto& weights = f.weights[columncount];
                    //std::cout << "i = " << i << " column = " << column << " columncount = " << columncount << std::endl;

                    f.partialsequence[columncount] = 'N';
                    if(isValidColumnIndexInMSASequence(column)){
                        f.partialsequence[columncount] = sequence[column - subjectColumnsBegin_incl];
                    }

                    std::fill(weights.begin(), weights.end(), 0.0f);
                    if(isValidColumnIndexInMSA(column)){
                        for(int row = 0; row < nRows; ++row){
                            const char base = multiple_sequence_alignment[row * nColumns + column];
                            const float weight = canUseWeights ? multiple_sequence_alignment_weights[row * nColumns + column] : 1.0;
                            switch(base){
                                case 'A': weights[0] += weight; break;
                                case 'C': weights[1] += weight; break;
                                case 'G': weights[2] += weight; break;
                                case 'T': weights[3] += weight; break;
                                //default: weights[4] += weight; break;
                            }
                        }
                    }

                }

                result.emplace_back(f);
            }
        }

        return result;
    }
}
