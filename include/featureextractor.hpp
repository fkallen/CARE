#ifndef CARE_FEATURE_EXTRACTOR
#define CARE_FEATURE_EXTRACTOR

#include <vector>
#include <iostream>
#include <array>

namespace care{

    struct MSAFeature{
        int position = -1;

        double position_support = 0.0; // support of the center of k-region (at read position "position")
        int position_coverage = 0; // coverage of the base in read at center of k-region (at read position "position")
        int alignment_coverage = 0; // number of sequences in MSA. equivalent to the max possible value of coverage.
        int dataset_coverage = 0; // estimated coverage of dataset

        double min_support = 0.0;
        double min_coverage = 0.0;
        double max_support = 0.0;
        double max_coverage = 0.0;
        double mean_support = 0.0;
        double mean_coverage = 0.0;
        double median_support = 0.0;
        double median_coverage = 0.0;
    };

    std::ostream& operator<<(std::ostream& os, const MSAFeature& f);

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
                                            int dataset_coverage);

    struct MSAFeature2{
        using Column = std::array<float, 4>;

        int position = -1;
        char base = 'N';
        int alignment_coverage = 0;
        int dataset_coverage = 0;
        std::array<char, 17> partialsequence;
        std::array<Column, 17> weights;
    };

    std::ostream& operator<<(std::ostream& os, const MSAFeature2& f);

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
                                    int dataset_coverage);

}



#endif
