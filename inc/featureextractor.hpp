#ifndef CARE_FEATURE_EXTRACTOR
#define CARE_FEATURE_EXTRACTOR

#include "multiple_sequence_alignment.hpp"

#include <vector>
#include <iostream>

namespace care{

    struct Feature{
        double A_weight_normalized = 0.0f;
        double C_weight_normalized= 0.0f;
        double G_weight_normalized= 0.0f;
        double T_weight_normalized= 0.0f;
        double support = 0.0f; // support of the center of k-region
        double col_support = 0.0f; // support of the column of this feature
        int original_base_coverage = 0; // original_base_coverage of the center of k-region
        int col_coverage = 0; //coverage of the oclumn of this feature
        int alignment_coverage = 0; // number of sequences in MSA. equivalent to the max possible value of coverage.
        int dataset_coverage = 0;
        int position_in_read = 0; // the position in the center of k-region
        int k = 0;
        char original_base = 'N';

        friend std::ostream& operator<<(std::ostream& os, const Feature& f);
    };

    struct MSAFeature{
        std::vector<Feature> features;
        int position;
    };

    void writeFeatures(std::ostream& os, const std::vector<MSAFeature>& vec);

    std::vector<MSAFeature> extractFeatures(const pileup::PileupImage& pileup, const std::string& sequence,
                                    int k, double support_threshold,
                                    int dataset_coverage);

}



#endif
