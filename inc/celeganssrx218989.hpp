#ifndef C_ELEGANS_SRX218989_HPP
#define C_ELEGANS_SRX218989_HPP

#include <utility>

namespace celegans_srx218989{
    //coverage is normalized to number of reads in msa
    bool shouldCorrect(double min_col_support, double min_col_coverage,
        double max_col_support, double max_col_coverage,
        double mean_col_support, double mean_col_coverage,
        double median_col_support, double median_col_coverage,
        double maxgini);


    std::pair<int, int> shouldCorrect_forest(double min_col_support, double min_col_coverage,
        double max_col_support, double max_col_coverage,
        double mean_col_support, double mean_col_coverage,
        double median_col_support, double median_col_coverage,
        double maxgini);


    bool shouldCorrect(double position_support,
        double position_coverage,
        double alignment_coverage,
        double dataset_coverage,
        double min_support,
        double min_coverage,
        double max_support,
        double max_coverage,
        double mean_support,
        double mean_coverage,
        double median_support,
        double median_coverage,
        double maxgini);


    std::pair<int, int> shouldCorrect_forest(double position_support,
        double position_coverage,
        double alignment_coverage,
        double dataset_coverage,
        double min_support,
        double min_coverage,
        double max_support,
        double max_coverage,
        double mean_support,
        double mean_coverage,
        double median_support,
        double median_coverage,
        double maxgini);
}

#endif
