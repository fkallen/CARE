#ifndef C_ELEGANS_SRR543736_HPP
#define C_ELEGANS_SRR543736_HPP

#include <utility>

namespace celegans_srr543736{

    bool shouldCorrect(double position_support,
                        double position_coverage,
                        double alignment_coverage,
                        double dataset_coverage,
                        double min_support,
                        double min_coverage, //normalized to number of reads in msa
                        double max_support,
                        double max_coverage, //normalized to number of reads in msa
                        double mean_support,
                        double mean_coverage, //normalized to number of reads in msa
                        double median_support,
                        double median_coverage, //normalized to number of reads in msa
                        double maxgini);

    std::pair<int, int> shouldCorrect_forest(double position_support,
                        double position_coverage,
                        double alignment_coverage,
                        double dataset_coverage,
                        double min_support,
                        double min_coverage, //normalized to number of reads in msa
                        double max_support,
                        double max_coverage, //normalized to number of reads in msa
                        double mean_support,
                        double mean_coverage, //normalized to number of reads in msa
                        double median_support,
                        double median_coverage, //normalized to number of reads in msa
                        double maxgini);
}

#endif
