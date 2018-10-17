#ifndef HUMAN_ERX069715_HPP
#define HUMAN_ERX069715_HPP

#include <utility>

namespace human_erx069715{

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
