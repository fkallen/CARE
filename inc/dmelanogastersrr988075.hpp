#ifndef D_MELANOGASTER_SRR988075_HPP
#define D_MELANOGASTER_SRR988075_HPP

#include <utility>

namespace dmelanogaster_srr988075{
    
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
