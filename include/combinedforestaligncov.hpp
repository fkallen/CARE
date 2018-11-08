#ifndef CARE_COMBINED_FOREST_ALIGN_COV
#define CARE_COMBINED_FOREST_ALIGN_COV

#include <utility>

namespace combinedforestaligncov{
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
