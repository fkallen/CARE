#include <utility> //std::pair

extern "C"{  // required for loading as shared object

    // returns voting of all trees in the forest
    // pair.first counts number of votes against correction
    // pair.second counts number of vots for correction
    std::pair<int, int> shouldCorrect_forest(
        double position_support, 
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
        double maxgini
    ) {
        std::pair<int,int> result{13,42};

        return result;
    }


}