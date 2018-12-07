#include "../include/treeclassifier.hpp"

#include "../include/celeganssrr543736.hpp"
#include "../include/celeganssrx218989.hpp"
#include "../include/ecolisrr490124.hpp"
#include "../include/dmelanogastersrr82337.hpp"

#include <stdexcept>

namespace care{
namespace treeclassifier{

    bool shouldCorrect(Mode mode,
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
                                double maxgini){

        bool result = false;

        switch(mode){
        case care::treeclassifier::Mode::ecoli_srr490124:
            result = ecoli_srr490124::shouldCorrect(position_support,
                                            position_coverage,
                                            alignment_coverage,
                                            dataset_coverage,
                                            min_support,
                                            min_coverage / alignment_coverage,
                                            max_support,
                                            max_coverage / alignment_coverage,
                                            mean_support,
                                            mean_coverage / alignment_coverage,
                                            median_support,
                                            median_coverage / alignment_coverage,
                                            maxgini);
            break;
        case care::treeclassifier::Mode::celegans_srx218989:
            result = celegans_srx218989::shouldCorrect(position_support,
                                            position_coverage,
                                            alignment_coverage,
                                            dataset_coverage,
                                            min_support,
                                            min_coverage / alignment_coverage,
                                            max_support,
                                            max_coverage / alignment_coverage,
                                            mean_support,
                                            mean_coverage / alignment_coverage,
                                            median_support,
                                            median_coverage / alignment_coverage,
                                            maxgini);
            break;
        case care::treeclassifier::Mode::dmelanogaster_srr82337:
            result = dmelanogaster_srr82337::shouldCorrect(position_support,
                                            position_coverage,
                                            alignment_coverage,
                                            dataset_coverage,
                                            min_support,
                                            min_coverage / alignment_coverage,
                                            max_support,
                                            max_coverage / alignment_coverage,
                                            mean_support,
                                            mean_coverage / alignment_coverage,
                                            median_support,
                                            median_coverage / alignment_coverage,
                                            maxgini);
            break;
        case care::treeclassifier::Mode::celegans_srr543736:
            result = celegans_srr543736::shouldCorrect(position_support,
                                            position_coverage,
                                            alignment_coverage,
                                            dataset_coverage,
                                            min_support,
                                            min_coverage / alignment_coverage,
                                            max_support,
                                            max_coverage / alignment_coverage,
                                            mean_support,
                                            mean_coverage / alignment_coverage,
                                            median_support,
                                            median_coverage / alignment_coverage,
                                            maxgini);
            break;
        default: throw std::runtime_error("care::treeclassifier::shouldCorrect: Invalid mode!");
        }

        return result;
    }

}
}
