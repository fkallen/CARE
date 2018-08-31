#include "../inc/forestclassifier.hpp"

#include "../inc/combinedforestaligncov.hpp"
#include "../inc/combinedforestdatacov.hpp"

#include "../inc/celeganssrr543736.hpp"
#include "../inc/celeganssrx218989.hpp"
#include "../inc/ecolisrr490124.hpp"
#include "../inc/dmelanogastersrr82337.hpp"
#include "../inc/dmelanogastersrr988075.hpp"

#include <utility>
#include <stdexcept>

namespace care{
namespace forestclassifier{

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
                        double maxgini,
                        double correction_fraction){

        std::pair<int, int> forestresult;

        switch(mode){
        case care::forestclassifier::Mode::CombinedAlignCov:
                forestresult = combinedforestaligncov::shouldCorrect_forest(
                                            position_support,
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
        case care::forestclassifier::Mode::CombinedDataCov:
                forestresult = combinedforestdatacov::shouldCorrect_forest(
                                            position_support,
                                            position_coverage,
                                            alignment_coverage / dataset_coverage,
                                            dataset_coverage / dataset_coverage,
                                            min_support,
                                            min_coverage / (alignment_coverage * dataset_coverage),
                                            max_support,
                                            max_coverage / (alignment_coverage * dataset_coverage),
                                            mean_support,
                                            mean_coverage / (alignment_coverage * dataset_coverage),
                                            median_support,
                                            median_coverage / (alignment_coverage * dataset_coverage),
                                            maxgini);
                break;
        case care::forestclassifier::Mode::Species:
                //namespace speciestype = ecoli_srr490124;
                //namespace speciestype = celegans_srx218989;
                //namespace speciestype = dmelanogaster_srr82337;
                //namespace speciestype = celegans_srr543736;
                namespace speciestype = dmelanogaster_srr988075;

                forestresult = speciestype::shouldCorrect_forest(
                                            position_support,
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
        default: throw std::runtime_error("care::forestclassifier::shouldCorrect: Invalid mode!");
        }

        const int& count_correct = forestresult.second;
        const int& count_dontcorrect = forestresult.first;

        bool result = count_correct / (count_dontcorrect + count_correct) > correction_fraction;

        return result;
    }

}
}
