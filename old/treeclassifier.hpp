#ifndef CARE_TREE_CLASSIFIER
#define CARE_TREE_CLASSIFIER

namespace care{
namespace treeclassifier{

    enum class Mode{
        ecoli_srr490124,
        celegans_srx218989,
        dmelanogaster_srr82337,
        celegans_srr543736,
    };

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
                            double maxgini);

}
}


#endif
