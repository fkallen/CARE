#ifndef CARE_FOREST_CLASSIFIER
#define CARE_FOREST_CLASSIFIER

namespace care{
namespace forestclassifier{

    enum class Mode{
        CombinedAlignCov,
        CombinedDataCov,
        Species,
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
                                double maxgini,
                                double correction_fraction);

}
}


#endif
