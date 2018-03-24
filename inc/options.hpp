#ifndef CARE_OPTIONS_HPP
#define CARE_OPTIONS_HPP

namespace care{
    enum class CorrectionMode {Hamming, Graph};

    struct MinhashOptions {
        int maps = 2;
        int k = 16;
    };

    struct AlignmentOptions{
        int alignmentscore_match = 1;
        int alignmentscore_sub = -1;
        int alignmentscore_ins = -100;
        int alignmentscore_del = -100;
    };

    struct GoodAlignmentProperties{
        int min_overlap = 35;
        double max_mismatch_ratio = 0.2;
        double min_overlap_ratio = 0.35;
    };

    struct CorrectionOptions{
        CorrectionMode correctionMode = CorrectionMode::Hamming;
        bool correctCandidates = false;
        bool useQualityScores = true;
        double estimatedCoverage = 1.0;
        double errorrate = 0.01;
        double m_coverage = 0.6;
        double graphalpha = 1.0;
        double graphx = 1.5;
        int kmerlength = 16;
    };
}



#endif
