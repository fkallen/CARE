#ifndef CARE_ARGS_HPP
#define CARE_ARGS_HPP

#include "cxxopts/cxxopts.hpp"
#include "options.hpp"



namespace care{

struct Args{
    cxxopts::Options options;
    bool useQScores = false;
	bool help = false;

    Args(int argc, char** argv);

    Args(const cxxopts::Options& options);

    MinhashOptions getMinhashOptions() const;
    AlignmentOptions getAlignmentOptions() const;
    GoodAlignmentProperties getGoodAlignmentProperties() const;
    CorrectionOptions getCorrectionOptions() const;

    bool isValid() const;
};



}


#endif
