#ifndef CARE_ARGS_HPP
#define CARE_ARGS_HPP

#include "cxxopts/cxxopts.hpp"



namespace care{

    struct MinhashOptions {
    	int maps;
    	int k;
    };

struct Args{
    cxxopts::Options options;
    bool useQScores = false;
	bool help = false;

    Args(int argc, char** argv);

    Args(const cxxopts::Options& options);

    MinhashOptions getMinhashOptions() const;

    bool isValid() const;
};



}


#endif
