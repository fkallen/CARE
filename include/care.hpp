#ifndef CARE_HPP
#define CARE_HPP

#include <options.hpp>

namespace care{

//void performCorrection(const cxxopts::ParseResult& args);

void performCorrection(MinhashOptions minhashOptions,
                        AlignmentOptions alignmentOptions,
                        CorrectionOptions correctionOptions,
                        RuntimeOptions runtimeOptions,
                        FileOptions fileOptions,
                        GoodAlignmentProperties goodAlignmentProperties);
}

#endif
