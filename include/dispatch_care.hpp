#ifndef CARE_DISPATCH_CARE_HPP
#define CARE_DISPATCH_CARE_HPP

#include <config.hpp>
#include <options.hpp>

namespace care{

    void performCorrection(MinhashOptions minhashOptions,
                            AlignmentOptions alignmentOptions,
                            CorrectionOptions correctionOptions,
                            RuntimeOptions runtimeOptions,
                            FileOptions fileOptions,
                            GoodAlignmentProperties goodAlignmentProperties);


}







#endif
