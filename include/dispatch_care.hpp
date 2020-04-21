#ifndef CARE_DISPATCH_CARE_HPP
#define CARE_DISPATCH_CARE_HPP

#include <config.hpp>
#include <options.hpp>

namespace care{

    void performCorrection(
                            CorrectionOptions correctionOptions,
                            RuntimeOptions runtimeOptions,
                            MemoryOptions memoryOptions,
                            FileOptions fileOptions,
                            GoodAlignmentProperties goodAlignmentProperties);


}







#endif
