#ifndef CARE_DISPATCH_CARE_CORRECT_CPU_HPP
#define CARE_DISPATCH_CARE_CORRECT_CPU_HPP

#include <options.hpp>

namespace care{

    void performCorrection(
        CorrectionOptions correctionOptions,
        RuntimeOptions runtimeOptions,
        MemoryOptions memoryOptions,
        FileOptions fileOptions,
        GoodAlignmentProperties goodAlignmentProperties
    );

}



#endif
