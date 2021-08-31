#ifndef CARE_DISPATCH_CARE_EXTEND_CPU_HPP
#define CARE_DISPATCH_CARE_EXTEND_CPU_HPP

#include <options.hpp>

namespace care{

    void performExtension(
        CorrectionOptions correctionOptions,
        ExtensionOptions extensionOptions,
        RuntimeOptions runtimeOptions,
        MemoryOptions memoryOptions,
        FileOptions fileOptions,
        GoodAlignmentProperties goodAlignmentProperties
    );
}







#endif
