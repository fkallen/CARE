#ifndef CARE_DISPATCH_CARE_HPP
#define CARE_DISPATCH_CARE_HPP

#include <config.hpp>
#include <options.hpp>

#include <vector>

namespace care{

    std::vector<int> getUsableDeviceIds(std::vector<int> deviceIds);

    void performCorrection(
                            CorrectionOptions correctionOptions,
                            RuntimeOptions runtimeOptions,
                            MemoryOptions memoryOptions,
                            FileOptions fileOptions,
                            GoodAlignmentProperties goodAlignmentProperties);

    void performExtension(
                            CorrectionOptions correctionOptions,
                            ExtensionOptions extensionOptions,
                            RuntimeOptions runtimeOptions,
                            MemoryOptions memoryOptions,
                            FileOptions fileOptions,
                            GoodAlignmentProperties goodAlignmentProperties);


}







#endif
