#ifndef CARE_CPUMINHASHERCONSTRUCTION_HPP
#define CARE_CPUMINHASHERCONSTRUCTION_HPP


#include <cpuminhasher.hpp>
#include <cpureadstorage.hpp>
#include <options.hpp>

#include <memory>
#include <utility>
#include <string>

namespace care{

    enum class CpuMinhasherType{
        Ordinary,
        OrdinarySingleHash,
        None
    };

    std::string to_string(CpuMinhasherType type);

    std::pair<std::unique_ptr<CpuMinhasher>, CpuMinhasherType>
    constructCpuMinhasherFromCpuReadStorage(
        const FileOptions &fileOptions,
        const RuntimeOptions &runtimeOptions,
        const MemoryOptions& memoryOptions,
        const CorrectionOptions& correctionOptions,
        const CpuReadStorage& cpuReadStorage,
        CpuMinhasherType requestedType = CpuMinhasherType::None
    );

}


#endif