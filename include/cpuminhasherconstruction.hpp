#ifndef CARE_CPUMINHASHERCONSTRUCTION_HPP
#define CARE_CPUMINHASHERCONSTRUCTION_HPP


#include <cpuminhasher.hpp>
#include <cpureadstorage.hpp>

#include <options.hpp>

#include <memory>
#include <utility>

namespace care{

    enum class CpuMinhasherType{
        Ordinary,
        None
    };

    std::pair<std::unique_ptr<CpuMinhasher>, CpuMinhasherType>
    constructCpuMinhasherFromCpuReadStorage(
        const FileOptions &fileOptions,
        const RuntimeOptions &runtimeOptions,
        const MemoryOptions& memoryOptions,
        const CorrectionOptions& correctionOptions,
        const SequenceFileProperties& totalInputFileProperties,
        const CpuReadStorage& cpuReadStorage,
        CpuMinhasherType requestedType = CpuMinhasherType::None
    );

}


#endif