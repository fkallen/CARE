#ifndef CARE_READEXTENSION_CPU_HPP
#define CARE_READEXTENSION_CPU_HPP


#include <config.hpp>
#include <options.hpp>
#include <memoryfile.hpp>
#include <cpuminhasher.hpp>
#include <cpureadstorage.hpp>
#include <correctionresultprocessing.hpp>
#include <extensionresultprocessing.hpp>

#include <vector>

namespace care{
    MemoryFileFixedSize<ExtendedRead> 
    //std::vector<ExtendedRead>
    extend_cpu(
        const GoodAlignmentProperties& goodAlignmentProperties,
        const CorrectionOptions& correctionOptions,
        const ExtensionOptions& extensionOptions,
        const RuntimeOptions& runtimeOptions,
        const FileOptions& fileOptions,
        const MemoryOptions& memoryOptions,
        const CpuMinhasher& minhasher,
        const CpuReadStorage& readStorage
    );





} //namespace care






#endif
