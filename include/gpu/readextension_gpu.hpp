#ifndef CARE_READEXTENSION_GPU_HPP
#define CARE_READEXTENSION_GPU_HPP


#include <config.hpp>
#include <options.hpp>
#include <memoryfile.hpp>
#include <minhasher.hpp>
#include <readstorage.hpp>
#include <correctionresultprocessing.hpp>
#include <extensionresultprocessing.hpp>

#include <vector>

namespace care{
namespace gpu{
    
    MemoryFileFixedSize<ExtendedRead> 
    extend_gpu(
        const GoodAlignmentProperties& goodAlignmentProperties,
        const CorrectionOptions& correctionOptions,
        const ExtensionOptions& extensionOptions,
        const RuntimeOptions& runtimeOptions,
        const FileOptions& fileOptions,
        const MemoryOptions& memoryOptions,
        const SequenceFileProperties& sequenceFileProperties,
        Minhasher& minhasher,
        cpu::ContiguousReadStorage& readStorage
    );



} //namespace gpu

} //namespace care






#endif
