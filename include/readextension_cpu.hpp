#ifndef CARE_READEXTENSION_CPU_HPP
#define CARE_READEXTENSION_CPU_HPP


#include <config.hpp>
#include <options.hpp>
#include <memoryfile.hpp>
#include <minhasher.hpp>
#include <readstorage.hpp>
#include <correctionresultprocessing.hpp>

namespace care{
    MemoryFileFixedSize<EncodedTempCorrectedSequence> 
    extend_cpu(
        const GoodAlignmentProperties& goodAlignmentProperties,
        const CorrectionOptions& correctionOptions,
        const RuntimeOptions& runtimeOptions,
        const FileOptions& fileOptions,
        const MemoryOptions& memoryOptions,
        const SequenceFileProperties& sequenceFileProperties,
        Minhasher& minhasher,
        cpu::ContiguousReadStorage& readStorage
    );





} //namespace care






#endif
