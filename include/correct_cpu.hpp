#ifndef CARE_CORRECT_CPU_HPP
#define CARE_CORRECT_CPU_HPP

#include <config.hpp>
#include <options.hpp>
#include <readlibraryio.hpp>
#include <minhasher.hpp>
#include <readstorage.hpp>

#include <mutex>
#include <memory>
#include <vector>

namespace care{
namespace cpu{

    void correct_cpu(
    				  const GoodAlignmentProperties& goodAlignmentProperties,
    				  const CorrectionOptions& correctionOptions,
    				  const RuntimeOptions& runtimeOptions,
    				  const FileOptions& fileOptions,
					  const MemoryOptions& memoryOptions,
                      const SequenceFileProperties& sequenceFileProperties,
                      Minhasher& minhasher,
                      cpu::ContiguousReadStorage& readStorage);


}
}

#endif
