#ifndef CARE_CORRECT_CPU_HPP
#define CARE_CORRECT_CPU_HPP

#include <config.hpp>
#include <correctionresultprocessing.hpp>
#include <minhasher.hpp>
#include <options.hpp>
#include <readlibraryio.hpp>
#include <readstorage.hpp>


namespace care{
namespace cpu{

    MemoryFileFixedSize<EncodedTempCorrectedSequence>
	correct_cpu(
		const GoodAlignmentProperties& goodAlignmentProperties,
		const CorrectionOptions& correctionOptions,
		const RuntimeOptions& runtimeOptions,
		const FileOptions& fileOptions,
		const MemoryOptions& memoryOptions,
		const SequenceFileProperties& sequenceFileProperties,
		Minhasher& minhasher,
		cpu::ContiguousReadStorage& readStorage
	);


}
}

#endif
