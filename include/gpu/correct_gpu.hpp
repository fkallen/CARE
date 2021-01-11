#ifndef CARE_CORRECT_GPU_HPP
#define CARE_CORRECT_GPU_HPP

#include <gpu/gpuminhasher.cuh>
#include <gpu/gpureadstorage.cuh>

#include <config.hpp>
#include <correctionresultprocessing.hpp>
#include <memoryfile.hpp>

#include <options.hpp>

#include <mutex>
#include <memory>
#include <vector>

namespace care {
namespace gpu {


MemoryFileFixedSize<EncodedTempCorrectedSequence> 
correct_gpu(
	const GoodAlignmentProperties& goodAlignmentProperties,
	const CorrectionOptions& correctionOptions,
	const RuntimeOptions& runtimeOptions,
	const FileOptions& fileOptions,
	const MemoryOptions& memoryOptions,
	const SequenceFileProperties& sequenceFileProperties,
	GpuMinhasher& minhasher,
	GpuReadStorage& readStorage
);

}
}

#endif
