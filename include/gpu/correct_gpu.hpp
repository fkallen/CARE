#ifndef CARE_CORRECT_GPU_HPP
#define CARE_CORRECT_GPU_HPP

#include <gpu/distributedreadstorage.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/singlegpuminhasher.cuh>

#include <config.hpp>
#include <correctionresultprocessing.hpp>

#include <options.hpp>
#include <readlibraryio.hpp>
#include <readstorage.hpp>

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
	SingleGpuMinhasher& minhasher,
	DistributedReadStorage& readStorage
);

MemoryFileFixedSize<EncodedTempCorrectedSequence> 
correct_gpu(
	const GoodAlignmentProperties& goodAlignmentProperties,
	const CorrectionOptions& correctionOptions,
	const RuntimeOptions& runtimeOptions,
	const FileOptions& fileOptions,
	const MemoryOptions& memoryOptions,
	const SequenceFileProperties& sequenceFileProperties,
	GpuMinhasher& minhasher,
	DistributedReadStorage& readStorage
);


}
}

#endif
