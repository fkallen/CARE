#ifndef CARE_CORRECT_GPU_HPP
#define CARE_CORRECT_GPU_HPP

#include <gpu/distributedreadstorage.hpp>
#include <gpu/fakegpuminhasher.cuh>
#include <gpu/singlegpuminhasher.cuh>
#include <gpu/multigpuminhasher.cuh>

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

#if 1
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
#endif

#if 1
MemoryFileFixedSize<EncodedTempCorrectedSequence> 
correct_gpu(
	const GoodAlignmentProperties& goodAlignmentProperties,
	const CorrectionOptions& correctionOptions,
	const RuntimeOptions& runtimeOptions,
	const FileOptions& fileOptions,
	const MemoryOptions& memoryOptions,
	const SequenceFileProperties& sequenceFileProperties,
	MultiGpuMinhasher& minhasher,
	DistributedReadStorage& readStorage
);
#endif

MemoryFileFixedSize<EncodedTempCorrectedSequence> 
correct_gpu(
	const GoodAlignmentProperties& goodAlignmentProperties,
	const CorrectionOptions& correctionOptions,
	const RuntimeOptions& runtimeOptions,
	const FileOptions& fileOptions,
	const MemoryOptions& memoryOptions,
	const SequenceFileProperties& sequenceFileProperties,
	FakeGpuMinhasher& minhasher,
	DistributedReadStorage& readStorage
);


}
}

#endif
