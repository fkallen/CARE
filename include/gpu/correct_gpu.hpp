#ifndef CARE_CORRECT_GPU_HPP
#define CARE_CORRECT_GPU_HPP

#include <config.hpp>
#include <options.hpp>
#include <readlibraryio.hpp>
#include <minhasher.hpp>
#include <readstorage.hpp>
#include <gpu/distributedreadstorage.hpp>

#include <mutex>
#include <memory>
#include <vector>

namespace care {
namespace gpu {


void correct_gpu(
	const GoodAlignmentProperties& goodAlignmentProperties,
	const CorrectionOptions& correctionOptions,
	const RuntimeOptions& runtimeOptions,
	const FileOptions& fileOptions,
	const MemoryOptions& memoryOptions,
	const SequenceFileProperties& sequenceFileProperties,
	Minhasher& minhasher,
	DistributedReadStorage& readStorage
);


}
}

#endif
