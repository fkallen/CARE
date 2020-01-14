#ifndef CARE_CORRECT_GPU_HPP
#define CARE_CORRECT_GPU_HPP

#include <config.hpp>
#include <options.hpp>
#include <sequencefileio.hpp>
#include <minhasher.hpp>
#include <readstorage.hpp>
#include <gpu/distributedreadstorage.hpp>

#include <mutex>
#include <memory>
#include <vector>

namespace care {
namespace gpu {


void correct_gpu(const MinhashOptions& minhashOptions,
			const AlignmentOptions& alignmentOptions,
			const GoodAlignmentProperties& goodAlignmentProperties,
			const CorrectionOptions& correctionOptions,
			const RuntimeOptions& runtimeOptions,
			const FileOptions& fileOptions,
			const SequenceFileProperties& sequenceFileProperties,
            Minhasher& minhasher,
            DistributedReadStorage& readStorage,
            std::uint64_t maxCandidatesPerRead);


namespace test{

void correct_gpu(const MinhashOptions& minhashOptions,
			const AlignmentOptions& alignmentOptions,
			const GoodAlignmentProperties& goodAlignmentProperties,
			const CorrectionOptions& correctionOptions,
			const RuntimeOptions& runtimeOptions,
			const FileOptions& fileOptions,
			const SequenceFileProperties& sequenceFileProperties,
            Minhasher& minhasher,
            DistributedReadStorage& readStorage,
            std::uint64_t maxCandidatesPerRead);			

}

}
}

#endif
