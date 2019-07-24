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
            cpu::ContiguousReadStorage& cpuReadStorage,
            std::uint64_t maxCandidatesPerRead,
			std::vector<char>& readIsCorrectedVector,
			std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
			std::size_t nLocksForProcessedFlags);

void correct_gpu2(const MinhashOptions& minhashOptions,
			const AlignmentOptions& alignmentOptions,
			const GoodAlignmentProperties& goodAlignmentProperties,
			const CorrectionOptions& correctionOptions,
			const RuntimeOptions& runtimeOptions,
			const FileOptions& fileOptions,
			const SequenceFileProperties& sequenceFileProperties,
            Minhasher& minhasher,
            DistributedReadStorage& readStorage,
            std::uint64_t maxCandidatesPerRead,
			std::vector<char>& readIsCorrectedVector,
			std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
			std::size_t nLocksForProcessedFlags);

}
}

#endif
