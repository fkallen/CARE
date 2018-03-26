#ifndef CARE_CORRECT_HPP
#define CARE_CORRECT_HPP

#include "options.hpp"

#include "minhasher.hpp"
#include "readstorage.hpp"

#include <memory>
#include <mutex>
#include <vector>

namespace care{

void correct(const MinhashOptions& minhashOptions,
				  const AlignmentOptions& alignmentOptions,
				  const GoodAlignmentProperties& goodAlignmentProperties,
				  const CorrectionOptions& correctionOptions,
				  const RuntimeOptions& runtimeOptions,
				  const FileOptions& fileOptions,
                  Minhasher& minhasher,
                  ReadStorage& readStorage,
				  std::vector<char>& readIsProcessedVector,
				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
				  size_t nLocksForProcessedFlags,
				  const std::vector<int>& deviceIds);


}

#endif
