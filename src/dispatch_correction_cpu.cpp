#include <dispatch_correction.hpp>

#include <config.hpp>
#include <options.hpp>
#include <sequencefileio.hpp>
#include <minhasher.hpp>
#include <readstorage.hpp>

#include <correct_cpu.hpp>

#include <iostream>
#include <mutex>
#include <memory>
#include <vector>

namespace care{

    void dispatch_correction(const MinhashOptions& minhashOptions,
                			const AlignmentOptions& alignmentOptions,
                			const GoodAlignmentProperties& goodAlignmentProperties,
                			const CorrectionOptions& correctionOptions,
                			const RuntimeOptions& runtimeOptions,
                			const FileOptions& fileOptions,
                			const SequenceFileProperties& sequenceFileProperties,
                            Minhasher& minhasher,
                            cpu::ContiguousReadStorage& readStorage,
                            std::uint64_t maxCandidatesPerRead,
                			std::vector<char>& readIsCorrectedVector,
                			std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
                			std::size_t nLocksForProcessedFlags){

        std::cout << "Running CARE CPU" << std::endl;

        cpu::correct_cpu(minhashOptions, alignmentOptions,
                    goodAlignmentProperties, correctionOptions,
                    runtimeOptions, fileOptions, sequenceFileProperties,
                    minhasher, readStorage,
                    maxCandidatesPerRead,
                    readIsCorrectedVector, locksForProcessedFlags,
                    nLocksForProcessedFlags);
    }

}
