#include <dispatch_correction.hpp>

#include <config.hpp>
#include <options.hpp>
#include <sequencefileio.hpp>
#include <minhasher.hpp>
#include <readstorage.hpp>

#include <correct_cpu.hpp>
#include <gpu/correct_gpu.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
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

        if(runtimeOptions.canUseGpu && runtimeOptions.deviceIds.size() > 0) {
            std::cout << "Running CARE GPU" << std::endl;

            std::cout << "Can use the following GPU device Ids: ";

            for(int i : runtimeOptions.deviceIds)
                std::cout << i << " ";

            std::cout << std::endl;

            gpu::correct_gpu(minhashOptions, alignmentOptions,
                        goodAlignmentProperties, correctionOptions,
                        runtimeOptions, fileOptions, sequenceFileProperties,
                        minhasher, readStorage,
                        maxCandidatesPerRead,
                        readIsCorrectedVector, locksForProcessedFlags,
                        nLocksForProcessedFlags);
        }else{
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

}
