#ifndef CARE_BUILD_HPP
#define CARE_BUILD_HPP

#include <config.hpp>

#include <options.hpp>

#include <minhasher.hpp>
#include <minhasher_transform.hpp>
#include "readstorage.hpp"
#include "sequencefileio.hpp"
#include "sequence.hpp"
#include "threadsafe_buffer.hpp"

#ifdef __NVCC__
#include <gpu/distributedreadstorage.hpp>
#endif

#include <stdexcept>
#include <iostream>
#include <limits>
#include <thread>
#include <future>
#include <mutex>
#include <iterator>
#include <random>
#include <omp.h>

namespace care{

    enum class BuiltType {Constructed, Loaded};

    template<class T>
    struct BuiltDataStructure{
        T data;
        BuiltType builtType;
    };

    struct BuiltDataStructures{
        BuiltDataStructure<cpu::ContiguousReadStorage> builtReadStorage;
        BuiltDataStructure<Minhasher> builtMinhasher;

        SequenceFileProperties sequenceFileProperties;
    };




    BuiltDataStructure<cpu::ContiguousReadStorage> build_readstorage(const FileOptions& fileOptions,
                                                const RuntimeOptions& runtimeOptions,
                                                bool useQualityScores,
                                                read_number expectedNumberOfReads,
                                                int expectedMaximumReadLength);

    BuiltDataStructure<Minhasher> build_minhasher(const FileOptions& fileOptions,
                                			   const RuntimeOptions& runtimeOptions,
                                			   std::uint64_t nReads,
                                               const MinhashOptions& minhashOptions,
                                			   cpu::ContiguousReadStorage& readStorage);

    BuiltDataStructures buildDataStructures(const MinhashOptions& minhashOptions,
                                			const CorrectionOptions& correctionOptions,
                                			const RuntimeOptions& runtimeOptions,
                                			const FileOptions& fileOptions);

    BuiltDataStructures buildAndSaveDataStructures(const MinhashOptions& minhashOptions,
                                			const CorrectionOptions& correctionOptions,
                                			const RuntimeOptions& runtimeOptions,
                                			const FileOptions& fileOptions);

#ifdef __NVCC__

    namespace gpu{

        struct GpuReadStorageWithFlags{
            //std::vector<bool> readIsValidFlags;
            DistributedReadStorage readStorage;
        };

        struct BuiltGpuDataStructures{
            BuiltDataStructure<GpuReadStorageWithFlags> builtReadStorage;
            BuiltDataStructure<Minhasher> builtMinhasher;

            SequenceFileProperties sequenceFileProperties;
        };

        BuiltDataStructure<GpuReadStorageWithFlags> buildGpuReadStorage(const FileOptions& fileOptions,
                                                                        const RuntimeOptions& runtimeOptions,
                                                                        bool useQualityScores,
                                                                        read_number expectedNumberOfReads,
                                                                        int expectedMaximumReadLength);

        BuiltDataStructure<Minhasher> build_minhasher(const FileOptions& fileOptions,
                                    			   const RuntimeOptions& runtimeOptions,
                                    			   std::uint64_t nReads,
                                                   const MinhashOptions& minhashOptions,
                                    			   const GpuReadStorageWithFlags& readStorage);

        BuiltGpuDataStructures buildGpuDataStructures(const MinhashOptions& minhashOptions,
                                    			const CorrectionOptions& correctionOptions,
                                    			const RuntimeOptions& runtimeOptions,
                                    			const FileOptions& fileOptions);

        BuiltGpuDataStructures buildAndSaveGpuDataStructures(const MinhashOptions& minhashOptions,
                                                            const CorrectionOptions& correctionOptions,
                                                            const RuntimeOptions& runtimeOptions,
                                                            const FileOptions& fileOptions);
    }
#endif
}



#endif
