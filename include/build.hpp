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

#include <string>
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

    namespace detail{

        inline
        SequenceFileProperties getSequenceFilePropertiesFromFileOptions(const FileOptions& fileOptions){
            if(fileOptions.nReads == 0 || fileOptions.maximum_sequence_length == 0 || fileOptions.minimum_sequence_length < 0) {
                std::cout << "Scanning file to get number of reads and min/max sequence length." << std::endl;

                return getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);
            }else{
                std::cout << "Using the supplied number of reads and min/max sequence length." << std::endl;

                SequenceFileProperties sequenceFileProperties;
                sequenceFileProperties.maxSequenceLength = fileOptions.maximum_sequence_length;
                sequenceFileProperties.minSequenceLength = fileOptions.minimum_sequence_length;
                sequenceFileProperties.nReads = fileOptions.nReads;
                return sequenceFileProperties;
            }
        }

        inline
        void printInputFileProperties(std::ostream& os, const std::string& filename, const SequenceFileProperties& props){
            os << "----------------------------------------\n";
            os << "File: " << filename << "\n";
            os << "Reads: " << props.nReads << "\n";
            os << "Minimum sequence length: " << props.minSequenceLength << "\n";
            os << "Maximum sequence length: " << props.maxSequenceLength << "\n";
            os << "----------------------------------------\n";
        }; 

    }

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
                                               const MemoryOptions& memoryOptions,
                                			   std::uint64_t nReads,
                                               const MinhashOptions& minhashOptions,
                                			   cpu::ContiguousReadStorage& readStorage);

    BuiltDataStructures buildDataStructures(const MinhashOptions& minhashOptions,
                                			const CorrectionOptions& correctionOptions,
                                			const RuntimeOptions& runtimeOptions,
                                            const MemoryOptions& memoryOptions,
                                			const FileOptions& fileOptions);

    BuiltDataStructures buildAndSaveDataStructures(const MinhashOptions& minhashOptions,
                                			const CorrectionOptions& correctionOptions,
                                			const RuntimeOptions& runtimeOptions,
                                            const MemoryOptions& memoryOptions,
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
                                                   const MemoryOptions& memoryOptions,
                                    			   std::uint64_t nReads,
                                                   const MinhashOptions& minhashOptions,
                                    			   const GpuReadStorageWithFlags& readStorage);

        BuiltGpuDataStructures buildGpuDataStructures(const MinhashOptions& minhashOptions,
                                    			const CorrectionOptions& correctionOptions,
                                    			const RuntimeOptions& runtimeOptions,
                                                const MemoryOptions& memoryOptions,
                                    			const FileOptions& fileOptions);

        BuiltGpuDataStructures buildAndSaveGpuDataStructures(const MinhashOptions& minhashOptions,
                                                            const CorrectionOptions& correctionOptions,
                                                            const RuntimeOptions& runtimeOptions,
                                                            const MemoryOptions& memoryOptions,
                                                            const FileOptions& fileOptions);
    }
#endif
}



#endif
