#ifndef CARE_BUILD_HPP
#define CARE_BUILD_HPP

#include <config.hpp>

#include <options.hpp>

#include <minhasher.hpp>
#include "readstorage.hpp"
#include <readlibraryio.hpp>
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

        // inline
        // SequenceFileProperties getSequenceFilePropertiesFromFileOptions(const FileOptions& fileOptions){
        //     if(fileOptions.nReads == 0 || fileOptions.maximum_sequence_length == 0 || fileOptions.minimum_sequence_length < 0) {
        //         std::cout << "Scanning file to get number of reads and min/max sequence length." << std::endl;

        //         return getSequenceFileProperties(fileOptions.inputfile);
        //     }else{
        //         std::cout << "Using the supplied number of reads and min/max sequence length." << std::endl;

        //         SequenceFileProperties sequenceFileProperties;
        //         sequenceFileProperties.maxSequenceLength = fileOptions.maximum_sequence_length;
        //         sequenceFileProperties.minSequenceLength = fileOptions.minimum_sequence_length;
        //         sequenceFileProperties.nReads = fileOptions.nReads;
        //         return sequenceFileProperties;
        //     }
        // }


        inline
        std::vector<SequenceFileProperties> getSequenceFilePropertiesFromFileOptions2(const FileOptions& fileOptions){
            std::vector<SequenceFileProperties> result;

            std::cout << "Scanning file to get number of reads and min/max sequence length." << std::endl;

            for(const auto& inputfile : fileOptions.inputfiles){
                result.emplace_back(getSequenceFileProperties(inputfile));

                std::cout << "----------------------------------------\n";
                std::cout << "File: " << inputfile << "\n";
                std::cout << "Reads: " << result.back().nReads << "\n";
                std::cout << "Minimum sequence length: " << result.back().minSequenceLength << "\n";
                std::cout << "Maximum sequence length: " << result.back().maxSequenceLength << "\n";
                std::cout << "----------------------------------------\n";
            }
            return result;
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

        SequenceFileProperties totalInputFileProperties;
        //std::vector<SequenceFileProperties> inputFileProperties;
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
                                			   cpu::ContiguousReadStorage& readStorage);

    BuiltDataStructures buildDataStructures2(
                                			const CorrectionOptions& correctionOptions,
                                			const RuntimeOptions& runtimeOptions,
                                            const MemoryOptions& memoryOptions,
                                			const FileOptions& fileOptions);

    BuiltDataStructures buildAndSaveDataStructures2(
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

            SequenceFileProperties totalInputFileProperties;
            //std::vector<SequenceFileProperties> inputFileProperties;
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
                                    			   const GpuReadStorageWithFlags& readStorage);


        BuiltGpuDataStructures buildGpuDataStructures2(
            const CorrectionOptions& correctionOptions,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const FileOptions& fileOptions);

        BuiltGpuDataStructures buildAndSaveGpuDataStructures2(
            const CorrectionOptions& correctionOptions,
            const RuntimeOptions& runtimeOptions,
            const MemoryOptions& memoryOptions,
            const FileOptions& fileOptions);
    }
#endif
}



#endif
