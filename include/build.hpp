#ifndef CARE_BUILD_HPP
#define CARE_BUILD_HPP

#include <config.hpp>

#include <options.hpp>

#include <minhasher.hpp>
#include "readstorage.hpp"
#include <readlibraryio.hpp>
#include "sequence.hpp"
#include "threadsafe_buffer.hpp"


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

    namespace builddetail{
        inline 
        int getKmerSizeForHashing(int maximumReadLength){
            if(maximumReadLength < 160){
                return 20;
            }else{
                return 32;
            }
        }
    }

    enum class BuiltType {Constructed, Loaded};

    template<class T>
    struct BuiltDataStructure{
        T data;
        BuiltType builtType;
    };

    struct BuiltDataStructures{
        int kmerlength;

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


}



#endif
