#ifndef CARE_READEXTENSION_CPU_HPP
#define CARE_READEXTENSION_CPU_HPP


#include <config.hpp>
#include <options.hpp>
#include <serializedobjectstorage.hpp>
#include <cpuminhasher.hpp>
#include <cpureadstorage.hpp>
#include <extendedread.hpp>

#include <vector>
#include <functional>
namespace care{

    using SubmitReadyExtensionResultsCallback = std::function<void(
        std::vector<ExtendedRead> extendedReads, 
        std::vector<EncodedExtendedRead> encodedExtendedReads,
        std::vector<read_number> idsOfNotExtendedReads
    )>;
    
    void extend_cpu(
        const ProgramOptions& programOptions,
        const CpuMinhasher& minhasher,
        const CpuReadStorage& readStorage,
        SubmitReadyExtensionResultsCallback submitReadyResults //needs to be thread-safe
    );





} //namespace care






#endif
