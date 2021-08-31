#ifndef CARE_EXTENSION_RESULT_OUTPUT_HPP
#define CARE_EXTENSION_RESULT_OUTPUT_HPP

#include <options.hpp>

#include <serializedobjectstorage.hpp>
#include <readlibraryio.hpp>

#include <string>
#include <vector>


namespace care{

void constructOutputFileFromExtensionResults(
    const std::vector<std::string>& originalReadFiles,
    SerializedObjectStorage& partialResults, 
    FileFormat outputFormat,
    const std::string& extendedOutputfile,
    const std::vector<std::string>& outputfiles,
    SequencePairType pairmode,
    bool outputToSingleFile
);

}



#endif