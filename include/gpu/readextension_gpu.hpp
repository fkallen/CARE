#ifndef CARE_READEXTENSION_GPU_HPP
#define CARE_READEXTENSION_GPU_HPP


#include <config.hpp>
#include <options.hpp>
#include <serializedobjectstorage.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/gpureadstorage.cuh>
#include <correctionresultprocessing.hpp>
#include <extensionresultprocessing.hpp>

#include <gpu/gpuminhasher.cuh>

#include <vector>

namespace care{
namespace gpu{
    
    SerializedObjectStorage extend_gpu(
        const GoodAlignmentProperties& goodAlignmentProperties,
        const CorrectionOptions& correctionOptions,
        const ExtensionOptions& extensionOptions,
        const RuntimeOptions& runtimeOptions,
        const FileOptions& fileOptions,
        const MemoryOptions& memoryOptions,
        const GpuMinhasher& minhasher,
        const GpuReadStorage& gpuReadStorage
    );



} //namespace gpu

} //namespace care






#endif
