#ifndef CARE_KERNEL_LAUNCH_HPP
#define CARE_KERNEL_LAUNCH_HPP

#include <hpc_helpers.cuh>
#include <gpu/cudaerrorcheck.cuh>

#include <map>

namespace care{
namespace gpu{

enum class KernelId {
    Conversion2BitTo2BitHiLo,
    Conversion2BitTo2BitHiLoNN,
    Conversion2BitTo2BitHiLoNT,
    Conversion2BitTo2BitHiLoTT,
    SelectIndicesOfGoodCandidates,
    GetNumCorrectedCandidatesPerAnchor,
    PopcountSHDSmem,
    PopcountSHDReg,
    PopcountRightSHDSmem,
    PopcountRightSHDReg,
	FilterAlignmentsByMismatchRatio,
	MSACorrectSubject,
	MSACorrectCandidates,
    msaCorrectCandidatesAndComputeEdits,
    MSACorrectSubjectImplicit,
    ConstructSequenceCorrectionResults,
    MSAConstruction,
    MSACandidateRefinementSingleIter,
    MSACandidateRefinementMultiIter,
    FlagCandidatesToBeCorrected,
    FlagCandidatesToBeCorrectedWithExcludeFlags,
};

struct KernelLaunchConfig {
	int threads_per_block;
	int smem;
};

constexpr bool operator<(const KernelLaunchConfig& lhs, const KernelLaunchConfig& rhs){
    if(lhs.threads_per_block < rhs.threads_per_block){
        return true;
    }else if(lhs.threads_per_block > rhs.threads_per_block){
        return false;
    }else{
        return lhs.smem < rhs.smem;
    }
}

struct KernelProperties {
	int max_blocks_per_SM = 1;
};

struct KernelLaunchHandle {
	int deviceId;
	cudaDeviceProp deviceProperties;
	std::map<KernelId, std::map<KernelLaunchConfig, KernelProperties> > kernelPropertiesMap;
};


__inline__
KernelLaunchHandle make_kernel_launch_handle(int deviceId){
    KernelLaunchHandle handle;
    handle.deviceId = deviceId;
    CUDACHECK(cudaGetDeviceProperties(&handle.deviceProperties, deviceId));
    return handle;
}




}
}





#endif