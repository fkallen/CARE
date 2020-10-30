#ifndef CARE_KERNEL_LAUNCH_HPP
#define CARE_KERNEL_LAUNCH_HPP

#include <hpc_helpers.cuh>
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
    MSACorrectSubjectImplicit,
    ConstructAnchorResults,
    MSAConstruction,
    MSACandidateRefinementSingleIter,
    MSACandidateRefinementMultiIter,
    FlagCandidatesToBeCorrected,
};

struct KernelLaunchConfig {
	int threads_per_block;
	int smem;
};

constexpr bool operator<(const KernelLaunchConfig& lhs, const KernelLaunchConfig& rhs){
	return lhs.threads_per_block < rhs.threads_per_block
	       && lhs.smem < rhs.smem;
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
    cudaGetDeviceProperties(&handle.deviceProperties, deviceId); CUERR;
    return handle;
}




}
}





#endif