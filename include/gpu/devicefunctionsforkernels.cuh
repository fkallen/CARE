#ifndef DEVICEFUNCTIONSFORKERNELS_CUH
#define DEVICEFUNCTIONSFORKERNELS_CUH

#include <hpc_helpers.cuh>
#include <cmath>

namespace care{
namespace gpu{

    HOSTDEVICEQUALIFIER
    INLINEQUALIFIER
    constexpr bool feq(float l, float r){
        constexpr float threshold = 1e-5;

        return abs(l-r) < threshold;
    }

    HOSTDEVICEQUALIFIER
    INLINEQUALIFIER
    constexpr bool fleq(float l, float r){
        
        return (l < r) || feq(l,r);
    }

    HOSTDEVICEQUALIFIER
    INLINEQUALIFIER
    constexpr bool fgeq(float l, float r){
        
        return (l > r) || feq(l,r);
    }


    __device__
    __forceinline__
    float getQualityWeight(char qualitychar){
        constexpr int ascii_base = 33;
        constexpr float min_weight = 0.001f;

        const int q(qualitychar);
        const float errorprob = exp10f(-(q-ascii_base)/10.0f);

        return max(min_weight, 1.0f - errorprob);
    }

    __device__
    __forceinline__
    float calculateOverlapWeightnew(int anchorlength, int nOps, int overlapsize){
        constexpr float maxErrorPercentInOverlap = 0.2f;

        return (float(overlapsize) / anchorlength) * ((float(overlapsize) / anchorlength + (overlapsize - nOps) * maxErrorPercentInOverlap)
                                                        / (1 + overlapsize * maxErrorPercentInOverlap));
    }

    __device__
    __forceinline__
    float calculateOverlapWeight(int anchorlength, int nOps, int overlapsize){
        constexpr float maxErrorPercentInOverlap = 0.2f;

        return 1.0f - sqrtf(nOps / (overlapsize * maxErrorPercentInOverlap));
    }


}
}

#endif