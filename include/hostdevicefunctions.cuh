#ifndef CARE_HOSTDEVICE_FUNCTIONS_CUH
#define CARE_HOSTDEVICE_FUNCTIONS_CUH

#include <hpc_helpers.cuh>
#include <cmath>

namespace care{

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

    HOSTDEVICEQUALIFIER
    INLINEQUALIFIER
    float getQualityWeight(char qualitychar){
        constexpr int ascii_base = 33;
        constexpr float min_weight = 0.001f;

        auto base10expf = [](float p){
            #ifdef __CUDA_ARCH__
            return exp10f(p);
            #else
            return std::pow(10.0f, p);
            #endif
        };

        const int q(qualitychar);
        const float errorprob = base10expf(-(q-ascii_base)/10.0f);

        return min_weight > 1.0f - errorprob ? min_weight : 1.0f - errorprob;
    }

    HOSTDEVICEQUALIFIER
    INLINEQUALIFIER
    float calculateOverlapWeightnew(int anchorlength, int nOps, int overlapsize){
        constexpr float maxErrorPercentInOverlap = 0.2f;

        return (float(overlapsize) / anchorlength) * ((float(overlapsize) / anchorlength + (overlapsize - nOps) * maxErrorPercentInOverlap)
                                                        / (1 + overlapsize * maxErrorPercentInOverlap));
    }

    HOSTDEVICEQUALIFIER
    INLINEQUALIFIER
    float calculateOverlapWeight(int anchorlength, int nOps, int overlapsize){
        constexpr float maxErrorPercentInOverlap = 0.2f;

        return 1.0f - sqrtf(nOps / (overlapsize * maxErrorPercentInOverlap));
    }

}

#endif