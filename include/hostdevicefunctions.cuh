#ifndef CARE_HOSTDEVICE_FUNCTIONS_CUH
#define CARE_HOSTDEVICE_FUNCTIONS_CUH

#include <hpc_helpers.cuh>
#include <cmath>

namespace care{

    HOSTDEVICEQUALIFIER
    INLINEQUALIFIER
    constexpr bool feq(float l, float r){
        constexpr float threshold = 1e-5f;
        const float absdiff = l-r < 0 ? -(l-r) : l-r;
        return absdiff < threshold;
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
    float calculateOverlapWeight(int anchorlength, int nOps, int overlapsize, float maxMismatchRatio){
        return 1.0f - sqrtf(nOps / (overlapsize * maxMismatchRatio));
    }

    HD_WARNING_DISABLE
    template<class IndexTransformation>
    HOSTDEVICEQUALIFIER
    void shiftBitArrayLeftBy(unsigned int* array, int size, int shiftamount, IndexTransformation indextrafo){
        if(shiftamount == 0) return;

        const int completeInts = shiftamount / (8 * sizeof(unsigned int));

        for(int i = 0; i < size - completeInts; i += 1) {
            array[indextrafo(i)] = array[indextrafo(completeInts + i)];
        }

        for(int i = size - completeInts; i < size; i += 1) {
            array[indextrafo(i)] = 0;
        }

        shiftamount -= completeInts * 8 * sizeof(unsigned int);

        for(int i = 0; i < size - completeInts - 1; i += 1) {
            const unsigned int a = array[indextrafo(i)];
            const unsigned int b = array[indextrafo(i+1)];

            array[indextrafo(i)] = (a << shiftamount) | (b >> (8 * sizeof(unsigned int) - shiftamount));
        }

        array[indextrafo(size - completeInts - 1)] <<= shiftamount;
    }

    HD_WARNING_DISABLE
    template<int shiftamount, class IndexTransformation>
    HOSTDEVICEQUALIFIER
    void shiftBitArrayLeftBy(unsigned int* array, int size, IndexTransformation indextrafo){
        if(shiftamount == 0) return;

        constexpr int completeInts = shiftamount / (8 * sizeof(unsigned int));

        for(int i = 0; i < size - completeInts; i += 1) {
            array[indextrafo(i)] = array[indextrafo(completeInts + i)];
        }

        for(int i = size - completeInts; i < size; i += 1) {
            array[indextrafo(i)] = 0;
        }

        constexpr int remainingShift = shiftamount - completeInts * 8 * sizeof(unsigned int);

        for(int i = 0; i < size - completeInts - 1; i += 1) {
            const unsigned int a = array[indextrafo(i)];
            const unsigned int b = array[indextrafo(i+1)];

            array[indextrafo(i)] = (a << remainingShift) | (b >> (8 * sizeof(unsigned int) - remainingShift));
        }

        array[indextrafo(size - completeInts - 1)] <<= remainingShift;
    }



    HD_WARNING_DISABLE
    template<class IndexTransformation1,
             class IndexTransformation2,
             class PopcountFunc>
    HOSTDEVICEQUALIFIER
    int hammingdistanceHiLo(const unsigned int* lhi,
                            const unsigned int* llo,
                            const unsigned int* rhi,
                            const unsigned int* rlo,
                            int lhi_bitcount,
                            int rhi_bitcount,
                            int max_errors,
                            IndexTransformation1 indextrafoL,
                            IndexTransformation2 indextrafoR,
                            PopcountFunc popcount){

        const int overlap_bitcount = std::min(lhi_bitcount, rhi_bitcount);

        if(overlap_bitcount == 0)
            return max_errors+1;

        const int partitions = SDIV(overlap_bitcount, (8 * sizeof(unsigned int)));
        const int remaining_bitcount = partitions * sizeof(unsigned int) * 8 - overlap_bitcount;

        int result = 0;

        for(int i = 0; i < partitions - 1 && result < max_errors; i += 1) {
            const unsigned int hixor = lhi[indextrafoL(i)] ^ rhi[indextrafoR(i)];
            const unsigned int loxor = llo[indextrafoL(i)] ^ rlo[indextrafoR(i)];
            const unsigned int bits = hixor | loxor;
            result += popcount(bits);
        }

        if(result >= max_errors)
            return result;

        // i == partitions - 1

        const unsigned int mask = remaining_bitcount == 0 ? 0xFFFFFFFF : 0xFFFFFFFF << (remaining_bitcount);
        const unsigned int hixor = lhi[indextrafoL(partitions - 1)] ^ rhi[indextrafoR(partitions - 1)];
        const unsigned int loxor = llo[indextrafoL(partitions - 1)] ^ rlo[indextrafoR(partitions - 1)];
        const unsigned int bits = hixor | loxor;
        result += popcount(bits & mask);

        return result;
    };

}

#endif