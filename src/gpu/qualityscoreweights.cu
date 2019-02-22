#include <qualityscoreweights.hpp>
#include <gpu/qualityscoreweights.hpp>
#include <hpc_helpers.cuh>

#include <cmath>
#include <cassert>
#include <vector>

namespace care{
namespace gpu{

    #ifdef __NVCC__
    __device__ float d_qscore_to_weight[256];
    #endif

    void init_weights(const std::vector<int>& deviceIds){
        cpu::QualityScoreConversion conversion;
        #ifdef __NVCC__


            auto weights = conversion.getWeights();
            assert(weights.size() == 256);

            int oldId;
            cudaGetDevice(&oldId); CUERR;

            for(auto deviceId : deviceIds){
                cudaSetDevice(deviceId);
                cudaMemcpyToSymbol(d_qscore_to_weight, weights.data(), 256*sizeof(float)); CUERR;
            }

            cudaSetDevice(oldId); CUERR;
        #endif
    }


}
}
