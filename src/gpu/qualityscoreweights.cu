#include <qualityscoreweights.hpp>
#include <gpu/qualityscoreweights.hpp>
#include <hpc_helpers.cuh>

#include <cmath>
#include <cassert>
#include <vector>

namespace care{
namespace gpu{

    #ifdef __NVCC__
    __device__ double d_qscore_to_weight[256];
    #endif

    void init_weights(const std::vector<int>& deviceIds){
        #ifdef __NVCC__

            cpu::QualityScoreConversion conversion;

            auto weights = conversion.getWeights();
            assert(weights.size() == 256);

            int oldId;
            cudaGetDevice(&oldId); CUERR;

            for(auto deviceId : deviceIds){
                cudaSetDevice(deviceId);
                cudaMemcpyToSymbol(d_qscore_to_weight, weights.data(), 256*sizeof(double)); CUERR;
            }

            cudaSetDevice(oldId); CUERR;
        #endif
    }


}
}
