#include "../inc/qualityscoreweights.hpp"

#include "../inc/ganja/hpc_helpers.cuh"

#include <cmath>

namespace care{

    double	qscore_to_error_prob[256];
    double	qscore_to_weight[256];

    #ifdef __NVCC__
    __device__ double d_qscore_to_weight[256];
    #endif

    void init_weights(){

        constexpr int ASCII_BASE = 33;
        constexpr double MIN_WEIGHT = 0.001;

        for(int i = 0; i < 256; i++){
            if(i < ASCII_BASE)
                qscore_to_error_prob[i] = 1.0;
            else
                qscore_to_error_prob[i] = std::pow(10.0, -(i-ASCII_BASE)/10.0);
        }

        for(int i = 0; i < 256; i++){
            qscore_to_weight[i] = std::max(MIN_WEIGHT, 1.0 - qscore_to_error_prob[i]);
        }

        #ifdef __NVCC__
            int devices;
            cudaGetDeviceCount(&devices); CUERR;
            for(int i = 0; i < devices; i++){
                cudaSetDevice(i);
                cudaMemcpyToSymbol(d_qscore_to_weight, qscore_to_weight, 256*sizeof(double)); CUERR;
            }
        #endif
    }



}
