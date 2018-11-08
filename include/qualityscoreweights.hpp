#ifndef CARE_QUALITY_SCORE_WEIGHTS_HPP
#define CARE_QUALITY_SCORE_WEIGHTS_HPP

namespace care{

    extern double qscore_to_error_prob[256];
    extern double qscore_to_weight[256];

    #ifdef __NVCC__
    extern __device__ double d_qscore_to_weight[256];
    #endif

    void init_weights();

}

#endif
