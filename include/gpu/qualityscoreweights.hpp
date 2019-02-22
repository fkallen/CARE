#ifndef CARE_GPU_QUALITY_SCORE_WEIGHTS_HPP
#define CARE_GPU_QUALITY_SCORE_WEIGHTS_HPP

#include <vector>

namespace care {
namespace gpu {

#ifdef __NVCC__
extern __device__ float d_qscore_to_weight[256];
#endif

void init_weights(const std::vector<int>& deviceIds);

}
}

#endif
