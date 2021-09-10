#include <string>
#include <iostream>



// void cudaCheck(cudaError_t status, const char *file, int line, bool abort=true){
//     using namespace std::string_literals;

//     if (status != cudaSuccess){
//         std::string msg = "CUDA Error: "s + cudaGetErrorString(status) + " "s + file + " "s + std::to_string(line);
//         std::cerr << msg << "\n";
//         if(abort){
//             throw std::runtime_error(msg);
//         }
//     }
// }

//#define CUDACHECK(ans) { cudaCheck((ans), __FILE__, __LINE__); }
//#define CUDACHECKASYNC { CUDACHECK((cudaPeekAtLastError()), __FILE__, __LINE__); }

#ifdef __CUDACC__

#if 0
#define CUDACHECK(ans) { \
    using namespace std::string_literals; \
    constexpr bool abort = true; \
    cudaError_t MY_CUDA_CHECK_status = (ans);                                 \
    if (MY_CUDA_CHECK_status != cudaSuccess){              \
        cudaGetLastError();                 \
        std::string msg = "CUDA Error: "s + cudaGetErrorString(MY_CUDA_CHECK_status) + " "s + __FILE__ + " "s + std::to_string(__LINE__); \
        std::cerr << msg << "\n"; \
        if(abort){ \
            throw std::runtime_error(msg); \
        } \
    } \
}
#endif

#define CUDACHECKIMPL(ans, async) { \
    using namespace std::string_literals; \
    constexpr bool abort = true; \
    cudaError_t MY_CUDA_CHECK_status = (ans);                                 \
    if (MY_CUDA_CHECK_status != cudaSuccess){              \
        cudaGetLastError();                 \
        std::string msg = (async ? "Asynchronous "s : "Synchronous "s) + "CUDA Error: "s + cudaGetErrorString(MY_CUDA_CHECK_status) + " "s + __FILE__ + " "s + std::to_string(__LINE__); \
        std::cerr << msg << "\n"; \
        if(abort){ \
            throw std::runtime_error(msg); \
        } \
    } \
}

#define CUDACHECK(ans) { CUDACHECKIMPL((ans), false); }

#define CUDACHECKASYNC { CUDACHECKIMPL((cudaPeekAtLastError()), true); }

#else

#define CUDACHECK(ANS) (ANS)
#define CUDACHECKASYNC 

#endif