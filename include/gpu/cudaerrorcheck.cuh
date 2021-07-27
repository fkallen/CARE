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

#define CUDACHECK(ans) { \
    using namespace std::string_literals; \
    constexpr bool abort = true; \
    cudaError_t status = (ans);                                 \
    if (status != cudaSuccess){              \
        cudaGetLastError();                 \
        std::string msg = "CUDA Error: "s + cudaGetErrorString(status) + " "s + __FILE__ + " "s + std::to_string(__LINE__); \
        std::cerr << msg << "\n"; \
        if(abort){ \
            throw std::runtime_error(msg); \
        } \
    } \
}

#define CUDACHECKASYNC { CUDACHECK((cudaPeekAtLastError())); }