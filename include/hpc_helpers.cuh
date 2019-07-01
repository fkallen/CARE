#ifndef HPC_HELPERS_CUH
#define HPC_HELPERS_CUH

#include <iostream>
#include <cstdint>
#include <chrono>

#ifndef HOSTDEVICEQUALIFIER
    #ifdef __CUDACC__
        #define HOSTDEVICEQUALIFIER  __host__ __device__
    #else
        #define HOSTDEVICEQUALIFIER
    #endif
#endif

#ifndef INLINEQUALIFIER
    #ifdef __CUDACC__
        #define INLINEQUALIFIER  __forceinline__
    #else
        #define INLINEQUALIFIER inline
    #endif
#endif

#ifndef GLOBALQUALIFIER
    #ifdef __CUDACC__
        #define GLOBALQUALIFIER  __global__
    #else
        #define GLOBALQUALIFIER
    #endif
#endif

#ifndef DEVICEQUALIFIER
    #ifdef __CUDACC__
        #define DEVICEQUALIFIER  __device__
    #else
        #define DEVICEQUALIFIER
    #endif
#endif

#ifndef HOSTQUALIFIER
    #ifdef __CUDACC__
        #define HOSTQUALIFIER  __host__
    #else
        #define HOSTQUALIFIER
    #endif
#endif

#ifndef HD_WARNING_DISABLE
    #ifdef __CUDACC__
        #define HD_WARNING_DISABLE #pragma hd_warning_disable
    #else
        #define HD_WARNING_DISABLE
    #endif
#endif


#ifndef TIMERSTARTCPU
    #define TIMERSTARTCPU(label)                                                  \
        std::chrono::time_point<std::chrono::system_clock> a##label, b##label; \
        a##label = std::chrono::system_clock::now();
#endif

#ifndef TIMERSTOPCPU
    #define TIMERSTOPCPU(label)                                                   \
        b##label = std::chrono::system_clock::now();                           \
        std::chrono::duration<double> delta##label = b##label-a##label;        \
        std::cout << "# elapsed time ("<< #label <<"): "                       \
                  << delta##label.count()  << " s" << std::endl;
#endif








#ifndef TIMERSTART

#ifndef __CUDACC__
    #define TIMERSTART(label)                                                  \
        std::chrono::time_point<std::chrono::system_clock> a##label, b##label; \
        a##label = std::chrono::system_clock::now();
#else
    #define TIMERSTART(label)                                                  \
        cudaEvent_t start##label, stop##label;                                 \
        float time##label;                                                     \
        cudaEventCreate(&start##label);                                        \
        cudaEventCreate(&stop##label);                                         \
        cudaEventRecord(start##label, 0);
#endif

#endif

#ifndef TIMERSTOP

#ifndef __CUDACC__
    #define TIMERSTOP(label)                                                   \
        b##label = std::chrono::system_clock::now();                           \
        std::chrono::duration<double> delta##label = b##label-a##label;        \
        std::cout << "# elapsed time ("<< #label <<"): "                       \
                  << delta##label.count()  << " s" << std::endl;
#else
    #define TIMERSTOP(label)                                                   \
            cudaEventRecord(stop##label, 0);                                   \
            cudaEventSynchronize(stop##label);                                 \
            cudaEventElapsedTime(&time##label, start##label, stop##label);     \
	    std::cout << "# elapsed time (" << #label << "): "	               \
                      << time##label << " ms" << std::endl;
#endif

#endif

#ifndef CUERR

    #define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                      << __FILE__ << ", line " << __LINE__ << std::endl;       \
            exit(1);                                                           \
        }                                                                      \
    }

#endif

    // transfer constants
    #ifndef H2D
        #define H2D (cudaMemcpyHostToDevice)
    #endif
    #ifndef D2H
        #define D2H (cudaMemcpyDeviceToHost)
    #endif
    #ifndef H2H
        #define H2H (cudaMemcpyHostToHost)
    #endif
    #ifndef D2D
        #define D2D (cudaMemcpyDeviceToDevice)
    #endif
    #ifndef SDIV
        #define SDIV(x,y)(((x)+(y)-1)/(y))
    #endif



    // safe division


#endif
