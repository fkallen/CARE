#ifndef GANJA_QUALIFIERS_CUH
#define GANJA_QUALIFIERS_CUH

    ///////////////////////////////////////////////////////////////////////
    // cross platform classifiers
    ///////////////////////////////////////////////////////////////////////

    #ifdef __CUDACC__
        #define HOSTDEVICEQUALIFIER  __host__ __device__
    #else
        #define HOSTDEVICEQUALIFIER
    #endif

    #ifdef __CUDACC__
        #define INLINEQUALIFIER  __forceinline__
    #else
        #define INLINEQUALIFIER inline
    #endif

    #ifdef __CUDACC__
        #define GLOBALQUALIFIER  __global__
    #else
        #define GLOBALQUALIFIER
    #endif

    #ifdef __CUDACC__
        #define DEVICEQUALIFIER  __device__
    #else
        #define DEVICEQUALIFIER
    #endif

    #ifdef __CUDACC__
        #define HOSTQUALIFIER  __host__
    #else
        #define HOSTQUALIFIER
    #endif

#endif
