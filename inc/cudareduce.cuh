#ifndef CUDA_REDUCE_CUH
#define CUDA_REDUCE_CUH

#ifdef __NVCC__

#ifdef WARP_SIZE
#undef WARP_SIZE
#endif

#define WARP_SIZE 32

#if __CUDACC_VER_MAJOR__ >= 9
#include <cooperative_groups.h>
using namespace cooperative_groups;
#endif

__host__ __device__
constexpr bool power_of_two(unsigned int x) {
   return x && !(x & (x - 1));
}


#if __CUDACC_VER_MAJOR__ < 9

    template<unsigned int TileSize, class T, class Func>
    __device__ T reduceTile(T val, Func func){
        static_assert(power_of_two(TileSize) && TileSize <= 32,
            "reduceTile is only available if TileSize < 32 and TileSize is power of 2");

        for (unsigned int offset = TileSize/2; offset > 0; offset /= 2){
            myValue = func(myValue, __shfl_down_sync(__activemask(), myValue, offset));
        }
        return val;
    }

#else

template<unsigned int TileSize, class T, class Func>
__device__ T reduceTile(thread_block_tile<TileSize>& g, T val, Func func){
    static_assert(power_of_two(TileSize) && TileSize <= 32,
        "reduceTile is only available if TileSize < 32 and TileSize is power of 2");

    for (unsigned int offset = TileSize/2; offset > 0; offset /= 2){
        val = func(val, g.shfl_down(val, offset));
    }
    return val;
}

#endif



template <unsigned int MAX_BLOCK_DIM_X_, class S, class Func>
__device__ void blockreduce(S *result, S localvalue, Func func){

    __shared__ S sdata[MAX_BLOCK_DIM_X_];

    unsigned int tid = threadIdx.x;
    S myValue = localvalue;

    // each thread puts its local sum into shared memory
    sdata[tid] = localvalue;
    __syncthreads();
    /*if(threadIdx.x == 0){
        for(int i = 0; i < MAX_BLOCK_DIM_X_; i++)
            printf("%d %d\n", i, (*((int2*)&sdata[i])).x);
    }
    __syncthreads();*/


    // do reduction in shared mem
    if ((blockDim.x >= 1024) && (tid < 512)){
        sdata[tid] = myValue = func(myValue, sdata[tid + 512]);
    }
    __syncthreads();
    /*if(threadIdx.x == 0)
        printf("step\n");
    __syncthreads();*/

    if ((blockDim.x >= 512) && (tid < 256)){
        sdata[tid] = myValue = func(myValue, sdata[tid + 256]);
    }
    __syncthreads();
    /*if(threadIdx.x == 0)
        printf("step\n");
    __syncthreads();*/

    if ((blockDim.x >= 256) && (tid < 128)){
            sdata[tid] = myValue = func(myValue, sdata[tid + 128]);
    }
     __syncthreads();
    /*if(threadIdx.x == 0)
        printf("step\n");
    __syncthreads();*/

    if ((blockDim.x >= 128) && (tid <  64)){
       sdata[tid] = myValue = func(myValue, sdata[tid +  64]);
    }
    __syncthreads();
    /*if(threadIdx.x == 0)
        printf("step\n");
    __syncthreads();*/
#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 ){
        // Fetch final intermediate sum from 2nd warp
        if (blockDim.x >=  64) myValue = func(myValue, sdata[tid + 32]);
    /*if(threadIdx.x == 0)
        printf("step\n");
    __syncthreads();*/
        // Reduce final warp using shuffle
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
#if __CUDACC_VER_MAJOR__ < 9
            myValue = func(myValue, __shfl_down(myValue, offset));
#else
            unsigned mask = __activemask();
            myValue = func(myValue, __shfl_down_sync(mask, myValue, offset));
#endif
    /*if(threadIdx.x == 0)
        printf("step\n");
    __syncthreads();*/
        }

    }
#else
    // fully unroll reduction within a single warp
    if ((blockDim.x >=  64) && (tid < 32)){
        sdata[tid] = myValue = func(myValue, sdata[tid + 32]);
    }
    __syncthreads();

    if ((blockDim.x >=  32) && (tid < 16)){
        sdata[tid] = myValue = func(myValue, sdata[tid + 16]);
    }
    __syncthreads();

    if ((blockDim.x >=  16) && (tid <  8)){
        sdata[tid] = myValue = func(myValue, sdata[tid +  8]);
    }
    __syncthreads();

    if ((blockDim.x >=   8) && (tid <  4)){
        sdata[tid] = myValue = func(myValue, sdata[tid +  4]);
    }
    __syncthreads();

    if ((blockDim.x >=   4) && (tid <  2)){
        sdata[tid] = myValue = func(myValue, sdata[tid +  2]);
    }
    __syncthreads();

    if ((blockDim.x >=   2) && ( tid <  1)){
        sdata[tid] = myValue = func(myValue,sdata[tid +  1]);
    }
    __syncthreads();
#endif

    if (tid == 0) *result = myValue;
}


template <class S, class Func>
__device__ void blockreduce(S* tmpstorage, S *result, S localvalue, Func func){

    unsigned int tid = threadIdx.x;
    S myValue = localvalue;

    // each thread puts its local sum into shared memory
    tmpstorage[tid] = localvalue;
    __syncthreads();

    // do reduction in shared mem
    if ((blockDim.x >= 1024) && (tid < 512)){
        tmpstorage[tid] = myValue = func(myValue, tmpstorage[tid + 512]);
    }
    __syncthreads();

    if ((blockDim.x >= 512) && (tid < 256)){
        tmpstorage[tid] = myValue = func(myValue, tmpstorage[tid + 256]);
    }
    __syncthreads();

    if ((blockDim.x >= 256) && (tid < 128)){
            tmpstorage[tid] = myValue = func(myValue, tmpstorage[tid + 128]);
    }
     __syncthreads();

    if ((blockDim.x >= 128) && (tid <  64)){
       tmpstorage[tid] = myValue = func(myValue, tmpstorage[tid + 64]);
    }
    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 ){
        // Fetch final intermediate sum from 2nd warp
        if (blockDim.x >=  64) myValue = func(myValue, tmpstorage[tid + 32]);

        // Reduce final warp using shuffle
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
#if __CUDACC_VER_MAJOR__ < 9
            myValue = func(myValue, __shfl_down(myValue, offset));
#else
            unsigned mask = __activemask();
            myValue = func(myValue, __shfl_down_sync(mask, myValue, offset));
#endif
        }

    }
#else
    // fully unroll reduction within a single warp
    if ((blockDim.x >=  64) && (tid < 32)){
        tmpstorage[tid] = myValue = func(myValue, tmpstorage[tid + 32]);
    }
    __syncthreads();

    if ((blockDim.x >=  32) && (tid < 16)){
        tmpstorage[tid] = myValue = func(myValue, tmpstorage[tid + 16]);
    }
    __syncthreads();

    if ((blockDim.x >=  16) && (tid <  8)){
        tmpstorage[tid] = myValue = func(myValue, tmpstorage[tid +  8]);
    }
    __syncthreads();

    if ((blockDim.x >=   8) && (tid <  4)){
        tmpstorage[tid] = myValue = func(myValue, tmpstorage[tid +  4]);
    }
    __syncthreads();

    if ((blockDim.x >=   4) && (tid <  2)){
        tmpstorage[tid] = myValue = func(myValue, tmpstorage[tid +  2]);
    }
    __syncthreads();

    if ((blockDim.x >=   2) && ( tid <  1)){
        tmpstorage[tid] = myValue = func(myValue, tmpstorage[tid +  1]);
    }
    __syncthreads();
#endif

    if (tid == 0) *result = myValue;
}


#endif
#endif
