#ifndef CUDA_REDUCE_CUH
#define CUDA_REDUCE_CUH

#ifdef __NVCC__

#ifdef WARP_SIZE
#undef WARP_SIZE
#endif

#define WARP_SIZE 32


template <unsigned int MAX_BLOCK_DIM_X_, class S, class Func>
__device__ void blockreduce(S *result, S localvalue, Func func){

    __shared__ S sdata[MAX_BLOCK_DIM_X_];

    unsigned int tid = threadIdx.x;
    S myValue = localvalue;

    // each thread puts its local sum into shared memory
    sdata[tid] = localvalue;
    __syncthreads();

    // do reduction in shared mem
    if ((blockDim.x >= 1024) && (tid < 512)){
        sdata[tid] = myValue = func(myValue, sdata[tid + 512]);
    }
    __syncthreads();

    if ((blockDim.x >= 512) && (tid < 256)){
        sdata[tid] = myValue = func(myValue, sdata[tid + 256]);
    }
    __syncthreads();

    if ((blockDim.x >= 256) && (tid < 128)){
            sdata[tid] = myValue = func(myValue, sdata[tid + 128]);
    }
     __syncthreads();

    if ((blockDim.x >= 128) && (tid <  64)){
       sdata[tid] = myValue = func(myValue, sdata[tid +  64]);
    }
    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 ){
        // Fetch final intermediate sum from 2nd warp
        if (blockDim.x >=  64) myValue = func(myValue, sdata[tid + 32]);
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
















#endif

#endif
