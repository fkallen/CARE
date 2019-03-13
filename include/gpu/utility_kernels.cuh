#ifndef UTILITY_KERNELS_HPP
#define UTILITY_KERNELS_HPP

#include "../hpc_helpers.cuh"

#ifdef __NVCC__

template<class T>
__global__
void fill_kernel(T* data, int elements, T value){
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for(; index < elements; index += stride){
        data[index] = value;
    }
}

template<class T>
void call_fill_kernel_async(T* d_data, int elements, const T& value, cudaStream_t stream){
    const int blocksize = 128;
    const int blocks = SDIV(elements, blocksize);
    dim3 block(blocksize,1,1);
    dim3 grid(blocks,1,1);

    fill_kernel<<<grid, block, 0, stream>>>(d_data, elements, value); CUERR;
}


template<class T>
__global__
void set_kernel(T* data, int index, T value){
    data[index] = value;
}

template<class T>
void call_set_kernel_async(T* d_data, int index, const T& value, cudaStream_t stream){
    fill_kernel<<<1, 1, 0, stream>>>(d_data, index, value); CUERR;
}

#endif

#endif
