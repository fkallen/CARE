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
    set_kernel<<<1, 1, 0, stream>>>(d_data, index, value); CUERR;
}

template<class Func>
__global__
void generic_kernel(Func f){
    f();
}

template<class T>
__global__
void compact_kernel(T* out, const T* in, const int* indices, int n){

    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x){
        const int srcindex = indices[i];
        out[i] = in[srcindex];
    }
}

template<class T>
__global__
void compact_kernel_nptr(T* out, const T* in, const int* indices, const int* n){

    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < *n; i += blockDim.x * gridDim.x){
        const int srcindex = indices[i];
        out[i] = in[srcindex];
    }
}

template<class T>
void call_compact_kernel_async(T* out, const T* in, const int* indices, int n, cudaStream_t stream){
    dim3 block(128,1,1);
    dim3 grid(SDIV(n, block.x),1,1);

    compact_kernel<<<grid, block, 0, stream>>>(out, in, indices, n); CUERR;
}

template<class T>
void call_compact_kernel_async(T* out, const T* in, const int* indices, const int* n, int maxN, cudaStream_t stream){
    dim3 block(128,1,1);
    dim3 grid(SDIV(maxN, block.x),1,1);

    compact_kernel_nptr<<<grid, block, 0, stream>>>(out, in, indices, n); CUERR;
}


template<int blocksize_x, int blocksize_y, class T>
__global__
void transpose_kernel(T* __restrict__ output, const T* __restrict__ input, int numRows, int numColumns, int columnpitchelements){
    constexpr int tilesize = 32;
    __shared__ T tile[tilesize][tilesize+1];

    const int requiredTilesX = SDIV(numColumns, tilesize);
    const int requiredTilesY = SDIV(numRows, tilesize);
    const int dstNumRows = numColumns;
    const int dstNumColumns = numRows;

    for(int blockId = blockIdx.x; blockId < requiredTilesX * requiredTilesY; blockId += gridDim.x){
        const int tile_id_x = blockId % requiredTilesX;
        const int tile_id_y = blockId / requiredTilesX;

        for(int tile_x = threadIdx.x; tile_x < tilesize; tile_x += blocksize_x){
            for(int tile_y = threadIdx.y; tile_y < tilesize; tile_y += blocksize_y){
                const int srcColumn = tile_id_x * tilesize + tile_x;
                const int srcRow = tile_id_y * tilesize + tile_y;

                if(srcColumn < numColumns && srcRow < numRows){
                    tile[tile_y][tile_x] = input[srcRow * columnpitchelements + srcColumn];
                }
            }
        }

        __syncthreads(); //wait for tile to be loaded

        for(int tile_x = threadIdx.x; tile_x < tilesize; tile_x += blocksize_x){
            for(int tile_y = threadIdx.y; tile_y < tilesize; tile_y += blocksize_y){
                const int dstColumn = tile_id_y * tilesize + tile_x;
                const int dstRow = tile_id_x * tilesize + tile_y;

                if(dstRow < dstNumRows && dstColumn < dstNumColumns){
                    output[dstRow * dstNumColumns + dstColumn] = tile[tile_x][tile_y];
                }
            }
        }

        __syncthreads(); //wait before reusing shared memory
    }
}

template<class T>
void call_transpose_kernel(T* output, const T* input, int numRows, int numColumns, int columnpitchelements, cudaStream_t stream){
    dim3 block(32,8);
    const int blocks_x = SDIV(numColumns, block.x);
    const int blocks_y = SDIV(numRows, block.y);
    dim3 grid(min(65535, blocks_x * blocks_y), 1, 1);

    transpose_kernel<32,8><<<grid, block, 0, stream>>>(output,
                                                input,
                                                numRows,
                                                numColumns,
                                                columnpitchelements); CUERR;
}

#endif

#endif
