#include <iostream>
#include "kiss_rng.cuh"

template<class T, class Rng>
HOSTQUALIFIER  INLINEQUALIFIER
void uniform_distribution(
    T * const out, 
    const std::uint64_t n, 
    const std::uint32_t seed) noexcept
{
    // execute kernel
    lambda_kernel<<<4096, 32>>>
    ([=] DEVICEQUALIFIER
    {
        const std::uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        // generate initial local seed per thread
        const std::uint32_t local_seed = 
            hashers::MurmurHash<std::uint32_t>::hash(seed+tid);
        
        Rng rng{local_seed};

        // grid-stride loop
        const auto grid_stride = blockDim.x * gridDim.x;
        for(std::uint64_t i = tid; i < n; i += grid_stride)
        {
            // generate random element and write to output
            out[i] = rng.template next<T>();
        }
    })
    ; CUERR
}

// This example shows the easy generation of gigabytes of uniform random values
// in only a few milliseconds.

int main ()
{
    // define the data types to be generated
    using data_t = std::uint64_t; 
    using rng_t = Kiss<data_t>;

    // number of values to draw
    static constexpr std::uint64_t n = 1UL << 28;

    // random seed
    static constexpr std::uint32_t seed = 42;

    // allocate host memory for the result
    data_t * data_h = nullptr;
    cudaMallocHost(&data_h, sizeof(data_t)*n); CUERR

    // allocate GPU memory for the result
    data_t * data_d = nullptr;
    cudaMalloc(&data_d, sizeof(data_t)*n); CUERR

    // initialize th allocated memory (contents my inner paranoia)
    THROUGHPUTSTART(memset_zeroes)
    cudaMemset(data_d, 0, sizeof(data_t)*n); CUERR
    THROUGHPUTSTOP(memset_zeroes, sizeof(data_t), n)

    // generate uniform random numbers and measure throughput
    THROUGHPUTSTART(generate_random)
    uniform_distribution<data_t, rng_t>(
        data_d, 
        n, 
        seed); CUERR
    THROUGHPUTSTOP(generate_random, sizeof(data_t), n)

    cudaMemcpy(data_h, data_d, sizeof(data_t)*n, D2H); CUERR

    // do something with drawn random numbers
    for(std::uint64_t i = 0; i < 10; i++)
    {
        std::cout << data_h[i] << std::endl;
    }

    // free any allocated memory
    cudaFreeHost(data_h); CUERR
    cudaFree(data_d); CUERR
}
