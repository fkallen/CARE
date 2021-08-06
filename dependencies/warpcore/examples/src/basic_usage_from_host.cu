#include <iostream>
#include <warpcore/single_value_hash_table.cuh>
#include <helpers/timers.cuh>

// This example shows the basic usage of a single value hash table using
// host-sided table operations provided by warpcore
int main()
{
    // key type of the hash table (uint32_t or uint64_t)
    using key_t   = std::uint32_t;

    // value type of the hash table
    using value_t = std::uint32_t;

    using namespace warpcore;

    // type of the hash table (with default parameters)
    using hash_table_t = SingleValueHashTable<key_t, value_t>;

    // in this example we use dummy key/value pairs which will be inserted
    // into the hast table
    const uint64_t input_size = 1UL << 27;

    std::cout << "num elements " << input_size << std::endl;

    // allocate host-sided (pinned) arrays for our input data
    key_t*   keys_h;   cudaMallocHost(&keys_h,   sizeof(key_t)*input_size);
    value_t* values_h; cudaMallocHost(&values_h, sizeof(value_t)*input_size);

    // lets generate some random data
    // (key, val)->(1, 2), (2, 3), .., (input_size, input_size+1)
    // NOTE: since we are using default parameters, key_t(0) and key_t(0)-1 are
    // invalid keys since they internally map to the empty key specifier and
    // tombstone key specifier respectively
    for(key_t i = 0; i < input_size; i++)
    {
        keys_h[i]   = i+1;
        values_h[i] = i+2;
    }

    // allocate device-sided arrays for our input data
    key_t*   keys_d;   cudaMalloc(&keys_d,   sizeof(key_t)*input_size); CUERR
    value_t* values_d; cudaMalloc(&values_d, sizeof(value_t)*input_size); CUERR

    // copy input key/value pairs from the host to the device
    cudaMemcpy(keys_d,   keys_h,   sizeof(key_t)*input_size, cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(values_d, values_h, sizeof(value_t)*input_size, cudaMemcpyHostToDevice); CUERR

    // the target load factor of the table after inserting our dummy data
    const float load = 0.9;

    // which results in the following capacity of the hash table
    const uint64_t capacity = input_size/load;

    // INITIALIZE the hash table
    hash_table_t hash_table(capacity); CUERR

    std::cout << "capacity " << hash_table.capacity() << std::endl;

    {
        helpers::GpuTimer timer("insert");
        // INSERT the input data into the hash_table
        hash_table.insert(keys_d, values_d, input_size);
        timer.print();
    } CUERR

    cudaDeviceSynchronize(); CUERR
    // check if any errors occured
    std::cout  << "insertion errors: " << hash_table.pop_status().get_errors() << std::endl;

    std::cout << "load " << hash_table.load_factor() << std::endl;
    std::cout << "size " << hash_table.size() << std::endl;

    // now, we want to retrieve our dummy data from the
    // hash table again

    // first, allocate some device-sided memory to hold the result
    value_t* result_d; cudaMalloc(&result_d, sizeof(value_t)*input_size);

    {
        helpers::GpuTimer timer("retrieve");
        // RETRIEVE the corresponding values of the rear half of the input keys
        // from the hash table
        hash_table.retrieve(keys_d, input_size, result_d);
        timer.print();
    } CUERR

    // check again if any errors occured
    std::cout << "retrieval errors: " << hash_table.pop_status().get_errors() << std::endl;

    // allocate host-sided memory to copy the result back to the host
    // in order to perform a unit test
    value_t* result_h; cudaMallocHost(&result_h, sizeof(value_t)*input_size); CUERR

    // copy the result back to the host
    cudaMemcpy(result_h, result_d, sizeof(value_t)*input_size, cudaMemcpyDeviceToHost); CUERR

    // check the result
    uint64_t errors = 0;
    for(uint64_t i = 0; i < input_size; i++)
    {
        // check if output matches the input
        if(result_h[i] != values_h[i])
        {
            errors++;
        }
    }
    std::cout << "check result: " << errors << " errors occured" << std::endl;

    // free all allocated recources
    cudaFreeHost(keys_h);
    cudaFree(keys_d);
    cudaFreeHost(values_h);
    cudaFree(values_d);
    cudaFreeHost(result_h);
    cudaFree(result_d);

    // check for any CUDA errors
    cudaDeviceSynchronize(); CUERR
}
