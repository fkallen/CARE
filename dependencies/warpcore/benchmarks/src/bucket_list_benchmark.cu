#include <warpcore/bucket_list_hash_table.cuh>
#include "../include/benchmark_common.cuh"

template<class HashTable>
HOSTQUALIFIER INLINEQUALIFIER
void bucket_list_benchmark(
    const typename HashTable::key_type * keys_d,
    const uint64_t max_keys,
    float key_load_factor,
    float value_load_factor,
    std::vector<uint64_t> input_sizes,
    std::vector<std::tuple<float, uint64_t, uint64_t>> slab_list_configs,
    typename HashTable::key_type seed = warpcore::defaults::seed<key_t>(),
    bool print_headers = true,
    uint8_t iters = 1,
    std::chrono::milliseconds thermal_backoff = std::chrono::milliseconds(100))
{
    using index_t = typename HashTable::index_type;
    using key_t = typename HashTable::key_type;
    using value_t = typename HashTable::value_type;

    const uint64_t max_unique_size = num_unique(keys_d, max_keys);

    std::cout << "unique_keys: " << max_unique_size << "\tvalues: " << max_keys << std::endl;

    const uint64_t key_store_capacity = max_unique_size / key_load_factor;
    const uint64_t value_store_capacity = max_keys / value_load_factor;

    key_t* query_keys_d = nullptr;
    cudaMalloc(&query_keys_d, sizeof(key_t)*max_keys); CUERR
    value_t* values_d = nullptr;
    cudaMalloc(&values_d, sizeof(value_t)*max_keys); CUERR
    index_t * offsets_d = nullptr;
    cudaMalloc(&offsets_d, sizeof(index_t)*(max_keys+1)); CUERR

    const auto max_input_size =
        *std::max_element(input_sizes.begin(), input_sizes.end());

    if(max_input_size > max_keys)
    {
        std::cerr << "Maximum input size exceeded." << std::endl;
        exit(1);
    }

    if(!sufficient_memory_bl<HashTable>(key_store_capacity, value_store_capacity))
    {
        std::cerr << "Not enough GPU memory." << std::endl;
        exit(1);
    }

    for(const auto& size : input_sizes)
    {
        for(const auto& slab_list_config : slab_list_configs)
        {
            const float slab_grow_factor = std::get<0>(slab_list_config);
            const index_t min_slab_size = std::get<1>(slab_list_config);
            const index_t max_slab_size = std::get<2>(slab_list_config);

            HashTable hash_table(
                key_store_capacity,
                value_store_capacity,
                seed,
                slab_grow_factor,
                min_slab_size);

            Output<key_t,value_t> output;
            output.sample_size = size;
            output.key_capacity = hash_table.key_capacity();
            output.value_capacity = hash_table.value_capacity();

            output.insert_ms = benchmark_insert(
                hash_table, keys_d, values_d, size,
                iters, thermal_backoff);

            output.query_ms = benchmark_query_multi(
                hash_table, query_keys_d, size,
                offsets_d, values_d,
                iters, thermal_backoff);

            // output.query_ms = benchmark_query_unique(
            //     hash_table, query_keys_d, offsets_d, values_d,
            //     iters, thermal_backoff);

            output.key_load_factor = hash_table.key_load_factor();
            output.value_load_factor = hash_table.value_load_factor();
            output.density = hash_table.storage_density();
            output.relative_density = hash_table.relative_storage_density();
            output.status = hash_table.pop_status();

            std::cout << std::fixed
                << "grow_factor=" << slab_grow_factor
                << output.d << "min_slab_size=" << min_slab_size
                << output.d << "max_slab_size=" << max_slab_size
                << output.d;

            if(print_headers)
                output.print_with_headers();
            else
                output.print_without_headers();
        }
    }

    cudaFree(query_keys_d); CUERR
    cudaFree(values_d); CUERR
    cudaFree(offsets_d); CUERR
}

int main(int argc, char* argv[])
{
    using namespace warpcore;

    using key_t = std::uint32_t;
    using value_t = std::uint32_t;

    const uint64_t max_keys = 1UL << 27;

    const bool print_headers = true;

    uint64_t dev_id = 0;
    if(argc > 2) dev_id = std::atoi(argv[2]);
    cudaSetDevice(dev_id); CUERR

    key_t * keys_d = nullptr;
    if(argc > 1)
        keys_d = load_keys<key_t>(argv[1], max_keys);
    else
        keys_d = generate_keys<key_t>(max_keys, 8);

    using hash_table_t = BucketListHashTable<
        key_t,
        value_t,
        defaults::empty_key<key_t>(),
        defaults::tombstone_key<key_t>(),
        storage::multi_value::BucketListStore<value_t, 29, 18, 15>,
        defaults::probing_scheme_t<key_t, 8>>;

    bucket_list_benchmark<hash_table_t>(
        keys_d, max_keys,
        0.90,
        0.50,
        {max_keys},
        {{1.1, 1, 0}},
        0x5ad0ded,
        print_headers);

    cudaFree(keys_d); CUERR
}
