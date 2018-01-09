#ifndef GANJA_ROBIN_HOOD_MULTI_HASH_MAP_CUH
#define GANJA_ROBIN_HOOD_MULTI_HASH_MAP_CUH

#include <omp.h>
#include <atomic>
#include <vector>
#include "data_types.cuh"
#include "atomic_helpers.cuh"

template <
    typename index_t,
    index_t bits_key,
    index_t bits_val,
    index_t bits_cnt,
    typename funct_t,
    typename probe_t>
struct RobinHoodMultiHashMap {

    typedef KeyValueCountTriple_t<index_t,bits_key,bits_val,bits_cnt> entry_t;
    typedef std::atomic<entry_t> atomic_t;

    static constexpr index_t bits_for_key = bits_key;
    static constexpr index_t bits_for_val = bits_val;
    static constexpr index_t bits_for_cnt = bits_cnt;

    atomic_t * data;
    const funct_t hash_func;
    const probe_t prob_func;
    const index_t capacity;
    const index_t probe_length;
    std::atomic<index_t> size = {0};

    RobinHoodMultiHashMap(
        const index_t capacity_,
        const funct_t hash_func_,
        const probe_t prob_func_) : capacity(capacity_),
                                    probe_length(capacity_),
                                    hash_func(hash_func_),
                                    prob_func(prob_func_) {

        data = new atomic_t[capacity];

        # pragma omp parallel for
        for (index_t index = 0; index < capacity; index++)
            data[index].store(entry_t::get_empty());
    }

    ~RobinHoodMultiHashMap() {

        delete [] data;
    }

    std::vector<index_t> get(
        const index_t& key) const {

        std::vector<index_t> result;
        index_t index = hash_func(key) % capacity;

        if (key >= entry_t::mask_key)
            return result;

        for (index_t iters = 0; iters < probe_length; ++iters) {

            entry_t probed = data[index].load(std::memory_order_relaxed);

            if (probed == entry_t::get_empty() ||
               (probed.get_cnt() != entry_t::mask_cnt &&
                probed.get_cnt() < iters))
                return result;

            if (probed.get_key() == key)
                result.push_back(probed.get_val());

            index = prob_func(index, iters, key) % capacity;
        }

        return result;
    }

    bool add(
        index_t key,
        index_t val) {

        if (key >= entry_t::mask_key || val >= entry_t::mask_val)
            return false;

        entry_t nil = entry_t::get_empty(), entry;
        index_t index = hash_func(key) % capacity;

        for (index_t iters = 0; iters < probe_length; ++iters) {

            entry_t triple = data[index].load(std::memory_order_relaxed);
            entry.set_triple_safe(key, val, iters);

            if (triple == nil) {
                std::atomic_compare_exchange_strong(&data[index], &nil, entry);

                if (nil == entry_t::get_empty())
                    return ++size;
                else
                    nil = entry_t::get_empty();
            }

            triple = data[index].load(std::memory_order_relaxed);

            // take from the rich!
            while(triple.get_cnt() < entry.get_cnt()) {
                if (std::atomic_compare_exchange_weak(&data[index],
                                                      &triple,
                                                      entry)) {
                    key   = triple.get_key();
                    val   = triple.get_val();
                    iters = triple.get_cnt();
                }
            }

            index = prob_func(index, iters, key) % capacity;
        }

        return false;
    }
};

#endif
