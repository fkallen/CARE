#ifndef GANJA_HASH_FUNCTIONS_CUH
#define GANJA_HASH_FUNCTIONS_CUH

#include <cstdint>
#include "qualifiers.cuh"

///////////////////////////////////////////////////////////////////////////////
// uint32_t hashes
///////////////////////////////////////////////////////////////////////////////

struct nvidia_hash_uint32_t {

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    uint32_t operator() (
        uint32_t x) const {

        x = (x + 0x7ed55d16) + (x << 12);
        x = (x ^ 0xc761c23c) ^ (x >> 19);
        x = (x + 0x165667b1) + (x <<  5);
        x = (x + 0xd3a2646c) ^ (x <<  9);
        x = (x + 0xfd7046c5) + (x <<  3);
        x = (x ^ 0xb55a4f09) ^ (x >> 16);

        return x;
    }
};

struct mueller_hash_uint32_t {

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    uint32_t operator() (
        uint32_t x) const {

        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x);

        return x;
    }
};


struct murmur_integer_finalizer_hash_uint32_t {

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    uint32_t operator() (
        uint32_t x) const {

        x ^= x >> 16;
        x *= 0x85ebca6b;
        x ^= x >> 13;
        x *= 0xc2b2ae35;
        x ^= x >> 16;

        return x;
    }
};

struct identity_map_t {

    template <
        typename index_t> HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_t operator() (
        index_t x) const {

        return x;
    }
};

///////////////////////////////////////////////////////////////////////////////
// probin schemes
///////////////////////////////////////////////////////////////////////////////

struct linear_probing_scheme_t {

    template <
        typename index_t> HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_t operator() (
        const index_t& index,
        const index_t& iters,
        const index_t& key) const {

        return index + 1;
    }
};


#endif
