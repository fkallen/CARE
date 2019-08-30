#ifndef CARE_CONFIG_HPP
#define CARE_CONFIG_HPP

#include <cstdint>
#include <type_traits>


namespace care{




    //unsigned integral type large enough to enumerate all reads
    using read_number = std::uint32_t;

    static_assert(std::is_integral<read_number>::value, "read_number must be integral.");
    static_assert(std::is_unsigned<read_number>::value, "read_number must be unsigned.");

    //unsigned integral type of a kmer in the hash map.
    using kmer_type = std::uint32_t;

    static_assert(std::is_integral<kmer_type>::value, "kmer_type must be integral.");
    static_assert(std::is_unsigned<kmer_type>::value, "kmer_type must be unsigned.");

    //maximum number of minhash maps
    constexpr int maximum_number_of_maps = 48;
    static_assert(maximum_number_of_maps > 0);

    //At least gpuReadStorageHeadroomPerGPU bytes per GPU will not be used by gpuReadStorage
    constexpr size_t gpuReadStorageHeadroomPerGPU = size_t(1) << 30;

//##################################################

}

#endif
