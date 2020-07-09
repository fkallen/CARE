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
    using kmer_type = std::uint64_t;

    static_assert(std::is_integral<kmer_type>::value, "kmer_type must be integral.");
    static_assert(std::is_unsigned<kmer_type>::value, "kmer_type must be unsigned.");

    //unsigned integral type to enumerate hits in a hash table
    using BucketSize = std::uint16_t;
    static_assert(std::is_integral<BucketSize>::value, "BucketSize must be integral.");
    static_assert(std::is_unsigned<BucketSize>::value, "BucketSize must be unsigned.");

    //maximum number of minhash maps
    constexpr int maximum_number_of_maps = 48;
    static_assert(maximum_number_of_maps > 0, "");


    //At least gpuReadStorageHeadroomPerGPU bytes per GPU will not be used by gpuReadStorage
    constexpr std::size_t gpuReadStorageHeadroomPerGPU = std::size_t(1) << 30;

    //During construction of minhasher, minhasherConstructionNumMaps maps will be constructed before each of those is transformed into
    //a space efficient format.
    //greater number means faster construction time and greater memory usage during construction.
    //smaller number means slower construction time and less memory usage during construction.
    constexpr int minhasherConstructionNumMaps = 16;


    //Controls file size of temporary results.
    //tmpresultfileformat = 0 -> use a plain text file. use gnu sort for sorting
    //tmpresultfileformat = 1 -> use space efficient format. use custom sort function
    constexpr int tmpresultfileformat = 1;
    static_assert(0 <= tmpresultfileformat && tmpresultfileformat <= 1, "");

//##################################################

    template<class T> struct max_k;
    template<> struct max_k<std::uint8_t>{static constexpr int value = 4;};
    template<> struct max_k<std::uint16_t>{static constexpr int value = 8;};
    template<> struct max_k<std::uint32_t>{static constexpr int value = 16;};
    template<> struct max_k<std::uint64_t>{static constexpr int value = 32;};
}

#endif
