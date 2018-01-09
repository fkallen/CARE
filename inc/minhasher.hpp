#ifndef MINHASHER_HPP
#define MINHASHER_HPP

#include "read.hpp"
#include "countminsketch.hpp"

#include "ganja/open_addressing_multi_hash_map.cuh"
#include "ganja/hash_functions.cuh"

#include <set>
#include <cstdint>
#include <memory>
#include <map>
#include <atomic>
#include <chrono>
#include <stdexcept>

struct MinhashParameters {
	int maps;
	int k;

	MinhashParameters() : MinhashParameters(1, 1)
	{
	}

	MinhashParameters(const int number_maps, const int k_)
		:  maps(number_maps), k(k_)
	{
		if(maps < 1 || k < 1){
			throw std::runtime_error("constructor arguments of MinhashParameters must be greater than zero.");
		}
	}
};

struct Minhasher {

	// configure hash map

	// bits_key bits are used to store kmer hash value. if hash value has more bits, leftover bits are discarded
	// bits_val - 1 bits are used to store read id
	// last bit is used internally
	static constexpr int bits_key = 32;
	static constexpr int bits_val = 32;

	static_assert(bits_key > 0, "bits_key == 0!!!");
	static_assert(bits_val > 0, "bits_val == 0!!!");
	static_assert((bits_key + bits_val) == 64, "invalid key/value partition in hashmap");

	static constexpr std::uint64_t hv_bitmask = (std::uint64_t(1) << bits_key) - 1;
	static constexpr std::uint64_t MAX_READ_NUM = (std::uint64_t(1) << (bits_val-1)) - 1;

	using hash_func = mueller_hash_uint32_t;
	using prob_func = linear_probing_scheme_t;
	using oa_hash_t = OpenAddressingMultiHashMap<uint64_t, bits_key, bits_val,
					   hash_func,prob_func>;

	// the actual hash maps
	std::vector<std::unique_ptr<oa_hash_t> > minhashTables;
	MinhashParameters minparams;
	double load;
	std::chrono::duration<double> minhashtime;
	std::chrono::duration<double> maptime;

	Minhasher();

	Minhasher(const MinhashParameters& parameters);

	void init();
	void init(std::uint64_t nReads, double load);

	void clear();

	int insertSequence(const std::string& sequence, const std::uint64_t readnum);

	std::vector<std::pair<std::uint64_t, int>> getCandidates(const std::string& sequence) const;



private:
	// calculate band hash values of a read
	int make_minhash_band_hashes(const std::string& sequence, std::uint64_t* bandHashValues, std::uint32_t* isForwardStrand) const;
	int minhashfunc(const std::string& sequence, std::uint64_t* minhashSignature, std::uint32_t* isForwardStrand) const; // calculates the hash values
	void bandhashfunc(const std::uint64_t* minhashSignature, std::uint64_t* bandHashValues) const; // combines multiple hash values to keys
};

#endif
