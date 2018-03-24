#ifndef CARE_MINHASHER_HPP
#define CARE_MINHASHER_HPP

#include "options.hpp"
#include "read.hpp"
#include "kvmapfixed.hpp"

#include <set>
#include <cstdint>
#include <memory>
#include <map>
#include <atomic>
#include <chrono>
#include <stdexcept>

namespace care{

struct MinhasherBuffers{
	std::uint64_t* allMinhashResults = nullptr;
	std::uint64_t* d_allMinhashResults = nullptr;
	size_t size = 0;
	size_t capacity = 0;
	int deviceId = -1;
#ifdef __NVCC__
	cudaStream_t stream;
#endif

	MinhasherBuffers(int id);
	void grow(size_t newcapacity);
};

void cuda_cleanup_MinhasherBuffers(MinhasherBuffers& buffer);

struct Minhasher {

	// configure hash map
	using key_t = std::uint64_t;
	static constexpr int bits_key = sizeof(key_t) * 8;
	static constexpr std::uint64_t key_mask = (std::uint64_t(1) << (bits_key - 1)) | ((std::uint64_t(1) << (bits_key - 1)) - 1);


	// the actual hash maps
	std::vector<std::unique_ptr<KVMapFixed<key_t>>> minhashTables;
	MinhashOptions minparams;
	std::uint64_t nReads;

	std::chrono::duration<double> minhashtime;
	std::chrono::duration<double> maptime;

	Minhasher();

	Minhasher(const MinhashOptions& parameters);

	void init(std::uint64_t nReads);

	void clear();

	int insertSequence(const std::string& sequence, const std::uint64_t readnum);

	std::vector<std::pair<std::uint64_t, int>> getCandidatesWithFlag(const std::string& sequence) const;
	std::vector<std::uint64_t> getCandidates(MinhasherBuffers& buffers, const std::string& sequence) const;

	void saveTablesToFile(std::string filename) const;

	bool loadTablesFromFile(std::string filename);

	void transform();



private:
	// calculate band hash values of a read
	int make_minhash_band_hashes(const std::string& sequence, std::uint64_t* bandHashValues, std::uint32_t* isForwardStrand) const;
	int minhashfunc(const std::string& sequence, std::uint64_t* minhashSignature, std::uint32_t* isForwardStrand) const; // calculates the hash values
	void bandhashfunc(const std::uint64_t* minhashSignature, std::uint64_t* bandHashValues) const; // combines multiple hash values to keys
};

}

#endif
