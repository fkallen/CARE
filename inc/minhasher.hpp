#ifndef CARE_MINHASHER_HPP
#define CARE_MINHASHER_HPP

#include "options.hpp"
#include "kvmapfixed.hpp"
#include "ganja/hpc_helpers.cuh"

#include <set>
#include <cstdint>
#include <memory>
#include <map>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <limits>

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

    using Key_t = std::uint64_t; //hash value type
    using Index_t = std::uint64_t; //read id type

    struct Value{
        HOSTDEVICEQUALIFIER
        Value() : Value(0){}
        HOSTDEVICEQUALIFIER
        Value(Index_t payload) : payload(payload){}
        Value(const Value& other) = default;
        Value(Value&& other) = default;
        Value& operator=(const Value& other) = default;
        Value& operator=(Value&& other) = default;
        HOSTDEVICEQUALIFIER
        bool operator==(const Value& other){ return payload == other.payload;}
        HOSTDEVICEQUALIFIER
        bool operator!=(const Value& other){ return !(*this == other);}
        HOSTDEVICEQUALIFIER
        bool operator<(const Value& other){ return payload < other.payload;}

        HOSTDEVICEQUALIFIER
        Index_t getReadId() const{
            return payload;
        }

        Index_t payload;
    };

    using Value_t = Index_t; //Value type for hashmap
    using Result_t = Index_t; // Return value for minhash query

	static constexpr int bits_key = sizeof(Key_t) * 8;
	static constexpr std::uint64_t key_mask = (std::uint64_t(1) << (bits_key - 1)) | ((std::uint64_t(1) << (bits_key - 1)) - 1);
    static constexpr std::uint64_t max_read_num = std::numeric_limits<Index_t>::max();
    static constexpr int maximum_number_of_maps = 16;


	// the actual hash maps
	std::vector<std::unique_ptr<KVMapFixed<Key_t, Value_t, Index_t>>> minhashTables;
	MinhashOptions minparams;
	std::uint64_t nReads;

	std::chrono::duration<double> minhashtime;
	std::chrono::duration<double> maptime;

	Minhasher();

	Minhasher(const MinhashOptions& parameters);

	void init(Index_t nReads);

	void clear();

	void insertSequence(const std::string& sequence, const std::uint64_t readnum);

    std::vector<Result_t> getCandidates(const std::string& sequence) const;
	//std::vector<Result_t> getCandidates(MinhasherBuffers& buffers, const std::string& sequence) const;

	void saveTablesToFile(std::string filename) const;

	bool loadTablesFromFile(std::string filename);

	void transform();



private:
	void minhashfunc(const std::string& sequence, std::uint64_t* minhashSignature, bool* isForwardStrand) const;
};

}

#endif
