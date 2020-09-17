#ifndef CARE_MINHASHER_HPP
#define CARE_MINHASHER_HPP

#include <hpc_helpers.cuh>
#include <util.hpp>
#include <config.hpp>
#include <memorymanagement.hpp>
#include <options.hpp>
#include <readstorage.hpp>
#include <cpuhashtable.hpp>

#include <cstdint>
#include <memory>
#include <map>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <numeric>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <array>
#include <fstream>
#include <cassert>
#include <iterator>
#include <unordered_map>


namespace care{


struct Minhasher {

    using Key_t = kmer_type;
    using Value_t = read_number;

    using Map_t = CpuReadOnlyMultiValueHashTable<kmer_type, read_number>;

    using Range_t = std::pair<const Value_t*, const Value_t*>;

    static constexpr int bits_key = sizeof(Key_t) * 8;
	static constexpr std::uint64_t key_mask = (std::uint64_t(1) << (bits_key - 1)) | ((std::uint64_t(1) << (bits_key - 1)) - 1);
    static constexpr int maximum_kmer_length = max_k<kmer_type>::value;

    struct Handle{        
        std::vector<Range_t> ranges;
		std::vector<Value_t> allUniqueResults;
        SetUnionHandle<Value_t> suHandle;

        

        std::vector<Value_t> contiguousDataOfRanges;
        std::vector<Range_t> multiranges;
		std::vector<Value_t> multiallUniqueResults;
        std::vector<std::uint64_t> multiminhashSignatures;
        std::vector<int> numResultsPerSequence;
        std::vector<int> numResultsPerSequencePrefixSum;

        std::vector<Value_t>& result() noexcept{
            return allUniqueResults;
        }

        std::vector<Value_t>& multiresults() noexcept{
            return multiallUniqueResults;
        }

        // int numResultsOfSequence(int i) const{
        //     assert(i < int(numResultsPerSequence.size()));
        //     return numResultsPerSequence[i];
        // }
    };

    int kmerSize;
    int resultsPerMapThreshold;
    std::vector<std::unique_ptr<Map_t>> minhashTables;

    // Minhasher();

    // Minhasher(const MinhashOptions& parameters);

    Minhasher() : Minhasher(16, 50){

    }

    Minhasher(int kmerSize, int resultsPerMapThreshold)
        : kmerSize(kmerSize), resultsPerMapThreshold(resultsPerMapThreshold){

    }

    Minhasher(const Minhasher&) = delete;
    Minhasher& operator=(const Minhasher&) = delete;

    Minhasher(Minhasher&& rhs);
    Minhasher& operator=(Minhasher&& rhs);

    bool operator==(const Minhasher& rhs) const;

    bool operator!=(const Minhasher& rhs) const;

    void construct(
        const FileOptions& fileOptions,
        const RuntimeOptions& runtimeOptions,
        const MemoryOptions& memoryOptions,
        std::uint64_t nReads,
        const CorrectionOptions& correctionOptions,
        care::cpu::ContiguousReadStorage& readStorage
    );

    int getNumberOfMaps() const{
        return minhashTables.size();
    }

    int getKmerSize() const{
        return kmerSize;
    }

    int getNumResultsPerMapThreshold() const{
        return resultsPerMapThreshold;
    }

    MemoryUsage getMemoryInfo() const;

    void writeToStream(std::ostream& os) const;

    int loadFromStream(std::ifstream& is, int numMapsUpperLimit = std::numeric_limits<int>::max());

	void clear();

	void destroy();

    std::pair<const Value_t*, const Value_t*> queryMap(int mapid,
                                                        Key_t key) const noexcept;                               


    void getCandidates_any_map(
            Minhasher::Handle& handle,
            const std::string& sequence,
            std::uint64_t) const noexcept;

    void getCandidates_any_map(
            Minhasher::Handle& handle,
            const char* sequence,
            int sequenceLength,
            std::uint64_t) const noexcept;

private:

    std::array<std::uint64_t, maximum_number_of_maps>
    minhashfunc(const std::string& sequence) const noexcept{
        return minhashfunc(sequence, getNumberOfMaps());
    }

    std::array<std::uint64_t, maximum_number_of_maps> 
    minhashfunc(const char* sequence, int sequenceLength) const noexcept{
        return minhashfunc(sequence, sequenceLength, getNumberOfMaps());
    }


	std::array<std::uint64_t, maximum_number_of_maps>
    minhashfunc(const std::string& sequence, int numHashfuncs) const noexcept;

    std::array<std::uint64_t, maximum_number_of_maps> 
    minhashfunc(const char* sequence, int sequenceLength, int numHashfuncs) const noexcept;

};



int calculateResultsPerMapThreshold(int coverage);




}

#endif
