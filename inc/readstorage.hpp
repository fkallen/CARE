#ifndef READ_STORAGE_HPP
#define READ_STORAGE_HPP

#include "read.hpp"

#include <map>
#include <vector>

/*
    Data structure to store reads, either compressed or uncompressed
*/

struct ReadStorage{

    int nThreads;
    bool isReadOnly;

    std::vector<std::vector<std::string>> headers;
    std::vector<std::vector<std::string>> qualityscores;
    std::vector<std::vector<std::string>> reverseComplqualityscores;
    std::vector<std::vector<Sequence>> sequences;
    std::vector<std::vector<Sequence>> reverseComplSequences;
    std::vector<Sequence*> sequencepointers;
    std::vector<Sequence> sequences_dedup;

    std::vector<std::uint64_t> bytecounts;

    //std::vector<std::unordered_map<std::string>> dedupqscores;

    ReadStorage();

    // threads: maximum number of threads that insert into ReadStorage in parallel
    ReadStorage(int threads);

    ReadStorage(const ReadStorage& other);

    ReadStorage& operator=(const ReadStorage& other);

    ReadStorage& operator=(const ReadStorage&& other);

    void clear();

    void insertRead(std::uint32_t readNumber, const Read& read);
    void noMoreInserts();
    Read fetchRead(std::uint32_t readNumber) const;

    const std::string* fetchHeader_ptr(std::uint32_t readNumber) const;
    const std::string* fetchQuality_ptr(std::uint32_t readNumber) const;
    const std::string* fetchReverseComplementQuality_ptr(std::uint32_t readNumber) const;
    const Sequence* fetchSequence_ptr(std::uint32_t readNumber) const;
    const Sequence* fetchReverseComplementSequence_ptr(std::uint32_t readNumber) const;
};


#endif
