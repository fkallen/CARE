#ifndef READ_STORAGE_HPP
#define READ_STORAGE_HPP

#include "read.hpp"

#include <map>
#include <vector>

/*
    Data structure to store reads, either compressed or uncompressed
*/

struct ReadStorage{

    bool isReadOnly;
    bool useQualityScores;

    std::vector<std::string> headers;
    std::vector<std::string> qualityscores;
    std::vector<std::string> reverseComplqualityscores;
    std::vector<Sequence> sequences;

    std::vector<Sequence*> sequencepointers;
    std::vector<Sequence*> reverseComplSequencepointers;

    std::vector<Sequence> all_unique_sequences; //forward and reverse complement

    ReadStorage();

    ReadStorage& operator=(const ReadStorage&& other);

    void clear();
    void init(size_t nReads);

    void setUseQualityScores(bool use);

    void insertRead(size_t readNumber, const Read& read);
    void noMoreInserts();
    Read fetchRead(size_t readNumber) const;

    const std::string* fetchHeader_ptr(size_t readNumber) const;
    const std::string* fetchQuality_ptr(size_t readNumber) const;
    const std::string* fetchReverseComplementQuality_ptr(size_t readNumber) const;
    const Sequence* fetchSequence_ptr(size_t readNumber) const;
    const Sequence* fetchReverseComplementSequence_ptr(size_t readNumber) const;

    double getMemUsageMB() const;
};


#endif

