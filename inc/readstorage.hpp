#ifndef READ_STORAGE_HPP
#define READ_STORAGE_HPP

#include "sequencefileio.hpp"
#include "sequence.hpp"
#include "types.hpp"

#include <map>
#include <vector>

namespace care{

/*
    Data structure to store reads, either compressed or uncompressed
*/

struct ReadStorage{

    bool isReadOnly;
    bool useQualityScores;

    std::vector<std::string> headers;
    std::vector<std::string> qualityscores;
    std::vector<std::string> reverseComplqualityscores;
    std::vector<Sequence_t> sequences;

    std::vector<Sequence_t*> sequencepointers;
    std::vector<Sequence_t*> reverseComplSequencepointers;

    std::vector<Sequence_t> all_unique_sequences; //forward and reverse complement

    ReadStorage();

    ReadStorage& operator=(const ReadStorage&& other);

    void clear();
    void init(ReadId_t nReads);
    void destroy();

    void setUseQualityScores(bool use);

    void insertRead(ReadId_t readNumber, const Read& read);
    void noMoreInserts();
    Read fetchRead(ReadId_t readNumber) const;

    const std::string* fetchHeader_ptr(ReadId_t readNumber) const;
    const std::string* fetchQuality_ptr(ReadId_t readNumber) const;
    const std::string* fetchReverseComplementQuality_ptr(ReadId_t readNumber) const;
    const Sequence_t* fetchSequence_ptr(ReadId_t readNumber) const;
    const Sequence_t* fetchReverseComplementSequence_ptr(ReadId_t readNumber) const;

    double getMemUsageMB() const;
};

}

#endif
