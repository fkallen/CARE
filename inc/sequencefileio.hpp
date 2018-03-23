#ifndef CARE_SEQUENCEFILEIO_HPP
#define CARE_SEQUENCEFILEIO_HPP

#include "read.hpp"

#include <cstdint>
#include <string>
#include <fstream>

namespace care{

struct SequenceFileReader {

public:
	SequenceFileReader(const std::string& filename_) : filename(filename_), readnum(0)
	{
	};
	virtual ~SequenceFileReader()
	{
	}

    //return false if EOF or if error occured. true otherwise
    bool getNextRead(Read* read);

    std::uint64_t getReadnum() const{
        return readnum;
    }

protected:
    //return false if EOF or if error occured while reading file. true otherwise
    virtual bool getNextRead_impl(Read* read) = 0;
	std::string filename;
private:



    std::uint64_t readnum; // 1 bases read id of read returned by previous successful call to getNextRead
};


struct FastqReader : public SequenceFileReader {
public:
	FastqReader(const std::string& filename);

	~FastqReader() override;

protected:
	bool getNextRead_impl(Read* read) override;

private:
	std::ifstream is;
	std::string stmp;
};

}

#endif
