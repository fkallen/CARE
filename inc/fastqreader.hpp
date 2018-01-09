#ifndef FASTQ_READER_HPP
#define FASTQ_READER_HPP

#include "read.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <cstdint>

struct FastqReader : public ReadReader {
private:
	std::ifstream is;
	int readnum;
	std::string stmp;

public:
	FastqReader(const std::string& filename);

	~FastqReader() override;

	bool getNextRead(Read* read, std::uint32_t* readnumber) override;

	void seekToRead(const int readnumber) override;

	void reset() override;
};




#endif
