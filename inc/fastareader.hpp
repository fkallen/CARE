#ifndef FASTA_READER_HPP
#define FASTA_READER_HPP

#include "read.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <cstdint>

struct FastaReader : public ReadReader {
private:
	std::ifstream is;
	int readnum;
	std::string stmp;

public:
	FastaReader(const std::string& filename);

	~FastaReader() override;

	bool getNextRead(Read* read, std::uint32_t* readnumber) override;

	void seekToRead(const int readnumber) override;

	void reset() override;
};




#endif
