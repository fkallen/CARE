#include "../inc/fastqreader.hpp"

#include "../inc/read.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <cstdint>

//#define BE_SAVE

	FastqReader::FastqReader(const std::string& filename_) : ReadReader(filename_), readnum(0)
	{
		is.open(filename);
		if (!(bool)is)
			throw std::runtime_error("could not open file " + filename);
	}

	FastqReader::~FastqReader()
	{
		is.close();
	}

	bool FastqReader::getNextRead(Read* read, std::uint32_t* readnumber)
	{
		if (!is.good())
			return false;

		read->reset();
		std::getline(is, read->header);
		if (!is.good())
			return false;
#ifdef BE_SAVE
		if (read->header[0] != '@') {
			std::stringstream ss; ss << "unexpected file format of file " << filename << " at line " << (readnum * 4 + 1) << ". Header does not start with @";
			throw std::runtime_error(ss.str());
		}
#endif

		std::getline(is, read->sequence);
		if (!is.good())
			return false;
		std::getline(is, stmp);
		if (!is.good())
			return false;
#ifdef BE_SAVE
		if (stmp[0] != '+') {
			std::stringstream ss; ss << "unexpected file format of file " << filename << " at line " << (readnum * 4 + 3) << ". Line does not start with +";
			throw std::runtime_error(ss.str());
		}
#endif
		std::getline(is, read->quality);
		if (!is.good())
			return false;


		if (readnumber != nullptr) {
			*readnumber = readnum;
		}

		readnum++;

		return !(is.fail() || is.bad());
	}

	void FastqReader::seekToRead(const int readnumber)
	{
		long expectedLinesToSkip = readnumber * 4L;

		for (int i = 0; i < expectedLinesToSkip; i++) {
			is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
	}

	void FastqReader::reset()
	{
		is.seekg(std::ios::beg);
		readnum = 0;
	}

