#include "../inc/sequencefileio.hpp"

#include <string>
#include <sstream>
#include <stdexcept>

namespace care{

    bool SequenceFileReader::getNextRead(Read* read){
        if(read == nullptr) return false;

        bool b = getNextRead_impl(read);
        if(b)
            readnum++;
        return b;
    }

    FastqReader::FastqReader(const std::string& filename_) : SequenceFileReader(filename_){
		is.open(filename);
		if (!(bool)is)
			throw std::runtime_error("could not open file " + filename);
	}

	FastqReader::~FastqReader(){
		is.close();
	}

	bool FastqReader::getNextRead_impl(Read* read){
		if (!is.good())
			return false;

		read->reset();
		std::getline(is, read->header);
		if (!is.good())
			return false;
		if (read->header[0] != '@') {
			std::stringstream ss; ss << "unexpected file format of file " << filename << ". Header does not start with @";
			throw std::runtime_error(ss.str());
		}

		std::getline(is, read->sequence);
		if (!is.good())
			return false;
		std::getline(is, stmp);
		if (!is.good())
			return false;

		if (stmp[0] != '+') {
			std::stringstream ss; ss << "unexpected file format of file " << filename << ". Line does not start with +";
			throw std::runtime_error(ss.str());
		}

		std::getline(is, read->quality);
		if (!is.good())
			return false;

		return !(is.fail() || is.bad());
	}
}
