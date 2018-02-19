#ifndef KSEQ_READER_HPP
#define KSEQ_READER_HPP

#include <stdio.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "read.hpp"
#include "kseq/kseq.h"

/*size_t fread_wrapper(FILE* fp, void* buffer, size_t count){
    return fread(buffer, count, count, fp);
}*/

KSEQ_INIT(int, read);

struct KseqReader : public ReadReader {
private:
	FILE* fp;
	kseq_t* seq;
	int seqnum;
public:

	KseqReader(const std::string& filename_);

	~KseqReader() override;

	bool getNextRead(Read* read, std::uint32_t* readnumber) override;

	void seekToRead(const int readnumber) override;

	void reset() override;

};


#endif
