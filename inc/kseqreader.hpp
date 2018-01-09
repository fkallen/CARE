#ifndef KSEQ_READER_HPP
#define KSEQ_READER_HPP

#include <zlib.h>

#include "read.hpp"
#include "kseq/kseq.h"

KSEQ_INIT(gzFile, gzread, gzseek);

struct KseqReader : public ReadReader {
private:
	gzFile fp;
	kseq_t *seq;
	int seqnum;
public:

	KseqReader(const std::string& filename_);

	~KseqReader() override;

	bool getNextRead(Read* read, std::uint32_t* readnumber) override;

	void seekToRead(const int readnumber) override;

	void reset() override;

};


#endif
