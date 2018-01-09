#include "../inc/kseqreader.hpp"

#include <zlib.h>
#include <stdexcept>

#include "../inc/read.hpp"
#include "../inc/kseq/kseq.h"

	

	KseqReader::KseqReader(const std::string& filename_) : ReadReader(filename_), seqnum(0)
	{
		fp = gzopen(filename.c_str(), "r");
		if (!fp)
			throw std::runtime_error("could not open file " + filename);
		seq = kseq_init(fp);
	}

	KseqReader::~KseqReader()
	{
		kseq_destroy(seq);
		gzclose(fp);
	}

	bool KseqReader::getNextRead(Read* read, std::uint32_t* sequencenumber)
	{
		if (read == nullptr)
			throw std::invalid_argument("sequence must not be nullptr");
		read->header.clear();
		read->sequence.clear();
		read->quality.clear();

		int l = kseq_read(seq);
		if (l != -1) {
			read->header.assign(seq->name.s);
			read->sequence.assign(seq->seq.s);
			if (seq->qual.l) {
				read->quality.assign(seq->qual.s);
			}

      			seqnum++;

			if (sequencenumber != nullptr) {
				*sequencenumber = seqnum;
			}
		}

		return (l >= 0);
	}

	void KseqReader::seekToRead(const int sequencenumber)
	{
		throw std::runtime_error("seekToSequence not supported in KseqReader!");
	}

	void KseqReader::reset()
	{
		kseq_reset(seq);
		seqnum = 0;
	}

