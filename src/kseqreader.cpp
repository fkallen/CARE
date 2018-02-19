#include "../inc/kseqreader.hpp"

#include <stdexcept>
#include <stdio.h>

#include "../inc/read.hpp"
#include "../inc/kseq/kseq.h"


//int  read(  int  handle,  void  *buffer,  int  nbyte );

//fread(buf, 1, sizeof buf, file)

//__read(ks->f, ks->buf, __bufsize);





	KseqReader::KseqReader(const std::string& filename_) : ReadReader(filename_), seqnum(0)
	{
		fp = fopen(filename.c_str(), "r");
		if (!fp)
			throw std::runtime_error("could not open file " + filename);
		seq = kseq_init(fileno(fp));
	}

	KseqReader::~KseqReader()
	{
		kseq_destroy(seq);
		fclose(fp);
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

			if (sequencenumber != nullptr) {
				*sequencenumber = seqnum;
			}

            seqnum++;            
		}

		return (l >= 0);
	}

	void KseqReader::seekToRead(const int sequencenumber)
	{
		throw std::runtime_error("seekToSequence not supported in KseqReader!");
	}

	void KseqReader::reset()
	{
		throw std::runtime_error("seekToSequence not supported in KseqReader!");
	}
