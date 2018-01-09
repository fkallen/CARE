#ifndef FASTA_WRITER_HPP
#define FASTA_WRITER_HPP

struct FastaWriter {
public:
	FastaWriter(){}

	virtual ~FastaWriter(){}

	virtual void writeRead(std::ostream& stream, const Read& read) const
	{
		stream << read.header << '\n' << read.sequence << '\n';
	}
};





#endif
