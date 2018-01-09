#ifndef FASTQ_WRITER_HPP
#define FASTQ_WRITER_HPP

struct FastqWriter {
public:
	FastqWriter(){}

	virtual ~FastqWriter(){}

	virtual void writeRead(std::ostream& stream, const Read& read) const
	{
		stream << read.header << '\n' 
			<< read.sequence << '\n'
			<< '+' << '\n'
			<< read.quality << '\n';
	}
};





#endif
