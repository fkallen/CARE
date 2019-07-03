#ifndef CARE_SEQUENCEFILEIO_HPP
#define CARE_SEQUENCEFILEIO_HPP

#include <config.hpp>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <exception>
#include <memory>

#include <fcntl.h> // open
#include <zlib.h> //gzFile

namespace care{

enum class FileFormat {FASTA, FASTQ, FASTAGZ, FASTQGZ, NONE};

struct SequenceFileProperties{
        std::uint64_t nReads{};
        int minSequenceLength{};
        int maxSequenceLength{};
};

class SkipException : public std::exception {
public:
	SkipException() : std::exception()
	{
	}

	virtual const char* what() const noexcept{
		return std::exception::what();
	}
};

struct Read {
	std::string header = "";
	std::string sequence = "";
	std::string quality = "";

	bool operator==(const Read& other) const
	{
		return (header == other.header && sequence == other.sequence && quality == other.quality);
	}
	bool operator!=(const Read& other) const
	{
		return !(*this == other);
	}

	void reset()
	{
		header.clear();
		sequence.clear();
		quality.clear();
	}
};

struct SequenceFileWriter{
    SequenceFileWriter(const std::string& filename_) : filename(filename_)
	{
	};
	virtual ~SequenceFileWriter()
	{
	}

    void writeRead(const std::string& header, const std::string& sequence, const std::string& quality);

    virtual void writeReadImpl(const std::string& header, const std::string& sequence, const std::string& quality) = 0;

    std::string filename;
};

struct UncompressedWriter : public SequenceFileWriter{
    UncompressedWriter(const std::string& filename, FileFormat format);

    void writeReadImpl(const std::string& header, const std::string& sequence, const std::string& quality);

    std::ofstream ofs;
    FileFormat format;
};

struct GZipWriter : public SequenceFileWriter{
    GZipWriter(const std::string& filename, FileFormat format);
    ~GZipWriter();

    void writeReadImpl(const std::string& header, const std::string& sequence, const std::string& quality);

    gzFile fp;
    FileFormat format;
};

struct SequenceFileReader {

public:
	SequenceFileReader(const std::string& filename_) : filename(filename_), readnum(0)
	{
	};
	virtual ~SequenceFileReader()
	{
	}

    //return false if EOF or if error occured. true otherwise
    bool getNextRead(Read* read);
    bool getNextReadUnsafe(Read* read);

    std::uint64_t getReadnum() const{
        return readnum;
    }

    void skipBytes(std::uint64_t nBytes){
		skipBytes_impl(nBytes);
	}

	void skipReads(std::uint64_t nReads){
		skipReads_impl(nReads);
	}


    //return false if EOF or if error occured while reading file. true otherwise
    virtual bool getNextRead_impl(Read* read) = 0;
    virtual bool getNextReadUnsafe_impl(Read* read) = 0;
	virtual void skipBytes_impl(std::uint64_t nBytes) = 0;
	virtual void skipReads_impl(std::uint64_t nReads) = 0;
	std::string filename;




    std::uint64_t readnum; // 1 bases read id of read returned by previous successful call to getNextRead
};


struct FastqReader : public SequenceFileReader {
public:
	FastqReader(const std::string& filename);

	~FastqReader() override;


	bool getNextRead_impl(Read* read) override;
    bool getNextReadUnsafe_impl(Read* read) override;
	void skipBytes_impl(std::uint64_t nBytes) override;
	void skipReads_impl(std::uint64_t nBytes) override;

	std::ifstream is;
	std::string stmp;
};

struct KseqReader : public SequenceFileReader {
public:
	KseqReader(const std::string& filename);

	~KseqReader() override;


	bool getNextRead_impl(Read* read) override;
    bool getNextReadUnsafe_impl(Read* read) override;
	void skipBytes_impl(std::uint64_t nBytes) override;
	void skipReads_impl(std::uint64_t nBytes) override;

    int fp;
    void* seq; //pointer to kseq_t
};

struct KseqGzReader : public SequenceFileReader {
public:
	KseqGzReader(const std::string& filename);

	~KseqGzReader() override;


	bool getNextRead_impl(Read* read) override;
    bool getNextReadUnsafe_impl(Read* read) override;
	void skipBytes_impl(std::uint64_t nBytes) override;
	void skipReads_impl(std::uint64_t nBytes) override;

    gzFile fp;
    void* seq; //pointer to kseq_t
};

std::unique_ptr<SequenceFileReader> makeSequenceReader(const std::string& filename, FileFormat fileFormat);
std::unique_ptr<SequenceFileWriter> makeSequenceWriter(const std::string& filename, FileFormat fileFormat);

bool hasGzipHeader(const std::string& filename);
bool hasQualityScores(const std::unique_ptr<SequenceFileReader>& reader);
FileFormat getFileFormat(const std::string& filename);


SequenceFileProperties getSequenceFileProperties(const std::string& filename, FileFormat format);
std::uint64_t getNumberOfReadsFast(const std::string& filename, FileFormat format);
std::uint64_t getNumberOfReads(const std::string& filename, FileFormat format);

/*
    Deletes every file in vector filenames
*/
void deleteFiles(std::vector<std::string> filenames);

/*
    Merges temporary results with unordered reads into single file outputfile with ordered reads.
    Temporary result files are expected to be in format:

    readnumber
    sequence
    readnumber
    sequence
    ...
*/
void mergeResultFiles(std::uint32_t expectedNumReads, const std::string& originalReadFile,
                      FileFormat originalFormat,
                      const std::vector<std::string>& filesToMerge, const std::string& outputfile);

void mergeResultFiles2(std::uint32_t expectedNumReads, const std::string& originalReadFile,
                    FileFormat originalFormat,
                    const std::vector<std::string>& filesToMerge, const std::string& outputfile,
                    size_t tempbytes);

} //end namespace

#endif
