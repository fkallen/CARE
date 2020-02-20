#ifndef CARE_SEQUENCEFILEIO_HPP
#define CARE_SEQUENCEFILEIO_HPP

#include <config.hpp>
#include <memoryfile.hpp>

#include <kseqpp/kseqpp.hpp>

#include <hpc_helpers.cuh>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <exception>
#include <memory>
#include <sstream>

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
    read_number readNumber;
	std::string name = "";
    std::string comment = "";
	std::string sequence = "";
	std::string quality = "";

	bool operator==(const Read& other) const
	{
		return (readNumber == other.readNumber && name == other.name 
                && comment == other.comment && sequence == other.sequence 
                && quality == other.quality);
	}
	bool operator!=(const Read& other) const
	{
		return !(*this == other);
	}

	void reset()
	{
		name.clear();
        comment.clear();
		sequence.clear();
		quality.clear();
	}
};

struct SequenceFileWriter{

    SequenceFileWriter(const std::string& filename_, FileFormat format_) : filename(filename_), format(format_)
	{

	};
	virtual ~SequenceFileWriter()
	{

	}

    void writeRead(const Read& read);

    void writeRead(const std::string& name, const std::string& comment, const std::string& sequence, const std::string& quality);

    virtual void writeReadImpl(const std::string& name, const std::string& comment, const std::string& sequence, const std::string& quality) = 0;

    virtual void writeImpl(const std::string& data) = 0;

protected:

    std::string filename;
    FileFormat format;

};

struct UncompressedWriter : public SequenceFileWriter{
    UncompressedWriter(const std::string& filename, FileFormat format);

    void writeReadImpl(const std::string& name, const std::string& comment, const std::string& sequence, const std::string& quality) override;
    void writeImpl(const std::string& data) override;

    bool isFastq;
    char delimHeader;

    std::ofstream ofs;
};

struct GZipWriter : public SequenceFileWriter{
    static constexpr int maxBufferedReads = 128;

    GZipWriter(const std::string& filename, FileFormat format);
    ~GZipWriter();

    void writeReadImpl(const std::string& name, const std::string& comment, const std::string& sequence, const std::string& quality) override;
    void writeImpl(const std::string& data) override;

private:

    void bufferRead(const std::string& name, const std::string& comment, const std::string& sequence, const std::string& quality){
        buffer << delimHeader << name << ' ' << comment << '\n'
            << sequence << '\n';
        if(isFastq){
            buffer << '+' << '\n'
                << quality << '\n';
        }

        numBufferedReads++;
    }

    void writeBufferedReads(){
        writeImpl(buffer.str());
        numBufferedReads = 0;
        buffer.str(std::string());
        buffer.clear();
    }

    bool isFastq;
    char delimHeader;

    int numBufferedReads;
    std::stringstream buffer;

    gzFile fp;
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

template<class Func>
void forEachReadInFile(const std::string& filename, FileFormat format, Func f){
    // std::unique_ptr<SequenceFileReader> reader = makeSequenceReader(filename, format);

    // Read read;
    // while (reader->getNextRead(&read)) {
    //     std::uint64_t readnum = reader->getReadnum()-1;
    //     f(readnum, read);
    // }

    kseqpp::KseqPP reader(filename);

    Read read;

    std::int64_t readNumber = 0;

    auto getNextRead = [&](){
        const int status = reader.next();
        //std::cerr << "parser status = 0 in file " << filenames[i] << '\n';
        if(status >= 0){
            read.readNumber = readNumber;
            read.name = reader.name;
            read.comment = reader.comment;
            read.sequence = reader.seq;
            read.quality = reader.qual;
        }else if(status < -1){
            std::cerr << "parser error status " << status << " in file " << filename << '\n';
        }

        readNumber++;

        bool success = (status >= 0);

        return success;
    };

    bool success = getNextRead();

    while(success){

        f(read.readNumber, read);

        success = getNextRead();
    }
}


/*
    Deletes every file in vector filenames
*/
void deleteFiles(std::vector<std::string> filenames);


struct EncodedTempCorrectedSequence{
    std::uint32_t encodedflags; //contains size of data in bytes, and boolean flags
    read_number readId;
    std::unique_ptr<std::uint8_t[]> data;

    EncodedTempCorrectedSequence() = default;
    EncodedTempCorrectedSequence(EncodedTempCorrectedSequence&&) = default;
    EncodedTempCorrectedSequence& operator=(EncodedTempCorrectedSequence&&) = default;

    EncodedTempCorrectedSequence(const EncodedTempCorrectedSequence& rhs){
        *this = rhs;
    }

    EncodedTempCorrectedSequence& operator=(const EncodedTempCorrectedSequence& rhs){
        encodedflags = rhs.encodedflags;
        readId = rhs.readId;

        const int numBytes = rhs.getNumBytes();
        data = std::make_unique<std::uint8_t[]>(numBytes);
        std::memcpy(data.get(), rhs.data.get(), numBytes);

        return *this;
    }

    bool writeToBinaryStream(std::ostream& s) const;
    bool readFromBinaryStream(std::istream& s);

    std::uint8_t* copyToContiguousMemory(std::uint8_t*, std::uint8_t*) const;
    void copyFromContiguousMemory(const std::uint8_t*);

    bool operator==(const EncodedTempCorrectedSequence& rhs) const{
        std::uint32_t numBytes = 123;
        return encodedflags == rhs.encodedflags && readId == rhs.readId 
                && std::memcmp(data.get(), rhs.data.get(), numBytes);
    }

    bool operator!=(const EncodedTempCorrectedSequence& rhs) const{
        return !(operator==(rhs));
    }

    int getNumBytes() const{
        constexpr std::uint32_t mask = (std::uint32_t(1) << 29)-1;
        return (encodedflags & mask);
    }
};

// represents a sequence produced by the correction of a read.
// Will be saved to file during correction.
// Will be loaded from file during mergeResultFiles
struct TempCorrectedSequence{
    enum class Type : int {Anchor, Candidate};
    struct Edit{
        char base;
        int pos;

        Edit() = default;
        HOSTDEVICEQUALIFIER
        Edit(int p, char b) : base(b), pos(p){}

        HOSTDEVICEQUALIFIER
        bool operator==(const Edit& rhs) const{
            return base == rhs.base && pos == rhs.pos;
        }

        HOSTDEVICEQUALIFIER
        bool operator!=(const Edit& rhs) const{
            return !(operator==(rhs));
        }
    };
    static constexpr char AnchorChar = 'a';
    static constexpr char CandidateChar = 'c';

    TempCorrectedSequence() = default;
    TempCorrectedSequence(const TempCorrectedSequence&) = default;
    TempCorrectedSequence(TempCorrectedSequence&&) = default;
    TempCorrectedSequence& operator=(const TempCorrectedSequence&) = default;
    TempCorrectedSequence& operator=(TempCorrectedSequence&&) = default;

    TempCorrectedSequence(const EncodedTempCorrectedSequence&);
    TempCorrectedSequence& operator=(const EncodedTempCorrectedSequence&);

    EncodedTempCorrectedSequence encode() const;
    void decode(const EncodedTempCorrectedSequence&);

    bool writeToBinaryStream(std::ostream& s) const;
    bool readFromBinaryStream(std::istream& s);

    bool hq = false; //if anchor
    bool useEdits = false;
    Type type = Type::Anchor;
    int shift = 0; //if candidate
    read_number readId = 0;

    std::string sequence = "";
    std::vector<Edit> edits;
    std::vector<int> uncorrectedPositionsNoConsensus{}; //if anchor

    bool operator==(const TempCorrectedSequence& rhs) const{
        return hq == rhs.hq && useEdits == rhs.useEdits && type == rhs.type && shift == rhs.shift && readId == rhs.readId
            && sequence == rhs.sequence && edits == rhs.edits && uncorrectedPositionsNoConsensus == rhs.uncorrectedPositionsNoConsensus;
    }

    bool operator!=(const TempCorrectedSequence& rhs) const{
        return !(operator==(rhs));
    }

    
};





void mergeResultFiles(
                    const std::string& tempdir,
                    std::uint32_t expectedNumReads, 
                    const std::string& originalReadFile,
                    FileFormat originalFormat,
                    MemoryFile<EncodedTempCorrectedSequence>& partialResults, 
                    const std::string& outputfile,
                    bool isSorted);

void mergeResultFiles(
                    const std::string& tempdir,
                    std::uint32_t expectedNumReads, 
                    const std::string& originalReadFile,
                    FileFormat originalFormat,
                    MemoryFileFixedSize<EncodedTempCorrectedSequence>& partialResults, 
                    const std::string& outputfile,
                    bool isSorted);







std::ostream& operator<<(std::ostream& os, const TempCorrectedSequence& tmp);
std::istream& operator>>(std::istream& is, TempCorrectedSequence& tmp);

} //end namespace

#endif
