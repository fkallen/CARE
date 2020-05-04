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



namespace care{

enum class FileFormat {FASTA, FASTQ, FASTAGZ, FASTQGZ, NONE};

struct SequenceFileProperties{
        std::uint64_t nReads{};
        int minSequenceLength{};
        int maxSequenceLength{};
};


struct Read {
	std::string name = "";
    std::string comment = "";
	std::string sequence = "";
	std::string quality = "";

	bool operator==(const Read& other) const
	{
		return (name == other.name 
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



std::unique_ptr<SequenceFileWriter> makeSequenceWriter(const std::string& filename, FileFormat fileFormat);

bool hasQualityScores(const std::string& filename);
FileFormat getFileFormat(const std::string& filename);


SequenceFileProperties getSequenceFileProperties(const std::string& filename);
SequenceFileProperties getSequenceFileProperties(const std::string& filename, bool printProgress);

std::uint64_t getNumberOfReads(const std::string& filename);

template<class Func>
void forEachReadInFile(const std::string& filename, Func f){

    kseqpp::KseqPP reader(filename);

    Read read;

    std::int64_t readNumber = 0;

    auto getNextRead = [&](){
        const int status = reader.next();
        //std::cerr << "parser status = 0 in file " << filenames[i] << '\n';
        if(status >= 0){
            read.name = reader.getCurrentName();
            read.comment = reader.getCurrentComment();
            read.sequence = reader.getCurrentSequence();
            read.quality = reader.getCurrentQuality();
        }else if(status < -1){
            std::cerr << "parser error status " << status << " in file " << filename << '\n';
        }

        bool success = (status >= 0);

        return success;
    };

    bool success = getNextRead();

    while(success){

        f(readNumber, read);
        
        readNumber++;

        success = getNextRead();
    }
}



} //end namespace

#endif
