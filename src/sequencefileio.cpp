#include <sequencefileio.hpp>
#include <hpc_helpers.cuh>
#include <config.hpp>
#include <threadsafe_buffer.hpp>
#include <sequence.hpp>

#include <iterator>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <experimental/filesystem>
#include <future>

#include <zlib.h>
#include <fcntl.h> // open

#include <klib/kseq.h>

namespace care{

//###### BEGIN READER IMPLEMENTATION

    bool SequenceFileReader::getNextRead(Read* read){
        if(read == nullptr) return false;

        bool b = getNextRead_impl(read);
        if(b)
            readnum++;
        return b;
    }

    bool SequenceFileReader::getNextReadUnsafe(Read* read){
        bool b = getNextReadUnsafe_impl(read);
        if(b)
            readnum++;
        return b;
    }
#if 0
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
        while(std::getline(is, read->header).good() && read->header == ""){
            ;
        }
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

    bool FastqReader::getNextReadUnsafe_impl(Read* read){
		if (!is.good())
			return false;

		read->reset();
        while(std::getline(is, read->header).good() && read->header == ""){
            ;
        }
		if (!is.good())
			return false;

		std::getline(is, read->sequence);
		if (!is.good())
			return false;
		std::getline(is, stmp);
		if (!is.good())
			return false;

		std::getline(is, read->quality);
		if (!is.good())
			return false;

		return !(is.fail() || is.bad());
	}

	void FastqReader::skipBytes_impl(std::uint64_t nBytes){
		std::uint64_t currentPos = is.tellg();
		std::uint64_t newPos = currentPos + nBytes;

		std::experimental::filesystem::path path = this->filename;
		std::uint64_t size = std::experimental::filesystem::file_size(path);
		assert(size >= newPos);

		is.seekg(nBytes, std::ios::cur);

		/*
		 * search for the next read header. then skip one read
		 */
		//find nonempty line
		while(std::getline(is, stmp).good() && stmp == ""){
            ;
        }
        if (!is.good())
			throw SkipException();

		bool found = false;
		while(!found){
			bool foundPotentialHeader = false;
			//search line which starts with @ (may be quality scores, too)
			while(!foundPotentialHeader){
				std::getline(is, stmp);
				if (!is.good())
					throw SkipException();
				if(stmp[0] == '@')
					foundPotentialHeader = true;
			}
			std::getline(is, stmp);
			if (!is.good())
					throw SkipException();
			if(stmp[0] == '@'){
				//two consecutive lines starting with @. second line must be the header. skip sequence, check for +, skip quality
				std::getline(is, stmp);
				if (!is.good())
					throw SkipException();
				std::getline(is, stmp);
				if (!is.good())
					throw SkipException();
				if (stmp[0] == '+') {
					//found @ in first line and + in third line. skip quality scores and exit
					std::getline(is, stmp);
					if (!is.good())
						throw SkipException();
					found = true;
				}
			}else{
				std::getline(is, stmp);
				if (!is.good())
					throw SkipException();
				if (stmp[0] == '+') {
					//found @ in first line and + in third line. skip quality scores and exit
					std::getline(is, stmp);
					if (!is.good())
						throw SkipException();
					found = true;
				}
			}

		}
	}

	void FastqReader::skipReads_impl(std::uint64_t nReads){
		for(std::uint64_t counter = 0; counter < nReads; counter++){
			for(int i = 0; i < 4; i++){
				is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				if (!is.good())
					throw SkipException(); //std::runtime_error("Skipped too far in file");
			}
			++readnum;
		}
	}
#endif
    template<class KSEQ>
    void setReadFromKseqPtr(Read* read, const KSEQ* kseq){
        read->reset();

        read->name = kseq->name.s;
        if(kseq->comment.l > 0){
            read->comment = kseq->comment.s;
        }
        read->sequence = kseq->seq.s;
        if(kseq->qual.l > 0){
            read->quality = kseq->qual.s;
        }
    }

    namespace ksequncompressed{
        KSEQ_INIT(int, read)
    }

    namespace kseqgz{
        KSEQ_INIT(gzFile, gzread)
    }

    KseqReader::KseqReader(const std::string& filename_) : SequenceFileReader(filename_){
        fp = open(filename_.c_str(), O_RDONLY);
        if(fp == -1){
            throw std::runtime_error("could not open file " + filename_);
        }
        seq = (void*)ksequncompressed::kseq_init(fp);
    }

    KseqReader::~KseqReader(){
        kseq_destroy((ksequncompressed::kseq_t*)seq);
        close(fp);
    }

    bool KseqReader::getNextRead_impl(Read* read){
        ksequncompressed::kseq_t* typedseq = (ksequncompressed::kseq_t*)seq;

        int len = ksequncompressed::kseq_read(typedseq);
        if(len < 0)
            return false;

        setReadFromKseqPtr(read, typedseq);

        return true;
    }

    bool KseqReader::getNextReadUnsafe_impl(Read* read){
        throw std::runtime_error("KseqReader::getNextReadUnsafe_impl not implemented");
        return false;
    }

    void KseqReader::skipBytes_impl(std::uint64_t nBytes){
        throw std::runtime_error("KseqReader::skipBytes_impl not implemented");
    }

    void KseqReader::skipReads_impl(std::uint64_t nReads){
        throw std::runtime_error("KseqReader::skipReads_impl not implemented");
    }

    KseqGzReader::KseqGzReader(const std::string& filename_) : SequenceFileReader(filename_){
        fp = gzopen(filename_.c_str(), "r");
        if(fp == NULL){
            throw std::runtime_error("could not open file " + filename_);
        }
        seq = (void*)kseqgz::kseq_init(fp);
    }

    KseqGzReader::~KseqGzReader(){
        kseqgz::kseq_destroy((kseqgz::kseq_t*)seq);
        gzclose(fp);
    }

    bool KseqGzReader::getNextRead_impl(Read* read){
        kseqgz::kseq_t* typedseq = (kseqgz::kseq_t*)seq;

        int len = kseqgz::kseq_read(typedseq);
        if(len < 0)
            return false;

        setReadFromKseqPtr(read, typedseq);

        return true;
    }

    bool KseqGzReader::getNextReadUnsafe_impl(Read* read){
        throw std::runtime_error("KseqGzReader::getNextReadUnsafe_impl not implemented");
        return false;
    }

    void KseqGzReader::skipBytes_impl(std::uint64_t nBytes){
        throw std::runtime_error("KseqGzReader::skipBytes_impl not implemented");
    }

    void KseqGzReader::skipReads_impl(std::uint64_t nReads){
        throw std::runtime_error("KseqGzReader::skipReads_impl not implemented");
    }



//###### END READER IMPLEMENTATION

//###### BEGIN WRITER IMPLEMENTATION

void SequenceFileWriter::writeRead(const std::string& name, const std::string& comment, const std::string& sequence, const std::string& quality){
    //std::cerr << "Write " << header << "\n" << sequence << " " << "\n" << quality << "\n";
    writeReadImpl(name, comment, sequence, quality);


}

void SequenceFileWriter::writeRead(const Read& read){
    //std::cerr << "Write " << header << "\n" << sequence << " " << "\n" << quality << "\n";
    writeRead(read.name, read.comment, read.sequence, read. quality);
}

UncompressedWriter::UncompressedWriter(const std::string& filename, FileFormat format)
        : SequenceFileWriter(filename, format){

    assert(format == FileFormat::FASTA || format == FileFormat::FASTQ);

    ofs = std::ofstream(filename);
    if(!ofs){
        throw std::runtime_error("Cannot open file " + filename + " for writing.");
    }

    isFastq = format == FileFormat::FASTQ || format == FileFormat::FASTQGZ;
    delimHeader = '>';
    if(isFastq){
        delimHeader = '@';
    }

}

void UncompressedWriter::writeReadImpl(const std::string& name, const std::string& comment, const std::string& sequence, const std::string& quality){
    ofs << delimHeader << name << ' ' << comment << '\n'
        << sequence << '\n';
    if(format == FileFormat::FASTQ){
        ofs << '+' << '\n'
            << quality << '\n';
    }
}

void UncompressedWriter::writeImpl(const std::string& data){
    ofs << data;
}

GZipWriter::GZipWriter(const std::string& filename, FileFormat format)
        : SequenceFileWriter(filename, format){

    assert(format == FileFormat::FASTAGZ || format == FileFormat::FASTQGZ);

    fp = gzopen(filename.c_str(), "w");
    if(fp == NULL){
        throw std::runtime_error("Cannot open file " + filename + " for writing.");
    }

    isFastq = format == FileFormat::FASTQ || format == FileFormat::FASTQGZ;
    delimHeader = '>';
    if(isFastq){
        delimHeader = '@';
    }
}

GZipWriter::~GZipWriter(){
    if(numBufferedReads > 0){
        writeBufferedReads();
    }
    numBufferedReads = 0;
    gzclose(fp);
}

void GZipWriter::writeReadImpl(const std::string& name, const std::string& comment, const std::string& sequence, const std::string& quality){
    bufferRead(name, comment, sequence, quality);

    if(numBufferedReads > maxBufferedReads){
        writeBufferedReads();
    }

// #if 1
//     std::stringstream ss;
//     ss << delimHeader << name << ' ' << comment << '\n'
//         << sequence << '\n';
//     if(format == FileFormat::FASTQGZ){
//         ss << '+' << '\n'
//             << quality << '\n';
//     }
//
//     auto string = ss.str();
//     gzwrite(fp, string.c_str(), string.size());
// #else
//     gzputc(fp, delimHeader);
//     gzwrite(fp, name.c_str(), name.size());
//     gzputc(fp, ' ');
//     gzwrite(fp, comment.c_str(), comment.size());
//     gzputc(fp, '\n');
//     gzwrite(fp, sequence.c_str(), sequence.size());
//     gzputc(fp, '\n');
//     if(format == FileFormat::FASTQGZ){
//         gzwrite(fp, "+\n", 2);
//         gzwrite(fp, quality.c_str(), quality.size());
//         gzputc(fp, '\n');
//     }
// #endif
}

void GZipWriter::writeImpl(const std::string& data){
    gzwrite(fp, data.c_str(), data.size());
}


//###### END WRITER IMPLEMENTATION


    bool hasGzipHeader(const std::string& filename){
        std::ifstream is(filename, std::ios_base::binary);
        unsigned char buf[2];
        is.read(reinterpret_cast<char*>(&buf[0]), 2);

        if(buf[0] == 0x1f && buf[1] == 0x8b){
            return true;
        }else{
            return false;
        }
    }

    bool hasQualityScores(const std::unique_ptr<SequenceFileReader>& reader){
        Read read;
        int i = 0;
        int n = 5;
        int count = 0;
        while (reader->getNextRead(&read) && i < n){
            if(read.quality.size() > 0){
                count++;
            }
            i++;
        }
        if(count > 0 && count == i){
            return true;
        }else if(count == 0){
            return false;
        }else{
            throw std::runtime_error("Error. Some reads do not have quality scores");
        }
    }

    FileFormat getFileFormat(const std::string& filename){
        if(hasGzipHeader(filename)){
            std::unique_ptr<SequenceFileReader> reader = std::make_unique<KseqGzReader>(filename);
            if(hasQualityScores(reader)){
                return FileFormat::FASTQGZ;
            }else{
                return FileFormat::FASTAGZ;
            }
        }else{
            std::unique_ptr<SequenceFileReader> reader = std::make_unique<KseqReader>(filename);
            if(hasQualityScores(reader)){
                return FileFormat::FASTQ;
            }else{
                return FileFormat::FASTA;
            }
        }
    }

    std::unique_ptr<SequenceFileReader> makeSequenceReader(const std::string& filename, FileFormat fileFormat){
        switch (fileFormat) {
        case FileFormat::FASTA:
        case FileFormat::FASTQ:
            //reader.reset(new FastqReader(filename));
            return std::make_unique<KseqReader>(filename);
        case FileFormat::FASTAGZ:
        case FileFormat::FASTQGZ:
            return std::make_unique<KseqGzReader>(filename);
    	default:
    		throw std::runtime_error("makeSequenceReader: invalid format.");
    	}
    }

    std::unique_ptr<SequenceFileWriter> makeSequenceWriter(const std::string& filename, FileFormat fileFormat){
        switch (fileFormat) {
        case FileFormat::FASTA:
        case FileFormat::FASTQ:
            return std::make_unique<UncompressedWriter>(filename, fileFormat);
        case FileFormat::FASTAGZ:
        case FileFormat::FASTQGZ:
            return std::make_unique<GZipWriter>(filename, fileFormat);
    	default:
    		throw std::runtime_error("makeSequenceWriter: invalid format.");
    	}
    }

    SequenceFileProperties getSequenceFileProperties(const std::string& filename, FileFormat format){
#if 1
        std::unique_ptr<SequenceFileReader> reader = makeSequenceReader(filename, format);
        /*switch (format) {
        case FileFormat::FASTQ:
            //reader.reset(new FastqReader(filename));
            reader.reset(new KseqReader(filename));
            break;
        case FileFormat::GZIP:
            reader.reset(new KseqGzReader(filename));
            break;
    	default:
    		throw std::runtime_error("care::getNumberOfReads: invalid format.");
    	}*/

        SequenceFileProperties prop;

        prop.maxSequenceLength = 0;
        prop.minSequenceLength = std::numeric_limits<int>::max();

		std::chrono::time_point<std::chrono::system_clock> tpa, tpb;

		std::chrono::duration<double> duration;

        Read r;

		std::uint64_t countlimit = 1000000;
		std::uint64_t count = 0;
		std::uint64_t totalCount = 0;
		tpa = std::chrono::system_clock::now();

        while(reader->getNextRead(&r)){
            int len = int(r.sequence.length());
            if(len > prop.maxSequenceLength)
                prop.maxSequenceLength = len;
            if(len < prop.minSequenceLength)
                prop.minSequenceLength = len;

			++count;
			++totalCount;

			if(count == countlimit){
				tpb = std::chrono::system_clock::now();
				duration = tpb - tpa;
				std::cout << totalCount << " : " << duration.count() << " seconds." << std::endl;
				countlimit *= 2;
			}
        }

        if(count > 0){
            tpb = std::chrono::system_clock::now();
		    duration = tpb - tpa;
		    std::cout << totalCount << " : " << duration.count() << " seconds." << std::endl;
        }

        prop.nReads = reader->getReadnum();
#else

		/*TIMERSTARTCPU(asdf);
		std::ifstream myis(filename);
		std::uint64_t lines = 0;
		std::string tmp;
		while(myis.good()){
			myis.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			//std::getline(myis, tmp);
			++lines;
		}
		std::cout << "lines : " << lines << std::endl;
		std::cout << "reads : " << lines/4 << std::endl;
		TIMERSTOPCPU(asdf);

		SequenceFileProperties prop;

        prop.maxSequenceLength = -1;
        prop.minSequenceLength = -1;
		prop.nReads = lines/4;*/

		int nThreads = 4;
		std::vector<std::unique_ptr<FastqReader>> readers;
		for(int i = 0; i < nThreads; ++i){
			switch (format) {
			case FileFormat::FASTQ:
				readers.emplace_back(new FastqReader(filename));
				break;
			default:
				throw std::runtime_error("care::getSequenceFileProperties: invalid format.");
			}
		}

		std::experimental::filesystem::path path = filename;
		std::int64_t size = std::experimental::filesystem::file_size(path);

		using Result_t = std::tuple<int, int,std::uint64_t>;
		std::vector<std::future<Result_t>> futures;
		std::vector<std::int64_t> endings(nThreads);
		endings[nThreads-1] = size;

		std::chrono::time_point<std::chrono::system_clock> tpa, tpb;

		tpa = std::chrono::system_clock::now();

		std::int64_t sizePerThread = size / nThreads;
		for(int i = 1; i < nThreads; ++i){
			for(int j = i; j < nThreads; ++j){
				readers[j]->skipBytes(sizePerThread);
			}
			endings[i-1] = readers[i]->is.tellg();

			futures.emplace_back(std::async(std::launch::async, [&,i=i-1]{
				auto& reader = readers[i];
                int maxSequenceLength = 0;
				int minSequenceLength = std::numeric_limits<int>::max();

				Read r;

				while(reader->getNextRead(&r) && reader->is.tellg() < endings[i]){
					int len = int(r.sequence.length());
					if(len > maxSequenceLength)
						maxSequenceLength = len;
					if(len < minSequenceLength)
						minSequenceLength = len;
				}

				std::uint64_t nReads = reader->getReadnum();

				return Result_t(minSequenceLength, maxSequenceLength, nReads);
			}));
		}


		futures.emplace_back(std::async(std::launch::async, [&,i=nThreads-1]{
			auto& reader = readers[i];
			int maxSequenceLength = 0;
			int minSequenceLength = std::numeric_limits<int>::max();

			Read r;

			while(reader->getNextRead(&r) && reader->is.tellg() < endings[i]){
				int len = int(r.sequence.length());
				if(len > maxSequenceLength)
					maxSequenceLength = len;
				if(len < minSequenceLength)
					minSequenceLength = len;
			}

			std::uint64_t nReads = reader->getReadnum();

			return Result_t(minSequenceLength, maxSequenceLength, nReads);
		}));


		SequenceFileProperties prop;

        prop.maxSequenceLength = 0;
        prop.minSequenceLength = std::numeric_limits<int>::max();
		prop.nReads = 0;

		for(int i = 0; i < nThreads; ++i){
			futures[i].wait();
			Result_t result = futures[i].get();
			auto minSequenceLength = std::get<0>(result);
			auto maxSequenceLength = std::get<1>(result);
			auto nReads = std::get<2>(result);

			if(minSequenceLength < prop.minSequenceLength)
				prop.minSequenceLength = minSequenceLength;

			if(maxSequenceLength > prop.maxSequenceLength)
				prop.maxSequenceLength = maxSequenceLength;

			prop.nReads += nReads;
		}


		tpb = std::chrono::system_clock::now();
		std::chrono::duration<double> duration = tpb - tpa;
		std::cout << prop.nReads << " : " << duration.count() << " seconds." << std::endl;

#endif
        return prop;
    }

    std::uint64_t getNumberOfReadsFast(const std::string& filename, FileFormat format){

		if(format == FileFormat::FASTQ){
			std::ifstream myis(filename);
			std::uint64_t lines = 0;
			std::string tmp;
			while(myis.good()){
				myis.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				++lines;
			}
			//std::cout << "lines : " << lines << std::endl;
			//std::cout << "reads : " << lines/4 << std::endl;

			return lines / 4;
		}else{
			throw std::runtime_error("getNumberOfReads invalid format");
		}
	}

	std::uint64_t getNumberOfReads(const std::string& filename, FileFormat format){

		std::unique_ptr<SequenceFileReader> reader = makeSequenceReader(filename, format);

		//std::chrono::time_point<std::chrono::system_clock> tpa, tpb;

		//std::chrono::duration<double> duration;

        Read r;

		const std::uint64_t countlimit = 10000000;
		std::uint64_t count = 0;
		std::uint64_t totalCount = 0;
		//tpa = std::chrono::system_clock::now();

        while(reader->getNextRead(&r)){

			++count;
			++totalCount;

			if(count == countlimit){
				//tpb = std::chrono::system_clock::now();
				//duration = tpb - tpa;
				//std::cout << totalCount << " : " << duration.count() << " seconds." << std::endl;
				count = 0;
			}
        }

        //tpb = std::chrono::system_clock::now();
		//duration = tpb - tpa;
		//std::cout << totalCount << " : " << duration.count() << " seconds." << std::endl;

        return reader->getReadnum();
	}

    /*
        Deletes every file in vector filenames
    */
    void deleteFiles(std::vector<std::string> filenames){
        for (const auto& filename : filenames) {
            int ret = std::remove(filename.c_str());
            if (ret != 0){
                const std::string errormessage = "Could not remove file " + filename;
                std::perror(errormessage.c_str());
            }
        }
    }

    /*
        Merges temporary results with unordered reads into single file outputfile with ordered reads.
        Quality scores and missing sequences are taken from original file.
        Temporary result files are expected to be in format:

        readnumber
        sequence
        readnumber
        sequence
        ...
    */
#if 0
    void mergeResultFiles(std::uint32_t expectedNumReads, const std::string& originalReadFile,
                          FileFormat originalFormat,
                          const std::vector<std::string>& filesToMerge, const std::string& outputfile){
        std::vector<Read> reads(expectedNumReads);
        std::uint32_t nreads = 0;

        for(const auto& filename : filesToMerge){
            std::ifstream is(filename);
            if(!is)
                throw std::runtime_error("could not open tmp file: " + filename);

            std::string num;
            std::string seq;

            while(true){
                std::getline(is, num);
        		if (!is.good())
        			break;
                std::getline(is, seq);
                if (!is.good())
                    break;

                nreads++;

                auto readnum = std::stoull(num);
                reads[readnum].sequence = std::move(seq);
            }
        }

        std::unique_ptr<SequenceFileReader> reader;
    	switch (originalFormat) {
    	case FileFormat::FASTQ:
    		reader.reset(new FastqReader(originalReadFile));
    		break;
    	default:
    		throw std::runtime_error("Merging: Invalid file format.");
    	}

        Read read;

    	while (reader->getNextRead(&read)) {
            std::uint64_t readIndex = reader->getReadnum() - 1;
            reads[readIndex].header = std::move(read.header);
            reads[readIndex].quality = std::move(read.quality);
            if(reads[readIndex].sequence == ""){
                reads[readIndex].sequence = std::move(read.sequence);
            }
    	}

    	std::ofstream outputstream(outputfile);

    	for (const auto& read : reads) {
    		outputstream << read.header << '\n' << read.sequence << '\n';

    		if (originalFormat == FileFormat::FASTQ)
    			outputstream << '+' << '\n' << read.quality << '\n';
    	}

    	outputstream.flush();
    	outputstream.close();
    }

#else

std::uint64_t linecount(const std::string& filename){
	std::uint64_t count = 0;
	std::ifstream is(filename);
	if(is){
		std::string s;
		while(std::getline(is, s)){
			++count;
		}
	}
	return count;
}

void sortResultFileUsingDisk(const std::string& filename, std::uint32_t chunksize){
    const std::uint64_t resultsInFile = linecount(filename) / 2;

    for(std::uint64_t i = 0; i < SDIV(resultsInFile, chunksize); ++i){
        const std::uint64_t readNumBegin = i * chunksize;
        const std::uint64_t readNumEnd = std::min((i+1) * chunksize, resultsInFile);

        std::ofstream os("resultsortedtmp" + std::to_string(i));

        std::vector<std::pair<std::uint64_t, std::string>> tmpvec(readNumEnd - readNumBegin);

        std::ifstream is(filename);
        if(!is)
            throw std::runtime_error("could not open tmp file: " + filename);

        std::string num;
        std::string seq;

        while(true){
            std::getline(is, num);
            if (!is.good())
                break;
            std::getline(is, seq);
            if (!is.good())
                break;

            auto readnum = std::stoull(num);
            if(readNumBegin <= readnum && readnum < readNumEnd){
                tmpvec[readnum - readNumBegin] = {readnum, std::move(seq)};
                //std::cout << i << " " << readnum << std::endl;
            }
        }

        std::sort(tmpvec.begin(), tmpvec.end(), [](const auto& l, const auto& r){return l.first < r.first;});

        for(const auto& p : tmpvec){
            os << p.first << '\n';
            os << p.second << '\n';
        }
    }

    // now merge all the sorted file chunks into one file

}













struct SequenceWriterThread{

    std::unique_ptr<SequenceFileWriter> writer;

    ThreadsafeBuffer<Read, 1000> queue;
    std::thread t;

    SequenceWriterThread(const std::string& filename, FileFormat format){
        writer = std::move(makeSequenceWriter(filename, format));

        t = std::move(std::thread([&](){

            auto popresult = queue.getNew();
            while(!popresult.foreverEmpty){
                writer->writeRead(popresult.value);
                popresult = std::move(queue.getNew());
            }
        }));
    }

    ~SequenceWriterThread(){
        t.join();
    }


    void push(Read&& data){
        queue.add(std::move(data));
    }

    void push(Read data){
        queue.add(std::move(data));
    }

    void producerDone(){
        queue.done();
    }

};


void mergeResultFiles(std::uint32_t expectedNumReads, const std::string& originalReadFile,
                      FileFormat originalFormat,
                      const std::vector<std::string>& filesToMerge, const std::string& outputfile,
                        bool isSorted){


    bool oldsyncflag = true;//std::ios::sync_with_stdio(false);

    std::string tempfile = outputfile + "mergetempfile";
    std::stringstream commandbuilder;

    //sort the result files and save sorted result file in tempfile.
    //Then, merge original file and tempfile, replacing the reads in
    //original file by the corresponding reads in the tempfile.



    if(isSorted && filesToMerge.size() == 1){
        tempfile = filesToMerge[0];
    }else{
        if(isSorted){
            std::stringstream commandbuilder;
            commandbuilder << "sort --parallel=4 -k1,1 -n -m ";
            for(const auto& filename : filesToMerge){
                commandbuilder << "\"" << filename << "\" ";
            }
            commandbuilder << " > " << tempfile;

            std::string command = commandbuilder.str();
            std::cerr << "Running shell command: " << command << "\n";
            TIMERSTARTCPU(sort_during_merge);
            int r1 = std::system(command.c_str());

            TIMERSTOPCPU(sort_during_merge);
            if(r1 != 0){
                throw std::runtime_error("Merge of result files failed! sort returned " + std::to_string(r1));
            }
        }else{
            std::stringstream commandbuilder;
            commandbuilder << "sort --parallel=4 -k1,1 -n ";
            for(const auto& filename : filesToMerge){
                commandbuilder << "\"" << filename << "\" ";
            }
            commandbuilder << " > " << tempfile;

            std::string command = commandbuilder.str();
            std::cerr << "Running shell command: " << command << "\n";
            TIMERSTARTCPU(sort_during_merge);
            int r1 = std::system(command.c_str());

            TIMERSTOPCPU(sort_during_merge);
            if(r1 != 0){
                throw std::runtime_error("Merge of result files failed! sort returned " + std::to_string(r1));
            }
        }
    }

    std::unique_ptr<SequenceFileReader> reader = makeSequenceReader(originalReadFile, originalFormat);
    //std::unique_ptr<SequenceFileReader> reader = std::make_unique<FastqReader>(originalReadFile);

    //only output uncompressed for now
    FileFormat outputformat = originalFormat;
    if(outputformat == FileFormat::FASTQGZ)
        outputformat = FileFormat::FASTQ;
    if(outputformat == FileFormat::FASTAGZ)
        outputformat = FileFormat::FASTA;

    std::unique_ptr<SequenceFileWriter> writer = makeSequenceWriter(outputfile, outputformat);
    //SequenceWriterThread swt(outputfile, outputformat);

    std::ifstream correctionsstream(tempfile);

    std::string correctionline;
    //loop over correction sequences
    TIMERSTARTCPU(actualmerging);

    auto isValidSequence = [](const std::string& s){
        return std::all_of(s.begin(), s.end(), [](char c){
            return (c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N');
        });
    };

    auto combineMultipleCorrectionResults2 = [](std::vector<TempCorrectedSequence>& tmpresults, const std::string& originalSequence){
        assert(!tmpresults.empty());

        constexpr bool outputHQ = true;
        constexpr bool outputLQAnchorDifferentCand = true;
        constexpr bool outputLQAnchorSameCand = true;
        constexpr bool outputLQAnchorNoCand = true;
        constexpr bool outputLQOnlyCand = true;

        auto isHQ = [](const auto& tcs){
            return tcs.type == TempCorrectedSequence::Type::Anchor && tcs.hq;
        };

        //if there is a correction using a high quality alignment, use it
        auto firstHqSequence = std::find_if(tmpresults.begin(), tmpresults.end(), isHQ);
        if(firstHqSequence != tmpresults.end()){
            if(firstHqSequence->isEqual){
                if(firstHqSequence->sequence != originalSequence){
                    std::cerr << firstHqSequence->sequence << "\n" << originalSequence << "\n";
                }
                assert(firstHqSequence->sequence == originalSequence);
                return std::make_pair(std::string{""}, false);
            }else{
                return std::make_pair(firstHqSequence->sequence, outputHQ);
            }
        }

        auto equalsFirstSequence = [&](const auto& result){
            return result.sequence == tmpresults[0].sequence;
        };

        auto getSequence = [&](int index){
            return tmpresults[index].sequence;
        };

        auto anchorIter = std::find_if(tmpresults.begin(), tmpresults.end(), [](const auto& r){
            return r.type == TempCorrectedSequence::Type::Anchor;
        });

        if(!std::all_of(tmpresults.begin()+1, tmpresults.end(), equalsFirstSequence)){
            // std::copy(sequences.begin(), sequences.end(), std::ostream_iterator<std::string>(std::cerr, "\n"));
            // std::cerr << "\n";
            // std::exit(0);
            std::string consensus(getSequence(0).size(), 'F');
            std::vector<int> countsA(getSequence(0).size(), 0);
            std::vector<int> countsC(getSequence(0).size(), 0);
            std::vector<int> countsG(getSequence(0).size(), 0);
            std::vector<int> countsT(getSequence(0).size(), 0);

            auto countBases = [&](const auto& result){
                const auto& sequence = result.sequence;
                assert(sequence.size() == consensus.size());
                for(size_t i = 0; i < sequence.size();  i++){
                    const char c = sequence[i];
                    if(c == 'A') countsA[i]++;
                    else if(c == 'C') countsC[i]++;
                    else if(c == 'G') countsG[i]++;
                    else if(c == 'T') countsT[i]++;
                    else if(c == 'N'){
                        ;
                    }else{
                        std::cerr << result.readId << " : " << sequence << "\n"; assert(false);
                    }
                }
            };

            auto findConsensusOfPosition = [&](int i){
                int count = countsA[i];
                char c = 'A';
                if(countsC[i] > count){
                    count = countsC[i];
                    c = 'C';
                }
                if(countsG[i] > count){
                    count = countsG[i];
                    c = 'G';
                }
                if(countsT[i] > count){
                    count = countsT[i];
                    c = 'T';
                }
                return c;
            };

            auto setConsensusOfPosition = [&](int position){
                consensus[position] = findConsensusOfPosition(position);
            };



            if(anchorIter != tmpresults.end() && (anchorIter->uncorrectedPositionsNoConsensus.size() >= 5)){

                const int maxShiftInResult = std::max_element(tmpresults.begin(),
                                                            tmpresults.end(),
                                                            [](const auto& l, const auto& r){
                                                                return l.shift < r.shift;
                                                            })->shift;
                // if(maxShiftInResult > 3){
                //     return std::make_pair(std::string{""}, false);
                // }
                // //return std::make_pair(anchorIter->sequence, true);
                // for(const auto& t : tmpresults){
                //     if(t.shift <= 3){
                //         //const int iters = maxShiftInResult - t.shift + 1;
                //         //for(int k = 0; k < iters; k++){
                //             countBases(t);
                //         //}
                //     }
                // }
                std::for_each(tmpresults.begin(), tmpresults.end(), countBases);

                // if(!anchorIter->uncorrectedPositionsNoConsensus.empty()){
                //     std::copy(anchorIter->sequence.begin(), anchorIter->sequence.end(), consensus.begin());
                //     const auto& positions = anchorIter->uncorrectedPositionsNoConsensus;
                //
                //     std::for_each(positions.begin(), positions.end(), setConsensusOfPosition);
                //     // std::copy(positions.begin(), positions.end(), std::ostream_iterator<int>(std::cerr, " "));
                //     // std::cerr << '\n';
                //     // std::copy(consensus.begin(), consensus.end(), std::ostream_iterator<char>(std::cerr, ""));
                //     // std::cerr << '\n';
                //     return std::make_pair(consensus, false);
                // }else{
                //     for(size_t i = 0; i < consensus.size();  i++){
                //         setConsensusOfPosition(i);
                //     }
                //     return std::make_pair(consensus, false);
                // }

                for(size_t i = 0; i < consensus.size();  i++){
                    setConsensusOfPosition(i);
                }
                return std::make_pair(consensus, outputLQAnchorDifferentCand);

            }else{
                //only candidates available

                return std::make_pair(std::string{""}, false); //always false
            }

        }else{
            //return std::make_pair(tmpresults[0].sequence, false);

            if(anchorIter != tmpresults.end()){
                //return std::make_pair(anchorIter->sequence, true);
                auto checkshift = [](const auto& r){
                    return r.type == TempCorrectedSequence::Type::Candidate && std::abs(r.shift) <= 15;
                    //return true;
                };
                if(0 < std::count_if(tmpresults.begin(), tmpresults.end(), checkshift)){
                    return std::make_pair(tmpresults[0].sequence, outputLQAnchorSameCand);
                }else{
                    if(tmpresults.size() == 1){
                        if(tmpresults[0].uncorrectedPositionsNoConsensus.size() < 1){
                            return std::make_pair(tmpresults[0].sequence, outputLQAnchorNoCand);
                        }else{
                            return std::make_pair(std::string{""}, false); //always false
                        }

                    }else{
                        return std::make_pair(std::string{""}, false); //always false
                    }
                }
            }else{
                //no correction as anchor. all corrections as candidate are equal.
                //only use the correction if at least one correction as candidate was performed with 0 new columns
                auto checkshift = [](const auto& r){
                    return std::abs(r.shift) <= 1;
                };
                if(0 < std::count_if(tmpresults.begin(), tmpresults.end(), checkshift)){
                    return std::make_pair(tmpresults[0].sequence, outputLQOnlyCand);
                }else{
                    return std::make_pair(std::string{""}, false); //always false
                }

                //return std::make_pair(std::string{""}, false);

                //return std::make_pair(tmpresults[0].sequence, true);
            }
        }

        //return tmpresults[0].sequence;
    };


    auto combineMultipleCorrectionResults3 = [](const std::vector<TempCorrectedSequence>& tmpresults, const std::string& originalSequence){
        assert(!tmpresults.empty());

        auto isHQ = [](const auto& tcs){
            return tcs.type == TempCorrectedSequence::Type::Anchor && tcs.hq;
        };

        static int hqcount = 0;

        //if there is a correction using a high quality alignment, use it
        auto firstHqSequence = std::find_if(tmpresults.begin(), tmpresults.end(), isHQ);
        if(firstHqSequence != tmpresults.end()){
            // if(firstHqSequence->isEqual){
            //     if(firstHqSequence->sequence != originalSequence){
            //         std::cerr << "orig " << originalSequence << "\n";
            //         std::cerr << "corr " << firstHqSequence->sequence << "\n";
            //     }
            // }
            // if(firstHqSequence->sequence == "-"){
            //     //if sequence is "", the high quality anchor correction equals the original read
            //     return std::make_pair(std::string{""}, false);
            // }else{
                hqcount++;
                //std::cerr << "hq " << hqcount << "\n";
                return std::make_pair(firstHqSequence->sequence, true);
            //}
        }

        auto anchorIter = std::find_if(tmpresults.begin(), tmpresults.end(), [](const auto& r){
            return r.type == TempCorrectedSequence::Type::Anchor;
        });

        auto getvalidrangeByShift = [](int length, int shift){
            const int begin = std::max(-shift, 0);
            const int end = length - std::max(shift, 0);
            return std::make_pair(begin, end);
            //return std::make_pair(0, length);
        };

        auto getvalidrangeTotal = [](int length, int shift){
            return std::make_pair(0, length);
        };

        //check if all corrections are equal.

        auto checkEquality = [&](auto getrange){
            bool equal = true;

            if(anchorIter != tmpresults.end()){
                //anchor exists. for candidates, only check positions with overlap to anchor in their respective alignment

                const auto& anchorseq = anchorIter->sequence;
                for(const auto& cand : tmpresults){
                    if(cand.type == TempCorrectedSequence::Type::Candidate){
                        auto range = getrange(cand.sequence.length(), cand.shift);
                        for(int i = range.first; i < range.second && equal; i++){
                            if(anchorseq[i] != cand.sequence[i]){
                                equal = false;
                            }
                        }
                    }
                    if(!equal){
                        break;
                    }
                }
            }else{
                //anchor does not exist. only check positions which are covered by all candidates

                int globalCheckBegin = 0;
                int globalCheckEnd = std::numeric_limits<int>::max();

                for(const auto& cand : tmpresults){
                    if(cand.type == TempCorrectedSequence::Type::Candidate){
                        auto range = getrange(cand.sequence.length(), cand.shift);
                        globalCheckBegin = std::max(range.first, globalCheckBegin);
                        globalCheckEnd = std::min(range.second, globalCheckEnd);
                    }
                }

                for(auto it = tmpresults.begin()+1; it != tmpresults.end() && equal; ++it){
                    for(int i = globalCheckBegin; i < globalCheckEnd && equal; i++){
                        if(tmpresults[0].sequence[i] != it->sequence[i]){
                            equal = false;
                        }
                    }
                }
            }

            return equal;
        };

        bool correctionsAreEqualByShift = checkEquality(getvalidrangeByShift);
        bool correctionsAreEqual = checkEquality(getvalidrangeTotal);

        static int equalcount = 0;
        static int differencecount = 0;



        if(correctionsAreEqualByShift != correctionsAreEqual){
            differencecount++;
            //std::cerr << equalcount << " " << differencecount << "\n";
        }else{
            equalcount++;
            //std::cerr << equalcount << " " << differencecount << "\n";
        }


        if(!correctionsAreEqual){
            // std::copy(sequences.begin(), sequences.end(), std::ostream_iterator<std::string>(std::cerr, "\n"));
            // std::cerr << "\n";
            // std::exit(0);

            //positions differ. if anchor exists, make consensus of anchor correction and candidate correction. else discard correction
            if(anchorIter == tmpresults.end()){
                //only candidates available. discard
                return std::make_pair(std::string{""}, false);
            }else{


                const int length = tmpresults[0].sequence.length();
                std::string consensus(length, 'F');
                std::vector<int> countsA(length, 0);
                std::vector<int> countsC(length, 0);
                std::vector<int> countsG(length, 0);
                std::vector<int> countsT(length, 0);

                for(const auto& res : tmpresults){
                    const auto& sequence = res.sequence;
                    assert(sequence.size() == consensus.size());
                    auto range = getvalidrangeByShift(sequence.length(), res.shift);

                    for(int i = range.first; i < range.second; i++){
                        const char c = sequence[i];
                        if(c == 'A') countsA[i]++;
                        else if(c == 'C') countsC[i]++;
                        else if(c == 'G') countsG[i]++;
                        else if(c == 'T') countsT[i]++;
                        else {
                            std::cerr << res.readId << " : " << sequence << "\n"; assert(false);
                        }
                    }
                }

                for(int i = 0; i < length; i++){
                    int count = countsA[i];
                    char c = 'A';
                    if(countsC[i] > count){
                        count = countsC[i];
                        c = 'C';
                    }
                    if(countsG[i] > count){
                        count = countsG[i];
                        c = 'G';
                    }
                    if(countsT[i] > count){
                        count = countsT[i];
                        c = 'T';
                    }
                    consensus[i] = c;
                }

                return std::make_pair(consensus, true);
            }
        }else{
            //return std::make_pair(tmpresults[0].sequence, false);

            if(anchorIter != tmpresults.end()){
                //return std::make_pair(anchorIter->sequence, true);
                auto checkshift = [](const auto& r){
                    return r.type == TempCorrectedSequence::Type::Candidate && std::abs(r.shift) <= 15;
                    //return true;
                };
                if(0 < std::count_if(tmpresults.begin(), tmpresults.end(), checkshift)){
                    return std::make_pair(tmpresults[0].sequence, true);
                }else{
                    if(tmpresults.size() == 1){
                        //return std::make_pair(std::string{""}, true);
                        return std::make_pair(tmpresults[0].sequence, true);
                    }else{
                        return std::make_pair(std::string{""}, false); //always false
                    }
                }
            }else{
                //no correction as anchor. all corrections as candidate are equal.
                //only use the correction if at least one correction as candidate was performed with 0 new columns
                auto checkshift = [](const auto& r){
                    return std::abs(r.shift) <= 2;
                };
                if(0 < std::count_if(tmpresults.begin(), tmpresults.end(), checkshift)){
                    return std::make_pair(tmpresults[0].sequence, true);
                }else{
                    return std::make_pair(std::string{""}, false); //always false
                }

                //return std::make_pair(std::string{""}, false);

                //return std::make_pair(tmpresults[0].sequence, true);
            }
        }

        //return tmpresults[0].sequence;
    };

    auto combineMultipleCorrectionResultsFunction = combineMultipleCorrectionResults2;


    std::uint64_t currentReadId = 0;
    std::vector<TempCorrectedSequence> correctionVector;
    correctionVector.reserve(256);
    //bool hqSubject = false;

    std::uint64_t currentReadId_tmp = 0;
    std::vector<TempCorrectedSequence> correctionVector_tmp;
    correctionVector_tmp.reserve(256);
    //bool hqSubject_tmp = false;

    bool firstiter = true;

    while(std::getline(correctionsstream, correctionline)){
        std::stringstream ss(correctionline);
        TempCorrectedSequence tcs;
        ss >> tcs;

        if(firstiter || tcs.readId == currentReadId){
            currentReadId = tcs.readId ;
            correctionVector.emplace_back(std::move(tcs));

            while(std::getline(correctionsstream, correctionline)){
                std::stringstream ss2(correctionline);
                TempCorrectedSequence tcs2;
                ss2 >> tcs2;

                if(tcs2.readId == currentReadId){
                    correctionVector.emplace_back(std::move(tcs2));
                }else{
                    currentReadId_tmp = tcs2.readId;
                    correctionVector_tmp.emplace_back(std::move(tcs2));
                    break;
                }
            }
        }else{
            currentReadId_tmp = tcs.readId;
            correctionVector_tmp.emplace_back(std::move(tcs));
        }

        std::uint64_t originalReadId = reader->getReadnum();
        Read read;
        //copy preceding reads from original file
        while(originalReadId < currentReadId){
            bool valid = reader->getNextRead(&read);

            assert(valid);

            assert(isValidSequence(read.sequence));

            writer->writeRead(read);
            //swt.push(read);

            originalReadId = reader->getReadnum();
        }
        //replace sequence of next read with corrected sequence
        bool valid = reader->getNextRead(&read);

        assert(valid);

        for(auto& tmpres : correctionVector){
            if(tmpres.isEqual){
                tmpres.sequence = read.sequence;
                // if(tmpres.sequence != read.sequence){
                //     std::cerr << currentReadId << "\n" << tmpres.sequence << "\n" << read.sequence << "\n";
                // }
                // assert(tmpres.sequence == read.sequence);
            }
        }

        auto correctedSequence = combineMultipleCorrectionResultsFunction(correctionVector, read.sequence);

        if(correctedSequence.second){
            assert(isValidSequence(correctedSequence.first));
            writer->writeRead(read.name, read.comment, correctedSequence.first, read.quality);
        }else{
            writer->writeRead(read.name, read.comment, read.sequence, read.quality);
        }

        correctionVector.clear();
        std::swap(correctionVector, correctionVector_tmp);
        std::swap(currentReadId, currentReadId_tmp);


        firstiter = false;
    }

    if(correctionVector.size() > 0){
        std::uint64_t originalReadId = reader->getReadnum();
        Read read;
        //copy preceding reads from original file
        while(originalReadId < currentReadId){
            bool valid = reader->getNextRead(&read);

            assert(valid);

            assert(isValidSequence(read.sequence));

            writer->writeRead(read);
            //swt.push(read);

            originalReadId = reader->getReadnum();
        }
        //replace sequence of next read with corrected sequence
        bool valid = reader->getNextRead(&read);

        assert(valid);

        for(auto& tmpres : correctionVector){
            if(tmpres.isEqual){
                tmpres.sequence = read.sequence;
                // if(tmpres.sequence != read.sequence){
                //     std::cerr << currentReadId << "\n" << tmpres.sequence << "\n" << read.sequence << "\n";
                // }
                // assert(tmpres.sequence == read.sequence);
            }
        }

        auto correctedSequence = combineMultipleCorrectionResultsFunction(correctionVector, read.sequence);

        if(correctedSequence.second){
            assert(isValidSequence(correctedSequence.first));
            writer->writeRead(read.name, read.comment, correctedSequence.first, read.quality);
        }else{
            writer->writeRead(read.name, read.comment, read.sequence, read.quality);
        }
    }

    //copy remaining reads from original file
    Read read;

    while(reader->getNextRead(&read)){
        assert(isValidSequence(read.sequence));

        writer->writeRead(read);
    }

    TIMERSTOPCPU(actualmerging);

    //deleteFiles({tempfile});

    std::ios::sync_with_stdio(oldsyncflag);
}


std::ostream& operator<<(std::ostream& os, const TempCorrectedSequence& tmp){
    // const int sequenceints = getEncodedNumInts2Bit(tmp.sequence.length());
    // std::vector<unsigned int> data(sequenceints);
    // encodeSequence2Bit(data.data(), tmp.sequence.c_str(), tmp.sequence.length());

    os << tmp.readId << ' ';

    os << tmp.sequence << ' ';
    // int length = tmp.sequence.length();
    // os.write((const char*)&length, sizeof(int));
    // os.write((const char*)data.data(), sizeof(unsigned int) * data.size());

    if(tmp.type == TempCorrectedSequence::Type::Anchor){
        os << TempCorrectedSequence::AnchorChar << ' ' << tmp.hq << ' ' << tmp.isEqual;
        const auto& vec = tmp.uncorrectedPositionsNoConsensus;
        os << ' ' << vec.size();
        if(!vec.empty()){
            os << ' ';
            std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(os, " "));
        }
    }else{
        os << TempCorrectedSequence::CandidateChar << ' ' << tmp.isEqual << ' ' << tmp.shift;
    }

    return os;
}

std::istream& operator>>(std::istream& is, TempCorrectedSequence& tmp){
    is >> tmp.readId;
    is >> tmp.sequence;

    // int length = 0;
    // is.read((char*)&length, sizeof(int));
    // const int sequenceints = getEncodedNumInts2Bit(length);
    // std::vector<unsigned int> data(sequenceints);
    // is.read((char*)data.data(), sizeof(unsigned int) * sequenceints);
    //
    // tmp.sequence = get2BitString(data.data(), length);

    char typechar;
    is >> typechar;
    if(typechar == TempCorrectedSequence::AnchorChar){
        tmp.type = TempCorrectedSequence::Type::Anchor;
        is >> tmp.hq >> tmp.isEqual;
        size_t vecsize;
        is >> vecsize;
        if(vecsize > 0){
            auto& vec = tmp.uncorrectedPositionsNoConsensus;
            vec.resize(vecsize);
            for(size_t i = 0; i < vecsize; i++){
                is >> vec[i];
            }
        }
    }else{
        tmp.type = TempCorrectedSequence::Type::Candidate;
        is >> tmp.isEqual;
        is >> tmp.shift;
        tmp.shift = std::abs(tmp.shift);
    }

    return is;
}


#endif
}
