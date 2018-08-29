#include "../inc/sequencefileio.hpp"
#include "../inc/hpc_helpers.cuh"

#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

namespace care{

    bool SequenceFileReader::getNextRead(Read* read){
        if(read == nullptr) return false;

        bool b = getNextRead_impl(read);
        if(b)
            readnum++;
        return b;
    }

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


    SequenceFileProperties getSequenceFileProperties(const std::string& filename, FileFormat format){
        std::unique_ptr<SequenceFileReader> reader;
        switch (format) {
        case FileFormat::FASTQ:
            reader.reset(new FastqReader(filename));
            break;
    	default:
    		throw std::runtime_error("care::getNumberOfReads: invalid format.");
    	}

        SequenceFileProperties prop;

        prop.maxSequenceLength = 0;
        prop.minSequenceLength = std::numeric_limits<int>::max();



        Read r;
        while(reader->getNextRead(&r)){
            int len = int(r.sequence.length());
            if(len > prop.maxSequenceLength)
                prop.maxSequenceLength = len;
            if(len < prop.minSequenceLength)
                prop.minSequenceLength = len;
        }

        prop.nReads = reader->getReadnum();

        return prop;
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


void mergeResultFiles(std::uint32_t expectedNumReads, const std::string& originalReadFile,
                      FileFormat originalFormat,
                      const std::vector<std::string>& filesToMerge, const std::string& outputfile){

    constexpr std::uint32_t chunksize = 5000000;

    std::vector<Read> reads;
    reads.reserve(chunksize);

    Read read;

    std::unique_ptr<SequenceFileReader> reader;
    switch (originalFormat) {
    case FileFormat::FASTQ:
        reader.reset(new FastqReader(originalReadFile));
        break;
    default:
        throw std::runtime_error("Merging: Invalid file format.");
    }

    std::ofstream outputstream(outputfile);

    for(std::uint32_t i = 0; i < SDIV(expectedNumReads, chunksize); ++i){
        const std::uint32_t readNumBegin = i * chunksize;
        const std::uint32_t readNumEnd = std::min((i+1) * chunksize, expectedNumReads);
        reads.resize(readNumEnd - readNumBegin);

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

                auto readnum = std::stoull(num);
                if(readNumBegin <= readnum && readnum < readNumEnd){
                    reads[readnum - readNumBegin].sequence = std::move(seq);
                    //std::cout << i << " " << readnum << std::endl;
                }
            }
        }

        while (reader->getNextRead(&read)) {
            std::uint64_t readnum = reader->getReadnum() - 1;
            if(readNumBegin <= readnum && readnum < readNumEnd){
                reads[readnum - readNumBegin].header = std::move(read.header);
                reads[readnum - readNumBegin].quality = std::move(read.quality);
                if(reads[readnum - readNumBegin].sequence == ""){
                    reads[readnum - readNumBegin].sequence = std::move(read.sequence);
                }
            }
            if(readnum == readNumEnd - 1)
                break;
        }

        for (const auto& read : reads) {
            outputstream << read.header << '\n' << read.sequence << '\n';

            if (originalFormat == FileFormat::FASTQ)
                outputstream << '+' << '\n' << read.quality << '\n';
        }

        reads.clear();
    }

    outputstream.flush();
    outputstream.close();
}


#endif
}
