#include "../inc/sequencefileio.hpp"
#include "../inc/hpc_helpers.cuh"

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


    SequenceFileProperties getSequenceFileProperties(const std::string& filename, FileFormat format){
#if 1
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

		std::chrono::time_point<std::chrono::system_clock> tpa, tpb;

		std::chrono::duration<double> duration;

        Read r;

		const std::uint64_t countlimit = 1000000;
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
				count = 0;
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

		std::unique_ptr<SequenceFileReader> reader;
        switch (format) {
        case FileFormat::FASTQ:
            reader.reset(new FastqReader(filename));
            break;
    	default:
    		throw std::runtime_error("care::getNumberOfReads: invalid format.");
    	}

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


void mergeResultFiles(std::uint32_t expectedNumReads, const std::string& originalReadFile,
                      FileFormat originalFormat,
                      const std::vector<std::string>& filesToMerge, const std::string& outputfile){
#if 0
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
#else

    std::string tempfile = outputfile + "mergetempfile";
    std::stringstream commandbuilder;

    //sort the result files and save sorted result file in tempfile.
    //Then, merge original file and tempfile, replacing the reads in
    //original file by the corresponding reads in the tempfile.

    commandbuilder << "sort --parallel=4 -k1,1 -n ";
    for(const auto& filename : filesToMerge){
        commandbuilder << "\"" << filename << "\" ";
    }
    commandbuilder << " > " << tempfile;

    std::string command = commandbuilder.str();
    TIMERSTARTCPU(sort_during_merge);
    int r1 = std::system(command.c_str());

    TIMERSTOPCPU(sort_during_merge);
    if(r1 != 0){
        throw std::runtime_error("Merge of result files failed! sort returned " + std::to_string(r1));
    }

    std::unique_ptr<SequenceFileReader> reader;
    switch (originalFormat) {
    case FileFormat::FASTQ:
        reader.reset(new FastqReader(originalReadFile));
        break;
    default:
        throw std::runtime_error("Merging: Invalid file format.");
    }

    std::ifstream correctionsstream(tempfile);
    std::ofstream outputstream(outputfile);

    std::string correctionline;
    //loop over correction sequences
    while(std::getline(correctionsstream, correctionline)){
        std::stringstream ss(correctionline);
        std::uint64_t correctionReadId;
        std::string correctedSequence;
        ss >> correctionReadId >> correctedSequence;

        std::uint64_t originalReadId = reader->getReadnum();
        Read read;
        //copy preceding reads from original file
        while(originalReadId < correctionReadId){
            bool valid = reader->getNextRead(&read);

            assert(valid);

            outputstream << read.header << '\n' << read.sequence << '\n';
            if (originalFormat == FileFormat::FASTQ)
                outputstream << '+' << '\n' << read.quality << '\n';

            originalReadId = reader->getReadnum();
        }
        //replace sequence of next read with corrected sequence
        bool valid = reader->getNextRead(&read);

        assert(valid);

        outputstream << read.header << '\n' << correctedSequence << '\n';
        if (originalFormat == FileFormat::FASTQ)
            outputstream << '+' << '\n' << read.quality << '\n';
    }

    //copy remaining reads from original file
    Read read;

    while(reader->getNextRead(&read)){
        outputstream << read.header << '\n' << read.sequence << '\n';
        if (originalFormat == FileFormat::FASTQ)
            outputstream << '+' << '\n' << read.quality << '\n';
    }

    outputstream.flush();
    outputstream.close();

    deleteFiles({tempfile});

#endif
}


#endif
}
