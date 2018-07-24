#ifndef CARE_BUILD_HPP
#define CARE_BUILD_HPP

#include "minhasher.hpp"
#include "readstorage.hpp"
#include "sequencefileio.hpp"
#include "sequence.hpp"
#include "threadsafe_buffer.hpp"

#include <stdexcept>
#include <iostream>
#include <limits>
#include <thread>

namespace care{

	namespace builddetail{
		template<class Minhasher_t,
				 class ReadStorage_t,
				 class Buffer_t>
		struct BuildThread{
			BuildThread(): progress(0), isRunning(false){}
			~BuildThread() = default;
			BuildThread(const BuildThread& other) = default;
			BuildThread(BuildThread&& other) = default;
			BuildThread& operator=(const BuildThread& other) = default;
			BuildThread& operator=(BuildThread&& other) = default;

			Buffer_t* buffer;
			ReadStorage_t* readStorage;
			Minhasher_t* minhasher;
			std::uint64_t totalNumberOfReads;

			std::uint64_t progress;
			bool isRunning;
			std::thread thread;

			void run(){
				if(isRunning) throw std::runtime_error("BuildThread::run: Is already running.");
				isRunning = true;
				thread = std::move(std::thread(&BuildThread::execute, this));
			}

			void join(){
				thread.join();
				isRunning = false;
			}

			void execute() {
				isRunning = true;
				progress = 0;

				auto pair = buffer->get();
				int Ncount = 0;
				char bases[4]{'A', 'C', 'G', 'T'};
				while (pair != buffer->defaultValue) {
					Read& read = pair.first;
					const auto readnum = pair.second;

					for(auto& c : read.sequence){
                        if(c == 'a') c = 'A';
                        if(c == 'c') c = 'C';
                        if(c == 'g') c = 'G';
                        if(c == 't') c = 'T';
						if(c == 'N' || c == 'n'){
							c = bases[Ncount];
							Ncount = (Ncount + 1) % 4;
						}
					}

					minhasher->insertSequence(read.sequence, readnum);
					readStorage->insertRead(readnum, read.sequence, read.quality);

					pair = buffer->get();

					progress += 1;
				}
			}
		};
	}

	template<class Minhasher_t,
			class ReadStorage_t>
    void build(const FileOptions& fileOptions,
			   const RuntimeOptions& runtimeOptions,
			   ReadStorage_t& readStorage,
			   Minhasher_t& minhasher){
        SequenceFileProperties props = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

        minhasher.init(props.nReads);
        readStorage.init(props.nReads);

        //std::string stmp;
        //std::cout << "build init done." << std::endl;
        //std::cin >> stmp;

        int nThreads = std::max(1, std::min(runtimeOptions.threads, 4));
        //int nThreads = 1;

        //single-threaded insertion
        if(nThreads == 1){
        	std::unique_ptr<SequenceFileReader> reader;

        	switch(fileOptions.format) {
        		case FileFormat::FASTQ: reader.reset(new FastqReader(fileOptions.inputfile)); break;
        		default: assert(false && "inputfileformat"); break;
        	}

        	Read read;
        	std::uint64_t progress = 0;
            int Ncount = 0;
            char bases[4]{'A', 'C', 'G', 'T'};

        	while (reader->getNextRead(&read)) {
                std::uint64_t readIndex = reader->getReadnum() - 1;

                for(auto& c : read.sequence){
                    if(c == 'a') c = 'A';
                    if(c == 'c') c = 'C';
                    if(c == 'g') c = 'G';
                    if(c == 't') c = 'T';
                    if(c == 'N' || c == 'n'){
                        c = bases[Ncount];
                        Ncount = (Ncount + 1) % 4;
                    }
                }

        		minhasher.insertSequence(read.sequence, readIndex);
        		readStorage.insertRead(readIndex, read.sequence, read.quality);
        		progress++;
        	}
        }else{
            //multi-threaded insertion

            using Buffer_t = ThreadsafeBuffer<std::pair<Read, std::uint64_t>, 30000>;
			using BuildThread_t = builddetail::BuildThread<Minhasher_t, ReadStorage_t, Buffer_t>;

            std::vector<BuildThread_t> buildthreads(nThreads);
            std::vector<Buffer_t> buffers(nThreads);

            for(int i = 0; i < nThreads; i++){
                buildthreads[i].buffer = &buffers[i];
                buildthreads[i].readStorage = &readStorage;
                buildthreads[i].minhasher = &minhasher;
                buildthreads[i].totalNumberOfReads = props.nReads;

                buildthreads[i].run();
            }

        	std::unique_ptr<SequenceFileReader> reader;

        	switch (fileOptions.format) {
        	case FileFormat::FASTQ:
        		reader.reset(new FastqReader(fileOptions.inputfile));
        		break;
        	default:
        		assert(false && "inputfileformat");
        		break;
        	}

        	Read read;
        	int target = 0;

        	while (reader->getNextRead(&read)) {
                std::uint64_t readnum = reader->getReadnum()-1;
        		target = readnum % nThreads;
        		buffers[target].add( { read, readnum });
        	}
        	for (auto& b : buffers) {
        		b.done();
        	}

            for(auto& t : buildthreads){
                t.join();
            }
        }
    }


}



#endif
