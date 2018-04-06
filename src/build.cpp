#include "../inc/build.hpp"
#include "../inc/sequencefileio.hpp"
#include "../inc/sequence.hpp"
#include "../inc/threadsafe_buffer.hpp"
#include "../inc/types.hpp"

#include <stdexcept>
#include <iostream>
#include <limits>
#include <thread>

namespace care{

    template<class Buffer_t>
    struct BuildThread{
        BuildThread(): progress(0), isRunning(false){}
        ~BuildThread() = default;
        BuildThread(const BuildThread& other) = default;
        BuildThread(BuildThread&& other) = default;
        BuildThread& operator=(const BuildThread& other) = default;
        BuildThread& operator=(BuildThread&& other) = default;

        Buffer_t* buffer;
        ReadStorage* readStorage;
        Minhasher* minhasher;
        ReadId_t totalNumberOfReads;

        ReadId_t progress;
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

            std::pair<Read, ReadId_t> pair = buffer->get();
            int Ncount = 0;
            char bases[4]{'A', 'C', 'G', 'T'};
            while (pair != buffer->defaultValue) {
                Read& read = pair.first;
                const ReadId_t readnum = pair.second;

                //replace 'N' with "random" base
                for(auto& c : read.sequence){
                    if(c == 'N'){
                        c = bases[Ncount];
                        Ncount = (Ncount + 1) % 4;
                    }
                }

                minhasher->insertSequence(read.sequence, readnum);
                readStorage->insertRead(readnum, read);

                pair = buffer->get();

                progress += 1;
            }
        }
    };

    void build(const FileOptions& fileOptions, ReadStorage& readStorage,
                Minhasher& minhasher, int nThreads){
        SequenceFileProperties props = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

        minhasher.init(props.nReads);
        readStorage.init(props.nReads);

        nThreads = std::max(1, std::min(nThreads, 4));

        //single-threaded insertion
        if(nThreads == 1){
        	std::unique_ptr<SequenceFileReader> reader;

        	switch(fileOptions.format) {
        		case FileFormat::FASTQ: reader.reset(new FastqReader(fileOptions.inputfile)); break;
        		default: assert(false && "inputfileformat"); break;
        	}

        	Read read;
        	ReadId_t progress = 0;
            int Ncount = 0;
            char bases[4]{'A', 'C', 'G', 'T'};

        	while (reader->getNextRead(&read)) {
                ReadId_t readIndex = reader->getReadnum() - 1;

        		//replace 'N' with 'A'
                for(auto& c : read.sequence){
                    if(c == 'N'){
                        c = bases[Ncount];
                        Ncount = (Ncount + 1) % 4;
                    }
                }

        		minhasher.insertSequence(read.sequence, readIndex);
        		readStorage.insertRead(readIndex, read);
        		progress++;
        	}
        }else{
            //multi-threaded insertion

            using Buffer_t = ThreadsafeBuffer<std::pair<Read, ReadId_t>, 30000>;

            std::vector<BuildThread<Buffer_t>> buildthreads(nThreads);
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
                ReadId_t readnum = reader->getReadnum()-1;
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
