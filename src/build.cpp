#include "../inc/build.hpp"
#include "../inc/sequencefileio.hpp"
#include "../inc/read.hpp"
#include "../inc/threadsafe_buffer.hpp"

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
        std::uint64_t totalNumberOfReads;

        int maxlength;
        int minlength;

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
            maxlength = 0;
            minlength = std::numeric_limits<int>::max();

            std::pair<Read, std::uint64_t> pair = buffer->get();
            int Ncount = 0;
            char bases[4]{'A', 'C', 'G', 'T'};
            while (pair != buffer->defaultValue) {
                Read& read = pair.first;
                const std::uint64_t readnum = pair.second;

                //replace 'N' with "random" base
                for(auto& c : read.sequence){
                    if(c == 'N'){
                        c = bases[Ncount];
                        Ncount = (Ncount + 1) % 4;
                    }
                }

                int len = int(read.sequence.length());
                if(len > maxlength)
                    maxlength = len;
                if(len < minlength)
                    minlength = len;

                minhasher->insertSequence(read.sequence, readnum);
                readStorage->insertRead(readnum, read);

                pair = buffer->get();

                progress += 1;
            }
        }
    };

    void build(const std::string& filename, Fileformat format, ReadStorage& readStorage,
                Minhasher& minhasher, int nThreads, int& minlen, int& maxlen){
        auto num = getNumberOfReads(filename, format);
        std::cout << "build found " << num << " reads." << std::endl;

        minhasher.init(num);
        readStorage.init(num);

        //single-threaded insertion
        if(nThreads == 1){
        	std::unique_ptr<SequenceFileReader> reader;

        	switch(format) {
        		case Fileformat::FASTQ: reader.reset(new FastqReader(filename)); break;
        		default: assert(false && "inputfileformat"); break;
        	}

        	Read read;
        	std::uint64_t progress = 0;
            int Ncount = 0;
            char bases[4]{'A', 'C', 'G', 'T'};
            int maxlength = 0;
            int minlength = std::numeric_limits<int>::max();

        	while (reader->getNextRead(&read)) {
                std::uint64_t readIndex = reader->getReadnum() - 1;

        		//replace 'N' with 'A'
                for(auto& c : read.sequence){
                    if(c == 'N'){
                        c = bases[Ncount];
                        Ncount = (Ncount + 1) % 4;
                    }
                }

                int len = int(read.sequence.length());
                if(len > maxlength)
                    maxlength = len;
                if(len < minlength)
                    minlength = len;

        		minhasher.insertSequence(read.sequence, readIndex);
        		readStorage.insertRead(readIndex, read);
        		progress++;
        	}

            minlen = minlength;
            maxlen = maxlength;
        }else{
            //multi-threaded insertion

            using Buffer_t = ThreadsafeBuffer<std::pair<Read, std::uint64_t>, 30000>;

            std::vector<BuildThread<Buffer_t>> buildthreads(nThreads);
            std::vector<Buffer_t> buffers(nThreads);

            for(int i = 0; i < nThreads; i++){
                buildthreads[i].buffer = &buffers[i];
                buildthreads[i].readStorage = &readStorage;
                buildthreads[i].minhasher = &minhasher;
                buildthreads[i].totalNumberOfReads = num;

                buildthreads[i].run();
            }

        	std::unique_ptr<SequenceFileReader> reader;

        	switch (format) {
        	case Fileformat::FASTQ:
        		reader.reset(new FastqReader(filename));
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

            maxlen = 0;
        	minlen = std::numeric_limits<int>::max();

            for(auto& t : buildthreads){
                t.join();
                if (t.maxlength > maxlen)
        			maxlen = t.maxlength;
        		if (t.minlength < minlen)
        			minlen = t.minlength;
            }
        }
    }

}
