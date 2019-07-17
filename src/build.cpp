#include <build.hpp>

#include <config.hpp>

#include <options.hpp>

#include <minhasher.hpp>
#include <minhasher_transform.hpp>
#include "readstorage.hpp"
#include "sequencefileio.hpp"
#include "sequence.hpp"
#include "threadsafe_buffer.hpp"

#include <stdexcept>
#include <iostream>
#include <limits>
#include <thread>
#include <future>
#include <mutex>
#include <iterator>
#include <random>
#include <omp.h>




namespace care{


	namespace builddetail{
		template<class Buffer_t>
		struct BuildThread{
			BuildThread(): progress(0), isRunning(false){}
			~BuildThread() = default;
			BuildThread(const BuildThread& other) = default;
			BuildThread(BuildThread&& other) = default;
			BuildThread& operator=(const BuildThread& other) = default;
			BuildThread& operator=(BuildThread&& other) = default;

			Buffer_t* buffer;
			cpu::ContiguousReadStorage* readStorage;
			Minhasher* minhasher;
			std::uint64_t totalNumberOfReads;

			int maxSequenceLength = 0;
			int minSequenceLength = std::numeric_limits<int>::max();

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

#if 0
                    SequenceString s1(read.sequence);
                    Sequence2Bit s2(read.sequence);
                    Sequence2BitHiLo s3(read.sequence);

                    assert(s1.toString() == read.sequence);
                    assert(s2.toString() == read.sequence);
                    assert(s3.toString() == read.sequence);

                    SequenceString rs1 = s1.reverseComplement();
                    Sequence2Bit rs2 = s2.reverseComplement();
                    Sequence2BitHiLo rs3 = s3.reverseComplement();

                    assert(rs1.toString() == rs2.toString());
                    assert(rs1.toString() == rs3.toString());

                    for(int i = 0; i < int(read.sequence.size()); i++){
                        assert(s1[i] == s2[i]);
                        assert(s1[i] == s3[i]);

                        assert(rs1[i] == rs2[i]);
                        assert(rs1[i] == rs3[i]);
                    }
#endif


					minhasher->insertSequence(read.sequence, readnum);
					readStorage->insertRead(readnum, read.sequence, read.quality);

					int len = int(read.sequence.length());
					if(len > maxSequenceLength)
						maxSequenceLength = len;
					if(len < minSequenceLength)
						minSequenceLength = len;

					pair = buffer->get();

					progress += 1;
				}
			}
		};
	}

#if 0

    SequenceFileProperties build(const FileOptions& fileOptions,
			   const RuntimeOptions& runtimeOptions,
			   std::uint64_t nReads,
			   cpu::ContiguousReadStorage& readStorage,
			   Minhasher& minhasher){

        minhasher.init(nReads);

        if(fileOptions.load_binary_reads_from != ""){
            readStorage.loadFromFile(fileOptions.load_binary_reads_from);

            std::cout << "Loaded binary reads from " << fileOptions.load_binary_reads_from << std::endl;

            int maxSequenceLength = 0;
            int minSequenceLength = std::numeric_limits<int>::max();

            const int oldnumthreads = omp_get_thread_num();

            omp_set_num_threads(runtimeOptions.threads);

            #pragma omp parallel for reduction(max:maxSequenceLength) reduction(min:minSequenceLength)
            for(std::size_t i = 0; i < readStorage.sequences.size(); i++){
                const auto& seq = readStorage.sequences[i];

                int len = seq.length();
                if(len > maxSequenceLength)
                    maxSequenceLength = len;
                if(len < minSequenceLength)
                    minSequenceLength = len;

                minhasher.insertSequence(seq.toString(), i);
            }

            omp_set_num_threads(oldnumthreads);

            //minhasher.loadFromFile("hashtabledump.bin");
            //std::cout << "Loaded hashtable from " << "hashtabledump.bin" << std::endl;

            SequenceFileProperties props;
            props.nReads = readStorage.sequences.size();
            props.maxSequenceLength = maxSequenceLength;
            props.minSequenceLength = minSequenceLength;

            return props;
        }else{


            readStorage.init(nReads);

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

    			int maxSequenceLength = 0;
    			int minSequenceLength = std::numeric_limits<int>::max();

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

    #if 0
                        SequenceString s1(read.sequence);
                        Sequence2Bit s2(read.sequence);
                        Sequence2BitHiLo s3(read.sequence);

                        assert(s1.toString() == read.sequence);
                        assert(s2.toString() == read.sequence);
                        assert(s3.toString() == read.sequence);

                        SequenceString rs1 = s1.reverseComplement();
                        Sequence2Bit rs2 = s2.reverseComplement();
                        Sequence2BitHiLo rs3 = s3.reverseComplement();

                        assert(rs1.toString() == rs2.toString());
                        assert(rs1.toString() == rs3.toString());

                        for(int i = 0; i < int(read.sequence.size()); i++){
                            assert(s1[i] == s2[i]);
                            assert(s1[i] == s3[i]);

                            assert(rs1[i] == rs2[i]);
                            assert(rs1[i] == rs3[i]);
                        }
    #endif

            		minhasher.insertSequence(read.sequence, readIndex);
            		readStorage.insertRead(readIndex, read.sequence, read.quality);
            		progress++;

    				int len = int(read.sequence.length());
    				if(len > maxSequenceLength)
    					maxSequenceLength = len;
    				if(len < minSequenceLength)
    					minSequenceLength = len;
            	}

                //minhasher.loadFromFile("hashtabledump.bin");
                //std::cout << "1Loaded hashtable from " << "hashtabledump.bin" << std::endl;

            	SequenceFileProperties props;
    			props.nReads = reader->getReadnum();
    			props.maxSequenceLength = maxSequenceLength;
    			props.minSequenceLength = minSequenceLength;

                if(fileOptions.save_binary_reads_to != ""){
                    readStorage.saveToFile(fileOptions.save_binary_reads_to);
                    std::cout << "Saved binary reads to file " << fileOptions.save_binary_reads_to << std::endl;
                }

    			return props;
            }else{
                //multi-threaded insertion
    #if 1
                using Buffer_t = ThreadsafeBuffer<std::pair<Read, std::uint64_t>, 30000>;
    			using BuildThread_t = builddetail::BuildThread<Buffer_t>;
    #if 1
                std::vector<BuildThread_t> buildthreads(nThreads);
                std::vector<Buffer_t> buffers(nThreads);

                for(int i = 0; i < nThreads; i++){
                    buildthreads[i].buffer = &buffers[i];
                    buildthreads[i].readStorage = &readStorage;
                    buildthreads[i].minhasher = &minhasher;
                    buildthreads[i].totalNumberOfReads = nReads;

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

            	SequenceFileProperties props;
    			props.nReads = nReads;//reader->getReadnum();
    			props.maxSequenceLength = 0;
    			props.minSequenceLength = std::numeric_limits<int>::max();

                for(auto& t : buildthreads){
                    t.join();

    				auto minSequenceLength = t.minSequenceLength;
    				auto maxSequenceLength = t.maxSequenceLength;

    				if(minSequenceLength < props.minSequenceLength)
    					props.minSequenceLength = minSequenceLength;

    				if(maxSequenceLength > props.maxSequenceLength)
    					props.maxSequenceLength = maxSequenceLength;
                }

                //minhasher.loadFromFile("hashtabledump.bin");
                //std::cout << "2Loaded hashtable from " << "hashtabledump.bin" << std::endl;

                if(fileOptions.save_binary_reads_to != ""){
                    readStorage.saveToFile(fileOptions.save_binary_reads_to);
                    std::cout << "Saved binary reads to file " << fileOptions.save_binary_reads_to << std::endl;
                }

    			return props;
    #else
    			auto random_string = [](std::size_t len) {
    				std::mt19937_64 gen { std::random_device()() };

    				const std::string allowed_chars = "ACGT";
    				std::uniform_int_distribution<size_t> dist { 0, allowed_chars.length()-1 };

    				std::string ret;
    				ret.reserve(len);

    				std::generate_n(std::back_inserter(ret), len, [&] { return allowed_chars[dist(gen)]; });
    				return ret;
    			};

    			#pragma omp parallel for num_threads(16)
    			for(std::uint64_t readnum = 0; readnum < nReads; readnum++){
    				//const int target = readnum % nThreads;
    				//Read read;
    				//read.sequence = random_string(101);
    				//buffers[target].add( { read, readnum });
    				minhasher.insertSequence(random_string(101), readnum);
            	}

            	SequenceFileProperties props;
    			props.nReads = nReads;//reader->getReadnum();
    			props.maxSequenceLength = 0;
    			props.minSequenceLength = std::numeric_limits<int>::max();

    			return props;
    #endif



    #else


    			nThreads = std::max(1, std::min(runtimeOptions.threads, 6));
    			std::vector<std::unique_ptr<FastqReader>> readers;
    			for(int i = 0; i < nThreads; ++i){
    				switch (fileOptions.format) {
    				case FileFormat::FASTQ:
    					readers.emplace_back(new FastqReader(fileOptions.inputfile));
    					break;
    				default:
    					throw std::runtime_error("care::getSequenceFileProperties: invalid format.");
    				}
    			}

    			std::uint64_t readsPerThread = (nReads + nThreads - 1)/ nThreads;
    			using Result_t = std::pair<int,int>;
    			std::vector<Result_t> results(nThreads, {std::numeric_limits<int>::max(), 0});
    			std::vector<std::thread> threads;
    			std::vector<std::uint64_t> endings(nThreads);
    			endings[nThreads-1] = nReads;

    			std::mutex mutex;
    #if 0
    			for(int i = 1; i < nThreads; ++i){
    				for(int j = i; j < nThreads; ++j){
    					readers[j]->skipReads(readsPerThread);
    				}
    				endings[i-1] = readers[i]->getReadnum();

    				threads.emplace_back(std::thread([&,i=i-1]{

    					auto& reader = readers[i];
    					mutex.lock();
    					std::cout << i << " is running. current read num : " << reader->getReadnum() << ", ending : " << endings[i] << std::endl;
    					mutex.unlock();
    					int Ncount = 0;
    					char bases[4]{'A', 'C', 'G', 'T'};

    					Read read;
    					int maxSequenceLength = 0;
    					int minSequenceLength = std::numeric_limits<int>::max();

    					while(reader->getNextRead(&read) && reader->getReadnum() <= endings[i]){
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

    						readStorage.insertRead(readIndex, read.sequence, read.quality);
    						minhasher.insertSequence(read.sequence, readIndex);

    						int len = int(read.sequence.length());
    						if(len > maxSequenceLength)
    							maxSequenceLength = len;
    						if(len < minSequenceLength)
    							minSequenceLength = len;
    					}

    					results[i] =  Result_t{minSequenceLength, maxSequenceLength};
    				}));

    			}

    			threads.emplace_back(std::thread([&,i=nThreads-1]{

    				auto& reader = readers[i];
    				mutex.lock();
    				std::cout << i << " is running. current read num : " << reader->getReadnum() << ", ending : " << endings[i] << std::endl;
    				mutex.unlock();
    				int Ncount = 0;
    				char bases[4]{'A', 'C', 'G', 'T'};

    				Read read;
    				int maxSequenceLength = 0;
    				int minSequenceLength = std::numeric_limits<int>::max();

    				while(reader->getNextRead(&read) && reader->getReadnum() <= endings[i]){
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

    					readStorage.insertRead(readIndex, read.sequence, read.quality);
    					minhasher.insertSequence(read.sequence, readIndex);

    					int len = int(read.sequence.length());
    					if(len > maxSequenceLength)
    						maxSequenceLength = len;
    					if(len < minSequenceLength)
    						minSequenceLength = len;
    				}

    				results[i] =  Result_t{minSequenceLength, maxSequenceLength};
    			}));
    #else

    			for(int i = 0; i < nThreads; ++i){

    				threads.emplace_back(std::thread([&,i]{

    					auto& reader = readers[i];
    					std::uint64_t firstReadId_incl = readsPerThread * i;
    					std::uint64_t lastReadId_excl = i == nThreads-1 ? nReads : readsPerThread * (i+1);

    					try{
    						reader->skipReads(firstReadId_incl);
    					}catch(const SkipException& e){
    						return;
    					}

    					mutex.lock();
    					std::cout << i << " is running. current read num : " << reader->getReadnum() << ", ending : " << lastReadId_excl << std::endl;
    					mutex.unlock();
    					int Ncount = 0;
    					char bases[4]{'A', 'C', 'G', 'T'};

    					Read read;
    					int maxSequenceLength = 0;
    					int minSequenceLength = std::numeric_limits<int>::max();

    					while(reader->getNextRead(&read) && reader->getReadnum() <= lastReadId_excl){
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

    						readStorage.insertRead(readIndex, read.sequence, read.quality);
    						minhasher.insertSequence(read.sequence, readIndex);

    						int len = int(read.sequence.length());
    						if(len > maxSequenceLength)
    							maxSequenceLength = len;
    						if(len < minSequenceLength)
    							minSequenceLength = len;
    					}

    					results[i] =  Result_t{minSequenceLength, maxSequenceLength};
    				}));

    			}

    #endif

    			SequenceFileProperties props;
    			props.nReads = nReads;
    			props.maxSequenceLength = 0;
    			props.minSequenceLength = std::numeric_limits<int>::max();

    			for(int i = 0; i < nThreads; ++i){
    				threads[i].join();
    				Result_t result = results[i];
    				auto minSequenceLength = result.first;
    				auto maxSequenceLength = result.second;

    				if(minSequenceLength < props.minSequenceLength)
    					props.minSequenceLength = minSequenceLength;

    				if(maxSequenceLength > props.maxSequenceLength)
    					props.maxSequenceLength = maxSequenceLength;

    				props.nReads = readers[i]->getReadnum();
    			}

    			return props;

    		//std::cout << "props.minSequenceLength " << props.minSequenceLength << ", props.maxSequenceLength " << props.maxSequenceLength << std::endl;

    #endif
            }
        }

    }
#endif



    BuiltDataStructure<cpu::ContiguousReadStorage> build_readstorage(const FileOptions& fileOptions,
                                                const RuntimeOptions& runtimeOptions,
                                                bool useQualityScores,
                                                read_number expectedNumberOfReads,
                                                int expectedMaximumReadLength){



        if(fileOptions.load_binary_reads_from != ""){
            BuiltDataStructure<cpu::ContiguousReadStorage> result;
            auto& readStorage = result.data;

            readStorage.loadFromFile(fileOptions.load_binary_reads_from);
            result.builtType = BuiltType::Loaded;

            if(useQualityScores && !readStorage.hasQualityScores())
                throw std::runtime_error("Quality scores are required but not present in compressed sequence file!");
            if(!useQualityScores && readStorage.hasQualityScores())
                std::cerr << "Warning. The loaded compressed read file contains quality scores, but program does not use them!\n";

            std::cout << "Loaded binary reads from " << fileOptions.load_binary_reads_from << std::endl;

            return result;
        }else{
            int nThreads = std::max(1, std::min(runtimeOptions.threads, 4));

            constexpr std::array<char, 4> bases = {'A', 'C', 'G', 'T'};
            int Ncount = 0;

            BuiltDataStructure<cpu::ContiguousReadStorage> result;

            result.data = std::move(cpu::ContiguousReadStorage{expectedNumberOfReads, useQualityScores, expectedMaximumReadLength});
            result.builtType = BuiltType::Constructed;

            auto handle_read = [&](std::uint64_t readIndex, Read& read){
                const int readLength = int(read.sequence.size());

                if(readIndex >= expectedNumberOfReads){
                    throw std::runtime_error("Error! Expected " + std::to_string(expectedNumberOfReads)
                                            + " reads, but file contains at least "
                                            + std::to_string(readIndex) + " reads.");
                }

                if(readLength > expectedMaximumReadLength){
                    throw std::runtime_error("Error! Expected maximum read length = "
                                            + std::to_string(expectedMaximumReadLength)
                                            + ", but read " + std::to_string(readIndex)
                                            + "has length " + std::to_string(readLength));
                }

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

                result.data.insertRead(readIndex, read.sequence, read.quality);
#if 0
                const char* ptr = result.data.fetchSequenceData_ptr(readIndex);
                int length = result.data.fetchSequenceLength(readIndex);

                std::string s = get2BitHiLoString((const unsigned int*)ptr, length, [](auto i){return i;});
                assert(s == read.sequence);
#endif
            };

            if(nThreads == 1){
                std::unique_ptr<SequenceFileReader> reader = makeSequenceReader(fileOptions.inputfile, fileOptions.format);

                Read read;

                while (reader->getNextRead(&read)) {
                    std::uint64_t readIndex = reader->getReadnum() - 1;

                    handle_read(readIndex, read);
                }

            }else{

                using Buffer_t = ThreadsafeBuffer<std::pair<Read, std::uint64_t>, 30000>;

                std::vector<std::thread> threads;
                std::vector<Buffer_t> buffers(nThreads);

                for(int i = 0; i < nThreads; i++){
                    threads.emplace_back([&, i]{

                            auto pair = buffers[i].get();

                            while (pair != buffers[i].defaultValue) {
                                Read& read = pair.first;
                                const std::uint64_t readIndex = pair.second;

                                handle_read(readIndex, read);

                                pair = buffers[i].get();
                            }
                        });
                }

                std::unique_ptr<SequenceFileReader> reader = makeSequenceReader(fileOptions.inputfile, fileOptions.format);

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

                for(auto& thread : threads){
                    thread.join();
                }
            }

            return result;
        }

    }



    SequenceFileProperties build_readstorage(const FileOptions& fileOptions,
               const RuntimeOptions& runtimeOptions,
               cpu::ContiguousReadStorage& readStorage){

        if(fileOptions.load_binary_reads_from != ""){
            readStorage.loadFromFile(fileOptions.load_binary_reads_from);

            std::cout << "Loaded binary reads from " << fileOptions.load_binary_reads_from << std::endl;

            auto stats = readStorage.getSequenceStatistics(runtimeOptions.threads);

            SequenceFileProperties props;
            props.nReads = readStorage.getNumberOfSequences();
            props.maxSequenceLength = stats.maxSequenceLength;
            props.minSequenceLength = stats.minSequenceLength;

            readStorage.resize(props.nReads);

            return props;
        }else{

            int nThreads = std::max(1, std::min(runtimeOptions.threads, 4));
            //int nThreads = 1;

            //single-threaded insertion
            if(nThreads == 1){
                std::unique_ptr<SequenceFileReader> reader = makeSequenceReader(fileOptions.inputfile, fileOptions.format);

                Read read;
                std::uint64_t progress = 0;
                int Ncount = 0;
                char bases[4]{'A', 'C', 'G', 'T'};

                int maxSequenceLength = 0;
                int minSequenceLength = std::numeric_limits<int>::max();

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

    #if 0
                        SequenceString s1(read.sequence);
                        Sequence2Bit s2(read.sequence);
                        Sequence2BitHiLo s3(read.sequence);

                        assert(s1.toString() == read.sequence);
                        assert(s2.toString() == read.sequence);
                        assert(s3.toString() == read.sequence);

                        SequenceString rs1 = s1.reverseComplement();
                        Sequence2Bit rs2 = s2.reverseComplement();
                        Sequence2BitHiLo rs3 = s3.reverseComplement();

                        assert(rs1.toString() == rs2.toString());
                        assert(rs1.toString() == rs3.toString());

                        for(int i = 0; i < int(read.sequence.size()); i++){
                            assert(s1[i] == s2[i]);
                            assert(s1[i] == s3[i]);

                            assert(rs1[i] == rs2[i]);
                            assert(rs1[i] == rs3[i]);
                        }
    #endif

                    readStorage.insertRead(readIndex, read.sequence, read.quality);
                    progress++;

#if 0
					using Sequence_t = typename ReadStorage_t::Sequence_t;
					auto rsLength = readStorage.fetchSequenceLength(readIndex);
					auto rsSequenceptr = readStorage.fetchSequenceData_ptr(readIndex);
					auto rsSequenceString = Sequence_t::Impl_t::toString((const std::uint8_t*)rsSequenceptr, rsLength);

					assert(read.sequence.length() == size_t(rsLength));
					assert(read.sequence == rsSequenceString);
#endif

                    int len = int(read.sequence.length());
                    if(len > maxSequenceLength)
                        maxSequenceLength = len;
                    if(len < minSequenceLength)
                        minSequenceLength = len;
                }

                SequenceFileProperties props;
                props.nReads = reader->getReadnum();
                props.maxSequenceLength = maxSequenceLength;
                props.minSequenceLength = minSequenceLength;

                readStorage.resize(props.nReads);

				//TIMERSTARTCPU(readstoragetransform);
				//readStorage.transform(1);
				//TIMERSTOPCPU(readstoragetransform);

                return props;
            }else{
                //multi-threaded insertion

                using Buffer_t = ThreadsafeBuffer<std::pair<Read, std::uint64_t>, 30000>;

                std::vector<std::future<std::pair<int, int>>> futures;
                std::vector<Buffer_t> buffers(nThreads);

                for(int i = 0; i < nThreads; i++){
                    futures.emplace_back(std::async(std::launch::async, [&, i]{

                			int maxSequenceLength = 0;
                			int minSequenceLength = std::numeric_limits<int>::max();

            				auto pair = buffers[i].get();
            				int Ncount = 0;
            				char bases[4]{'A', 'C', 'G', 'T'};
            				while (pair != buffers[i].defaultValue) {
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

            #if 0
                                SequenceString s1(read.sequence);
                                Sequence2Bit s2(read.sequence);
                                Sequence2BitHiLo s3(read.sequence);

                                assert(s1.toString() == read.sequence);
                                assert(s2.toString() == read.sequence);
                                assert(s3.toString() == read.sequence);

                                SequenceString rs1 = s1.reverseComplement();
                                Sequence2Bit rs2 = s2.reverseComplement();
                                Sequence2BitHiLo rs3 = s3.reverseComplement();

                                assert(rs1.toString() == rs2.toString());
                                assert(rs1.toString() == rs3.toString());

                                for(int i = 0; i < int(read.sequence.size()); i++){
                                    assert(s1[i] == s2[i]);
                                    assert(s1[i] == s3[i]);

                                    assert(rs1[i] == rs2[i]);
                                    assert(rs1[i] == rs3[i]);
                                }
            #endif

            					readStorage.insertRead(readnum, read.sequence, read.quality);

#if 0
								using Sequence_t = typename ReadStorage_t::Sequence_t;
								auto rsLength = readStorage.fetchSequenceLength(readnum);
								auto rsSequenceptr = readStorage.fetchSequenceData_ptr(readnum);
								auto rsSequenceString = Sequence_t::Impl_t::toString((const std::uint8_t*)rsSequenceptr, rsLength);

								assert(read.sequence.length() == size_t(rsLength));
								assert(read.sequence == rsSequenceString);
#endif

            					int len = int(read.sequence.length());
            					if(len > maxSequenceLength)
            						maxSequenceLength = len;
            					if(len < minSequenceLength)
            						minSequenceLength = len;

            					pair = buffers[i].get();
            				}

                            return std::pair<int, int>(minSequenceLength, maxSequenceLength);
            			}
                    ));
                }

                std::unique_ptr<SequenceFileReader> reader = makeSequenceReader(fileOptions.inputfile, fileOptions.format);

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

                SequenceFileProperties props;
                props.nReads = reader->getReadnum();
                props.maxSequenceLength = 0;
                props.minSequenceLength = std::numeric_limits<int>::max();

                for(auto& f : futures){
                    auto pair = f.get();

                    auto minSequenceLength = pair.first;
                    auto maxSequenceLength = pair.second;

                    if(minSequenceLength < props.minSequenceLength)
                        props.minSequenceLength = minSequenceLength;

                    if(maxSequenceLength > props.maxSequenceLength)
                        props.maxSequenceLength = maxSequenceLength;
                }

                readStorage.resize(props.nReads);

                return props;
            }
        }

    };



    BuiltDataStructure<Minhasher> build_minhasher(const FileOptions& fileOptions,
                                			   const RuntimeOptions& runtimeOptions,
                                			   std::uint64_t nReads,
                                               const MinhashOptions& minhashOptions,
                                			   cpu::ContiguousReadStorage& readStorage){

        BuiltDataStructure<Minhasher> result;
        auto& minhasher = result.data;

        auto identity = [](auto i){return i;};

        minhasher = std::move(Minhasher{minhashOptions});

        minhasher.init(nReads);

        if(fileOptions.load_hashtables_from != ""){
            minhasher.loadFromFile(fileOptions.load_hashtables_from);
            result.builtType = BuiltType::Loaded;

            std::cout << "Loaded hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }else{
            result.builtType = BuiltType::Constructed;

            const int oldnumthreads = omp_get_thread_num();

            omp_set_num_threads(runtimeOptions.threads);
            constexpr int numMapsPerBatch = 16;

            const int numBatches = SDIV(minhashOptions.maps, numMapsPerBatch);

            for(int batch = 0; batch < numBatches; batch++){
                const int firstMap = batch * numMapsPerBatch;
                const int lastMap = std::min(minhashOptions.maps, (batch+1) * numMapsPerBatch);
                const int numMaps = lastMap - firstMap;
                std::vector<int> mapIds(numMaps);
                std::iota(mapIds.begin(), mapIds.end(), firstMap);

                for(auto mapId : mapIds){
                    minhasher.initMap(mapId);
                }

                #pragma omp parallel for
                for(std::size_t readId = 0; readId < readStorage.getNumberOfSequences(); readId++){
                    //const auto& seq = readStorage.sequences[readId];
    				const std::uint8_t* sequenceptr = (const std::uint8_t*)readStorage.fetchSequenceData_ptr(readId);
    				const int sequencelength = readStorage.fetchSequenceLength(readId);
    				std::string sequencestring;
                    sequencestring.resize(sequencelength);
                    decode2BitHiLoSequence(&sequencestring[0], (const unsigned int*)sequenceptr, sequencelength, identity);
                    minhasher.insertSequence(sequencestring, readId, mapIds);
                }

                for(auto mapId : mapIds){
                    transform_minhasher(minhasher, mapId, runtimeOptions.deviceIds);
                }

            }

            omp_set_num_threads(oldnumthreads);
        }

        //TIMERSTARTCPU(finalize_hashtables);
        //minhasher.transform();
        //TIMERSTOPCPU(finalize_hashtables);

        return result;
    }



















    BuiltDataStructures buildDataStructures(const MinhashOptions& minhashOptions,
                                			const CorrectionOptions& correctionOptions,
                                			const RuntimeOptions& runtimeOptions,
                                			const FileOptions& fileOptions){

        BuiltDataStructures result;

        auto& sequenceFileProperties = result.sequenceFileProperties;

        if(fileOptions.load_binary_reads_from == "") {
            if(fileOptions.nReads == 0 || fileOptions.maximum_sequence_length == 0) {
                std::cout << "Scanning file to get number of reads and maximum sequence length." << std::endl;
                sequenceFileProperties = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);
            }else{
                sequenceFileProperties.maxSequenceLength = fileOptions.maximum_sequence_length;
                sequenceFileProperties.minSequenceLength = 0;
                sequenceFileProperties.nReads = fileOptions.nReads;
            }
        }

        TIMERSTARTCPU(build_readstorage);
        result.builtReadStorage = build_readstorage(fileOptions,
                                                  runtimeOptions,
                                                  correctionOptions.useQualityScores,
                                                  sequenceFileProperties.nReads,
                                                  sequenceFileProperties.maxSequenceLength);
        TIMERSTOPCPU(build_readstorage);

        auto& readStorage = result.builtReadStorage.data;

        if(result.builtReadStorage.builtType == BuiltType::Loaded) {
            auto stats = readStorage.getSequenceStatistics(runtimeOptions.threads);
            sequenceFileProperties.nReads = readStorage.getNumberOfSequences();
            sequenceFileProperties.maxSequenceLength = stats.maxSequenceLength;
            sequenceFileProperties.minSequenceLength = stats.minSequenceLength;
        }

        TIMERSTARTCPU(build_minhasher);
        result.builtMinhasher = build_minhasher(fileOptions, runtimeOptions, sequenceFileProperties.nReads, minhashOptions, readStorage);
        TIMERSTOPCPU(build_minhasher);

        //auto& minhasher = result.builtMinhasher.data;

        //TIMERSTARTCPU(finalize_hashtables);
        //transform_minhasher(minhasher, runtimeOptions.deviceIds);
        //TIMERSTOPCPU(finalize_hashtables);

        return result;

    }
}
