#include "../inc/care.hpp"

#include "../inc/args.hpp"
#include "../inc/build.hpp"
#include "../inc/correct.hpp"
#include "../inc/minhasher.hpp"
#include "../inc/options.hpp"
#include "../inc/readstorage.hpp"
#include "../inc/sequence.hpp"

#include <vector>
#include <iostream>
#include <mutex>
#include <thread>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;


#ifdef __NVCC__
#include "../inc/gpu_only_path/correct.hpp"
#endif

namespace care{

    enum class Mode{
        CPU,
        GPU,
    };

    template<class minhasher_t,
    		 class readStorage_t,
    		 bool indels,
             class StartCorrectionFunction>
    void correctFileWithMode_impl(const MinhashOptions& minhashOptions,
    				  const AlignmentOptions& alignmentOptions,
    				  const GoodAlignmentProperties& goodAlignmentProperties,
    				  const CorrectionOptions& correctionOptions,
    				  const RuntimeOptions& runtimeOptions,
    				  const FileOptions& fileOptions,
    				  std::uint64_t nReads,
    				  std::vector<char>& readIsCorrectedVector,
    				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
    				  std::size_t nLocksForProcessedFlags,
    				  StartCorrectionFunction startCorrection){

    	constexpr bool indelAlignment = indels;

    	using Minhasher_t = minhasher_t;
    	using ReadStorage_t = readStorage_t;
        using Sequence_t = typename ReadStorage_t::Sequence_t;

        auto toGB = [](std::size_t bytes){
            double gb = bytes / 1024. / 1024. / 1024.0;
            return gb;
        };

        std::cout << "Sequence type: " << getSequenceType<Sequence_t>() << std::endl;

        Minhasher_t minhasher(minhashOptions, runtimeOptions.canUseGpu);
        ReadStorage_t readStorage(correctionOptions.useQualityScores);

        std::cout << "loading file and building data structures..." << std::endl;

        TIMERSTARTCPU(load_and_build);
        const SequenceFileProperties props = build_readstorage(fileOptions, runtimeOptions, nReads, readStorage);
        build_minhasher(fileOptions, runtimeOptions, props.nReads, readStorage, minhasher);
        TIMERSTOPCPU(load_and_build);

    	std::cout << "----------------------------------------" << std::endl;
        std::cout << "File: " << fileOptions.inputfile << std::endl;
        std::cout << "Reads: " << props.nReads << std::endl;
        std::cout << "Minimum sequence length: " << props.minSequenceLength << std::endl;
        std::cout << "Maximum sequence length: " << props.maxSequenceLength << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        if(fileOptions.save_binary_reads_to != ""){
            readStorage.saveToFile(fileOptions.save_binary_reads_to);
            std::cout << "Saved binary reads to file " << fileOptions.save_binary_reads_to << std::endl;
        }

        if(fileOptions.save_hashtables_to != ""){
            minhasher.saveToFile(fileOptions.save_hashtables_to);
            std::cout << "Saved hash tables to file " << fileOptions.save_hashtables_to << std::endl;
        }

        readIsCorrectedVector.resize(props.nReads, 0);


        std::cout << "reads take up " << toGB(readStorage.size()) << " GB." << std::endl;
        std::cout << "hash maps take up " << toGB(minhasher.numBytes()) << " GB." << std::endl;


        startCorrection(minhasher, readStorage, props);

        /*correct<Minhasher_t,
    			ReadStorage_t,
    			indelAlignment>(minhashOptions, alignmentOptions,
    							goodAlignmentProperties, correctionOptions,
    							runtimeOptions, fileOptions, props,
    							minhasher, readStorage,
    							readIsCorrectedVector, locksForProcessedFlags,
    							nLocksForProcessedFlags, runtimeOptions.deviceIds);*/

    }


    void correctFileWithMode(Mode mode,
                    const MinhashOptions& minhashOptions,
    				  const AlignmentOptions& alignmentOptions,
    				  const GoodAlignmentProperties& goodAlignmentProperties,
    				  const CorrectionOptions& correctionOptions,
    				  const RuntimeOptions& runtimeOptions,
    				  const FileOptions& fileOptions,
    				  std::uint64_t nReads,
    				  std::vector<char>& readIsCorrectedVector,
    				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
    				  std::size_t nLocksForProcessedFlags){

        using Key_t = std::uint32_t; // asume minhashOptions.k <= 16
        using ReadId_t = std::uint32_t; // asume nReads <= std::numeric_limits<std::uint32_t>::max()

        using Minhasher_t = Minhasher<Key_t, ReadId_t>;

        if(mode == Mode::GPU){
#ifdef __NVCC__
            using NoIndelSequence_t = Sequence2BitHiLo;
            //using IndelSequence_t = Sequence2BitHiLo;
            using NoIndelReadStorage_t = ReadStorageMinMemory<NoIndelSequence_t, ReadId_t>;
            //using IndelReadStorage_t = ReadStorageMinMemory<IndelSequence_t, ReadId_t>;

            if(correctionOptions.correctionMode == CorrectionMode::Hamming){
        		constexpr bool indels = false;

                auto func = [&](Minhasher_t& minhasher, NoIndelReadStorage_t& readStorage, SequenceFileProperties props){
                    //using Minhasher_t = decltype(minhasher);
                    //using ReadStorage_t = decltype(readStorage);
                    gpu::correct_gpu<Minhasher_t,
            			NoIndelReadStorage_t,
            			indels>(minhashOptions, alignmentOptions,
            							goodAlignmentProperties, correctionOptions,
            							runtimeOptions, fileOptions, props,
            							minhasher, readStorage,
            							readIsCorrectedVector, locksForProcessedFlags,
            							nLocksForProcessedFlags);
                };

        		correctFileWithMode_impl<Minhasher_t, NoIndelReadStorage_t, indels>
        						(
        							minhashOptions,
        							alignmentOptions,
        							goodAlignmentProperties,
        							correctionOptions,
        							runtimeOptions,
        							fileOptions,
        							nReads,
        							readIsCorrectedVector,
        							locksForProcessedFlags,
        							nLocksForProcessedFlags,
                                    func
        						);
        	}else{
        		//constexpr bool indels = true;

        		std::cout << "Cannot correct indels with GPU version" << std::endl;
                return;
        	}
#else
            throw std::runtime_error("This should not happen in correctFileWithMode");
#endif
        }else if(mode == Mode::CPU){
            using NoIndelSequence_t = Sequence2BitHiLo;
            //using IndelSequence_t = Sequence2BitHiLo;
            using NoIndelReadStorage_t = ReadStorageMinMemory<NoIndelSequence_t, ReadId_t>;
            //using IndelReadStorage_t = ReadStorageMinMemory<IndelSequence_t, ReadId_t>;

            if(correctionOptions.correctionMode == CorrectionMode::Hamming){
                constexpr bool indels = false;

                auto func = [&](Minhasher_t& minhasher, NoIndelReadStorage_t& readStorage, SequenceFileProperties props){
                    correct_cpu<Minhasher_t,
            			NoIndelReadStorage_t,
            			indels>(minhashOptions, alignmentOptions,
            							goodAlignmentProperties, correctionOptions,
            							runtimeOptions, fileOptions, props,
            							minhasher, readStorage,
            							readIsCorrectedVector, locksForProcessedFlags,
            							nLocksForProcessedFlags);
                };

                correctFileWithMode_impl<Minhasher_t,
                                NoIndelReadStorage_t,
                                indels>
                                (
                                    minhashOptions,
                                    alignmentOptions,
                                    goodAlignmentProperties,
                                    correctionOptions,
                                    runtimeOptions,
                                    fileOptions,
                                    nReads,
                                    readIsCorrectedVector,
                                    locksForProcessedFlags,
                                    nLocksForProcessedFlags,
                                    func
                                );
            }else{
                //constexpr bool indels = true;

                std::cout << "Cannot correct indels with CPU version" << std::endl;
                return;
            }
        }


    }





void performCorrection(const cxxopts::ParseResult& args) {
	//check arguments
    /*if(!args::areValid(args)){
        throw std::runtime_error("care::performCorrection: Invalid arguments!");
    }*/

	//parse options from arguments
	MinhashOptions minhashOptions = args::to<MinhashOptions>(args);
	AlignmentOptions alignmentOptions = args::to<AlignmentOptions>(args);
	GoodAlignmentProperties goodAlignmentProperties = args::to<GoodAlignmentProperties>(args);
    CorrectionOptions correctionOptions = args::to<CorrectionOptions>(args);
	RuntimeOptions runtimeOptions = args::to<RuntimeOptions>(args);
	FileOptions fileOptions = args::to<FileOptions>(args);

    {
        if(!args::isValid(minhashOptions)) throw std::runtime_error("care::performCorrection: Invalid minhashOptions!");
        if(!args::isValid(alignmentOptions)) throw std::runtime_error("care::performCorrection: Invalid alignmentOptions!");
        if(!args::isValid(goodAlignmentProperties)) throw std::runtime_error("care::performCorrection: Invalid goodAlignmentProperties!");
        if(!args::isValid(correctionOptions)) throw std::runtime_error("care::performCorrection: Invalid correctionOptions!");
        if(!args::isValid(runtimeOptions)) throw std::runtime_error("care::performCorrection: Invalid runtimeOptions!");
        if(!args::isValid(fileOptions)) throw std::runtime_error("care::performCorrection: Invalid fileOptions!");
    }

    if(correctionOptions.correctCandidates && correctionOptions.extractFeatures){
        std::cout << "Warning! correctCandidates=true cannot be used with extractFeatures=true. Using correctCandidates=false" << std::endl;
        correctionOptions.correctCandidates = false;
    }

#ifndef __NVCC__
        std::cout << "Running CARE CPU" << std::endl;
#else
    if(runtimeOptions.canUseGpu){
        std::cout << "Running CARE GPU" << std::endl;
        std::cout << "Can use the following GPU device Ids: ";

        for(int i : runtimeOptions.deviceIds)
            std::cout << i << " ";

        std::cout << std::endl;
    }else{
        std::cout << "Running CARE CPU" << std::endl;
    }
#endif

	//create output directory
	filesys::create_directories(fileOptions.outputdirectory);

	std::uint64_t nReads = fileOptions.nReads;
	if(nReads == 0 && fileOptions.load_binary_reads_from == ""){ // if load_binary_reads_from != "", we use number of reads from binaryreadfile
		std::cout << "Determining number of reads" << std::endl;
		nReads = getNumberOfReadsFast(fileOptions.inputfile, fileOptions.format);
	}

	std::vector<char> readIsCorrectedVector;
	std::size_t nLocksForProcessedFlags = runtimeOptions.nCorrectorThreads * 1000;
	std::unique_ptr<std::mutex[]> locksForProcessedFlags(new std::mutex[nLocksForProcessedFlags]);

	const int iters = 1;
	int iter = 0;

    auto thread_id = std::this_thread::get_id();
    std::string thread_id_string;
    {
        std::stringstream ss;
        ss << thread_id;
        thread_id_string = ss.str();
    }

#define DO_ALTERNATE

	// correct file in multiple passes
	do{
		FileOptions iterFileOptions = fileOptions;

#ifdef DO_ALTERNATE
		//alternate between two output files
		// on even iteration, correct file _iter_odd and save to _iter_even
		// on odd iteration, correct file _iter_even and save to _iter_odd
		if(iter == 0){
			//inputfile remains original input file
			iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
		}else{
			if(iter % 2 == 0){
				iterFileOptions.inputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd";
				iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
			}else{
				iterFileOptions.inputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
				iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd";
			}
		}

#else
		if(iter == 0){
			//inputfile remains original input file
			iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_0";
		}else{
			iterFileOptions.inputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iter-1);
			iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iter);
		}
#endif

        Mode mode = runtimeOptions.canUseGpu ? Mode::GPU : Mode::CPU;

		correctFileWithMode(mode, minhashOptions, alignmentOptions,
            goodAlignmentProperties, correctionOptions,
            runtimeOptions, iterFileOptions,
			nReads,
            readIsCorrectedVector, locksForProcessedFlags,
            nLocksForProcessedFlags);//, runtimeOptions.deviceIds);

		iter++;

	}while(iter < iters);


	//rename final result to requested output file name and delete intermediate files
	bool keepIntermediateResults = false;

#ifdef DO_ALTERNATE
	if(iters % 2 == 0){
		std::string toRename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd";
		std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

		if(!keepIntermediateResults && iters > 1)
			deleteFiles({fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even"});
	}else{
		std::string toRename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
		std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

		if(!keepIntermediateResults && iters > 1)
			deleteFiles({fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd"});
	}
#else
	std::string toRename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iters-1);
	std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

	if(!keepIntermediateResults){
		std::vector<std::string> filestodelete;
		for(int i = 0; i < iters-1; i++)
			filestodelete.push_back(fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(i));
		deleteFiles(filestodelete);
	}
#endif


}


}
