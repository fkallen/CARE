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

namespace care{

/*
    Correct fileOptions.inputfile and save result to fileOptions.outputfile
*/
template<class minhasher_t,
		 class readStorage_t,
		 bool indels>
void correctFile_impl(const MinhashOptions& minhashOptions,
				  const AlignmentOptions& alignmentOptions,
				  const GoodAlignmentProperties& goodAlignmentProperties,
				  const CorrectionOptions& correctionOptions,
				  const RuntimeOptions& runtimeOptions,
				  const FileOptions& fileOptions,
                  const SequenceFileProperties& props,
				  std::uint64_t nReads,
				  std::vector<char>& readIsCorrectedVector,
				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
				  std::size_t nLocksForProcessedFlags,
				  const std::vector<int>& deviceIds){

	constexpr bool indelAlignment = indels;

	using Minhasher_t = minhasher_t;
	using ReadStorage_t = readStorage_t;

    auto toGB = [](std::size_t bytes){
        double gb = bytes / 1024. / 1024. / 1024.0;
        return gb;
    };

    Minhasher_t minhasher(minhashOptions);
    ReadStorage_t readStorage(correctionOptions.useQualityScores);

    std::string stmp;

    std::cout << "loading file and building data structures..." << std::endl;

    //std::cin >> stmp;
	TIMERSTARTCPU(load_and_build);
    build(fileOptions, runtimeOptions, readStorage, minhasher);
	TIMERSTOPCPU(load_and_build);

    //std::cin >> stmp;

    TIMERSTARTCPU(finalize_datastructures);

    readStorage.transform();
    minhasher.transform();

	TIMERSTOPCPU(finalize_datastructures);

    std::cout << "reads take up " << toGB(readStorage.size()) << " GB." << std::endl;
    std::cout << "hash maps take up " << toGB(minhasher.numBytes()) << " GB." << std::endl;

    correct<Minhasher_t,
			ReadStorage_t,
			indelAlignment>(minhashOptions, alignmentOptions,
							goodAlignmentProperties, correctionOptions,
							runtimeOptions, fileOptions, props,
							minhasher, readStorage,
							readIsCorrectedVector, locksForProcessedFlags,
							nLocksForProcessedFlags, deviceIds);

}

void correctFile(const MinhashOptions& minhashOptions,
				  const AlignmentOptions& alignmentOptions,
				  const GoodAlignmentProperties& goodAlignmentProperties,
				  const CorrectionOptions& correctionOptions,
				  const RuntimeOptions& runtimeOptions,
				  const FileOptions& fileOptions,
                  const SequenceFileProperties& props,
				  std::uint64_t nReads,
				  std::vector<char>& readIsCorrectedVector,
				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
				  std::size_t nLocksForProcessedFlags,
				  const std::vector<int>& deviceIds){

    using Key_t = std::uint32_t; // asume minhashOptions.k <= 16
    using ReadId_t = std::uint32_t; // asume nReads <= std::numeric_limits<std::uint32_t>::max()

    using Minhasher_t = Minhasher<Key_t, ReadId_t>;

    if(runtimeOptions.canUseGpu){
        using NoIndelSequence_t = Sequence2Bit;
        using IndelSequence_t = Sequence2Bit;
        using NoIndelReadStorage_t = ReadStorageMinMemory<NoIndelSequence_t, ReadId_t>;
        using IndelReadStorage_t = ReadStorageMinMemory<IndelSequence_t, ReadId_t>;

        if(correctionOptions.correctionMode == CorrectionMode::Hamming){
    		constexpr bool indels = false;

    		correctFile_impl<Minhasher_t,
    						NoIndelReadStorage_t,
    						indels>
    						(
    							minhashOptions,
    							alignmentOptions,
    							goodAlignmentProperties,
    							correctionOptions,
    							runtimeOptions,
    							fileOptions,
                                props,
    							nReads,
    							readIsCorrectedVector,
    							locksForProcessedFlags,
    							nLocksForProcessedFlags,
    							deviceIds
    						);
    	}else{
    		constexpr bool indels = true;

    		correctFile_impl<Minhasher_t,
    						IndelReadStorage_t,
    						indels>
    						(
    							minhashOptions,
    							alignmentOptions,
    							goodAlignmentProperties,
    							correctionOptions,
    							runtimeOptions,
    							fileOptions,
                                props,
    							nReads,
    							readIsCorrectedVector,
    							locksForProcessedFlags,
    							nLocksForProcessedFlags,
    							deviceIds
    						);
    	}
    }else{
        using NoIndelSequence_t = Sequence2BitHiLo;
        using IndelSequence_t = Sequence2Bit;
        using NoIndelReadStorage_t = ReadStorageMinMemory<NoIndelSequence_t, ReadId_t>;
        using IndelReadStorage_t = ReadStorageMinMemory<IndelSequence_t, ReadId_t>;

        if(correctionOptions.correctionMode == CorrectionMode::Hamming){
            constexpr bool indels = false;

            correctFile_impl<Minhasher_t,
                            NoIndelReadStorage_t,
                            indels>
                            (
                                minhashOptions,
                                alignmentOptions,
                                goodAlignmentProperties,
                                correctionOptions,
                                runtimeOptions,
                                fileOptions,
                                props,
                                nReads,
                                readIsCorrectedVector,
                                locksForProcessedFlags,
                                nLocksForProcessedFlags,
                                deviceIds
                            );
        }else{
            constexpr bool indels = true;

            correctFile_impl<Minhasher_t,
                            IndelReadStorage_t,
                            indels>
                            (
                                minhashOptions,
                                alignmentOptions,
                                goodAlignmentProperties,
                                correctionOptions,
                                runtimeOptions,
                                fileOptions,
                                props,
                                nReads,
                                readIsCorrectedVector,
                                locksForProcessedFlags,
                                nLocksForProcessedFlags,
                                deviceIds
                            );
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

    std::cout << "Determining read properties..." << std::endl;

    SequenceFileProperties props = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "File: " << fileOptions.inputfile << std::endl;
    std::cout << "Reads: " << props.nReads << std::endl;
    std::cout << "Minimum sequence length: " << props.minSequenceLength << std::endl;
    std::cout << "Maximum sequence length: " << props.maxSequenceLength << std::endl;
    std::cout << "----------------------------------------" << std::endl;

	std::vector<char> readIsCorrectedVector(props.nReads, 0);
	std::size_t nLocksForProcessedFlags = correctionOptions.batchsize * runtimeOptions.nCorrectorThreads * 1000;
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
        SequenceFileProperties iterprops = props;

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

            // with indel correction, corrected sequence lengths may be different from original sequence length. cannot reuse min / max sequence length from props
            if(correctionOptions.correctionMode == CorrectionMode::Graph)
                iterprops = getSequenceFileProperties(iterFileOptions.inputfile, iterFileOptions.format);
		}

#else
		if(iter == 0){
			//inputfile remains original input file
			iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_0";
		}else{
			iterFileOptions.inputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iter-1);
			iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iter);

            // with indel correction, corrected sequence lengths may be different from original sequence length. cannot reuse min / max sequence length from props
            if(correctionOptions.correctionMode == CorrectionMode::Graph)
                iterprops = getSequenceFileProperties(iterFileOptions.inputfile, iterFileOptions.format);
		}
#endif



		correctFile(minhashOptions, alignmentOptions,
            goodAlignmentProperties, correctionOptions,
            runtimeOptions, iterFileOptions, iterprops,
			props.nReads,
            readIsCorrectedVector, locksForProcessedFlags,
            nLocksForProcessedFlags, runtimeOptions.deviceIds);

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
