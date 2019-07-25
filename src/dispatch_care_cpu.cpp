#include <dispatch_care.hpp>

#include <config.hpp>
#include <options.hpp>
#include <sequencefileio.hpp>
#include <readstorage.hpp>

#include <correct_cpu.hpp>

#include <build.hpp>

#include <minhasher.hpp>
#include <minhasher_transform.hpp>
#include <candidatedistribution.hpp>

#include <sequence.hpp>

#include <vector>
#include <iostream>
#include <mutex>
#include <thread>
#include <memory>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;



namespace care{

    template<class minhasher_t,
             class readStorage_t>
    void printDataStructureMemoryUsage(const minhasher_t& minhasher, const readStorage_t& readStorage){
    	auto toGB = [](std::size_t bytes){
    			    double gb = bytes / 1024. / 1024. / 1024.0;
    			    return gb;
    		    };

    	std::cout << "reads take up " << toGB(readStorage.size()) << " GB." << std::endl;
    	std::cout << "hash maps take up " << toGB(minhasher.numBytes()) << " GB." << std::endl;
    }

    void printFileProperties(const std::string& filename, const SequenceFileProperties& props){
    	std::cout << "----------------------------------------" << std::endl;
    	std::cout << "File: " << filename << std::endl;
    	std::cout << "Reads: " << props.nReads << std::endl;
    	std::cout << "Minimum sequence length: " << props.minSequenceLength << std::endl;
    	std::cout << "Maximum sequence length: " << props.maxSequenceLength << std::endl;
    	std::cout << "----------------------------------------" << std::endl;
    }

    template<class readStorage_t>
    void saveReadStorageToFile(const readStorage_t& readStorage, const FileOptions& fileOptions){
    	if(fileOptions.save_binary_reads_to != "") {
    		readStorage.saveToFile(fileOptions.save_binary_reads_to);
    		std::cout << "Saved binary reads to file " << fileOptions.save_binary_reads_to << std::endl;
    	}
    }

    template<class minhasher_t>
    void saveMinhasherToFile(const minhasher_t& minhasher, const FileOptions& fileOptions){
    	if(fileOptions.save_hashtables_to != "") {
    		minhasher.saveToFile(fileOptions.save_hashtables_to);
    		std::cout << "Saved hash tables to file " << fileOptions.save_hashtables_to << std::endl;
    	}
    }

    template<class minhasher_t,
             class readStorage_t>
    void saveDataStructuresToFile(const minhasher_t& minhasher, const readStorage_t& readStorage, const FileOptions& fileOptions){
    	saveReadStorageToFile(readStorage, fileOptions);
    	saveMinhasherToFile(minhasher, fileOptions);
    }

    void performCorrection(MinhashOptions minhashOptions,
                            AlignmentOptions alignmentOptions,
                            CorrectionOptions correctionOptions,
                            RuntimeOptions runtimeOptions,
                            FileOptions fileOptions,
                            GoodAlignmentProperties goodAlignmentProperties){

        std::cout << "Running CARE CPU" << std::endl;

    	//create output directory
    	filesys::create_directories(fileOptions.outputdirectory);

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
    	do {
    		FileOptions iterFileOptions = fileOptions;

    	#ifdef DO_ALTERNATE
    		//alternate between two output files
    		// on even iteration, correct file _iter_odd and save to _iter_even
    		// on odd iteration, correct file _iter_even and save to _iter_odd
    		if(iter == 0) {
    			//inputfile remains original input file
    			iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
    		}else{
    			if(iter % 2 == 0) {
    				iterFileOptions.inputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd";
    				iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
    			}else{
    				iterFileOptions.inputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
    				iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd";
    			}
    		}

    	#else
    		if(iter == 0) {
    			//inputfile remains original input file
    			iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_0";
    		}else{
    			iterFileOptions.inputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iter-1);
    			iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iter);
    		}
    	#endif

            std::cout << "loading file and building data structures..." << std::endl;

            TIMERSTARTCPU(load_and_build);

            BuiltDataStructures dataStructures = buildDataStructures(minhashOptions,
                                                                    correctionOptions,
                                                                    runtimeOptions,
                                                                    iterFileOptions);

            TIMERSTOPCPU(load_and_build);

            auto& readStorage = dataStructures.builtReadStorage.data;
            auto& minhasher = dataStructures.builtMinhasher.data;
            auto& sequenceFileProperties = dataStructures.sequenceFileProperties;

            saveReadStorageToFile(readStorage, iterFileOptions);
            saveMinhasherToFile(minhasher, iterFileOptions);

            printFileProperties(fileOptions.inputfile, sequenceFileProperties);

            TIMERSTARTCPU(candidateestimation);
            std::uint64_t maxCandidatesPerRead = runtimeOptions.max_candidates;

            if(maxCandidatesPerRead == 0){
                maxCandidatesPerRead = calculateMaxCandidatesPerReadThreshold(minhasher,
                                                        readStorage,
                                                        sequenceFileProperties.nReads / 10,
                                                        correctionOptions.hits_per_candidate,
                                                        runtimeOptions.threads
                                                        //,"ncandidates.txt"
                                                        );

                std::cout << "maxCandidates option not specified. Using estimation: " << maxCandidatesPerRead << std::endl;
            }



            TIMERSTOPCPU(candidateestimation);

            readIsCorrectedVector.resize(sequenceFileProperties.nReads, 0);

            std::cerr << "readIsCorrectedVector bytes: " << readIsCorrectedVector.size() / 1024. / 1024. << " MB\n";

            printDataStructureMemoryUsage(minhasher, readStorage);

            std::cout << "Running CARE CPU" << std::endl;

            cpu::correct_cpu(minhashOptions, alignmentOptions,
                        goodAlignmentProperties, correctionOptions,
                        runtimeOptions, fileOptions, sequenceFileProperties,
                        minhasher, readStorage,
                        maxCandidatesPerRead,
                        readIsCorrectedVector, locksForProcessedFlags,
                        nLocksForProcessedFlags);

    		iter++;

    	} while(iter < iters);


    	//rename final result to requested output file name and delete intermediate files
    	bool keepIntermediateResults = false;

        TIMERSTARTCPU(finalizing_files);

    	#ifdef DO_ALTERNATE
    	if(iters % 2 == 0) {
    		std::string toRename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd";
    		std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

            //rename feature file
            if(correctionOptions.extractFeatures){
                std::string tmpfeaturename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd_features";
                std::string outputfeaturename = fileOptions.outputfile + "_features";

        		std::rename(tmpfeaturename.c_str(), outputfeaturename.c_str());
            }

    		if(!keepIntermediateResults && iters > 1)
    			deleteFiles({fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even"});
    	}else{
    		std::string toRename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
    		std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

            if(correctionOptions.extractFeatures){
                std::string tmpfeaturename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even_features";
                std::string outputfeaturename = fileOptions.outputfile + "_features";

        		std::rename(tmpfeaturename.c_str(), outputfeaturename.c_str());
            }

    		if(!keepIntermediateResults && iters > 1)
    			deleteFiles({fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd"});
    	}
    	#else
    	std::string toRename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iters-1);
    	std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

        if(correctionOptions.extractFeatures){
            std::string tmpfeaturename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iters-1) + "_features";
            std::string outputfeaturename = fileOptions.outputfile + "_features";

            std::rename(tmpfeaturename.c_str(), outputfeaturename.c_str());
        }

    	if(!keepIntermediateResults) {
    		std::vector<std::string> filestodelete;
    		for(int i = 0; i < iters-1; i++)
    			filestodelete.push_back(fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(i));
    		deleteFiles(filestodelete);
    	}
    	#endif

        TIMERSTOPCPU(finalizing_files);
    }

}
