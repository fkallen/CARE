#include <care.hpp>

//#include <args.hpp>
#include <build.hpp>

#include <minhasher.hpp>
#include <minhasher_transform.hpp>
#include <dispatch_correction.hpp>
#include <options.hpp>
#include <candidatedistribution.hpp>

#include <sequence.hpp>
#include <readstorage.hpp>
#include <config.hpp>

#include <vector>
#include <iostream>
#include <mutex>
#include <thread>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;



#include <gpu/correct_gpu.hpp>
#include <gpu/readstorage.hpp>

namespace care {
namespace gpu {


template<class minhasher_t,
         class readStorage_t>
void printDataStructureMemoryUsage(const minhasher_t& minhasher, const readStorage_t& readStorage){
	auto toGB = [](std::size_t bytes){
			    double gb = bytes / 1024. / 1024. / 1024.0;
			    return gb;
		    };

    auto memInfo = readStorage.getMemoryInfo();

    assert(memInfo.deviceIds.size() == memInfo.deviceSizeInBytes.size());

    std::cout << "Reads occupy " << toGB(memInfo.hostSizeInBytes) << " GB on host\n";
    for(size_t i = 0; i < memInfo.deviceIds.size(); i++){
        std::cout << "Reads occupy " << toGB(memInfo.deviceSizeInBytes[i]) << " GB on device " << memInfo.deviceIds[i] << '\n';
    }

	//std::cout << "reads take up " << toGB(readStorage.size()) << " GB." << std::endl;
	std::cout << "hash maps take up " << toGB(minhasher.numBytes()) << " GB on host." << std::endl;
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



void checkBuiltDataStructures(BuiltGpuDataStructures& gpudata, BuiltDataStructures& cpudata){
    auto& cpurs = cpudata.builtReadStorage.data;
    auto& gpurs = gpudata.builtReadStorage.data;
    //auto& oldMinhasher = cpudata.builtMinhasher.data;
    //auto& newMinhasher = gpudata.builtMinhasher.data;

    gpu::ContiguousReadStorage oldgpurs(&cpurs, {0});
    oldgpurs.initGPUData();
    std::cerr << "CHECKING....\n";


    // #pragma omp parallel for num_threads(16)
    // for(read_number i = 0; i < gpurs.getNumberOfReads(); i++){
    //     std::vector<unsigned int> oldRsData(8,0);
    //     std::vector<unsigned int> newRsData(8,0);
    //
    //     oldgpurs.distributedSequenceData2.get(i, oldRsData.data());
    //     gpurs.distributedSequenceData2.get(i, newRsData.data());
    //
    //     const char* ptr = cpurs.fetchSequenceData_ptr(i);
    //
    //     std::vector<unsigned int> cpuRsData(8,0);
    //     std::copy((const unsigned int*)ptr, ((const unsigned int*)ptr)+8, cpuRsData.begin());
    //
    //     if(oldRsData != newRsData || oldRsData != cpuRsData){
    //         std::cerr << "Error read " << i << "\n";
    //         std::cerr << "Old rs\n";
    //         std::copy(oldRsData.begin(), oldRsData.end(), std::ostream_iterator<unsigned int>(std::cerr, " "));
    //         std::cerr << "\n";
    //         std::cerr << get2BitHiLoString(oldRsData.data(), 100) << '\n';
    //         std::cerr << "New rs\n";
    //         std::copy(newRsData.begin(), newRsData.end(), std::ostream_iterator<unsigned int>(std::cerr, " "));
    //         std::cerr << "\n";
    //         std::cerr << get2BitHiLoString(newRsData.data(), 100) << '\n';
    //         std::cerr << "Cpu rs\n";
    //         std::copy(cpuRsData.begin(), cpuRsData.end(), std::ostream_iterator<unsigned int>(std::cerr, " "));
    //         std::cerr << "\n";
    //         std::cerr << get2BitHiLoString(cpuRsData.data(), 100) << '\n';
    //         assert(oldRsData == newRsData);
    //         assert(oldRsData == cpuRsData);
    //     }
    //
    //     assert(oldRsData == newRsData);
    //     assert(oldRsData == cpuRsData);
    // }


    for(read_number i = 0; i < 1024; i++){
        int num = 50000;
        int maxlen = 128;
        int enclen = getEncodedNumInts2BitHiLo(maxlen);
        std::mt19937 gen;
        gen.seed(std::random_device()());
        std::uniform_int_distribution<read_number> dist(0, gpurs.getNumberOfReads()-1); // distribution in range [1, 6]
        std::vector<read_number> ids(num);
        std::generate(ids.begin(), ids.end(), [&](){return dist(gen);});

        SimpleAllocationDevice<read_number> d_readIds;
        d_readIds.resize(num);
        cudaMemcpy(d_readIds.get(), ids.data(), sizeof(read_number) * num, H2D); CUERR;


        SimpleAllocationDevice<char> d_data;
        d_data.resize(maxlen * num);
        cudaMemset(d_data.get(), 0, sizeof(char) * maxlen * num); CUERR;
        oldgpurs.copyGpuSequenceDataToGpuBufferAsync(d_data.get(), maxlen, d_readIds.get(), num, 0, 0);
        cudaDeviceSynchronize(); CUERR;

        auto handle = gpurs.makeGatherHandleSequences();
        SimpleAllocationDevice<char> d_datanew;
        d_datanew.resize(maxlen * num);
        cudaMemset(d_datanew.get(), 0, sizeof(char) * maxlen * num); CUERR;
        gpurs.gatherSequenceDataToGpuBufferAsync(
                                    handle,
                                    d_datanew.get(),
                                    maxlen,
                                    ids.data(),
                                    d_readIds.get(),
                                    num,
                                    0,
                                    0,
                                    1);

        cudaDeviceSynchronize(); CUERR;

        SimpleAllocationPinnedHost<char> h_data; h_data.resize(maxlen * num);
        SimpleAllocationPinnedHost<char> h_datanew; h_datanew.resize(maxlen * num);

        cudaMemcpy(h_data.get(), d_data.get(), sizeof(char) * maxlen * num, D2H); CUERR;
        cudaMemcpy(h_datanew.get(), d_datanew.get(), sizeof(char) * maxlen * num, D2H); CUERR;

        // std::vector<unsigned int> oldRsData(8,0);
        // std::vector<unsigned int> newRsData(8,0);
        //
        // oldgpurs.distributedSequenceData2.get(ids[0], oldRsData.data());
        // gpurs.distributedSequenceData2.get(ids[0], newRsData.data());
        //
        //
        // std::cerr << get2BitHiLoString(oldRsData.data(), 100) << '\n';
        // std::cerr << get2BitHiLoString(newRsData.data(), 100) << '\n';
        // std::cerr << get2BitHiLoString((unsigned int*)h_data.get(), 100) << '\n';
        // std::cerr << get2BitHiLoString((unsigned int*)h_datanew.get(), 100) << '\n';



        auto* oldd = d_data.get();
        auto* newd = d_datanew.get();
        SimpleAllocationDevice<bool> flag;
        flag.resize(1);
        auto* f = flag.get();
        generic_kernel<<<num,128>>>([=] __device__ (){
            //bool error = false;
            for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < num * maxlen; i += blockDim.x * gridDim.x){
                if(oldd[i] != newd[i]){
                    printf("error i %d, %d, %d\n", i, int(oldd[i]), int(newd[i]));
                    //error = true;
                    break;
                }
            }
            //*f = error;
        });
        cudaDeviceSynchronize();
        //SimpleAllocationPinnedHost<bool> hflag; hflag.resize(1);
        //cudaMemcpy(hflag.get(), flag.get(), sizeof(bool), D2H); CUERR;
        //cudaDeviceSynchronize();

        //if(hflag[0]){
        //    break;
        //}
    }
    std::exit(0);
    std::cerr << "READ CHECKING DONE\n";

    // for(read_number i = 0; i < gpurs.getNumberOfReads(); i++){
    //     std::vector<unsigned int> newRsData(8,0);
    //     gpurs.distributedSequenceData2.get(i, newRsData.data());
    //
    //     std::string sequencestring = get2BitHiLoString(newRsData.data(), 100);
    //
    //     auto oldCandidates = oldMinhasher.getCandidates(sequencestring,
    //                                                         1,
    //                                                         3200,
    //                                                         50 * 2.5);
    //
    //     auto newCandidates = newMinhasher.getCandidates(sequencestring,
    //                                                         1,
    //                                                         3200,
    //                                                         50 * 2.5);
    //
    //     if(oldCandidates != newCandidates){
    //         std::cerr << "Error candidates read " << i << "\n";
    //         std::copy(oldCandidates.begin(), oldCandidates.end(), std::ostream_iterator<read_number>(std::cerr, " "));
    //         std::cerr << "\n";
    //         std::copy(newCandidates.begin(), newCandidates.end(), std::ostream_iterator<read_number>(std::cerr, " "));
    //         std::cerr << "\n";
    //         assert(oldCandidates == newCandidates);
    //     }
    //     assert(oldCandidates == newCandidates);
    // }
    //
    // std::cerr << "CANDIDATE CHECKING DONE\n";
}


void performCorrection_gpu(MinhashOptions minhashOptions,
			AlignmentOptions alignmentOptions,
			CorrectionOptions correctionOptions,
			RuntimeOptions runtimeOptions,
			FileOptions fileOptions,
			GoodAlignmentProperties goodAlignmentProperties){

	filesys::create_directories(fileOptions.outputdirectory);

	std::vector<char> readIsCorrectedVector;
	std::size_t nLocksForProcessedFlags = runtimeOptions.nCorrectorThreads * 1000;
	std::unique_ptr<std::mutex[]> locksForProcessedFlags(new std::mutex[nLocksForProcessedFlags]);

	auto thread_id = std::this_thread::get_id();
	std::string thread_id_string;
	{
		std::stringstream ss;
		ss << thread_id;
		thread_id_string = ss.str();
	}

	FileOptions iterFileOptions = fileOptions;

	iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";

    std::cout << "loading file and building data structures..." << std::endl;

    TIMERSTARTCPU(load_and_build);

    BuiltGpuDataStructures dataStructuresgpu = buildGpuDataStructures(minhashOptions,
                                                                    correctionOptions,
                                                                    runtimeOptions,
                                                                    fileOptions);

    TIMERSTOPCPU(load_and_build);

    // {
    //     BuiltDataStructures dataStructurescpu = buildDataStructures(minhashOptions,
    //                                                                     correctionOptions,
    //                                                                     runtimeOptions,
    //                                                                     fileOptions);
    //
    //     checkBuiltDataStructures(dataStructuresgpu, dataStructurescpu);
    //
    // }

    auto& readStorage = dataStructuresgpu.builtReadStorage.data;
    auto& minhasher = dataStructuresgpu.builtMinhasher.data;
    auto& sequenceFileProperties = dataStructuresgpu.sequenceFileProperties;

    saveReadStorageToFile(readStorage, iterFileOptions);
    saveMinhasherToFile(minhasher, iterFileOptions);

    printFileProperties(fileOptions.inputfile, sequenceFileProperties);

    TIMERSTARTCPU(candidateestimation);
    std::uint64_t maxCandidatesPerRead = runtimeOptions.max_candidates;

    if(maxCandidatesPerRead == 0){
        // maxCandidatesPerRead = cpu::calculateMaxCandidatesPerReadThreshold(minhasher,
        //                                         readStorage,
        //                                         sequenceFileProperties.nReads / 10,
        //                                         correctionOptions.hits_per_candidate,
        //                                         runtimeOptions.threads
        //                                         //,"ncandidates.txt"
        //                                         );
        assert(maxCandidatesPerRead != 0);
        std::cout << "maxCandidates option not specified. Using estimation: " << maxCandidatesPerRead << std::endl;
    }



    TIMERSTOPCPU(candidateestimation);

    readIsCorrectedVector.resize(sequenceFileProperties.nReads, 0);

    std::cerr << "readIsCorrectedVector bytes: " << readIsCorrectedVector.size() / 1024. / 1024. << " MB\n";

    printDataStructureMemoryUsage(minhasher, readStorage);

    correct_gpu2(minhashOptions, alignmentOptions,
                        goodAlignmentProperties, correctionOptions,
                        runtimeOptions, iterFileOptions, sequenceFileProperties,
                        minhasher, readStorage,
                        maxCandidatesPerRead,
                        readIsCorrectedVector, locksForProcessedFlags,
                        nLocksForProcessedFlags);

    TIMERSTARTCPU(finalizing_files);

	std::string toRename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
	std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

    //rename feature file
    if(correctionOptions.extractFeatures){
        std::string tmpfeaturename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even_features";
        std::string outputfeaturename = fileOptions.outputfile + "_features";

		std::rename(tmpfeaturename.c_str(), outputfeaturename.c_str());
    }

    TIMERSTOPCPU(finalizing_files);
}


}
}
