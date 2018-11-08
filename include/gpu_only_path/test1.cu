#include "correct_only_gpu.hpp"

#include "../minhasher.hpp"
#include "../readstorage.hpp"
#include "../sequence.hpp"
#include "../options.hpp"

#include <vector>

using Key_t = std::uint32_t; // asume minhashOptions.k <= 16
using ReadId_t = std::uint32_t; // asume nReads <= std::numeric_limits<std::uint32_t>::max()

using Minhasher_t = care::Minhasher<Key_t, ReadId_t>;
using NoIndelSequence_t = care::Sequence2Bit;
using ReadStorage_t = care::ReadStorageMinMemory<NoIndelSequence_t, ReadId_t>;


struct MinhasherDummy{
	
	std::vector<ReadId_t> getCandidates(const std::string& sequence, std::uint64_t max_number_candidates) const noexcept{
		
		if(sequence == "AGCAGTTAACCGGTGCACCGCCATACAGTTGGGTTTGATCCGGATCGACCACCACAATATCCACCAGATAACCCGGAATACGGACAGATTTAGGATGCAGC"){
				return {0,1,2,3,4,5,6,7,8,9};
		}
		
		return {};
	}
	
};

struct ReadStorageDummy : public ReadStorage_t{
	
	ReadStorageDummy() : ReadStorageDummy(false){
		
	}
	
	ReadStorageDummy(bool q) : ReadStorage_t(q){
		init(10);
		
		std::string s0 = "AGCAGTTAACCGGTGCACCGCCATACAGTTGGGTTTGATCCGGATCGACCACCACAATATCCACCAGATAACCCGGAATACGGACAGATTTAGGATGCAGC";
		std::string s1 = "AGCAGTTAACCGGTGCACCGCCATACAGTTGGGTATGATCCGGATCGACCACCACAATATCCACCAGATAACCCGGAATACGGACAGATTTAGGATGCAGC";
		std::string q0 = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
		std::string q1 = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
		
		insertRead(0, s0, q0);
		insertRead(1, s1, q1);
		insertRead(2, s1, q1);
		insertRead(3, s1, q1);
		insertRead(4, s1, q1);
		insertRead(5, s1, q1);
		insertRead(6, s1, q1);
		insertRead(7, s1, q1);
		insertRead(8, s1, q1);
		insertRead(9, s1, q1);
		
	}
	
};


int main(){
	
	
	/*
	 * Options
	 */
	
	care::RuntimeOptions runtimeOptions;
	runtimeOptions.threads = 1;
	runtimeOptions.nInserterThreads = 4;
	runtimeOptions.nCorrectorThreads = 1;
	runtimeOptions.showProgress = true;
	runtimeOptions.canUseGpu = true;
	runtimeOptions.max_candidates = 0;
	runtimeOptions.deviceIds = {0};
	
	care::AlignmentOptions alignmentOptions;
	
	care::GoodAlignmentProperties goodAlignmentProperties;
	
	goodAlignmentProperties.min_overlap = 30;
    goodAlignmentProperties.maxErrorRate = 0.2;
    goodAlignmentProperties.min_overlap_ratio = 0.30;
	
	care::CorrectionOptions correctionOptions;
	
	correctionOptions.correctionMode = care::CorrectionMode::Hamming;
    correctionOptions.correctCandidates = false;
    correctionOptions.useQualityScores = false;
    correctionOptions.estimatedCoverage = 1.0;
    correctionOptions.estimatedErrorrate = 0.01;
    correctionOptions.m_coverage = 0.6;
    correctionOptions.graphalpha = 1.0;
    correctionOptions.graphx = 1.5;
    correctionOptions.kmerlength = 16;
	correctionOptions.batchsize = 1;
    correctionOptions.new_columns_to_correct = 0;
    correctionOptions.extractFeatures = false;
    correctionOptions.classicMode = true;
	
	care::SequenceFileProperties fileProperties;
	fileProperties.nReads = 10;
	fileProperties.minSequenceLength = 101;
	fileProperties.maxSequenceLength = 101;

	
	using CorThread = care::gpu::ErrorCorrectionThreadOnlyGPU<MinhasherDummy, ReadStorageDummy, care::gpu::BatchGenerator<ReadId_t>>;
	//using CorThread = care::gpu::ErrorCorrectionThreadOnlyGPU<MinhasherDummy, ReadStorageDummy>;
	
	
	MinhasherDummy minhasher;
	ReadStorageDummy readStorage;
	care::gpu::BatchGenerator<ReadId_t> batchGen(0, 1, 1);
	
	/*void build(const FileOptions& fileOptions,
			   const RuntimeOptions& runtimeOptions,
			   ReadStorage_t& readStorage,
			   Minhasher_t& minhasher)*/


	std::mutex coutLock;
	
	std::vector<char> readIsProcessedVector(1000000);
	std::vector<char> readIsCorrectedVector(1000000);
	std::vector<std::mutex> locksForProcessedFlags(1000000);
	std::size_t nLocksForProcessedFlags = 1000000;
	
	
		
	CorThread::CorrectionThreadOptions correctionThreadOptions;
	
	correctionThreadOptions.threadId = 0;
	correctionThreadOptions.deviceId = 0;
	correctionThreadOptions.outputfile = "testout.txt";
	correctionThreadOptions.batchGen = &batchGen;
	correctionThreadOptions.minhasher = &minhasher;
	correctionThreadOptions.readStorage = &readStorage;
	correctionThreadOptions.coutLock = &coutLock;
	correctionThreadOptions.readIsProcessedVector = &readIsProcessedVector;
	correctionThreadOptions.readIsCorrectedVector = &readIsCorrectedVector;
	correctionThreadOptions.locksForProcessedFlags = locksForProcessedFlags.data();
	correctionThreadOptions.nLocksForProcessedFlags = nLocksForProcessedFlags;
	
	
	
	
	CorThread corThread;
		
	corThread.alignmentOptions = alignmentOptions; 
	corThread.goodAlignmentProperties = goodAlignmentProperties;
	corThread.correctionOptions = correctionOptions;
	corThread.threadOpts = correctionThreadOptions;
	corThread.fileProperties = fileProperties;
	corThread.max_candidates = 1000;	
	
	corThread.run();
	
	bool showProgress = true;

    std::thread progressThread = std::thread([&]() -> void{
        if(!showProgress)
            return;

        std::chrono::time_point<std::chrono::system_clock> timepoint_begin = std::chrono::system_clock::now();
        std::chrono::duration<double> runtime = std::chrono::seconds(0);
        std::chrono::duration<int> sleepinterval = std::chrono::seconds(1);

        while(showProgress){
            ReadId_t progress = 0;
            ReadId_t correctorProgress = 0;

            //for(int i = 0; i < nCorrectorThreads; i++){
                correctorProgress += corThread.nProcessedReads;
            //}

            progress = correctorProgress;

            printf("Progress: %3.2f %% %10u %10lu (Runtime: %03d:%02d:%02d)\r",
                    ((progress * 1.0 / fileProperties.nReads) * 100.0),
                    correctorProgress, fileProperties.nReads,
                    int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
                    int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count()) % 60,
                    int(runtime.count()) % 60);
            std::cout << std::flush;

            if(progress < fileProperties.nReads){
                  std::this_thread::sleep_for(sleepinterval);
                  runtime = std::chrono::system_clock::now() - timepoint_begin;
            }
        }
    });
	
	corThread.join();

	cudaDeviceReset(); CUERR;
}
