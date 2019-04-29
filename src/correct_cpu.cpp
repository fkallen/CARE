#include <correct_cpu.hpp>

#include <config.hpp>

#include "options.hpp"

#include "graph.hpp"

#include "sequence.hpp"
#include "sequencefileio.hpp"
#include "qualityscoreweights.hpp"
#include "tasktiming.hpp"

#include "featureextractor.hpp"
#include "bestalignment.hpp"
#include "readstorage.hpp"
#include "minhasher.hpp"
#include "rangegenerator.hpp"


#include <array>
#include <cstdint>
#include <sstream>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>
#include <thread>
#include <future>
#include <chrono>


#include "cpu_correction_thread.hpp"

#include "candidatedistribution.hpp"

namespace care{
namespace cpu{

    void correct_cpu(const MinhashOptions& minhashOptions,
    				  const AlignmentOptions& alignmentOptions,
    				  const GoodAlignmentProperties& goodAlignmentProperties,
    				  const CorrectionOptions& correctionOptions,
    				  const RuntimeOptions& runtimeOptions,
    				  const FileOptions& fileOptions,
                      const SequenceFileProperties& sequenceFileProperties,
                      Minhasher& minhasher,
                      cpu::ContiguousReadStorage& readStorage,
    				  std::vector<char>& readIsCorrectedVector,
    				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
    				  std::size_t nLocksForProcessedFlags){

        std::cout << "correct_cpu" << std::endl;

        using CPUErrorCorrectionThread_t = cpu::CPUCorrectionThread;

    #if 1
        const int nCorrectorThreads = runtimeOptions.nCorrectorThreads;
    #else
    	const int nCorrectorThreads = 1;
    #endif

    	std::cout << "Using " << nCorrectorThreads << " corrector threads" << std::endl;

          // initialize qscore-to-weight lookup table
      	//cpu::init_weights();

        //SequenceFileProperties sequenceFileProperties = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

        /*
            Make candidate statistics
        */

        std::uint64_t max_candidates = runtimeOptions.max_candidates;
        //std::uint64_t max_candidates = std::numeric_limits<std::uint64_t>::max();

        if(max_candidates == 0){
            std::cout << "estimating candidate cutoff" << std::endl;

            Dist<std::int64_t, std::int64_t> candidateDistribution;
            cpu::Dist2<std::int64_t, std::int64_t> candidateDistribution2;

            {
                TIMERSTARTCPU(candidateestimation);
                std::map<std::int64_t, std::int64_t> candidateHistogram
                        = getCandidateCountHistogram(minhasher,
                                                    readStorage,
                                                    sequenceFileProperties.nReads / 10,
                                                    correctionOptions.hits_per_candidate,
                                                    runtimeOptions.threads);

                TIMERSTOPCPU(candidateestimation);

                candidateDistribution = estimateDist(candidateHistogram);
                //candidateDistribution2 = cpu::estimateDist2(candidateHistogram);

                std::vector<std::pair<std::int64_t, std::int64_t>> vec(candidateHistogram.begin(), candidateHistogram.end());
                std::sort(vec.begin(), vec.end(), [](auto p1, auto p2){ return p1.second < p2.second;});

                std::ofstream of("ncandidates.txt");
                for(const auto& p : vec)
                    of << p.first << " " << p.second << '\n';
                of.flush();
            }

            std::cout << "candidates.max " << candidateDistribution.max << std::endl;
            std::cout << "candidates.average " << candidateDistribution.average << std::endl;
            std::cout << "candidates.stddev " << candidateDistribution.stddev << std::endl;

            const std::uint64_t estimatedMeanAlignedCandidates = candidateDistribution.max;
            const std::uint64_t estimatedDeviationAlignedCandidates = candidateDistribution.stddev;
            const std::uint64_t estimatedAlignmentCountThreshold = estimatedMeanAlignedCandidates
                                                            + 2.5 * estimatedDeviationAlignedCandidates;

            max_candidates = estimatedAlignmentCountThreshold;
            //max_candidates = candidateDistribution2.percentRanges[90].first;
            //std::exit(0);
        }

        std::cout << "Using candidate cutoff: " << max_candidates << std::endl;

        /*
            Spawn correction threads
        */

        std::vector<std::string> tmpfiles;
        for(int i = 0; i < nCorrectorThreads; i++){
            tmpfiles.emplace_back(fileOptions.outputfile + "_tmp_" + std::to_string(1000 + i));
        }

        cpu::RangeGenerator<read_number> readIdGenerator(sequenceFileProperties.nReads);

        NN_Correction_Classifier_Base nnClassifierBase;//{"./nn_sources", fileOptions.nnmodelfilename};

        std::vector<CPUErrorCorrectionThread_t> cpucorrectorThreads(nCorrectorThreads);
        std::mutex writelock;

    	for(int threadId = 0; threadId < nCorrectorThreads; threadId++){

            //cpubatchgenerators[threadId] = BatchGenerator<read_number>(ncpuReads, 1, threadId, nCpuThreads);
            typename CPUErrorCorrectionThread_t::CorrectionThreadOptions threadOpts;
            threadOpts.threadId = threadId;

            threadOpts.outputfile = tmpfiles[threadId];
            threadOpts.readIdGenerator = &readIdGenerator;
            threadOpts.minhasher = &minhasher;
            threadOpts.readStorage = &readStorage;
            threadOpts.coutLock = &writelock;
            threadOpts.readIsCorrectedVector = &readIsCorrectedVector;
            threadOpts.locksForProcessedFlags = locksForProcessedFlags.get();
            threadOpts.nLocksForProcessedFlags = nLocksForProcessedFlags;

            cpucorrectorThreads[threadId].alignmentOptions = alignmentOptions;
            cpucorrectorThreads[threadId].goodAlignmentProperties = goodAlignmentProperties;
            cpucorrectorThreads[threadId].correctionOptions = correctionOptions;
            cpucorrectorThreads[threadId].fileOptions = fileOptions;
            cpucorrectorThreads[threadId].threadOpts = threadOpts;
            cpucorrectorThreads[threadId].fileProperties = sequenceFileProperties;
            cpucorrectorThreads[threadId].max_candidates = max_candidates;

            cpucorrectorThreads[threadId].classifierBase = &nnClassifierBase;

            cpucorrectorThreads[threadId].run();
        }

        std::cout << "Correcting..." << std::endl;

        bool showProgress = runtimeOptions.showProgress;

        std::thread progressThread = std::thread([&]() -> void{
            if(!showProgress)
                return;

            std::chrono::time_point<std::chrono::system_clock> timepoint_begin = std::chrono::system_clock::now();
            std::chrono::duration<double> runtime = std::chrono::seconds(0);
            std::chrono::duration<int> sleepinterval = std::chrono::seconds(1);

            while(showProgress){
                read_number progress = readIdGenerator.getCurrentUnsafe() - readIdGenerator.getBegin();

                printf("Progress: %3.2f %% %10u %10lu (Runtime: %03d:%02d:%02d)\r",
                        ((progress * 1.0 / sequenceFileProperties.nReads) * 100.0),
                        progress, sequenceFileProperties.nReads,
                        int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
                        int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count()) % 60,
                        int(runtime.count()) % 60);
                std::cout << std::flush;

                if(progress < sequenceFileProperties.nReads){
                      std::this_thread::sleep_for(sleepinterval);
                      runtime = std::chrono::system_clock::now() - timepoint_begin;
                }
            }
        });

    TIMERSTARTCPU(correction);

        for (auto& thread : cpucorrectorThreads)
            thread.join();

        showProgress = false;
        progressThread.join();
        if(runtimeOptions.showProgress)
            printf("Progress: %3.2f %%\n", 100.00);

    TIMERSTOPCPU(correction);

    std::chrono::duration<double> getCandidatesTimeTotal{0};
    std::chrono::duration<double> copyCandidateDataToBufferTimeTotal{0};
    std::chrono::duration<double> getAlignmentsTimeTotal{0};
    std::chrono::duration<double> findBestAlignmentDirectionTimeTotal{0};
    std::chrono::duration<double> gatherBestAlignmentDataTimeTotal{0};
    std::chrono::duration<double> mismatchRatioFilteringTimeTotal{0};
    std::chrono::duration<double> compactBestAlignmentDataTimeTotal{0};
    std::chrono::duration<double> fetchQualitiesTimeTotal{0};
    std::chrono::duration<double> makeCandidateStringsTimeTotal{0};
    std::chrono::duration<double> msaAddSequencesTimeTotal{0};
    std::chrono::duration<double> msaFindConsensusTimeTotal{0};
    std::chrono::duration<double> msaMinimizationTimeTotal{0};
    std::chrono::duration<double> msaCorrectSubjectTimeTotal{0};
    std::chrono::duration<double> msaCorrectCandidatesTimeTotal{0};
    std::chrono::duration<double> correctWithFeaturesTimeTotal{0};

    for(const auto& cput : cpucorrectorThreads){
        getCandidatesTimeTotal += cput.getCandidatesTimeTotal;
        copyCandidateDataToBufferTimeTotal += cput.copyCandidateDataToBufferTimeTotal;
        getAlignmentsTimeTotal += cput.getAlignmentsTimeTotal;
        findBestAlignmentDirectionTimeTotal += cput.findBestAlignmentDirectionTimeTotal;
        gatherBestAlignmentDataTimeTotal += cput.gatherBestAlignmentDataTimeTotal;
        mismatchRatioFilteringTimeTotal += cput.mismatchRatioFilteringTimeTotal;
        compactBestAlignmentDataTimeTotal += cput.compactBestAlignmentDataTimeTotal;
        fetchQualitiesTimeTotal += cput.fetchQualitiesTimeTotal;
        makeCandidateStringsTimeTotal += cput.makeCandidateStringsTimeTotal;
        msaAddSequencesTimeTotal += cput.msaAddSequencesTimeTotal;
        msaFindConsensusTimeTotal += cput.msaFindConsensusTimeTotal;
        msaMinimizationTimeTotal += cput.msaMinimizationTimeTotal;
        msaCorrectSubjectTimeTotal += cput.msaCorrectSubjectTimeTotal;
        msaCorrectCandidatesTimeTotal += cput.msaCorrectCandidatesTimeTotal;
        correctWithFeaturesTimeTotal += cput.correctWithFeaturesTimeTotal;
    }

    getCandidatesTimeTotal /= cpucorrectorThreads.size();
    copyCandidateDataToBufferTimeTotal /= cpucorrectorThreads.size();
    getAlignmentsTimeTotal /= cpucorrectorThreads.size();
    findBestAlignmentDirectionTimeTotal /= cpucorrectorThreads.size();
    gatherBestAlignmentDataTimeTotal /= cpucorrectorThreads.size();
    mismatchRatioFilteringTimeTotal /= cpucorrectorThreads.size();
    compactBestAlignmentDataTimeTotal /= cpucorrectorThreads.size();
    fetchQualitiesTimeTotal /= cpucorrectorThreads.size();
    makeCandidateStringsTimeTotal /= cpucorrectorThreads.size();
    msaAddSequencesTimeTotal /= cpucorrectorThreads.size();
    msaFindConsensusTimeTotal /= cpucorrectorThreads.size();
    msaMinimizationTimeTotal /= cpucorrectorThreads.size();
    msaCorrectSubjectTimeTotal /= cpucorrectorThreads.size();
    msaCorrectCandidatesTimeTotal /= cpucorrectorThreads.size();
    correctWithFeaturesTimeTotal /= cpucorrectorThreads.size();

    std::chrono::duration<double> totalDuration = getCandidatesTimeTotal
                                                + copyCandidateDataToBufferTimeTotal
                                                + getAlignmentsTimeTotal
                                                + findBestAlignmentDirectionTimeTotal
                                                + gatherBestAlignmentDataTimeTotal
                                                + mismatchRatioFilteringTimeTotal
                                                + compactBestAlignmentDataTimeTotal
                                                + fetchQualitiesTimeTotal
                                                + makeCandidateStringsTimeTotal
                                                + msaAddSequencesTimeTotal
                                                + msaFindConsensusTimeTotal
                                                + msaMinimizationTimeTotal
                                                + msaCorrectSubjectTimeTotal
                                                + msaCorrectCandidatesTimeTotal
                                                + correctWithFeaturesTimeTotal;

    auto printDuration = [&](const auto& name, const auto& duration){
        std::cout << "# elapsed time ("<< name << "): "
                  << duration.count()  << " s. "
                  << (100.0 * duration / totalDuration) << " %."<< std::endl;
    };

    #define printme(x) printDuration((#x),(x));

    printme(getCandidatesTimeTotal);
    printme(copyCandidateDataToBufferTimeTotal);
    printme(getAlignmentsTimeTotal);
    printme(findBestAlignmentDirectionTimeTotal);
    printme(gatherBestAlignmentDataTimeTotal);
    printme(mismatchRatioFilteringTimeTotal);
    printme(compactBestAlignmentDataTimeTotal);
    printme(fetchQualitiesTimeTotal);
    printme(makeCandidateStringsTimeTotal);
    printme(msaAddSequencesTimeTotal);
    printme(msaFindConsensusTimeTotal);
    printme(msaMinimizationTimeTotal);
    printme(msaCorrectSubjectTimeTotal);
    printme(msaCorrectCandidatesTimeTotal);
    printme(correctWithFeaturesTimeTotal);

    #undef printme



        //std::cout << "threads done" << std::endl;

        minhasher.destroy();
    	readStorage.destroy();

        std::cout << "begin merge" << std::endl;

        if(!correctionOptions.extractFeatures){

            std::cout << "begin merging reads" << std::endl;

            TIMERSTARTCPU(merge);

            mergeResultFiles(sequenceFileProperties.nReads, fileOptions.inputfile, fileOptions.format, tmpfiles, fileOptions.outputfile);

            TIMERSTOPCPU(merge);

            std::cout << "end merging reads" << std::endl;

        }

        deleteFiles(tmpfiles);

        std::vector<std::string> featureFiles(tmpfiles);
        for(auto& s : featureFiles)
            s = s + "_features";

        //concatenate feature files of each thread into one file

        if(correctionOptions.extractFeatures){
            std::cout << "begin merging features" << std::endl;

            std::stringstream commandbuilder;

            commandbuilder << "cat";

            for(const auto& featureFile : featureFiles){
                commandbuilder << " \"" << featureFile << "\"";
            }

            commandbuilder << " > \"" << fileOptions.outputfile << "_features\"";

            const std::string command = commandbuilder.str();
            TIMERSTARTCPU(concat_feature_files);
            int r1 = std::system(command.c_str());
            TIMERSTOPCPU(concat_feature_files);

            if(r1 != 0){
                std::cerr << "Warning. Feature files could not be concatenated!\n";
                std::cerr << "This command returned a non-zero error value: \n";
                std::cerr << command +  '\n';
                std::cerr << "Please concatenate the following files manually\n";
                for(const auto& s : featureFiles)
                    std::cerr << s << '\n';
            }else{
                deleteFiles(featureFiles);
            }

            std::cout << "end merging features" << std::endl;
        }else{
            deleteFiles(featureFiles);
        }

        std::cout << "end merge" << std::endl;
    }


}
}
