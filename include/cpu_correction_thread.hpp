#ifndef CARE_GPU_CPU_CORRECTION_THREAD_HPP
#define CARE_GPU_CPU_CORRECTION_THREAD_HPP

#include <config.hpp>

#include <options.hpp>
#include <minhasher.hpp>
#include <readstorage.hpp>
#include <cpu_alignment.hpp>
#include <bestalignment.hpp>
#include <rangegenerator.hpp>

#include <cpu_correction_core.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include <vector>

namespace care{
namespace cpu{

    /*template<bool indels>
    struct alignment_result_type;

    template<>
    struct alignment_result_type<true>{using type = SGAResult;};

    template<>
    struct alignment_result_type<false>{using type = SHDResult;};*/

    struct CPUCorrectionThread{

        struct CorrectionTask{
            CorrectionTask(){}

            CorrectionTask(read_number readId)
                :   active(true),
                    corrected(false),
                    readId(readId)
                    {}

            CorrectionTask(const CorrectionTask& other)
                : active(other.active),
                corrected(other.corrected),
                readId(other.readId),
                candidate_read_ids(other.candidate_read_ids),
                candidate_read_ids_begin(other.candidate_read_ids_begin),
                candidate_read_ids_end(other.candidate_read_ids_end),
                original_subject_string(other.original_subject_string),
                subject_string(other.subject_string),
                clipping_begin(other.clipping_begin),
                clipping_end(other.clipping_end),
                corrected_subject(other.corrected_subject),
                corrected_candidates(other.corrected_candidates),
                corrected_candidates_read_ids(other.corrected_candidates_read_ids){

                candidate_read_ids_begin = &(candidate_read_ids[0]);
                candidate_read_ids_end = &(candidate_read_ids[candidate_read_ids.size()]);

            }

            CorrectionTask(CorrectionTask&& other){
                operator=(other);
            }

            CorrectionTask& operator=(const CorrectionTask& other){
                CorrectionTask tmp(other);
                swap(*this, tmp);
                return *this;
            }

            CorrectionTask& operator=(CorrectionTask&& other){
                swap(*this, other);
                return *this;
            }

            friend void swap(CorrectionTask& l, CorrectionTask& r) noexcept{
                using std::swap;

                swap(l.active, r.active);
                swap(l.corrected, r.corrected);
                swap(l.readId, r.readId);
                swap(l.original_subject_string, r.original_subject_string);
                swap(l.subject_string, r.subject_string);
                swap(l.candidate_read_ids, r.candidate_read_ids);
                swap(l.candidate_read_ids_begin, r.candidate_read_ids_begin);
                swap(l.candidate_read_ids_end, r.candidate_read_ids_end);
                swap(l.clipping_begin, r.clipping_begin);
                swap(l.clipping_end, r.clipping_end);
                swap(l.corrected_subject, r.corrected_subject);
                swap(l.corrected_candidates_read_ids, r.corrected_candidates_read_ids);
            }

            bool active;
            bool corrected;
            read_number readId;

            std::vector<read_number> candidate_read_ids;
            read_number* candidate_read_ids_begin;
            read_number* candidate_read_ids_end; // exclusive

            std::string original_subject_string;
            std::string subject_string;

            int clipping_begin;
            int clipping_end;

            std::string corrected_subject;
            std::vector<std::string> corrected_candidates;
            std::vector<read_number> corrected_candidates_read_ids;
        };

        using Minhasher_t = Minhasher;
        using ReadStorage_t = cpu::ContiguousReadStorage;
    	using Sequence_t = typename ReadStorage_t::Sequence_t;
        using AlignmentResult_t = SHDResult;//typename alignment_result_type<indels>::type;

        using CorrectionTask_t = CorrectionTask;
        using RangeGenerator_t = RangeGenerator<read_number>;

    	struct CorrectionThreadOptions{
    		int threadId;

    		std::string outputfile;
    		RangeGenerator_t* readIdGenerator;
    		const Minhasher_t* minhasher;
    		const ReadStorage_t* readStorage;
    		std::mutex* coutLock;
    		std::vector<char>* readIsCorrectedVector;
    		std::mutex* locksForProcessedFlags;
    		std::size_t nLocksForProcessedFlags;
    	};

        AlignmentOptions alignmentOptions;
        GoodAlignmentProperties goodAlignmentProperties;
        CorrectionOptions correctionOptions;
        CorrectionThreadOptions threadOpts;
        FileOptions fileOptions;

        SequenceFileProperties fileProperties;

        std::uint64_t max_candidates;


        std::uint64_t nProcessedReads = 0;

        std::uint64_t minhashcandidates = 0;
        std::uint64_t duplicates = 0;
    	int nProcessedQueries = 0;
    	int nCorrectedCandidates = 0; // candidates which were corrected in addition to query correction.

        int avgsupportfail = 0;
    	int minsupportfail = 0;
    	int mincoveragefail = 0;
    	int sobadcouldnotcorrect = 0;
    	int verygoodalignment = 0;

        std::map<int, int> numCandidatesOfUncorrectedSubjects;

        std::chrono::duration<double> getCandidatesTimeTotal;
        std::chrono::duration<double> copyCandidateDataToBufferTimeTotal;
    	std::chrono::duration<double> getAlignmentsTimeTotal;
        std::chrono::duration<double> findBestAlignmentDirectionTimeTotal;
        std::chrono::duration<double> gatherBestAlignmentDataTimeTotal;
        std::chrono::duration<double> mismatchRatioFilteringTimeTotal;
        std::chrono::duration<double> compactBestAlignmentDataTimeTotal;
        std::chrono::duration<double> fetchQualitiesTimeTotal;
        std::chrono::duration<double> makeCandidateStringsTimeTotal;
        std::chrono::duration<double> msaAddSequencesTimeTotal;
        std::chrono::duration<double> msaFindConsensusTimeTotal;
        std::chrono::duration<double> msaMinimizationTimeTotal;
        std::chrono::duration<double> msaCorrectSubjectTimeTotal;
        std::chrono::duration<double> msaCorrectCandidatesTimeTotal;
        std::chrono::duration<double> correctWithFeaturesTimeTotal;




        std::thread thread;
        bool isRunning = false;
        volatile bool stopAndAbort = false;

        void run();

        void join();

    private:

    	void execute();
    };

}
}




#endif
