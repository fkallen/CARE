#include "../inc/pileup.hpp"
#include "../inc/alignment.hpp"
#include "../inc/hammingtools.hpp"
#include "../inc/ganja/hpc_helpers.cuh"
#include "../inc/cudareduce.cuh"
#include "../inc/batchelem.hpp"

#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>

#ifdef __NVCC__
#include <cooperative_groups.h>
#include <cublas_v2.h>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>

using namespace cooperative_groups;
#endif

namespace hammingtools{

	namespace correction{

		constexpr int candidate_correction_new_cols = 0;

		double	qscore_to_error_prob[256];
		double	qscore_to_weight[256];

		#ifdef __NVCC__
		__device__ double d_qscore_to_weight[256];
		#endif

		void init_once(){

			constexpr int ASCII_BASE = 33;
			constexpr double MIN_WEIGHT = 0.001;

			for(int i = 0; i < 256; i++){
				if(i < ASCII_BASE)
					qscore_to_error_prob[i] = 1.0;
				else
					qscore_to_error_prob[i] = std::pow(10.0, -(i-ASCII_BASE)/10.0);
			}

			for(int i = 0; i < 256; i++){
				qscore_to_weight[i] = std::max(MIN_WEIGHT, 1.0 - qscore_to_error_prob[i]);
			}

			#ifdef __NVCC__
				int devices;
				cudaGetDeviceCount(&devices); CUERR;
				for(int i = 0; i < devices; i++){
					cudaSetDevice(i);
					cudaMemcpyToSymbol(d_qscore_to_weight, qscore_to_weight, 256*sizeof(double)); CUERR;
				}
			#endif
		}


std::chrono::duration<double>
cpu_add_weights(const CorrectionBuffers* buffers, const BatchElem& batchElem,
                const int startindex, const int endindex,
                const int columnsToCheck, const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
				const double maxErrorRate,
                const bool useQScores){

    std::chrono::duration<double> majorityvotetime(0);
    std::chrono::time_point<std::chrono::system_clock> tpa, tpb;

    tpa = std::chrono::system_clock::now();

    const int subjectlength = batchElem.fwdSequenceString.length();

    //add subject weights
    for(int i = 0; i < subjectlength; i++){
        const int globalIndex = subjectColumnsBegin_incl + i;
        double qw = 1.0;
        if(useQScores)
            qw *= qscore_to_weight[(unsigned char)(*batchElem.fwdQuality)[i]];

        const char base = batchElem.fwdSequenceString[i];
        switch(base){
            case 'A': buffers->h_Aweights[globalIndex] += qw; buffers->h_As[globalIndex] += 1; break;
            case 'C': buffers->h_Cweights[globalIndex] += qw; buffers->h_Cs[globalIndex] += 1; break;
            case 'G': buffers->h_Gweights[globalIndex] += qw; buffers->h_Gs[globalIndex] += 1; break;
            case 'T': buffers->h_Tweights[globalIndex] += qw; buffers->h_Ts[globalIndex] += 1; break;
            default: std::cout << "this A should not happen in pileup\n"; break;
        }
        buffers->h_coverage[globalIndex]++;
    }

    //add candidate weights
    
//TIMERSTARTCPU(addcandidates);    

#define WEIGHTMODE 3


#if WEIGHTMODE == 2

    for(size_t i = 0; i < batchElem.n_unique_candidates; i++){
		const auto& alignment = batchElem.bestAlignments[i];
		const std::string sequencestring = batchElem.bestSequences[i]->toString();

		const double defaultweight = 1.0 - std::sqrt(alignment.nOps / (alignment.overlap * maxErrorRate));
		const int len = sequencestring.length();
		const int freq = batchElem.candidateCountsPrefixSum[i+1] - batchElem.candidateCountsPrefixSum[i];

		for(int j = 0; j < len; j++){
			const int globalIndex = subjectColumnsBegin_incl + alignment.shift + j;
			double qw = 0.0;
			for(int f = 0; f < freq; f++){
				const std::string* scores = batchElem.bestQualities[batchElem.candidateCountsPrefixSum[i] + f];
				if(useQScores)
					qw += qscore_to_weight[(unsigned char)(*scores)[j]];
				else
					qw += 1.0;
			}
			qw *= defaultweight;
			const char base = sequencestring[j];
			switch(base){
				case 'A': buffers->h_Aweights[globalIndex] += qw; buffers->h_As[globalIndex] += freq; break;
				case 'C': buffers->h_Cweights[globalIndex] += qw; buffers->h_Cs[globalIndex] += freq; break;
				case 'G': buffers->h_Gweights[globalIndex] += qw; buffers->h_Gs[globalIndex] += freq; break;
				case 'T': buffers->h_Tweights[globalIndex] += qw; buffers->h_Ts[globalIndex] += freq; break;
				default: std::cout << "this B should not happen in pileup\n"; break;
			}
			buffers->h_coverage[globalIndex] += freq;
		}
    }


#elif WEIGHTMODE == 3

    for(size_t i = 0; i < batchElem.n_unique_candidates; i++){
//TIMERSTARTCPU(prepare);		
		const auto& alignment = batchElem.bestAlignments[i];
		//const std::string sequencestring = batchElem.bestSequences[i]->toString();

		const double defaultweight = 1.0 - std::sqrt(alignment.nOps / (alignment.overlap * maxErrorRate));
		//const int len = sequencestring.length();
		const int len = batchElem.bestSequences[i]->getNbases();
		const int freq = batchElem.candidateCountsPrefixSum[i+1] - batchElem.candidateCountsPrefixSum[i];
		const int defaultcolumnoffset = subjectColumnsBegin_incl + alignment.shift;
//TIMERSTOPCPU(prepare);
//TIMERSTARTCPU(addquality);		
		//use h_support as temporary storage to store sum of quality weights
		for(int f = 0; f < freq; f++){
			const std::string* scores = batchElem.bestQualities[batchElem.candidateCountsPrefixSum[i] + f];
			if(useQScores){
				for(int j = 0; j < len; j++){
					buffers->h_support[j] += qscore_to_weight[(unsigned char)(*scores)[j]];
				}
			}else{
				for(int j = 0; j < len; j++){
					buffers->h_support[j] += 1.0;
				}
			}
		}
//TIMERSTOPCPU(addquality);
//TIMERSTARTCPU(addbase);		
		for(int j = 0; j < len; j++){
			const int globalIndex = defaultcolumnoffset + j;
			const double qw = buffers->h_support[j] * defaultweight;
			//const char base = sequencestring[j];
			const char base = (*batchElem.bestSequences[i])[j];

			switch(base){
				case 'A': buffers->h_Aweights[globalIndex] += qw; buffers->h_As[globalIndex] += freq; break;
				case 'C': buffers->h_Cweights[globalIndex] += qw; buffers->h_Cs[globalIndex] += freq; break;
				case 'G': buffers->h_Gweights[globalIndex] += qw; buffers->h_Gs[globalIndex] += freq; break;
				case 'T': buffers->h_Tweights[globalIndex] += qw; buffers->h_Ts[globalIndex] += freq; break;
				default: std::cout << "this C should not happen in pileup\n"; break;
			}
			buffers->h_coverage[globalIndex] += freq;
			buffers->h_support[j] = 0;
		}
//TIMERSTOPCPU(addbase);		
    }

#elif
static_assert(false, "invalid WEIGHTMODE");
#endif

//TIMERSTOPCPU(addcandidates);   

    tpb = std::chrono::system_clock::now();

    majorityvotetime += tpb - tpa;
	
	return majorityvotetime;
}




void 
cpu_find_consensus(const CorrectionBuffers* buffers, const BatchElem& batchElem,
                const int columnsToCheck, const int subjectColumnsBegin_incl){

//TIMERSTARTCPU(findconsensus);
    //find consensus and support in each column
    for(int i = 0; i < columnsToCheck; i++){
        char cons = 'A';
        double consWeight = buffers->h_Aweights[i];
        if(buffers->h_Cweights[i] > consWeight){
            cons = 'C';
            consWeight = buffers->h_Cweights[i];
        }
        if(buffers->h_Gweights[i] > consWeight){
            cons = 'G';
            consWeight = buffers->h_Gweights[i];
        }
        if(buffers->h_Tweights[i] > consWeight){
            cons = 'T';
            consWeight = buffers->h_Tweights[i];
        }
        buffers->h_consensus[i] = cons;

        const double columnWeight = buffers->h_Aweights[i] + buffers->h_Cweights[i] + buffers->h_Gweights[i] + buffers->h_Tweights[i];
        buffers->h_support[i] = consWeight / columnWeight;
    }
    
    const int subjectlength = batchElem.fwdSequenceString.length();

    for(int i = 0; i < subjectlength; i++){
        const int globalIndex = subjectColumnsBegin_incl + i;
        switch(batchElem.fwdSequenceString[i]){
            case 'A':   buffers->h_origCoverage[globalIndex] = buffers->h_As[globalIndex];
                        buffers->h_origWeights[globalIndex] = buffers->h_Aweights[globalIndex];
                        break;
            case 'C':   buffers->h_origCoverage[globalIndex] = buffers->h_Cs[globalIndex];
                        buffers->h_origWeights[globalIndex] = buffers->h_Cweights[globalIndex];
                        break;
            case 'G':   buffers->h_origCoverage[globalIndex] = buffers->h_Gs[globalIndex];
                        buffers->h_origWeights[globalIndex] = buffers->h_Gweights[globalIndex];
                        break;
            case 'T':   buffers->h_origCoverage[globalIndex] = buffers->h_Ts[globalIndex];
                        buffers->h_origWeights[globalIndex] = buffers->h_Tweights[globalIndex];
                        break;
            default: std::cout << "this D should not happen in pileup\n"; break;
        }
    }
//TIMERSTOPCPU(findconsensus);

}



std::tuple<int,std::chrono::duration<double>>
cpu_correct(const CorrectionBuffers* buffers, BatchElem& batchElem,
                const int startindex, const int endindex,
                const int columnsToCheck, const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
                const double maxErrorRate,
                const bool correctQueries,
                const int estimatedCoverage,
                const double errorrate,
                const double m,
                const int k){

    std::chrono::duration<double> basecorrectiontime(0);
    std::chrono::time_point<std::chrono::system_clock> tpa, tpb;
    int status = 0;

    tpa = std::chrono::system_clock::now();

    double avg_support = 0;
    double min_support = 1.0;
    int max_coverage = 0;
    int min_coverage = std::numeric_limits<int>::max();
    //get stats for subject columns
    for(int i = subjectColumnsBegin_incl; i < subjectColumnsEnd_excl; i++){
        assert(i < columnsToCheck);

        avg_support += buffers->h_support[i];
        min_support = buffers->h_support[i] < min_support? buffers->h_support[i] : min_support;
        max_coverage = buffers->h_coverage[i] > max_coverage ? buffers->h_coverage[i] : max_coverage;
        min_coverage = buffers->h_coverage[i] < min_coverage ? buffers->h_coverage[i] : min_coverage;
    }
    const int subjectlength = batchElem.fwdSequenceString.length();
    avg_support /= subjectlength;


    batchElem.correctedSequence.resize(subjectlength);

    //TODO vary parameters
    bool isHQ = avg_support >= 1.0-errorrate
         && min_support >= 1.0-3.0*errorrate
         && min_coverage >= m / 2.0 * estimatedCoverage;

    if(isHQ){
#if 1
        //correct anchor
        for(int i = 0; i < subjectlength; i++){
            const int globalIndex = subjectColumnsBegin_incl + i;
            batchElem.correctedSequence[i] = buffers->h_consensus[globalIndex];
        }
        batchElem.corrected = true;
#endif
#if 0
        //correct candidates
        if(correctQueries){

            for(int i = 0; i < batchElem.n_unique_candidates; i++){
                int queryColumnsBegin_incl = alignments[i].shift - startindex;
                bool queryWasCorrected = false;
                //correct candidates which are shifted by at most candidate_correction_new_cols columns relative to subject
                if(queryColumnsBegin_incl >= subjectColumnsBegin_incl - candidate_correction_new_cols
                    && subjectColumnsEnd_excl + candidate_correction_new_cols >= queryColumnsBegin_incl + int(queries[i].length())){

                    double newColMinSupport = 1.0;
                    int newColMinCov = std::numeric_limits<int>::max();
                    //check new columns left of subject
                    for(int columnindex = subjectColumnsBegin_incl - candidate_correction_new_cols;
                        columnindex < subjectColumnsBegin_incl;
                        columnindex++){

                        assert(columnindex < columnsToCheck);
                        if(queryColumnsBegin_incl <= columnindex){
                            newColMinSupport = buffers->h_support[columnindex] < newColMinSupport ? buffers->h_support[columnindex] : newColMinSupport;
                            newColMinCov = buffers->h_coverage[columnindex] < newColMinCov ? buffers->h_coverage[columnindex] : newColMinCov;
                        }
                    }
                    //check new columns right of subject
                    for(int columnindex = subjectColumnsEnd_excl;
                        columnindex < subjectColumnsEnd_excl + candidate_correction_new_cols
                        && columnindex < columnsToCheck;
                        columnindex++){

                        newColMinSupport = buffers->h_support[columnindex] < newColMinSupport ? buffers->h_support[columnindex] : newColMinSupport;
                        newColMinCov = buffers->h_coverage[columnindex] < newColMinCov ? buffers->h_coverage[columnindex] : newColMinCov;
                    }

                    if(newColMinSupport >= 1-3*errorrate
                        && newColMinCov >= m / 2.0 * estimatedCoverage){
                        //assert(subjectColumnsBegin_incl == queryColumnsBegin_incl && subject.length() == queries[i].length());

                        for(int j = 0; j < int(queries[i].length()); j++){
                            int columnindex = queryColumnsBegin_incl + j;
                            queries[i][j] = buffers->h_consensus[columnindex];
                            queryWasCorrected = true;
                        }
                    }
                }
                if(queryWasCorrected){
                    correctedQueries[i] = true;
                }
            }
        }
#endif
    }else{
        if(avg_support < 1.0-errorrate)
            status |= (1 << 0);
        if(min_support < 1.0-3.0*errorrate)
            status |= (1 << 1);
        if(min_coverage < m / 2.0 * estimatedCoverage)
            status |= (1 << 2);
#if 1
        //correct anchor
//TODO vary parameters

        batchElem.correctedSequence = batchElem.fwdSequenceString;

        bool foundAColumn = false;
        for(int i = 0; i < subjectlength; i++){
            const int globalIndex = subjectColumnsBegin_incl + i;

#if 1
            if(buffers->h_support[globalIndex] >= 1.0-3.0*errorrate){
                batchElem.correctedSequence[i] = buffers->h_consensus[globalIndex];
                foundAColumn = true;
            }else{
//#else
            if(buffers->h_support[globalIndex] > 0.5 && buffers->h_origCoverage[globalIndex] < m / 2.0 * estimatedCoverage){
                double avgsupportkregion = 0;
                int c = 0;
                bool kregioncoverageisgood = true;
                for(int j = i - k/2; j <= i + k/2 && kregioncoverageisgood; j++){
                    if(j != i && j >= 0 && j < subjectlength){
                        avgsupportkregion += buffers->h_support[subjectColumnsBegin_incl + j];
                        kregioncoverageisgood &= (buffers->h_coverage[subjectColumnsBegin_incl + j] >= m / 2.0 * estimatedCoverage);
                        c++;
                    }
                }
                if(kregioncoverageisgood && avgsupportkregion / c >= 1.0-errorrate){
                    batchElem.correctedSequence[i] = buffers->h_consensus[globalIndex];
                    foundAColumn = true;
                }
            }

        }
#endif
        }

        batchElem.corrected = foundAColumn;

        if(!foundAColumn)
            status |= (1 << 3);

#endif
    }

    tpb = std::chrono::system_clock::now();

    basecorrectiontime += tpb - tpa;

    return std::tie(status, basecorrectiontime);
}















	}
}
