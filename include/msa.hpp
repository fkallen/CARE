#ifndef CARE_CPU_MSA_HPP
#define CARE_CPU_MSA_HPP

#include <config.hpp>

#include <util.hpp>
#include <qualityscoreweights.hpp>

#include <string>
#include <cassert>
#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <array>
#include <iostream>

namespace care{
    



struct MultipleSequenceAlignment{

public:

    std::vector<char> consensus;
    std::vector<float> support;
    std::vector<int> coverage;
    std::vector<float> origWeights;
    std::vector<int> origCoverages;

    std::vector<int> countsA;
    std::vector<int> countsC;
    std::vector<int> countsG;
    std::vector<int> countsT;

    std::vector<float> weightsA;
    std::vector<float> weightsC;
    std::vector<float> weightsG;
    std::vector<float> weightsT;

    int nCandidates;
    int nColumns;

    int subjectColumnsBegin_incl;
    int subjectColumnsEnd_excl;

    void build(const char* subject,
                int subjectLength,
                const char* candidates,
                const int* candidateLengths,
                int nCandidates_,
                const int* candidateShifts,
                const float* candidateDefaultWeightFactors,
                const char* subjectQualities,
                const char* candidateQualities,
                size_t candidatesPitch,
                size_t candidateQualitiesPitch,
                bool useQualityScores);

    void resize(int cols);

    void fillzero();

    void findConsensus();

    void findOrigWeightAndCoverage(const char* subject);

    void addSequence(bool useQualityScores, const char* sequence, const char* quality, int length, int shift, float defaultWeightFactor);

    //void removeSequence(bool useQualityScores, const char* sequence, const char* quality, int length, int shift, float defaultWeightFactor);

};

struct MSAProperties{
    float avg_support;
    float min_support;
    int max_coverage;
    int min_coverage;
    bool isHQ;
    bool failedAvgSupport;
    bool failedMinSupport;
    bool failedMinCoverage;
};

struct CorrectionResult{
    std::string correctedSequence;
    bool isCorrected;
};

struct CorrectedCandidate{
    int index;
    std::string sequence;
    CorrectedCandidate() noexcept{}
    CorrectedCandidate(int index, const std::string& sequence) noexcept
        : index(index), sequence(sequence){}
};

struct RegionSelectionResult{
    bool performedMinimization = false;
    std::vector<bool> differentRegionCandidate;

    int column = 0;
    char significantBase = 'F';
    char consensusBase = 'F';
    char originalBase = 'F';
    int significantCount = 0;
    int consensuscount = 0;
};

MSAProperties getMSAProperties(const float* support,
                            const int* coverage,
                            int nColumns,
                            float estimatedErrorrate,
                            float estimatedCoverage,
                            float m_coverage);

CorrectionResult getCorrectedSubject(const char* consensus,
                                    const float* support,
                                    const int* coverage,
                                    const int* originalCoverage,
                                    int nColumns,
                                    const char* subject,
                                    bool isHQ,
                                    float estimatedErrorrate,
                                    float estimatedCoverage,
                                    float m_coverage,
                                    int neighborRegionSize);

std::vector<CorrectedCandidate> getCorrectedCandidates(const char* consensus,
                                    const float* support,
                                    const int* coverage,
                                    int nColumns,
                                    int subjectColumnsBegin_incl,
                                    int subjectColumnsEnd_excl,
                                    const int* candidateShifts,
                                    const int* candidateLengths,
                                    int nCandidates,
                                    float estimatedErrorrate,
                                    float estimatedCoverage,
                                    float m_coverage,
                                    int new_columns_to_correct);

RegionSelectionResult findCandidatesOfDifferentRegion(const char* subject,
                                                    int subjectLength,
                                                    const char* candidates,
                                                    const int* candidateLengths,
                                                    int nCandidates,
                                                    size_t candidatesPitch,
                                                    const char* consensus,
                                                    const int* countsA,
                                                    const int* countsC,
                                                    const int* countsG,
                                                    const int* countsT,
                                                    const float* weightsA,
                                                    const float* weightsC,
                                                    const float* weightsG,
                                                    const float* weightsT,
                                                    int subjectColumnsBegin_incl,
                                                    int subjectColumnsEnd_excl,
                                                    const int* candidateShifts,
                                                    int dataset_coverage);

std::pair<int,int> findGoodConsensusRegionOfSubject(const char* subject,
                                                    int subjectLength,
                                                    const char* consensus,
                                                    const int* candidateShifts,
                                                    const int* candidateLengths,
                                                    int nCandidates);

std::pair<int,int> findGoodConsensusRegionOfSubject2(const char* subject,
                                                    int subjectLength,
                                                    const int* coverage,
                                                    int nColumns,
                                                    int subjectColumnsEnd_excl);




extern cpu::QualityScoreConversion qualityConversion;


void printSequencesInMSA(std::ostream& out,
                         const char* subject,
                         int subjectLength,
                         const char* candidates,
                         const int* candidateLengths,
                         int nCandidates,
                         const int* candidateShifts,
                         int subjectColumnsBegin_incl,
                         int subjectColumnsEnd_excl,
                         int nColumns,
                         size_t candidatesPitch);


}





#endif
