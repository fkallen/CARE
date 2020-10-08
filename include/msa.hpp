#ifndef CARE_CPU_MSA_HPP
#define CARE_CPU_MSA_HPP

#include <config.hpp>

#include <util.hpp>
#include <qualityscoreweights.hpp>
#include <bestalignment.hpp>
#include <string>
#include <cassert>
#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <array>
#include <iostream>

namespace care{


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
    bool isCorrected;
    std::string correctedSequence;

    void reset(){
        isCorrected = false;
        correctedSequence.clear();
    }
};

struct CorrectedCandidate{
    int index;
    int shift;
    std::string sequence;
    CorrectedCandidate() noexcept{}
    CorrectedCandidate(int index, int s, const std::string& sequence) noexcept
        : index(index), shift(s), sequence(sequence){}
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

struct MultipleSequenceAlignment{

public:

    struct InputData{
        bool useQualityScores;
        int subjectLength;
        int nCandidates;
        size_t candidatesPitch;
        size_t candidateQualitiesPitch;
        const char* subject;
        const char* candidates;
        const char* subjectQualities;
        const char* candidateQualities;
        const int* candidateLengths;
        const int* candidateShifts;
        const float* candidateDefaultWeightFactors;
    };

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


    InputData inputData;


    void build(const InputData& args);

    void resize(int cols);

    void fillzero();

    void findConsensus();

    void findOrigWeightAndCoverage(const char* subject);

    void addSequence(bool useQualityScores, const char* sequence, const char* quality, int length, int shift, float defaultWeightFactor);

    //void removeSequence(bool useQualityScores, const char* sequence, const char* quality, int length, int shift, float defaultWeightFactor);

    void print(std::ostream& os) const;
    void printWithDiffToConsensus(std::ostream& os) const;

    MSAProperties getMSAProperties(
        int firstCol,
        int lastCol, //exclusive
        float estimatedErrorrate,
        float estimatedCoverage,
        float m_coverage
    ) const;

    CorrectionResult getCorrectedSubject(
        MSAProperties msaProperties,
        float estimatedErrorrate,
        float estimatedCoverage,
        float m_coverage,
        int neighborRegionSize,
        read_number readId
    ) const;

    std::vector<CorrectedCandidate> getCorrectedCandidates(
        float estimatedErrorrate,
        float estimatedCoverage,
        float m_coverage,
        int new_columns_to_correct
    ) const;

    RegionSelectionResult findCandidatesOfDifferentRegion(
        int dataset_coverage
    ) const;
};



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





}





#endif
