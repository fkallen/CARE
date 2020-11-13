#ifndef CARE_CPUCORRECTOR_TASK_HPP
#define CARE_CPUCORRECTOR_TASK_HPP

#include <config.hpp>
#include <correctionresultprocessing.hpp>
#include <cpu_alignment.hpp>
#include <bestalignment.hpp>
#include <msa.hpp>

#include <vector>

namespace care{

    struct CpuErrorCorrectorInput; //forward declaration

    struct CpuErrorCorrectorTask{
        bool active{};

        std::vector<read_number> candidateReadIds{};
        std::vector<read_number> filteredReadIds{};
        std::vector<unsigned int> candidateSequencesData{};
        std::vector<unsigned int> candidateSequencesRevcData{};
        std::vector<int> candidateSequencesLengths{};
        std::vector<int> alignmentShifts{};
        std::vector<int> alignmentOps{};
        std::vector<int> alignmentOverlaps{};
        std::vector<float> alignmentWeights{};
        std::vector<char> candidateQualities{};
        std::vector<char> decodedAnchor{};
        std::vector<char> decodedCandidateSequences{};
        std::vector<cpu::SHDResult> alignments{};
        std::vector<cpu::SHDResult> revcAlignments{};
        std::vector<BestAlignment_t> alignmentFlags{};

        const CpuErrorCorrectorInput* input{};

        CorrectionResult subjectCorrection;
        std::vector<CorrectedCandidate> candidateCorrections;
        MSAProperties msaProperties;
        MultipleSequenceAlignment multipleSequenceAlignment;
    };

}

#endif