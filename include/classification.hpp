#ifndef CARE_CLASSIFICATION_HPP
#define CARE_CLASSIFICATION_HPP

#include <array>
#include <random>
#include <forest.hpp>
#include <logreg.hpp>
#include <msa.hpp>
#include <cpucorrectortask.hpp>
#include <options.hpp>

// This header allows toggling of feature transformations and classifiers,
// and seperates classification logic from the main corrector code.
// SEE BOTTOM OF FILE FOR TOGGLES! 

// TODO: implement logistic regression and investigate peformance
// Possibly same accuracy with VASTLY superior performance

// The current features were designed with the possibility of using logistic regression in mind
// but are highly redundant for any decision tree.
// However, since sklearn only supports float-type features, we might aswell do the one-hot-vector-encoding here for now.


namespace care {

struct ClfAgentDecisionInputData{
    const char* decodedAnchor{};
    int subjectColumnsBegin_incl{};
    int subjectColumnsEnd_excl{};
    const int* alignmentShifts{};
    const int* candidateSequencesLengths{};
    const char* decodedCandidateSequences{};
    const int* countsA{};
    const int* countsC{};
    const int* countsG{};
    const int* countsT{};
    const float* weightsA{};
    const float* weightsC{};
    const float* weightsG{};
    const float* weightsT{};
    const int* coverages{};
    const char* consensus{};

    std::function<MSAProperties(int,int,float,float,float)> getMSAProperties{};

    MSAProperties anchorMsaProperties{};
};

template<typename AnchorClf,
         typename CandClf,
         typename AnchorExtractor,
         typename CandsExtractor>
struct clf_agent
{

    //TODO: access permission
    std::shared_ptr<AnchorClf> classifier_anchor;
    std::shared_ptr<CandClf> classifier_cands;
    std::stringstream anchor_stream, cands_stream;
    std::shared_ptr<std::ofstream> anchor_file, cands_file;
    std::mt19937 rng;
    std::bernoulli_distribution coinflip_anchor, coinflip_cands;
    AnchorExtractor extract_anchor;
    CandsExtractor extract_cands;

    clf_agent(const CorrectionOptions& c_opts, const FileOptions& f_opts) :
        classifier_anchor(c_opts.correctionType == CorrectionType::Forest ? std::make_shared<AnchorClf>(f_opts.mlForestfileAnchor, c_opts.thresholdAnchor) : nullptr),
        classifier_cands(c_opts.correctionTypeCands == CorrectionType::Forest ? std::make_shared<CandClf>(f_opts.mlForestfileCands, c_opts.thresholdCands) : nullptr),
        anchor_file(c_opts.correctionType == CorrectionType::Print ? std::make_shared<std::ofstream>(f_opts.mlForestfileAnchor) : nullptr),
        cands_file(c_opts.correctionTypeCands == CorrectionType::Print ? std::make_shared<std::ofstream>(f_opts.mlForestfileCands) : nullptr),
        rng(44),
        coinflip_anchor(c_opts.sampleRateAnchor),
        coinflip_cands(c_opts.sampleRateCands)
    {
        if (c_opts.correctionType == CorrectionType::Print) {
            *anchor_file << extract_anchor << std::endl;
            *cands_file << extract_cands << std::endl;
        }
    }

    clf_agent(const clf_agent& other) :
        classifier_anchor(other.classifier_anchor),
        classifier_cands(other.classifier_cands),
        anchor_file(other.anchor_file),
        cands_file(other.cands_file),
        rng(44),
        coinflip_anchor(other.coinflip_anchor),
        coinflip_cands(other.coinflip_cands)
    {}

    void print_anchor(const CpuErrorCorrectorTask& task, size_t i, const CorrectionOptions& opt) {       
        if (!coinflip_anchor(rng)) return;

        anchor_stream << task.input.anchorReadId << ' ' << i << ' ';
        for (float j: extract_anchor(task, i, opt))
            anchor_stream << j << ' ';
        anchor_stream << '\n';
    }

    void print_cand(const CpuErrorCorrectorTask& task, int i, const CorrectionOptions& opt, size_t cand, size_t offset) {       
        if (!coinflip_cands(rng)) return;

        cands_stream << task.candidateReadIds[cand] << ' ' << (task.alignmentFlags[cand]==AlignmentOrientation::ReverseComplement?-i-1:i) << ' ';
        for (float j: extract_cands(task, i, opt, cand, offset))
            cands_stream << j << ' ';
        cands_stream << '\n';
    }

    template<typename... Args>
    bool decide_anchor(Args&&...args) {
        return classifier_anchor->decide(extract_anchor(std::forward<Args>(args)...));
    }

    template<typename... Args>
    bool decide_cand(Args&&...args) {
        return classifier_cands->decide(extract_cands(std::forward<Args>(args)...));
    }

    void flush() {
        #pragma omp critical
        {
            if (anchor_file)
                *anchor_file << anchor_stream.rdbuf();
            if (cands_file)
                *cands_file << cands_stream.rdbuf();
        }
        anchor_stream = std::stringstream();
        cands_stream = std::stringstream();
    }
};

namespace detail {

struct extract_anchor {
    using features_t = std::array<float, 21>;
    features_t operator()(const ClfAgentDecisionInputData& data, int i, const CorrectionOptions& opt) noexcept{   
        int a_begin = data.subjectColumnsBegin_incl;
        int a_end = data.subjectColumnsEnd_excl;
        int pos = a_begin + i;
        char orig = data.decodedAnchor[i];
        float countsACGT = data.coverages[pos];
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(data.consensus[pos] == 'A'),
            float(data.consensus[pos] == 'C'),
            float(data.consensus[pos] == 'G'),
            float(data.consensus[pos] == 'T'),
            data.weightsA[pos],
            data.weightsC[pos],
            data.weightsG[pos],
            data.weightsT[pos],
            data.countsA[pos]/countsACGT,
            data.countsC[pos]/countsACGT,
            data.countsG[pos]/countsACGT,
            data.countsT[pos]/countsACGT,
            data.anchorMsaProperties.avg_support,
            data.anchorMsaProperties.min_support,
            float(data.anchorMsaProperties.max_coverage)/opt.estimatedCoverage,
            float(data.anchorMsaProperties.min_coverage)/opt.estimatedCoverage,
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin)
        };
    }

    features_t operator()(const CpuErrorCorrectorTask& task, int i, const CorrectionOptions& opt) noexcept {   
        auto& msa = task.multipleSequenceAlignment;
        int a_begin = msa.subjectColumnsBegin_incl;
        int a_end = msa.subjectColumnsEnd_excl;
        int pos = a_begin + i;
        char orig = task.decodedAnchor[i];
        float countsACGT = msa.countsA[pos] + msa.countsC[pos] + msa.countsG[pos] + msa.countsT[pos];
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(msa.consensus[pos] == 'A'),
            float(msa.consensus[pos] == 'C'),
            float(msa.consensus[pos] == 'G'),
            float(msa.consensus[pos] == 'T'),
            msa.weightsA[pos],
            msa.weightsC[pos],
            msa.weightsG[pos],
            msa.weightsT[pos],
            msa.countsA[pos]/countsACGT,
            msa.countsC[pos]/countsACGT,
            msa.countsG[pos]/countsACGT,
            msa.countsT[pos]/countsACGT,
            task.msaProperties.avg_support,
            task.msaProperties.min_support,
            float(task.msaProperties.max_coverage)/opt.estimatedCoverage,
            float(task.msaProperties.min_coverage)/opt.estimatedCoverage,
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin)
        };
    }
};

struct extract_cands {
    using features_t = std::array<float, 26>;
    features_t operator()(const ClfAgentDecisionInputData& data, size_t i, const CorrectionOptions& opt, size_t cand, size_t offset) noexcept {   
        int a_begin = data.subjectColumnsBegin_incl;
        int a_end = data.subjectColumnsEnd_excl;
        int c_begin = a_begin + data.alignmentShifts[cand];
        int c_end = c_begin + data.candidateSequencesLengths[cand];
        int pos = c_begin + i;
        char orig = data.decodedCandidateSequences[offset+i];
        float countsACGT = data.coverages[pos];
        MSAProperties props = data.getMSAProperties(c_begin, c_end, opt.estimatedErrorrate, opt.estimatedCoverage, opt.m_coverage);
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(data.consensus[pos] == 'A'),
            float(data.consensus[pos] == 'C'),
            float(data.consensus[pos] == 'G'),
            float(data.consensus[pos] == 'T'),
            data.weightsA[pos],
            data.weightsC[pos],
            data.weightsG[pos],
            data.weightsT[pos],
            data.countsA[pos]/countsACGT,
            data.countsC[pos]/countsACGT,
            data.countsG[pos]/countsACGT,
            data.countsT[pos]/countsACGT,
            props.avg_support,
            props.min_support,
            float(props.max_coverage)/opt.estimatedCoverage,
            float(props.min_coverage)/opt.estimatedCoverage,
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(c_end-c_begin), // absolute shift (compatible with differing read lengths)
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(a_end-a_begin),
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(a_end-a_begin), // relative overlap (ratio of a or c length in case of diff. read len)
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(c_end-c_begin),
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin),
            float(std::max(a_begin-pos, pos-a_end))/(c_end-c_begin)
        };
    }

    features_t operator()(const CpuErrorCorrectorTask& task, size_t i, const CorrectionOptions& opt, size_t cand, size_t offset) noexcept {   
        auto& msa = task.multipleSequenceAlignment;
        int a_begin = msa.subjectColumnsBegin_incl;
        int a_end = msa.subjectColumnsEnd_excl;
        int c_begin = a_begin + task.alignmentShifts[cand];
        int c_end = c_begin + task.candidateSequencesLengths[cand];
        int pos = c_begin + i;
        char orig = task.decodedCandidateSequences[offset+i];
        float countsACGT = msa.countsA[pos] + msa.countsC[pos] + msa.countsG[pos] + msa.countsT[pos];
        MSAProperties props = msa.getMSAProperties(c_begin, c_end, opt.estimatedErrorrate, opt.estimatedCoverage, opt.m_coverage);
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(msa.consensus[pos] == 'A'),
            float(msa.consensus[pos] == 'C'),
            float(msa.consensus[pos] == 'G'),
            float(msa.consensus[pos] == 'T'),
            msa.weightsA[pos],
            msa.weightsC[pos],
            msa.weightsG[pos],
            msa.weightsT[pos],
            msa.countsA[pos]/countsACGT,
            msa.countsC[pos]/countsACGT,
            msa.countsG[pos]/countsACGT,
            msa.countsT[pos]/countsACGT,
            props.avg_support,
            props.min_support,
            float(props.max_coverage)/opt.estimatedCoverage,
            float(props.min_coverage)/opt.estimatedCoverage,
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(c_end-c_begin), // absolute shift (compatible with differing read lengths)
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(a_end-a_begin),
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(a_end-a_begin), // relative overlap (ratio of a or c length in case of diff. read len)
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(c_end-c_begin),
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin),
            float(std::max(a_begin-pos, pos-a_end))/(c_end-c_begin)
        };
    }
};

struct extract_anchor_transformed {
    using features_t = std::array<float, 37>;

    constexpr operator auto() {
        return u8"37 extract_anchor_transformed";
    }

    features_t operator()(const ClfAgentDecisionInputData& data, int i, const CorrectionOptions& opt) noexcept{
        int a_begin = data.subjectColumnsBegin_incl;
        int a_end = data.subjectColumnsEnd_excl;
        int pos = a_begin + i;
        char orig = data.decodedAnchor[i];
        float countsACGT = data.coverages[pos];

        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(data.consensus[pos] == 'A'),
            float(data.consensus[pos] == 'C'),
            float(data.consensus[pos] == 'G'),
            float(data.consensus[pos] == 'T'),
            orig == 'A'?data.countsA[pos]/countsACGT:0,
            orig == 'C'?data.countsC[pos]/countsACGT:0,
            orig == 'G'?data.countsG[pos]/countsACGT:0,
            orig == 'T'?data.countsT[pos]/countsACGT:0,
            orig == 'A'?data.weightsA[pos]:0,
            orig == 'C'?data.weightsC[pos]:0,
            orig == 'G'?data.weightsG[pos]:0,
            orig == 'T'?data.weightsT[pos]:0,
            data.consensus[pos] == 'A'?data.countsA[pos]/countsACGT:0,
            data.consensus[pos] == 'C'?data.countsC[pos]/countsACGT:0,
            data.consensus[pos] == 'G'?data.countsG[pos]/countsACGT:0,
            data.consensus[pos] == 'T'?data.countsT[pos]/countsACGT:0,
            data.consensus[pos] == 'A'?data.weightsA[pos]:0,
            data.consensus[pos] == 'C'?data.weightsC[pos]:0,
            data.consensus[pos] == 'G'?data.weightsG[pos]:0,
            data.consensus[pos] == 'T'?data.weightsT[pos]:0,
            data.weightsA[pos],
            data.weightsC[pos],
            data.weightsG[pos],
            data.weightsT[pos],
            data.countsA[pos]/countsACGT,
            data.countsC[pos]/countsACGT,
            data.countsG[pos]/countsACGT,
            data.countsT[pos]/countsACGT,
            data.anchorMsaProperties.avg_support,
            data.anchorMsaProperties.min_support,
            float(data.anchorMsaProperties.max_coverage)/opt.estimatedCoverage,
            float(data.anchorMsaProperties.min_coverage)/opt.estimatedCoverage,
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin)
        };
    }

    features_t operator()(const CpuErrorCorrectorTask& task, int i, const CorrectionOptions& opt) noexcept {   
        auto& msa = task.multipleSequenceAlignment;
        int a_begin = msa.subjectColumnsBegin_incl;
        int a_end = msa.subjectColumnsEnd_excl;
        int pos = a_begin + i;
        char orig = task.decodedAnchor[i];
        float countsACGT = msa.countsA[pos] + msa.countsC[pos] + msa.countsG[pos] + msa.countsT[pos];
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(msa.consensus[pos] == 'A'),
            float(msa.consensus[pos] == 'C'),
            float(msa.consensus[pos] == 'G'),
            float(msa.consensus[pos] == 'T'),
            orig == 'A'?msa.countsA[pos]/countsACGT:0,
            orig == 'C'?msa.countsC[pos]/countsACGT:0,
            orig == 'G'?msa.countsG[pos]/countsACGT:0,
            orig == 'T'?msa.countsT[pos]/countsACGT:0,
            orig == 'A'?msa.weightsA[pos]:0,
            orig == 'C'?msa.weightsC[pos]:0,
            orig == 'G'?msa.weightsG[pos]:0,
            orig == 'T'?msa.weightsT[pos]:0,
            msa.consensus[pos] == 'A'?msa.countsA[pos]/countsACGT:0,
            msa.consensus[pos] == 'C'?msa.countsC[pos]/countsACGT:0,
            msa.consensus[pos] == 'G'?msa.countsG[pos]/countsACGT:0,
            msa.consensus[pos] == 'T'?msa.countsT[pos]/countsACGT:0,
            msa.consensus[pos] == 'A'?msa.weightsA[pos]:0,
            msa.consensus[pos] == 'C'?msa.weightsC[pos]:0,
            msa.consensus[pos] == 'G'?msa.weightsG[pos]:0,
            msa.consensus[pos] == 'T'?msa.weightsT[pos]:0,
            msa.weightsA[pos],
            msa.weightsC[pos],
            msa.weightsG[pos],
            msa.weightsT[pos],
            msa.countsA[pos]/countsACGT,
            msa.countsC[pos]/countsACGT,
            msa.countsG[pos]/countsACGT,
            msa.countsT[pos]/countsACGT,
            task.msaProperties.avg_support,
            task.msaProperties.min_support,
            float(task.msaProperties.max_coverage)/opt.estimatedCoverage,
            float(task.msaProperties.min_coverage)/opt.estimatedCoverage,
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin)
        };
    }
};

struct extract_cands_transformed {
    using features_t = std::array<float, 42>;

    constexpr operator auto() {
        return u8"37 extract_cands_transformed";
    }

    features_t operator()(const ClfAgentDecisionInputData& data, size_t i, const CorrectionOptions& opt, size_t cand, size_t offset) noexcept {   

        int a_begin = data.subjectColumnsBegin_incl;
        int a_end = data.subjectColumnsEnd_excl;
        int c_begin = a_begin + data.alignmentShifts[cand];
        int c_end = c_begin + data.candidateSequencesLengths[cand];
        int pos = c_begin + i;
        char orig = data.decodedCandidateSequences[offset+i];
        float countsACGT = data.countsA[pos] + data.countsC[pos] + data.countsG[pos] + data.countsT[pos];
        MSAProperties props = data.getMSAProperties(c_begin, c_end, opt.estimatedErrorrate, opt.estimatedCoverage, opt.m_coverage);
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(data.consensus[pos] == 'A'),
            float(data.consensus[pos] == 'C'),
            float(data.consensus[pos] == 'G'),
            float(data.consensus[pos] == 'T'),
            orig == 'A'?data.countsA[pos]/countsACGT:0,
            orig == 'C'?data.countsC[pos]/countsACGT:0,
            orig == 'G'?data.countsG[pos]/countsACGT:0,
            orig == 'T'?data.countsT[pos]/countsACGT:0,
            orig == 'A'?data.weightsA[pos]:0,
            orig == 'C'?data.weightsC[pos]:0,
            orig == 'G'?data.weightsG[pos]:0,
            orig == 'T'?data.weightsT[pos]:0,
            data.consensus[pos] == 'A'?data.countsA[pos]/countsACGT:0,
            data.consensus[pos] == 'C'?data.countsC[pos]/countsACGT:0,
            data.consensus[pos] == 'G'?data.countsG[pos]/countsACGT:0,
            data.consensus[pos] == 'T'?data.countsT[pos]/countsACGT:0,
            data.consensus[pos] == 'A'?data.weightsA[pos]:0,
            data.consensus[pos] == 'C'?data.weightsC[pos]:0,
            data.consensus[pos] == 'G'?data.weightsG[pos]:0,
            data.consensus[pos] == 'T'?data.weightsT[pos]:0,
            data.weightsA[pos],
            data.weightsC[pos],
            data.weightsG[pos],
            data.weightsT[pos],
            data.countsA[pos]/countsACGT,
            data.countsC[pos]/countsACGT,
            data.countsG[pos]/countsACGT,
            data.countsT[pos]/countsACGT,
            props.avg_support,
            props.min_support,
            float(props.max_coverage)/opt.estimatedCoverage,
            float(props.min_coverage)/opt.estimatedCoverage,
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(c_end-c_begin), // absolute shift (compatible with differing read lengths)
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(a_end-a_begin),
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(a_end-a_begin), // relative overlap (ratio of a or c length in case of diff. read len)
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(c_end-c_begin),
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin),
            float(std::max(a_begin-pos, pos-a_end))/(c_end-c_begin)
        };
    }

    features_t operator()(const CpuErrorCorrectorTask& task, size_t i, const CorrectionOptions& opt, size_t cand, size_t offset) noexcept {   
        auto& msa = task.multipleSequenceAlignment;
        int a_begin = msa.subjectColumnsBegin_incl;
        int a_end = msa.subjectColumnsEnd_excl;
        int c_begin = a_begin + task.alignmentShifts[cand];
        int c_end = c_begin + task.candidateSequencesLengths[cand];
        int pos = c_begin + i;
        char orig = task.decodedCandidateSequences[offset+i];
        float countsACGT = msa.countsA[pos] + msa.countsC[pos] + msa.countsG[pos] + msa.countsT[pos];
        MSAProperties props = msa.getMSAProperties(c_begin, c_end, opt.estimatedErrorrate, opt.estimatedCoverage, opt.m_coverage);
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(msa.consensus[pos] == 'A'),
            float(msa.consensus[pos] == 'C'),
            float(msa.consensus[pos] == 'G'),
            float(msa.consensus[pos] == 'T'),
            orig == 'A'?msa.countsA[pos]/countsACGT:0,
            orig == 'C'?msa.countsC[pos]/countsACGT:0,
            orig == 'G'?msa.countsG[pos]/countsACGT:0,
            orig == 'T'?msa.countsT[pos]/countsACGT:0,
            orig == 'A'?msa.weightsA[pos]:0,
            orig == 'C'?msa.weightsC[pos]:0,
            orig == 'G'?msa.weightsG[pos]:0,
            orig == 'T'?msa.weightsT[pos]:0,
            msa.consensus[pos] == 'A'?msa.countsA[pos]/countsACGT:0,
            msa.consensus[pos] == 'C'?msa.countsC[pos]/countsACGT:0,
            msa.consensus[pos] == 'G'?msa.countsG[pos]/countsACGT:0,
            msa.consensus[pos] == 'T'?msa.countsT[pos]/countsACGT:0,
            msa.consensus[pos] == 'A'?msa.weightsA[pos]:0,
            msa.consensus[pos] == 'C'?msa.weightsC[pos]:0,
            msa.consensus[pos] == 'G'?msa.weightsG[pos]:0,
            msa.consensus[pos] == 'T'?msa.weightsT[pos]:0,
            msa.weightsA[pos],
            msa.weightsC[pos],
            msa.weightsG[pos],
            msa.weightsT[pos],
            msa.countsA[pos]/countsACGT,
            msa.countsC[pos]/countsACGT,
            msa.countsG[pos]/countsACGT,
            msa.countsT[pos]/countsACGT,
            props.avg_support,
            props.min_support,
            float(props.max_coverage)/opt.estimatedCoverage,
            float(props.min_coverage)/opt.estimatedCoverage,
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(c_end-c_begin), // absolute shift (compatible with differing read lengths)
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(a_end-a_begin),
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(a_end-a_begin), // relative overlap (ratio of a or c length in case of diff. read len)
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(c_end-c_begin),
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin),
            float(std::max(a_begin-pos, pos-a_end))/(c_end-c_begin)
        };
    }
};

struct extract_anchor_normed_weights {
    using features_t = std::array<float, 21>;

    features_t operator()(const ClfAgentDecisionInputData& data, int i, const CorrectionOptions& opt) noexcept {   
        int a_begin = data.subjectColumnsBegin_incl;
        int a_end = data.subjectColumnsEnd_excl;
        int pos = a_begin + i;
        char orig = data.decodedAnchor[i];
        float countsACGT = data.coverages[pos];
        float weightsACGT = data.weightsA[pos] + data.weightsC[pos] + data.weightsG[pos] + data.weightsT[pos];
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(data.consensus[pos] == 'A'),
            float(data.consensus[pos] == 'C'),
            float(data.consensus[pos] == 'G'),
            float(data.consensus[pos] == 'T'),
            data.weightsA[pos]/weightsACGT,
            data.weightsC[pos]/weightsACGT,
            data.weightsG[pos]/weightsACGT,
            data.weightsT[pos]/weightsACGT,
            data.countsA[pos]/countsACGT,
            data.countsC[pos]/countsACGT,
            data.countsG[pos]/countsACGT,
            data.countsT[pos]/countsACGT,
            data.anchorMsaProperties.avg_support,
            data.anchorMsaProperties.min_support,
            float(data.anchorMsaProperties.max_coverage)/opt.estimatedCoverage,
            float(data.anchorMsaProperties.min_coverage)/opt.estimatedCoverage,
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin)
        };
    }

    features_t operator()(const CpuErrorCorrectorTask& task, int i, const CorrectionOptions& opt) noexcept {   
        auto& msa = task.multipleSequenceAlignment;
        int a_begin = msa.subjectColumnsBegin_incl;
        int a_end = msa.subjectColumnsEnd_excl;
        int pos = a_begin + i;
        char orig = task.decodedAnchor[i];
        float countsACGT = msa.countsA[pos] + msa.countsC[pos] + msa.countsG[pos] + msa.countsT[pos];
        float weightsACGT = msa.weightsA[pos] + msa.weightsC[pos] + msa.weightsG[pos] + msa.weightsT[pos];
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(msa.consensus[pos] == 'A'),
            float(msa.consensus[pos] == 'C'),
            float(msa.consensus[pos] == 'G'),
            float(msa.consensus[pos] == 'T'),
            msa.weightsA[pos]/weightsACGT,
            msa.weightsC[pos]/weightsACGT,
            msa.weightsG[pos]/weightsACGT,
            msa.weightsT[pos]/weightsACGT,
            msa.countsA[pos]/countsACGT,
            msa.countsC[pos]/countsACGT,
            msa.countsG[pos]/countsACGT,
            msa.countsT[pos]/countsACGT,
            task.msaProperties.avg_support,
            task.msaProperties.min_support,
            float(task.msaProperties.max_coverage)/opt.estimatedCoverage,
            float(task.msaProperties.min_coverage)/opt.estimatedCoverage,
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin)
        };
    }
};

struct extract_cands_normed_weights {
    using features_t = std::array<float, 26>;
    features_t operator()(const ClfAgentDecisionInputData& data, size_t i, const CorrectionOptions& opt, size_t cand, size_t offset) noexcept {   

        int a_begin = data.subjectColumnsBegin_incl;
        int a_end = data.subjectColumnsEnd_excl;
        int c_begin = a_begin + data.alignmentShifts[cand];
        int c_end = c_begin + data.candidateSequencesLengths[cand];
        int pos = c_begin + i;
        char orig = data.decodedCandidateSequences[offset+i];
        float countsACGT = data.coverages[pos];
        float weightsACGT = data.weightsA[pos] + data.weightsC[pos] + data.weightsG[pos] + data.weightsT[pos];
        MSAProperties props = data.getMSAProperties(c_begin, c_end, opt.estimatedErrorrate, opt.estimatedCoverage, opt.m_coverage);
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(data.consensus[pos] == 'A'),
            float(data.consensus[pos] == 'C'),
            float(data.consensus[pos] == 'G'),
            float(data.consensus[pos] == 'T'),
            data.weightsA[pos]/weightsACGT,
            data.weightsC[pos]/weightsACGT,
            data.weightsG[pos]/weightsACGT,
            data.weightsT[pos]/weightsACGT,
            data.countsA[pos]/countsACGT,
            data.countsC[pos]/countsACGT,
            data.countsG[pos]/countsACGT,
            data.countsT[pos]/countsACGT,
            props.avg_support,
            props.min_support,
            float(props.max_coverage)/opt.estimatedCoverage,
            float(props.min_coverage)/opt.estimatedCoverage,
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(c_end-c_begin), // absolute shift (compatible with differing read lengths)
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(a_end-a_begin),
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(a_end-a_begin), // relative overlap (ratio of a or c length in case of diff. read len)
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(c_end-c_begin),
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin),
            float(std::max(a_begin-pos, pos-a_end))/(c_end-c_begin)
        };
    }

    features_t operator()(const CpuErrorCorrectorTask& task, size_t i, const CorrectionOptions& opt, size_t cand, size_t offset) noexcept {   
        auto& msa = task.multipleSequenceAlignment;
        int a_begin = msa.subjectColumnsBegin_incl;
        int a_end = msa.subjectColumnsEnd_excl;
        int c_begin = a_begin + task.alignmentShifts[cand];
        int c_end = c_begin + task.candidateSequencesLengths[cand];
        int pos = c_begin + i;
        char orig = task.decodedCandidateSequences[offset+i];
        float countsACGT = msa.countsA[pos] + msa.countsC[pos] + msa.countsG[pos] + msa.countsT[pos];
        float weightsACGT = msa.weightsA[pos] + msa.weightsC[pos] + msa.weightsG[pos] + msa.weightsT[pos];
        MSAProperties props = msa.getMSAProperties(c_begin, c_end, opt.estimatedErrorrate, opt.estimatedCoverage, opt.m_coverage);
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(msa.consensus[pos] == 'A'),
            float(msa.consensus[pos] == 'C'),
            float(msa.consensus[pos] == 'G'),
            float(msa.consensus[pos] == 'T'),
            msa.weightsA[pos]/weightsACGT,
            msa.weightsC[pos]/weightsACGT,
            msa.weightsG[pos]/weightsACGT,
            msa.weightsT[pos]/weightsACGT,
            msa.countsA[pos]/countsACGT,
            msa.countsC[pos]/countsACGT,
            msa.countsG[pos]/countsACGT,
            msa.countsT[pos]/countsACGT,
            props.avg_support,
            props.min_support,
            float(props.max_coverage)/opt.estimatedCoverage,
            float(props.min_coverage)/opt.estimatedCoverage,
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(c_end-c_begin), // absolute shift (compatible with differing read lengths)
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(a_end-a_begin),
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(a_end-a_begin), // relative overlap (ratio of a or c length in case of diff. read len)
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(c_end-c_begin),
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin),
            float(std::max(a_begin-pos, pos-a_end))/(c_end-c_begin)
        };
    }
};

struct extract_anchor_transformed_normed_weights {
    using features_t = std::array<float, 37>;

    features_t operator()(const ClfAgentDecisionInputData& data, int i, const CorrectionOptions& opt) noexcept {   
        int a_begin = data.subjectColumnsBegin_incl;
        int a_end = data.subjectColumnsEnd_excl;
        int pos = a_begin + i;
        char orig = data.decodedAnchor[i];
        float countsACGT = data.coverages[pos];
        float weightsACGT = data.weightsA[pos] + data.weightsC[pos] + data.weightsG[pos] + data.weightsT[pos];
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(data.consensus[pos] == 'A'),
            float(data.consensus[pos] == 'C'),
            float(data.consensus[pos] == 'G'),
            float(data.consensus[pos] == 'T'),
            orig == 'A'?data.countsA[pos]/countsACGT:0,
            orig == 'C'?data.countsC[pos]/countsACGT:0,
            orig == 'G'?data.countsG[pos]/countsACGT:0,
            orig == 'T'?data.countsT[pos]/countsACGT:0,
            orig == 'A'?data.weightsA[pos]/weightsACGT:0,
            orig == 'C'?data.weightsC[pos]/weightsACGT:0,
            orig == 'G'?data.weightsG[pos]/weightsACGT:0,
            orig == 'T'?data.weightsT[pos]/weightsACGT:0,
            data.consensus[pos] == 'A'?data.countsA[pos]/countsACGT:0,
            data.consensus[pos] == 'C'?data.countsC[pos]/countsACGT:0,
            data.consensus[pos] == 'G'?data.countsG[pos]/countsACGT:0,
            data.consensus[pos] == 'T'?data.countsT[pos]/countsACGT:0,
            data.consensus[pos] == 'A'?data.weightsA[pos]/weightsACGT:0,
            data.consensus[pos] == 'C'?data.weightsC[pos]/weightsACGT:0,
            data.consensus[pos] == 'G'?data.weightsG[pos]/weightsACGT:0,
            data.consensus[pos] == 'T'?data.weightsT[pos]/weightsACGT:0,
            data.weightsA[pos]/weightsACGT,
            data.weightsC[pos]/weightsACGT,
            data.weightsG[pos]/weightsACGT,
            data.weightsT[pos]/weightsACGT,
            data.countsA[pos]/countsACGT,
            data.countsC[pos]/countsACGT,
            data.countsG[pos]/countsACGT,
            data.countsT[pos]/countsACGT,
            data.anchorMsaProperties.avg_support,
            data.anchorMsaProperties.min_support,
            float(data.anchorMsaProperties.max_coverage)/opt.estimatedCoverage,
            float(data.anchorMsaProperties.min_coverage)/opt.estimatedCoverage,
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin)
        };
    }

    features_t operator()(const CpuErrorCorrectorTask& task, int i, const CorrectionOptions& opt) noexcept {   
        auto& msa = task.multipleSequenceAlignment;
        int a_begin = msa.subjectColumnsBegin_incl;
        int a_end = msa.subjectColumnsEnd_excl;
        int pos = a_begin + i;
        char orig = task.decodedAnchor[i];
        float countsACGT = msa.countsA[pos] + msa.countsC[pos] + msa.countsG[pos] + msa.countsT[pos];
        float weightsACGT = msa.weightsA[pos] + msa.weightsC[pos] + msa.weightsG[pos] + msa.weightsT[pos];
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(msa.consensus[pos] == 'A'),
            float(msa.consensus[pos] == 'C'),
            float(msa.consensus[pos] == 'G'),
            float(msa.consensus[pos] == 'T'),
            orig == 'A'?msa.countsA[pos]/countsACGT:0,
            orig == 'C'?msa.countsC[pos]/countsACGT:0,
            orig == 'G'?msa.countsG[pos]/countsACGT:0,
            orig == 'T'?msa.countsT[pos]/countsACGT:0,
            orig == 'A'?msa.weightsA[pos]/weightsACGT:0,
            orig == 'C'?msa.weightsC[pos]/weightsACGT:0,
            orig == 'G'?msa.weightsG[pos]/weightsACGT:0,
            orig == 'T'?msa.weightsT[pos]/weightsACGT:0,
            msa.consensus[pos] == 'A'?msa.countsA[pos]/countsACGT:0,
            msa.consensus[pos] == 'C'?msa.countsC[pos]/countsACGT:0,
            msa.consensus[pos] == 'G'?msa.countsG[pos]/countsACGT:0,
            msa.consensus[pos] == 'T'?msa.countsT[pos]/countsACGT:0,
            msa.consensus[pos] == 'A'?msa.weightsA[pos]/weightsACGT:0,
            msa.consensus[pos] == 'C'?msa.weightsC[pos]/weightsACGT:0,
            msa.consensus[pos] == 'G'?msa.weightsG[pos]/weightsACGT:0,
            msa.consensus[pos] == 'T'?msa.weightsT[pos]/weightsACGT:0,
            msa.weightsA[pos]/weightsACGT,
            msa.weightsC[pos]/weightsACGT,
            msa.weightsG[pos]/weightsACGT,
            msa.weightsT[pos]/weightsACGT,
            msa.countsA[pos]/countsACGT,
            msa.countsC[pos]/countsACGT,
            msa.countsG[pos]/countsACGT,
            msa.countsT[pos]/countsACGT,
            task.msaProperties.avg_support,
            task.msaProperties.min_support,
            float(task.msaProperties.max_coverage)/opt.estimatedCoverage,
            float(task.msaProperties.min_coverage)/opt.estimatedCoverage,
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin)
        };
    }
};

struct extract_cands_transformed_normed_weights {
    using features_t = std::array<float, 42>;

    features_t operator()(const ClfAgentDecisionInputData& data, size_t i, const CorrectionOptions& opt, size_t cand, size_t offset) noexcept {   
        int a_begin = data.subjectColumnsBegin_incl;
        int a_end = data.subjectColumnsEnd_excl;
        int c_begin = a_begin + data.alignmentShifts[cand];
        int c_end = c_begin + data.candidateSequencesLengths[cand];
        int pos = c_begin + i;
        char orig = data.decodedCandidateSequences[offset+i];
        float countsACGT = data.coverages[pos];
        float weightsACGT = data.weightsA[pos] + data.weightsC[pos] + data.weightsG[pos] + data.weightsT[pos];
        MSAProperties props = data.getMSAProperties(c_begin, c_end, opt.estimatedErrorrate, opt.estimatedCoverage, opt.m_coverage);
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(data.consensus[pos] == 'A'),
            float(data.consensus[pos] == 'C'),
            float(data.consensus[pos] == 'G'),
            float(data.consensus[pos] == 'T'),
            orig == 'A'?data.countsA[pos]/countsACGT:0,
            orig == 'C'?data.countsC[pos]/countsACGT:0,
            orig == 'G'?data.countsG[pos]/countsACGT:0,
            orig == 'T'?data.countsT[pos]/countsACGT:0,
            orig == 'A'?data.weightsA[pos]/weightsACGT:0,
            orig == 'C'?data.weightsC[pos]/weightsACGT:0,
            orig == 'G'?data.weightsG[pos]/weightsACGT:0,
            orig == 'T'?data.weightsT[pos]/weightsACGT:0,
            data.consensus[pos] == 'A'?data.countsA[pos]/countsACGT:0,
            data.consensus[pos] == 'C'?data.countsC[pos]/countsACGT:0,
            data.consensus[pos] == 'G'?data.countsG[pos]/countsACGT:0,
            data.consensus[pos] == 'T'?data.countsT[pos]/countsACGT:0,
            data.consensus[pos] == 'A'?data.weightsA[pos]/weightsACGT:0,
            data.consensus[pos] == 'C'?data.weightsC[pos]/weightsACGT:0,
            data.consensus[pos] == 'G'?data.weightsG[pos]/weightsACGT:0,
            data.consensus[pos] == 'T'?data.weightsT[pos]/weightsACGT:0,
            data.weightsA[pos]/weightsACGT,
            data.weightsC[pos]/weightsACGT,
            data.weightsG[pos]/weightsACGT,
            data.weightsT[pos]/weightsACGT,
            data.countsA[pos]/countsACGT,
            data.countsC[pos]/countsACGT,
            data.countsG[pos]/countsACGT,
            data.countsT[pos]/countsACGT,
            props.avg_support,
            props.min_support,
            float(props.max_coverage)/opt.estimatedCoverage,
            float(props.min_coverage)/opt.estimatedCoverage,
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(c_end-c_begin), // absolute shift (compatible with differing read lengths)
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(a_end-a_begin),
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(a_end-a_begin), // relative overlap (ratio of a or c length in case of diff. read len)
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(c_end-c_begin),
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin),
            float(std::max(a_begin-pos, pos-a_end))/(c_end-c_begin)
        };
    }

    features_t operator()(const CpuErrorCorrectorTask& task, size_t i, const CorrectionOptions& opt, size_t cand, size_t offset) noexcept {   
        auto& msa = task.multipleSequenceAlignment;
        int a_begin = msa.subjectColumnsBegin_incl;
        int a_end = msa.subjectColumnsEnd_excl;
        int c_begin = a_begin + task.alignmentShifts[cand];
        int c_end = c_begin + task.candidateSequencesLengths[cand];
        int pos = c_begin + i;
        char orig = task.decodedCandidateSequences[offset+i];
        float countsACGT = msa.countsA[pos] + msa.countsC[pos] + msa.countsG[pos] + msa.countsT[pos];
        float weightsACGT = msa.weightsA[pos] + msa.weightsC[pos] + msa.weightsG[pos] + msa.weightsT[pos];
        MSAProperties props = msa.getMSAProperties(c_begin, c_end, opt.estimatedErrorrate, opt.estimatedCoverage, opt.m_coverage);
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(msa.consensus[pos] == 'A'),
            float(msa.consensus[pos] == 'C'),
            float(msa.consensus[pos] == 'G'),
            float(msa.consensus[pos] == 'T'),
            orig == 'A'?msa.countsA[pos]/countsACGT:0,
            orig == 'C'?msa.countsC[pos]/countsACGT:0,
            orig == 'G'?msa.countsG[pos]/countsACGT:0,
            orig == 'T'?msa.countsT[pos]/countsACGT:0,
            orig == 'A'?msa.weightsA[pos]/weightsACGT:0,
            orig == 'C'?msa.weightsC[pos]/weightsACGT:0,
            orig == 'G'?msa.weightsG[pos]/weightsACGT:0,
            orig == 'T'?msa.weightsT[pos]/weightsACGT:0,
            msa.consensus[pos] == 'A'?msa.countsA[pos]/countsACGT:0,
            msa.consensus[pos] == 'C'?msa.countsC[pos]/countsACGT:0,
            msa.consensus[pos] == 'G'?msa.countsG[pos]/countsACGT:0,
            msa.consensus[pos] == 'T'?msa.countsT[pos]/countsACGT:0,
            msa.consensus[pos] == 'A'?msa.weightsA[pos]/weightsACGT:0,
            msa.consensus[pos] == 'C'?msa.weightsC[pos]/weightsACGT:0,
            msa.consensus[pos] == 'G'?msa.weightsG[pos]/weightsACGT:0,
            msa.consensus[pos] == 'T'?msa.weightsT[pos]/weightsACGT:0,
            msa.weightsA[pos]/weightsACGT,
            msa.weightsC[pos]/weightsACGT,
            msa.weightsG[pos]/weightsACGT,
            msa.weightsT[pos]/weightsACGT,
            msa.countsA[pos]/countsACGT,
            msa.countsC[pos]/countsACGT,
            msa.countsG[pos]/countsACGT,
            msa.countsT[pos]/countsACGT,
            props.avg_support,
            props.min_support,
            float(props.max_coverage)/opt.estimatedCoverage,
            float(props.min_coverage)/opt.estimatedCoverage,
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(c_end-c_begin), // absolute shift (compatible with differing read lengths)
            float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(a_end-a_begin),
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(a_end-a_begin), // relative overlap (ratio of a or c length in case of diff. read len)
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(c_end-c_begin),
            float(std::max(a_begin-pos, pos-a_end))/(a_end-a_begin),
            float(std::max(a_begin-pos, pos-a_end))/(c_end-c_begin)
        };
    }
};

} //namespace detail


//--------------------------------------------------------------------------------

using anchor_extractor = detail::extract_anchor_transformed;
using cands_extractor = detail::extract_cands_transformed;

using anchor_clf_t = ForestClf<anchor_extractor>;
using cands_clf_t = ForestClf<cands_extractor>;

using ClfAgent = clf_agent<anchor_clf_t, cands_clf_t, anchor_extractor, cands_extractor>;

} // namespace care

#endif
