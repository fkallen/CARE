#ifndef CARE_CLASSIFICATION_HPP
#define CARE_CLASSIFICATION_HPP

#include <array>
#include <random>
#include <forest.hpp>
#include <msa.hpp>


// This header allows toggling of feature transformations and classifiers,
// and seperates classification logic from the main corrector code.
// SEE BOTTOM OF FILE FOR TOGGLES! 

// TODO: implement logistic regression and investigate peformance
// Possibly same accuracy with VASTLY superior performance

// The current features were designed with the possibility of using logistic regression in mind
// but are highly redundant for any decision tree.

namespace care {

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
        classifier_anchor(c_opts.correctionType == CorrectionType::Forest ? std::make_shared<AnchorClf>(f_opts.mlForestfileAnchor) : nullptr),
        classifier_cands(c_opts.correctionTypeCands == CorrectionType::Forest ? std::make_shared<CandClf>(f_opts.mlForestfileCands) : nullptr),
        anchor_file(c_opts.correctionType == CorrectionType::Print ? std::make_shared<std::ofstream>(f_opts.mlForestfileAnchor) : nullptr),
        cands_file(c_opts.correctionTypeCands == CorrectionType::Print ? std::make_shared<std::ofstream>(f_opts.mlForestfileCands) : nullptr),
        rng(std::mt19937(std::chrono::system_clock::now().time_since_epoch().count())),
        coinflip_anchor(1.0/3.0),
        coinflip_cands(0.01/3.0)
    {}

    clf_agent(const clf_agent& other) :
        classifier_anchor(other.classifier_anchor),
        classifier_cands(other.classifier_cands),
        anchor_file(other.anchor_file),
        cands_file(other.cands_file),
        rng(std::mt19937(std::chrono::system_clock::now().time_since_epoch().count() + std::hash<std::thread::id>{}(std::this_thread::get_id()))),
        coinflip_anchor(other.coinflip_anchor),
        coinflip_cands(other.coinflip_cands)
    {}

    //TODO: if this could just get task as parameter, everthing would look much nicer, consider un-private-ing stuff?
    void print_anchor(const MultipleSequenceAlignment& msa, char orig, size_t i, const CorrectionOptions& opt, read_number read_id) {       
        if (!coinflip_anchor(rng)) return;

        anchor_stream << read_id << ' ' << i << ' ';
        for (float j: extract_anchor(msa, orig, i, opt))
            anchor_stream << j << ' ';
        anchor_stream << '\n';
    }

    void print_cand(const MultipleSequenceAlignment& msa, char orig, size_t i, const CorrectionOptions& opt, int shift, int cand_length, read_number read_id) {       
        if (!coinflip_cands(rng)) return;

        cands_stream << read_id << ' ' << i << ' ';
        for (float j: extract_cands(msa, orig, i, opt, shift, cand_length))
            cands_stream << j << ' ';
        cands_stream << '\n';
    }

    float decide_anchor(const MultipleSequenceAlignment& msa, char orig, size_t i, const CorrectionOptions& opt) {       
        return classifier_anchor->decide(extract_anchor(msa, orig, i, opt));
    }

    float decide_cand(const MultipleSequenceAlignment& msa, char orig, size_t i, const CorrectionOptions& opt, int shift, int cand_length) {       
        return classifier_cands->decide(extract_cands(msa, orig, i, opt, shift, cand_length));
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

struct extract_anchor_linear_37 {
    using features_t = std::array<float, 37>;
    features_t operator()(const MultipleSequenceAlignment& msa, char orig, int i, const CorrectionOptions& opt) noexcept {   
        int a_begin = msa.subjectColumnsBegin_incl;
        int a_end = msa.subjectColumnsEnd_excl;
        int pos = a_begin + i;
        float countsACGT = msa.countsA[pos] + msa.countsC[pos] + msa.countsG[pos] + msa.countsT[pos];
        MSAProperties props = msa.getMSAProperties(a_begin, a_end, opt.estimatedErrorrate, opt.estimatedCoverage, opt.m_coverage);
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
            float(std::max(a_begin-pos, pos-a_end))
        };
    }
};

struct extract_cands_linear_40 {
    using features_t = std::array<float, 40>;
    features_t operator()(const MultipleSequenceAlignment& msa, char orig, int i, const CorrectionOptions& opt, int shift, int c_len) noexcept {   
        int a_begin = msa.subjectColumnsBegin_incl;
        int a_end = msa.subjectColumnsEnd_excl;
        int c_begin = a_begin + shift;
        int c_end = c_begin + c_len;
        int pos = c_begin + i;
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
            float(std::max(std::abs(shift), std::abs(a_end-c_end))), // absolute shift (compatible with differing read lengths)
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(a_end-a_begin), // relative overlap (ratio of a or c length in case of diff. read len)
            float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(c_end-c_begin),
            float(std::max(a_begin-pos, pos-a_end))
        };
    }
};

} //namespace 


//--------------------------------------------------------------------------------

using anchor_extractor = detail::extract_anchor_linear_37;
using cands_extractor = detail::extract_cands_linear_40;

using anchor_clf_t = ForestClf;
using cands_clf_t = ForestClf;

using ClfAgent = clf_agent<anchor_clf_t, cands_clf_t, anchor_extractor, cands_extractor>;


} // namespace care

#endif
