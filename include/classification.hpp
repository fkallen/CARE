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
    void print_anchor(const care::MultipleSequenceAlignment& msa, const care::MSAProperties& props, char orig, size_t msa_pos, float norm, read_number read_id, int read_pos) {       
        if (!coinflip_anchor(rng)) return;


        anchor_stream << read_id << ' ' << read_pos << ' ';
        for (float j: extract_anchor(msa, props, orig, msa_pos, norm))
            anchor_stream << j << ' ';
        anchor_stream << '\n';
    }

    void print_cand(const care::MultipleSequenceAlignment& msa, const care::MSAProperties& props, char orig, size_t msa_pos, float norm, read_number read_id, int read_pos) {       
        if (!coinflip_cands(rng)) return;

        cands_stream << read_id << ' ' << read_pos << ' ';
        for (float j: extract_cands(msa, props, orig, msa_pos, norm))
            cands_stream << j << ' ';
        cands_stream << '\n';
    }

    float decide_anchor(const care::MultipleSequenceAlignment& msa, const care::MSAProperties& props, char orig, size_t pos, float norm) {       
        return classifier_anchor->decide(extract_anchor(msa, props, orig, pos, norm));
    }

    float decide_cand(const care::MultipleSequenceAlignment& msa, const care::MSAProperties& props, char orig, size_t pos, float norm) {       
        return classifier_cands->decide(extract_cands(msa, props, orig, pos, norm));
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

struct extract_anchor_linear_36 {
    using features_t = std::array<float, 36>;
    features_t operator()(const MultipleSequenceAlignment& msa, const MSAProperties& props, char orig, size_t pos, float norm) noexcept {   
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
            props.avg_support,
            props.min_support,
            float(props.max_coverage)/norm,
            float(props.min_coverage)/norm
        };
    }
};

} //namespace 


//--------------------------------------------------------------------------------

using anchor_extractor = detail::extract_anchor_linear_36;
using cands_extractor = detail::extract_anchor_linear_36;

using anchor_clf_t = ForestClf;
using cands_clf_t = ForestClf;

using ClfAgent = clf_agent<anchor_clf_t, cands_clf_t, anchor_extractor, cands_extractor>;


} // namespace care

#endif
