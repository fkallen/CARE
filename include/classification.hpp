#ifndef CARE_CLASSIFICATION_HPP
#define CARE_CLASSIFICATION_HPP

#include <array>
#include <random>
#include <forest.hpp>
#include <msa.hpp>


// This header allows toggling of feature transformations and classifiers.
// SEE BOTTOM OF FILE FOR TOGGLES! 

// TODO: implement logistic regression and investigate peformance
// Possibly same accuracy with VASTLY superior performance

// The current features were designed with the possibility of using logistic regression in mind
// but are highly redundant for any decision tree. This header serves to seperate
// feature extraction from the rest of the code.

namespace care {

template<typename AnchorFeatures,
         typename CandFeatures,
         typename AnchorClf,
         typename CandClf,
         AnchorFeatures (*anchor_extractor)(const care::MultipleSequenceAlignment&, const care::MSAProperties&, char, size_t, float), // c++17: "auto"
         CandFeatures (*cands_extractor)(const care::MultipleSequenceAlignment&, const care::MSAProperties&, char, size_t, float)>
struct clf_agent
{

    //TODO: access permission
    std::shared_ptr<AnchorClf> classifier_anchor;
    std::shared_ptr<CandClf> classifier_cands;
    std::mt19937 rng;
    std::bernoulli_distribution coinflip;
    std::stringstream anchor_stream, cands_stream;

    clf_agent(std::shared_ptr<AnchorClf> clf_a, std::shared_ptr<CandClf> clf_c) :
        classifier_anchor(clf_a),
        classifier_cands(clf_c),
        rng(std::mt19937(std::chrono::system_clock::now().time_since_epoch().count() + std::hash<std::thread::id>{}(std::this_thread::get_id()))),
        coinflip(0.01)
    {}

    //TODO: if this could just get task as parameter, everthing would look much nicer, consider un-private-ing stuff?
    void print_anchor(const care::MultipleSequenceAlignment& msa, const care::MSAProperties& props, char orig, size_t msa_pos, float norm, read_number read_id, int read_pos) {       
        AnchorFeatures sample = anchor_extractor(msa, props, orig, msa_pos, norm);
        anchor_stream << read_id << ' ' << read_pos << ' ';
        for (float j: sample) anchor_stream << j << ' ';
        anchor_stream << '\n';
    }

    void print_cand(const care::MultipleSequenceAlignment& msa, const care::MSAProperties& props, char orig, size_t msa_pos, float norm, read_number read_id, int read_pos) {       
        if (!coinflip(rng)) return;

        CandFeatures sample = cands_extractor(msa, props, orig, msa_pos, norm);
        cands_stream << read_id << ' ' << read_pos << ' ';
        for (float j: sample) cands_stream << j << ' ';
        cands_stream << '\n';
    }

    float decide_anchor(const care::MultipleSequenceAlignment& msa, const care::MSAProperties& props, char orig, size_t pos, float norm) {       
        return classifier_anchor->decide(anchor_extractor(msa, props, orig, pos, norm));
    }

    float decide_cand(const care::MultipleSequenceAlignment& msa, const care::MSAProperties& props, char orig, size_t pos, float norm) {       
        return classifier_cands->decide(cands_extractor(msa, props, orig, pos, norm));
    }
};


std::array<float, 36> extract_anchor_linear_36(const care::MultipleSequenceAlignment& msa, const care::MSAProperties& props, char orig, size_t pos, float norm) noexcept
{   
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


//--------------------------------------------------------------------------------

using anchor_features_t = std::array<float, 36>;
using cands_features_t = std::array<float, 36>;

using anchor_clf_t = ForestClf<anchor_features_t>;
using cands_clf_t = ForestClf<cands_features_t>;

// // c++17 only
// // c++14: "tHE tEmplAte aRgUMent dOenS't hAvE eXtErnAL liNKAGe" 
// constexpr auto& anchor_extractor = extract_anchor_linear_36;
// constexpr auto& cands_extractor = extract_anchor_linear_36;
// using ClfAgent = clf_agent<anchor_features_t, cands_features_t, anchor_clf_t, cands_clf_t, anchor_extractor, cands_extractor>;

using ClfAgent = clf_agent<anchor_features_t, cands_features_t, anchor_clf_t, cands_clf_t, extract_anchor_linear_36, extract_anchor_linear_36>;


} // namespace care

#endif

