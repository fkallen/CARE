#ifndef CARE_FOREST_HPP
#define CARE_FOREST_HPP


#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <array>
#include <deserialize.hpp>
// #include <numeric>

namespace care {

template<size_t feature_count>
class ForestClf {
    template<class CpuForest>
    friend class GpuForest;

    struct Node {
        uint8_t att;
        uint8_t flag;
        float thresh;
        union {
            uint32_t idx;
            float prob; 
        } lhs, rhs;
    };

    using Tree = std::vector<Node>;
    using features_t = std::array<float, feature_count>; // for now, only std::arrays are allowed

    void populate(std::ifstream& is, Tree& tree) {
        Node& node = *tree.emplace(tree.end());
        read_one(is, node.att);
        read_one(is, node.thresh);
        read_one(is, node.flag);
        if (node.flag / 2)
            new(&node.lhs.prob) float(read_one<float>(is));
        else {
            new(&node.lhs.idx) uint32_t(tree.size());
            populate(is, tree);
        }
        if (node.flag % 2)
            new(&node.rhs.prob) float(read_one<float>(is));
        else {
            new(&node.rhs.idx) uint32_t(tree.size());
            populate(is, tree);
        }
    }

    float decide(const features_t& features, const Tree& tree, size_t i = 0) const {
        if (features[tree[i].att] < tree[i].thresh) {
            if (tree[i].flag / 2)
                return tree[i].lhs.prob;
            return decide(features, tree, tree[i].lhs.idx);
        } else {
            if (tree[i].flag % 2)
                return tree[i].rhs.prob;
            return decide(features, tree, tree[i].rhs.idx);
        }
    }

    using Forest = std::vector<Tree>;
    
    Forest forest_;
    float thresh_;

public:

    ForestClf (const std::string& path, float t = 0.5f) : 
        thresh_(t) 
    {
        std::ifstream is(path, std::ios::binary);
        
        if (!is)
            throw std::runtime_error("Loading classifier file failed!");

        size_t file_feat_count = read_one<uint8_t>(is);
        if (file_feat_count != feature_count)
            throw std::runtime_error("Classifier feature shape does not match feature extractor! Expected: " +std::to_string(feature_count) + " Got: "+std::to_string(file_feat_count));

        forest_ = Forest(read_one<uint32_t>(is));
        for (Tree& tree: forest_) {
            tree.reserve(read_one<uint32_t>(is));
            populate(is, tree);
        }
    }

    void threshold(float t) {
        thresh_ = t;
    }

    float threshold() const {
        return thresh_;
    }

    bool decide(const features_t& features) const {
        float prob = 0.f;
        for (const Tree& tree: forest_)
            prob += decide(features, tree);
        return prob/forest_.size() >= thresh_;
    }
};




} // namespace care


#endif
