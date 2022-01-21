#ifndef CARE_FOREST_HPP
#define CARE_FOREST_HPP


#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <array>
#include <deserialize.hpp>

namespace care {

namespace gpu{
    class GpuForest;
}

template<typename extractor_t>
class ForestClf {
    friend class care::gpu::GpuForest;

    struct Node {
        uint8_t att;
        uint8_t flag;
        double thresh;
        union {
            uint32_t idx;
            float prob; 
        } lhs, rhs;
    };

    using Tree = std::vector<Node>;
    using features_t = typename extractor_t::features_t;

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

    float prob(const features_t& features, const Tree& tree, size_t i = 0) const {
        if (features[tree[i].att] <= tree[i].thresh) {
            if (tree[i].flag / 2)
                return tree[i].lhs.prob;
            return prob(features, tree, tree[i].lhs.idx);
        } else {
            if (tree[i].flag % 2)
                return tree[i].rhs.prob;
            return prob(features, tree, tree[i].rhs.idx);
        }
    }

    using Forest = std::vector<Tree>;
    
    Forest forest_;
    float thresh_;

public:

    ForestClf (const std::string& path, std::uint32_t maxNumTrees, float t = 0.5f) : 
        thresh_(t) 
    {
        std::ifstream is(path, std::ios::binary);
        if (!is)
            throw std::runtime_error("Loading classifier file failed! " + path);
        
        auto desc = read_str(is);
        auto expected = std::string(extractor_t());
        if (desc != expected)
            throw std::runtime_error("Classifier and extractor descriptors do not match! Expected: " + expected + " Received: " + desc);

        const auto numTrees = read_one<uint32_t>(is);
        const auto usableNumTrees = std::max(1u, std::min(numTrees, maxNumTrees));
        //std::cerr << "numTrees = " << numTrees << ", usableNumTrees = " << usableNumTrees << "\n";
        forest_ = Forest(usableNumTrees);
        for (Tree& tree: forest_) {
            const auto numNodesInTree = read_one<uint32_t>(is);
            tree.reserve(numNodesInTree);
            populate(is, tree);
        }
    }

    void threshold(float t) {
        thresh_ = t;
    }

    float threshold() const {
        return thresh_;
    }

    float prob(const features_t& features) const {
        float sum = 0.f;
        for (const Tree& tree: forest_)
            sum += prob(features, tree);
        return sum/forest_.size();
    }

    float prob_debug(const features_t& features) const {
        float sum = 0.f;
        std::cerr << "# ";
        for (const Tree& tree: forest_) {
            auto tmp = prob(features, tree);
            sum += tmp;
            std::cerr << tmp << " ";
        }
        std::cerr << "# ";
        return sum/forest_.size();
    }

    bool decide(const features_t& features) const {
        return prob(features) >= thresh_;
    }
};




} // namespace care


#endif
