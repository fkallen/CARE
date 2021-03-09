#ifndef CARE_FOREST_HPP
#define CARE_FOREST_HPP


#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <array>
#include <deserialize.hpp>
// #include <numeric>

namespace care {

namespace gpu{
    template<class CpuForest>
    class GpuForest;
}

class ForestClf {
    template<class CpuForest>
    friend class care::gpu::GpuForest;

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

    template<typename features_t>
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
        if(!bool(is)){
            throw std::runtime_error("Cannot open forest file " + path);
        }
        
        forest_ = Forest(read_one<uint32_t>(is));
        for (Tree& tree: forest_) {
            tree.reserve(read_one<uint32_t>(is));
            populate(is, tree);
        }
        is.close();
    }

    void threshold(float t) {
        thresh_ = t;
    }

    float threshold() const {
        return thresh_;
    }

    template<typename features_t>
    bool decide(const features_t& features) const {
        float prob = 0.f;
        for (const Tree& tree: forest_)
            prob += decide(features, tree);
        return prob/forest_.size() >= thresh_;
    }
};




} // namespace care


#endif
