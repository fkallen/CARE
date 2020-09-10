#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
// #include <numeric>

namespace care {

// de-serialization helpers
template<typename T>
T& read_one(std::ifstream& is, T& v) {
    char tmp[sizeof(v)];
    // this is only to be absolutely 100% standard-compliant no matter how read() is implemented
    // probably absolutely unnecessary but it will be optimized out
    is.read(tmp, sizeof(v));
    std::memcpy(&v, tmp, sizeof(v));
    return v;
}

template<typename T>
T read_one(std::ifstream& is) {
    T ret;
    read_one(is, ret);
    return ret;
}

using ml_sample_t = std::array<float, 32>;

class ForestClf {

    struct Node {
        uint8_t att;
        float thresh;
        uint8_t flag;
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

    float decide(const ml_sample_t& features, const Tree& tree, size_t i = 0) const {
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

public:

    ForestClf (const std::string& path) {
        std::ifstream is(path, std::ios::binary);
        forest_ = Forest(read_one<uint32_t>(is));
        for (Tree& tree: forest_) {
            tree.reserve(read_one<uint32_t>(is));
            populate(is, tree);
        }
        is.close();
    }

    float decide(const ml_sample_t& features) const {
        float prob = 0.f;
        for (const Tree& tree: forest_)
            prob += decide(features, tree);
        return prob/forest_.size();
    }
};

} // namespace care
