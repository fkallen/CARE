#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
// #include <numeric>

namespace care {

template<typename T>
T& read_one(std::ifstream& is, T& v) {
    char tmp[sizeof(v)];
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
using Forest = std::vector<Tree>;

float decide(const std::vector<float>& features, const Tree& tree, size_t i = 0) {
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

float decide(const std::vector<float>& features, const Forest& forest) {
    float prob = 0.f;
    for (const Tree& tree: forest)
        prob += decide(features, tree);
    return prob/forest.size();
}

void read_tree(std::ifstream& is, Tree& tree) {
    Node& node = *tree.emplace(tree.end());
    read_one(is, node.att);
    read_one(is, node.thresh);
    read_one(is, node.flag);
    if (node.flag / 2)
        new(&node.lhs.prob) float(read_one<float>(is));
    else {
        new(&node.lhs.idx) uint32_t(tree.size());
        read_tree(is, tree);
    }
    if (node.flag % 2)
        new(&node.rhs.prob) float(read_one<float>(is));
    else {
        new(&node.rhs.idx) uint32_t(tree.size());
        read_tree(is, tree);
    }
}

Forest read_forest(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    Forest forest(read_one<uint32_t>(is));
    for (Tree& tree: forest) {
        uint32_t num_nodes = read_one<uint32_t>(is);
        tree.reserve(num_nodes);
        read_tree(is, tree);
    }
    is.close();
    return forest;
}

} // namespace care
