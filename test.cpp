#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <numeric>
#include <cmath>

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
    uint32_t lhs, rhs;
};

using Tree = std::vector<Node>;
using Forest = std::vector<Tree>;

float decide(const std::vector<float>& features, const Tree& tree, size_t cur = 0) {
    const Node& node = tree[cur];
    uint32_t next;
    bool flag;
    if (features[node.att] < node.thresh) {
        next = node.lhs;
        flag = node.flag/2;
    } else {
        next = node.rhs;
        flag = node.flag%2;
    }
    if (flag) {
        float ret;
        std::memcpy(&ret, &next, sizeof(float));
        return ret;
    }
    return decide(features, tree, node.lhs);
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
        read_one(is, node.lhs);
    else {
        node.lhs = tree.size();
        read_tree(is, tree);
    }
    if (node.flag % 2)
        read_one(is, node.rhs);
    else {
        node.rhs = tree.size();
        read_tree(is, tree);
    }
}

void read_tree_(std::ifstream& is, Tree& tree) {
    std::vector<size_t> queue(1,0);
    while (!queue.empty()) {
        size_t i = queue.back();
        queue.pop_back();
        if (i == tree.size()) {
            queue.emplace_back(i);
            tree.emplace_back();
            read_one(is, tree[i].att);
            read_one(is, tree[i].thresh);
            read_one(is, tree[i].flag);
            if (tree[i].flag / 2)
                read_one(is, tree[i].lhs);
            else {
                tree[i].lhs = tree.size();
                queue.emplace_back(i+1);
            }
        } else {
            if (tree[i].flag % 2)
                read_one(is, tree[i].rhs);
            else {
                tree[i].rhs = tree.size();
                queue.emplace_back(tree.size());
            }
        }
    }
}

Forest read_forest() {
    std::ifstream is("/home/jc/Documents/errorcorrector/tree.bin", std::ios::binary);
    Forest forest(read_one<uint32_t>(is));
    for (Tree& tree: forest) {
        uint32_t num_nodes = read_one<uint32_t>(is);
        tree.reserve(num_nodes);
        read_tree(is, tree);
    }
    is.close();
    return forest;
}

int main() {
    Forest f = read_forest();
    // for (const Tree& t: f) {
    //     for (const Node& n: t) {
    //         std::cout << int(n.att) << " " << n.thresh << " " << int(n.flag) << " " << n.lhs << " " << n.rhs << std::endl;
    //     }
    // std::cout << std::endl;
    // }
}
