#ifndef CARE_FOREST_GPU_CUH
#define CARE_FOREST_GPU_CUH


#include <hpc_helpers.cuh>
#include <classification.hpp>

//#include <forest.hpp>

#if __CUDACC_VER_MAJOR__ >= 11

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#endif

namespace care{

template<class CpuForest>
class GpuForest {

    using Features = anchor_extractor::features_t;

    struct Node {
        uint8_t att;
        uint8_t flag;
        float thresh;
        union {
            uint32_t idx;
            float prob; 
        } lhs, rhs;
    };

    struct Clf{
        int numTrees;
        Node** data;

        HOSTDEVICEQUALIFIER
        float decide(const float* features, const Node* tree, size_t i = 0) const {
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

        HOSTDEVICEQUALIFIER
        float decide(const float* features) const {
            float prob = 0.f;
            for(int t = 0; t < numTrees; t++){
                prob += decide(features, data[t]);
            }
            return prob / numTrees;
        }

        #if __CUDACC_VER_MAJOR__ >= 11

            //correct result is returned by the first thread in the group
            template<class Group>
            DEVICEQUALIFIER
            float decide(Group& g, const float* features) const{
                float prob = 0.f;
                for(int t = g.thread_rank(); t < numTrees; t += g.size()){
                    prob += decide(features, data[t]);
                }
                prob = cg::reduce(g, prob, cg::plus<float>{});

                return prob / numTrees;
            }
        #endif
        
    };

    int deviceId;
    int numTrees;
    Node** h_data;


    GpuForest(const CpuForest& clf, int gpuId)
        : deviceId(gpuId) {

        int curgpu;
        cudaGetDevice(&curgpu); CUERR;
        cudaSetDevice(deviceId); CUERR;

        numTrees = clf.forest_.size();

        cudaMallocHost(&h_data, sizeof(Node*) * numTrees); CUERR;

        for(int t = 0; t < numTrees; t++){
            const int numNodes = clf.forest_[t].size();

            cudaMallocHost(&h_data[t], sizeof(Node) * numNodes); CUERR;

            for(int n = 0; n < numNodes; n++){
                const auto forestNode = clf.forest_[t][n];

                Node node;
                node.att = forestNode.att;
                node.flag = forestNode.flag;
                node.thresh = forestNode.thresh;
                node.lhs.idx = forestNode.lhs.idx;
                node.rhs.idx = forestNode.rhs.idx;

                h_data[t][n] = node;
            }
        }

        cudaSetDevice(curgpu); CUERR;
    }

    ~GpuForest(){
        int curgpu;
        cudaGetDevice(&curgpu); CUERR;
        cudaSetDevice(deviceId); CUERR;

        for(int t = 0; t < numTrees; t++){
            cudaFreeHost(h_data[t]); CUERR;
        }
        cudaFreeHost(h_data); CUERR;


        cudaSetDevice(curgpu); CUERR;
    }

    operator Clf() const{
        return getClf();
    }

    Clf getClf() const{
        return Clf{numTrees, h_data};
    }

    float decide(const float* features) const {
        Clf clf = getClf();

        float prob = 0.f;
        for(int t = 0; t < numTrees; t++){
            prob += clf.decide(features, h_data[t]);
        }
        return prob / numTrees;
    }

    float decide(const Features& features) const {
        return decide(features.data());
    }
};



}


#endif