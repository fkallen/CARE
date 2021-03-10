#ifndef CARE_FOREST_GPU_CUH
#define CARE_FOREST_GPU_CUH


#include <hpc_helpers.cuh>
#include <classification.hpp>
#include <utility>

//#include <forest.hpp>

#if __CUDACC_VER_MAJOR__ >= 11

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#endif

namespace care{
namespace gpu{

struct GpuForest {

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
        bool decide(const float* features, float thresh) const {
            float prob = 0.f;
            for(int t = 0; t < numTrees; t++){
                prob += decide(features, data[t]);
            }
            //printf("%f %d %f\n", prob, numTrees, thresh);
            return prob / numTrees >= thresh;
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

public:
    int deviceId{};
    int numTrees{};
    Node** d_data{};


    GpuForest() = default;

    template<class CpuForest>
    GpuForest(const CpuForest& clf, int gpuId)
        : deviceId(gpuId) {

        int curgpu;
        cudaGetDevice(&curgpu); CUERR;
        cudaSetDevice(deviceId); CUERR;

        numTrees = clf.forest_.size();

        std::vector<Node*> devicePointers(numTrees);

        for(int t = 0; t < numTrees; t++){
            const int numNodes = clf.forest_[t].size();

            std::vector<Node> nodes(numNodes);

            for(int n = 0; n < numNodes; n++){
                const auto forestNode = clf.forest_[t][n];

                Node node;
                node.att = forestNode.att;
                node.flag = forestNode.flag;
                node.thresh = forestNode.thresh;
                node.lhs.idx = forestNode.lhs.idx;
                node.rhs.idx = forestNode.rhs.idx;

                nodes[n] = node;
            }

            cudaMalloc(&devicePointers[t], sizeof(Node) * numNodes); CUERR;
            cudaMemcpy(
                devicePointers[t],
                nodes.data(),
                sizeof(Node) * numNodes,
                H2D
            );CUERR;
        }

        cudaMalloc(&d_data, sizeof(Node*) * numTrees);

        cudaMemcpy(
            d_data,
            devicePointers.data(),
            sizeof(Node*) * numTrees,
            H2D
        );CUERR;

        cudaSetDevice(curgpu); CUERR;
    }

    GpuForest(GpuForest&& rhs){
        deviceId = std::exchange(rhs.deviceId, 0);
        numTrees = std::exchange(rhs.numTrees, 0);
        d_data = std::exchange(rhs.d_data, nullptr);
    }

    GpuForest(const GpuForest& rhs) = delete;

    GpuForest& operator=(GpuForest&& rhs){
        deviceId = std::exchange(rhs.deviceId, 0);
        numTrees = std::exchange(rhs.numTrees, 0);
        d_data = std::exchange(rhs.d_data, nullptr);
        return *this;
    }

    GpuForest& operator=(const GpuForest& rhs) = delete;

    ~GpuForest(){
        int curgpu;
        cudaGetDevice(&curgpu); CUERR;
        cudaSetDevice(deviceId); CUERR;

        std::vector<Node*> devicePointers(numTrees);
        cudaMemcpy(
            devicePointers.data(),
            d_data,
            sizeof(Node*) * numTrees,
            D2H
        );CUERR;

        for(int t = 0; t < numTrees; t++){
            cudaFree(devicePointers[t]); CUERR;
        }
        cudaFree(d_data); CUERR;


        cudaSetDevice(curgpu); CUERR;
    }

    operator Clf() const{
        return getClf();
    }

    Clf getClf() const{
        return Clf{numTrees, d_data};
    }

};



}
}


#endif