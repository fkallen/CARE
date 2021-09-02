#include <iostream>
#include <set>
#include <utility>

void printarch(int major, int minor){
    std::cout << "-gencode=arch=compute_" << major << minor << ",code=sm_" << major << minor << " ";
}

void printDefault(){
    std::cout << "-gencode=arch=compute_60,code=compute_60";
}

int main(int argc, char** argv){
    int numGpus = 0;
    cudaError_t status = cudaSuccess;
    status = cudaGetDeviceCount(&numGpus);

    std::set<std::pair<int, int>> allgpusset;

    if(status == cudaSuccess && numGpus > 0){
        for(int d = 0; d < numGpus; d++){
            int major = 0;
            int minor = 0;
            status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, d);
            status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, d);
            if(status == cudaSuccess){
                allgpusset.insert(std::make_pair(major, minor));
            }
        }
    }

    std::set<std::pair<int, int>> usablegpusset;
    for(const auto& p : allgpusset){
        if(p.first >= 6){
            usablegpusset.insert(p);
        }
    }

    if(usablegpusset.empty()){
        printDefault();
    }else{
        for(const auto& p : usablegpusset){
            printarch(p.first, p.second);
        }
    }

    std::cout << "\n";
}
