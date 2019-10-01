#include <minhasher.hpp>
#include <minhasher_transform.hpp>

#include <config.hpp>

#include <iostream>
#include <vector>




namespace care{

    

    void transform_minhasher_gpu(Minhasher& minhasher, int map, const std::vector<int>& deviceIds){
        assert(map < int(minhasher.minhashTables.size()));

        auto& tableptr = minhasher.minhashTables[map];
        int maxValuesPerKey = minhasher.getResultsPerMapThreshold();
        if(!tableptr->noMoreWrites){
            std::cerr << "Transforming table " << map << ". ";
            transform_keyvaluemap_gpu(*tableptr, deviceIds, maxValuesPerKey);
        }
    }

    void transform_minhasher_gpu(Minhasher& minhasher, const std::vector<int>& deviceIds){
        for (std::size_t i = 0; i < minhasher.minhashTables.size(); ++i){
            transform_minhasher_gpu(minhasher, i, deviceIds);
        }
    }

}
