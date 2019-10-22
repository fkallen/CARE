#include <minhasher.hpp>
#include <minhasher_transform.hpp>

#include <config.hpp>

#include <iostream>
#include <vector>


namespace care{

    void transform_minhasher(Minhasher& minhasher, int map){
        assert(map < int(minhasher.minhashTables.size()));

        auto& tableptr = minhasher.minhashTables[map];
        int maxValuesPerKey = minhasher.getResultsPerMapThreshold();
        if(!tableptr->noMoreWrites){
            std::cout << "Transforming table " << map << ". ";
            transform_keyvaluemap(*tableptr, maxValuesPerKey);
        }
    }

    void transform_minhasher(Minhasher& minhasher){
        for (std::size_t i = 0; i < minhasher.minhashTables.size(); ++i){
            transform_minhasher(minhasher, i);
        }
    }

}
