#include <minhasher.hpp>
#include <minhasher_transform.hpp>

#include <config.hpp>

#include <iostream>
#include <vector>


namespace care{

    template<class KeyValueMap>
    void transform_keyvaluemap(KeyValueMap& map, const std::vector<int>& deviceIds){
        if(map.noMoreWrites) return;

        if(map.size == 0) return;

        cpu_transformation(map.keys, map.values, map.countsPrefixSum);

        map.nKeys = map.keys.size();
        map.noMoreWrites = true;


        /*keyIndexMap = KeyIndexMap(nKeys / load);
        for(Index_t i = 0; i < nKeys; i++){
            keyIndexMap.insert(keys[i], i);
        }
        for(Index_t i = 0; i < nKeys; i++){
            assert(keyIndexMap.get(keys[i]) == i);
        }*/
    }

    void transform_minhasher(Minhasher& minhasher, const std::vector<int>& deviceIds){
        for (std::size_t i = 0; i < minhasher.minhashTables.size(); ++i){
            std::cout << "Transforming table " << i << std::endl;
            auto& tableptr = minhasher.minhashTables[i];
            transform_keyvaluemap(*tableptr, deviceIds);
        }
    }

}
