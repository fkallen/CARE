#include <minhasher.hpp>
#include <minhasher_transform.hpp>

#include <config.hpp>

#include <iostream>
#include <vector>


namespace care{

    template<class KeyValueMap>
    void transform_keyvaluemap(KeyValueMap& map){
        if(map.noMoreWrites) return;

        if(map.size == 0) return;

        cpu_transformation(map.keys, map.values, map.countsPrefixSum);

        map.nKeys = map.keys.size();
        map.noMoreWrites = true;

        using Key_t = typename KeyValueMap::Key_t;
        using Index_t = typename KeyValueMap::Index_t;

        map.keyIndexMap = minhasherdetail::KeyIndexMap<Key_t, Index_t>(map.nKeys / map.load);
        for(Index_t i = 0; i < map.nKeys; i++){
            map.keyIndexMap.insert(map.keys[i], i);
        }

        {
            std::vector<Key_t> tmp;
            map.keys.swap(tmp);
        }

        /*keyIndexMap = KeyIndexMap(nKeys / load);
        for(Index_t i = 0; i < nKeys; i++){
            keyIndexMap.insert(keys[i], i);
        }
        for(Index_t i = 0; i < nKeys; i++){
            assert(keyIndexMap.get(keys[i]) == i);
        }*/
    }

    void transform_minhasher(Minhasher& minhasher, int map){
        assert(map < int(minhasher.minhashTables.size()));

        auto& tableptr = minhasher.minhashTables[map];
        if(!tableptr->noMoreWrites){
            std::cerr << "Transforming table " << map << std::endl;
            transform_keyvaluemap(*tableptr);
        }
    }

    void transform_minhasher(Minhasher& minhasher){
        for (std::size_t i = 0; i < minhasher.minhashTables.size(); ++i){
            transform_minhasher(minhasher, i);
        }
    }

}
