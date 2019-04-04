#include <minhasher.hpp>
#include <minhasher_transform.hpp>

#include <config.hpp>

#include <iostream>
#include <vector>


namespace care{

    void transform_minhasher(Minhasher& minhasher, const std::vector<int>& deviceIds){
        for (std::size_t i = 0; i < minhasher.minhashTables.size(); ++i){
            std::cout << "Transforming table " << i << std::endl;
            auto& tableptr = minhasher.minhashTables[i];
            transform_keyvaluemap(*tableptr, deviceIds);
        }
    }

}
