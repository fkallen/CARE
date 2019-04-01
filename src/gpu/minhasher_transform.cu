#include <minhasher.hpp>
#include <minhasher_transform.hpp>

#include <config.hpp>

#include <iostream>


namespace care{

    void Minhasher::transform(){

        for (std::size_t i = 0; i < minhashTables.size(); ++i){
            std::cout << "Transforming table " << i << std::endl;
            auto& tableptr = minhashTables[i];
            transform_keyvaluemap(*tableptr, deviceIds);
        }
    }

}
