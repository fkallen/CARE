#ifndef CARE_BUILD_HPP
#define CARE_BUILD_HPP

#include "minhasher.hpp"
#include "readstorage.hpp"
#include "sequencefileio.hpp"

namespace care{

void build(const FileOptions& fileOptions, ReadStorage& readStorage,
            Minhasher& minhasher, int nThreads);


}



#endif
