#ifndef CARE_GPU_MSA_HPP
#define CARE_GPU_MSA_HPP

#include "bestalignment.hpp"

namespace care{
namespace gpu{

    struct MSAColumnProperties{
        int startindex;
        int endindex;
        int columnsToCheck;
        int subjectColumnsBegin_incl;
        int subjectColumnsEnd_excl;
    };

    }
}


#endif
