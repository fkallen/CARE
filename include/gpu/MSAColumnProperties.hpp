#ifndef CARE_MSA_COLUMN_PROPERTIES_HPP
#define CARE_MSA_COLUMN_PROPERTIES_HPP

namespace care {
namespace gpu {



struct MSAColumnProperties{
    int subjectColumnsBegin_incl;
    int subjectColumnsEnd_excl;
    int firstColumn_incl;
    int lastColumn_excl;
};
 
}
}
#endif
