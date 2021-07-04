#ifndef CARE_UTIL_ITERATOR_HPP
#define CARE_UTIL_ITERATOR_HPP

#include <hpc_helpers.cuh>

#include <iterator>

template<class Iter>
struct IteratorMultiplier{
    using value_type = typename std::iterator_traits<Iter>::value_type;

    int factor{};
    Iter data{};

    HOSTDEVICEQUALIFIER
    IteratorMultiplier(Iter data_, int factor_)
        : factor(factor_), data(data_){

    }

    HOSTDEVICEQUALIFIER
    value_type operator()(int i) const{
        return *(data + (i / factor));
    }
};

template<class Iter>
HOSTDEVICEQUALIFIER
IteratorMultiplier<Iter> make_iterator_multiplier(Iter data, int factor){
    return IteratorMultiplier<Iter>{data, factor};
}




#endif