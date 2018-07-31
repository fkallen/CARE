#ifndef CARE_UTIL_HPP
#define CARE_UTIL_HPP

#include <iterator>
#include <cstdint>
#include <algorithm>
#include <iostream>

template<class InputIt1, class InputIt2, class OutputIt>
OutputIt set_union_n(InputIt1 first1, InputIt1 last1,
                   InputIt2 first2, InputIt2 last2,
                   std::size_t n,
                   OutputIt d_first){

    for (; first1 != last1 && n > 0; ++d_first, --n) {
        if (first2 == last2){
            const std::size_t remaining = std::distance(first1, last1);
            return std::copy_n(first1, std::min(remaining, n), d_first);
        }

        if (*first2 < *first1) {
            *d_first = *first2++;
        } else {
            *d_first = *first1;
            if (!(*first1 < *first2))
                ++first2;
            ++first1;
        }
    }

    const std::size_t remaining = std::distance(first2, last2);
    return std::copy_n(first2, std::min(remaining, n), d_first);
}

#endif
