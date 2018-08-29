#ifndef CARE_UTIL_HPP
#define CARE_UTIL_HPP

#include <algorithm>
#include <cstdint>
#include <iterator>

/*
    Essentially performs std::set_union(first1, last1, first2, last2, d_first)
    but limits the allowed result size to n.
    If the result would contain more than n elements, d_first is returned, i.e. the result range is empty
*/
template<class InputIt1, class InputIt2, class OutputIt>
OutputIt set_union_n_or_empty(InputIt1 first1, InputIt1 last1,
                   InputIt2 first2, InputIt2 last2,
                   std::size_t n,
                   OutputIt d_first){

    const OutputIt d_first_old = d_first;

    for (; first1 != last1 && n > 0; ++d_first, --n) {
        if (first2 == last2){
            const std::size_t remaining = std::distance(first1, last1);
            if(remaining > n)
                return d_first_old;
            else
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
    if(remaining > n)
        return d_first_old;
    else
        return std::copy_n(first2, std::min(remaining, n), d_first);
}

/*
    Essentially performs std::set_union(first1, last1, first2, last2, d_first)
    but limits the allowed result size to n.
*/
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








/*
    Bit shifts of bit array
*/

void shiftBitsLeftBy(unsigned char* array, int bytes, int shiftamount);

void shiftBitsRightBy(unsigned char* array, int bytes, int shiftamount);

void shiftBitsBy(unsigned char* array, int bytes, int shiftamount);

int hammingdistanceHiLo(const std::uint8_t* l, const std::uint8_t* r, int length_l, int length_r, int bytes);


#endif
