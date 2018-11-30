#ifndef CARE_UTIL_HPP
#define CARE_UTIL_HPP

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <functional>


/*
    Removes elements from sorted range which occure less than k times.
    Returns end of the new range
*/

template<class Iter, class Equal>
Iter remove_by_count(Iter first,
                    Iter last,
                    std::size_t k,
                    Equal equal){

    using T = typename Iter::value_type;

    if(std::distance(first, last) == 0) return first;

    Iter copytobegin = first;
    Iter copyfrombegin = first;
    Iter copyfromend = first;
    ++copyfromend;

    const T* curElem = &(*copyfrombegin);
    std::size_t count = 1;

    while(copyfromend != last){

        if(equal(*curElem, *copyfromend)){
            ++count;
        }else{
            if(count < k){
                copyfrombegin = copyfromend;
            }else{
                copytobegin = std::copy(copyfrombegin, copyfromend, copytobegin);
                copyfrombegin = copyfromend;
            }

            curElem = &(*copyfromend);
            count = 1;
        }

        ++copyfromend;
    }

    //handle last element
    if(count >= k)
        copytobegin = std::copy(copyfrombegin, copyfromend, copytobegin);

    return copytobegin;
}

template<class Iter>
Iter remove_by_count(Iter first,
                        Iter last,
                        std::size_t k){
    using T = typename Iter::value_type;
    return remove_by_count(first, last, k, std::equal_to<T>{});
}

/*
    Removes elements from sorted range which occure less than k times.
    If a range of equals elements is greater than or equal to k, only the first element of this range is kept.
    Returns end of the new range.
    The new range is empty if there are more than max_num_elements unique elements
*/

template<class Iter, class Equal>
Iter remove_by_count_unique_with_limit(Iter first,
                    Iter last,
                    std::size_t k,
                    std::size_t max_num_elements,
                    Equal equal){
    using T = typename Iter::value_type;

    constexpr std::size_t elements_to_copy = 1;

    if(std::distance(first, last) == 0) return first;
    if(elements_to_copy > k) return first;

    Iter copytobegin = first;
    Iter copyfrombegin = first;
    Iter copyfromend = first;
    ++copyfromend;

    std::size_t num_copied_elements = 0;

    const T* curElem = &(*copyfrombegin);
    std::size_t count = 1;

    while(copyfromend != last && num_copied_elements <= max_num_elements){

        if(equal(*curElem, *copyfromend)){
            ++count;
        }else{
            if(count < k){
                copyfrombegin = copyfromend;
            }else{
                copytobegin = std::copy_n(copyfrombegin, elements_to_copy, copytobegin);
                copyfrombegin = copyfromend;
                num_copied_elements += elements_to_copy;
            }

            curElem = &(*copyfromend);
            count = 1;
        }

        ++copyfromend;
    }

    //handle last element
    if(count >= k)
        copytobegin = std::copy_n(copyfrombegin, elements_to_copy, copytobegin);

    if(num_copied_elements > max_num_elements)
        return first;

    return copytobegin;
}

template<class Iter>
Iter remove_by_count_unique_with_limit(Iter first,
                    Iter last,
                    std::size_t k,
                    std::size_t max_num_elements){

    using T = typename Iter::value_type;
    return remove_by_count_unique_with_limit(first, last, k, max_num_elements, std::equal_to<T>{});
}


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
