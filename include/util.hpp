#ifndef CARE_UTIL_HPP
#define CARE_UTIL_HPP

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <functional>
#include <cmath>
#include <numeric>
#include <vector>

/*
    Merge ranges [first1, last1) and [first2, last2) into range beginning at d_first.
    If more than max_num_elements unique elements in the result range
    would occur >= threshold times, an empty range is returned.
*/
template<class OutputIt, class Iter1, class Iter2>
OutputIt merge_with_count_theshold(Iter1 first1, Iter1 last1,
                        Iter2 first2, Iter2 last2,
                        std::size_t threshold,
                        std::size_t max_num_elements,
                        OutputIt d_first){
    //static_assert(std::is_same<typename Iter1::value_type, typename Iter2::value_type>::value, "");
    static_assert(std::is_same<typename std::iterator_traits<Iter1>::value_type, typename std::iterator_traits<Iter2>::value_type>::value, "");

    using T = typename std::iterator_traits<Iter1>::value_type;

    OutputIt d_first_orig = d_first;

    T previous{};
    std::size_t count = 0;
    bool foundone = false;

    auto update = [&](){
        if(*d_first == previous){
            ++count;
        }else{
            if(count >= threshold){
                if(foundone){
                    --max_num_elements;
                }
                foundone = true;
            }
            previous = *d_first;
            count = 1;
        }
    };

    for (; first1 != last1 && max_num_elements > 0; ++d_first) {
        if (first2 == last2) {
            while(first1 != last1 && max_num_elements > 0){
                *d_first = *first1;
                update();
                ++d_first;
                ++first1;
            }
            break;
        }
        if (*first2 < *first1) {
            *d_first = *first2;
            ++first2;
        } else {
            *d_first = *first1;
            ++first1;
        }

        update();
    }

    while(first2 != last2 && max_num_elements > 0){
        *d_first = *first2;
        update();
        ++d_first;
        ++first2;
    }

    if(max_num_elements == 0 || (max_num_elements == 1 && count >= threshold))
        return d_first_orig;
    else
        return d_first;
}

template<class OutputIt, class Iter>
OutputIt k_way_set_intersection_naive(OutputIt destinationbegin, const std::vector<Iter>& iters){
    static_assert(std::is_same<typename OutputIt::value_type, typename Iter::value_type>::value, "");

    using T = typename Iter::value_type;

    // at least one range is invalid
    if(iters.size() % 2 == 1)
        return destinationbegin;

    if(iters.size() == 0)
        return destinationbegin;

    if(iters.size() == 4)
        return std::set_intersection(iters[0], iters[1], iters[2], iters[3], destinationbegin);

    std::size_t nranges = iters.size()/2;

    std::size_t num_elements = 0;
    for(std::size_t i = 0; i < iters.size() / 2; i++){
        num_elements += std::distance(iters[2*i + 0], iters[2*i+1]);
    }

    std::vector<T> tmpbuffer(num_elements);
    std::size_t merged_elements = 0;

    for(std::size_t i = 0; i < nranges; i++){
        auto src_begin = i % 2 == 0 ? tmpbuffer.begin() : destinationbegin;
        auto src_end = src_begin + merged_elements;
        auto dest_begin = i % 2 == 0 ? destinationbegin : tmpbuffer.begin();

        auto dest_end = std::set_intersection(src_begin, src_end,
                                    iters[2*i + 0], iters[2*i+1],
                                    dest_begin);

        merged_elements = std::distance(dest_begin, dest_end);
    }

    auto destinationend = destinationbegin + merged_elements;

    if(nranges % 2 == 0){
        destinationend = std::copy(tmpbuffer.begin(), tmpbuffer.begin() + merged_elements, destinationbegin);
    }

    return destinationend;
}


template<class OutputIt, class Iter>
OutputIt k_way_merge_naive(OutputIt destinationbegin, const std::vector<Iter>& iters){
    static_assert(std::is_same<typename OutputIt::value_type, typename Iter::value_type>::value, "");

    using T = typename Iter::value_type;

    // at least one range is invalid
    if(iters.size() % 2 == 1)
        return destinationbegin;

    if(iters.size() == 0)
        return destinationbegin;

    if(iters.size() == 2)
        return std::copy(iters[0], iters[1], destinationbegin);

    if(iters.size() == 4)
        return std::merge(iters[0], iters[1], iters[2], iters[3], destinationbegin);

    std::size_t nranges = iters.size()/2;

    std::size_t num_elements = 0;
    for(std::size_t i = 0; i < iters.size() / 2; i++){
        num_elements += std::distance(iters[2*i + 0], iters[2*i+1]);
    }

    std::vector<T> tmpbuffer(num_elements);
    std::size_t merged_elements = 0;

    for(std::size_t i = 0; i < nranges; i++){
        auto src_begin = i % 2 == 0 ? tmpbuffer.begin() : destinationbegin;
        auto src_end = src_begin + merged_elements;
        auto dest_begin = i % 2 == 0 ? destinationbegin : tmpbuffer.begin();

        auto dest_end = std::merge(src_begin, src_end,
                                    iters[2*i + 0], iters[2*i+1],
                                    dest_begin);

        merged_elements = std::distance(dest_begin, dest_end);
    }

    auto destinationend = destinationbegin + merged_elements;

    if(nranges % 2 == 0){
        destinationend = std::copy(tmpbuffer.begin(), tmpbuffer.begin() + merged_elements, destinationbegin);
    }

    return destinationend;
}

template<class OutputIt, class Iter>
OutputIt k_way_merge_naive_sortonce(OutputIt destinationbegin, const std::vector<Iter>& iters){
    static_assert(std::is_same<typename std::iterator_traits<OutputIt>::value_type, typename std::iterator_traits<Iter>::value_type>::value, "");

    using T = typename std::iterator_traits<Iter>::value_type;

    // at least one range is invalid
    if(iters.size() % 2 == 1)
        return destinationbegin;

    if(iters.size() == 0)
        return destinationbegin;

    if(iters.size() == 2)
        return std::copy(iters[0], iters[1], destinationbegin);

    if(iters.size() == 4)
        return std::merge(iters[0], iters[1], iters[2], iters[3], destinationbegin);

    std::size_t nranges = iters.size()/2;

    std::size_t num_elements = 0;
    for(std::size_t i = 0; i < iters.size() / 2; i++){
        num_elements += std::distance(iters[2*i + 0], iters[2*i+1]);
    }

    std::vector<int> indices(nranges);
    std::iota(indices.begin(), indices.end(), int(0));

    std::sort(indices.begin(), indices.end(), [&](auto l, auto r){
        auto ldist = std::distance(iters[2*l + 0], iters[2*l+1]);
        auto rdist = std::distance(iters[2*r + 0], iters[2*r+1]);
        return ldist < rdist;
    });

    std::vector<T> tmpbuffer(num_elements);
    std::size_t merged_elements = 0;

    for(std::size_t i = 0; i < nranges; i++){
        const int rangeid = indices[i];

        auto src_begin = i % 2 == 0 ? tmpbuffer.begin() : destinationbegin;
        auto src_end = src_begin + merged_elements;
        auto dest_begin = i % 2 == 0 ? destinationbegin : tmpbuffer.begin();

        auto dest_end = std::merge(src_begin, src_end,
                                    iters[2*rangeid + 0], iters[2*rangeid+1],
                                    dest_begin);

        merged_elements = std::distance(dest_begin, dest_end);
    }

    auto destinationend = destinationbegin + merged_elements;

    if(nranges % 2 == 0){
        destinationend = std::copy(tmpbuffer.begin(), tmpbuffer.begin() + merged_elements, destinationbegin);
    }

    return destinationend;
}

template<class OutputIt, class Iter>
OutputIt k_way_merge_sorted(OutputIt destinationbegin, std::vector<Iter> iters){
    static_assert(std::is_same<typename OutputIt::value_type, typename Iter::value_type>::value, "");

    using T = typename Iter::value_type;

    // at least one range is invalid
    if(iters.size() % 2 == 1)
        return destinationbegin;

    if(iters.size() == 0)
        return destinationbegin;

    if(iters.size() == 2)
        return std::copy(iters[0], iters[1], destinationbegin);

    if(iters.size() == 4)
        return std::merge(iters[0], iters[1], iters[2], iters[3], destinationbegin);

    std::size_t nranges = iters.size()/2;

    std::size_t num_elements = 0;
    for(std::size_t i = 0; i < iters.size() / 2; i++){
        num_elements += std::distance(iters[2*i + 0], iters[2*i+1]);
    }

    std::vector<int> indices(nranges);

    std::vector<std::vector<T>> buffers(nranges-1);
    //for(auto& buffer : buffers)
	//buffer.reserve(num_elements);

    auto destinationend = destinationbegin;

    int pending_merges = nranges-1;

    while(pending_merges > 0){
	//for(int i = 0; i < pending_merges+1; i++)
	//	std::cout << "range " << i << ", " << std::distance(iters[2*i + 0], iters[2*i+1]) << " elements" << std::endl;

    	if(pending_merges > 1){
    		indices.resize(pending_merges+1);
    	    	std::iota(indices.begin(), indices.end(), int(0));

    		std::sort(indices.begin(), indices.end(), [&](auto l, auto r){
    			auto ldist = std::distance(iters[2*l + 0], iters[2*l+1]);
    			auto rdist = std::distance(iters[2*r + 0], iters[2*r+1]);
    			return ldist < rdist;
    		});

    		int lindex = indices[0];
    		int rindex = indices[1];

    		std::size_t ldist = std::distance(iters[2*lindex + 0], iters[2*lindex+1]);
    		std::size_t rdist = std::distance(iters[2*rindex + 0], iters[2*rindex+1]);

    		int bufferindex = nranges-1 - pending_merges;
    		auto& buffer = buffers[bufferindex];
    		buffer.resize(ldist+rdist);

    		std::merge(iters[2*lindex + 0], iters[2*lindex+1],
    		           iters[2*rindex + 0], iters[2*rindex+1],
    		           buffer.begin());

    		iters[2*lindex+0] = buffer.begin();
    		iters[2*lindex+1] = buffer.end();
    		iters.erase(iters.begin() + (2*rindex+0), iters.begin() + (2*rindex+1) + 1);

    	}else{
    		int lindex = 0;
    		int rindex = 1;
    		destinationend = std::merge(iters[2*lindex + 0], iters[2*lindex+1],
    					   iters[2*rindex + 0], iters[2*rindex+1],
    					   destinationbegin);
    	}

    	--pending_merges;
    }

    return destinationend;
}

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
    Essentially performs std::set_intersection(first1, last1, first2, last2, d_first)
    but limits the allowed result size to n.
    If the result would contain more than n elements, d_first is returned, i.e. the result range is empty
*/
template<class InputIt1, class InputIt2, class OutputIt>
OutputIt set_intersection_n_or_empty(InputIt1 first1, InputIt1 last1,
                   InputIt2 first2, InputIt2 last2,
                   std::size_t n,
                   OutputIt d_first){

        const OutputIt d_first_old = d_first;
        ++n;
        while (first1 != last1 && first2 != last2 && n > 0) {
            if (*first1 < *first2) {
                ++first1;
            } else {
                if (!(*first2 < *first1)) {
                    *d_first++ = *first1++;
                    --n;
                }
                ++first2;
            }
        }
        if(n == 0){
           //intersection contains at least n+1 elements, return empty range
           return d_first_old;
        }

        return d_first;
}





/*
    Bit shifts of bit array
*/

void shiftBitsLeftBy(unsigned char* array, int bytes, int shiftamount);

void shiftBitsRightBy(unsigned char* array, int bytes, int shiftamount);

void shiftBitsBy(unsigned char* array, int bytes, int shiftamount);

int hammingdistanceHiLo(const std::uint8_t* l, const std::uint8_t* r, int length_l, int length_r, int bytes);


#endif
