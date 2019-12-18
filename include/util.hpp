#ifndef CARE_UTIL_HPP
#define CARE_UTIL_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <functional>
#include <cmath>
#include <numeric>
#include <vector>
#include <cassert>
#include <sstream>
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <type_traits>
#include <iostream>
#include <queue>


#include <unistd.h>
#include <sys/resource.h>


__inline__
std::size_t getAvailableMemoryInKB_linux(){
    //https://stackoverflow.com/questions/349889/how-do-you-determine-the-amount-of-linux-system-ram-in-c
    std::string token;
    std::ifstream file("/proc/meminfo");
    assert(bool(file));
    while(file >> token) {
        if(token == "MemAvailable:") {
            std::size_t mem;
            if(file >> mem) {
                return mem;
            } else {
                return 0;       
            }
        }
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0;
};

__inline__ 
std::size_t getCurrentRSS_linux(){
        std::ifstream in("/proc/self/statm");
        std::size_t tmp, rss;
        in >> tmp >> rss;
        
        return rss * sysconf(_SC_PAGESIZE);
}

__inline__
std::size_t getRSSLimit_linux(){
    rlimit rlim;
    int ret = getrlimit(RLIMIT_RSS, &rlim);
    if(ret != 0){
        std::perror("Could not get RSS limit!");
        return 0;
    }
    return rlim.rlim_cur;    
}


__inline__
std::size_t getAvailableMemoryInKB(){
    //return getAvailableMemoryInKB_linux();

    return std::min(getAvailableMemoryInKB_linux(), (getRSSLimit_linux() - getCurrentRSS_linux()) / 1024);
};


template<class T>
struct ProgressThread{

    template<class ProgressFunc>
    ProgressThread(T maxProgress_, ProgressFunc&& pfunc)
            : ProgressThread(maxProgress_, std::move(pfunc), [](auto seconds){return seconds;}){

    }

    template<class ProgressFunc, class SleepUpdateFunc>
    ProgressThread(T maxProgress_, ProgressFunc&& pfunc, SleepUpdateFunc&& sfunc)
            : starttime{std::chrono::system_clock::now()},
            sleepPeriod{std::chrono::seconds{1}},
            currentProgress{0},
            maxProgress{maxProgress_},
            showProgress{std::move(pfunc)},
            updateSleepPeriod{std::move(sfunc)},
            thread{[&](){threadFunc();}}{

        showProgress(0,0);
    }

    ~ProgressThread(){
        doCancel = true;
        thread.join();
    }

    ProgressThread(const ProgressThread&) = delete;
    ProgressThread(ProgressThread&&) = delete;
    ProgressThread operator=(const ProgressThread&) = delete;
    ProgressThread operator=(ProgressThread&&) = delete;

    void threadFunc(){
        std::this_thread::sleep_for(sleepPeriod);
        
        while(!doCancel && currentProgress < maxProgress){
            auto now = std::chrono::system_clock::now();
            const std::chrono::duration<double> duration = now - starttime;
            showProgress(currentProgress, duration.count());

            std::this_thread::sleep_for(sleepPeriod);
            sleepPeriod = updateSleepPeriod(sleepPeriod);
        }
    }

    void cancel(){
        doCancel = true;
    }

    void finished(){
        doCancel = true;
        auto now = std::chrono::system_clock::now();
        const std::chrono::duration<double> duration = now - starttime;
        showProgress(currentProgress, duration.count());
    }

    void setProgress(T newProgress){
        assert(newProgress >= currentProgress);
        currentProgress = newProgress;
    }

    void addProgress(T add){
        if(std::is_signed<T>::value){
            assert(add >= 0);
        }
        
        currentProgress += add;
    }

    //std::atomic<bool> doCancel = false;
    bool doCancel = false;
    std::chrono::time_point<std::chrono::system_clock> starttime;
    std::chrono::seconds sleepPeriod{1};
    std::atomic<T> currentProgress;
    std::atomic<T> maxProgress;
    std::function<void(T, double)> showProgress;
    std::function<std::chrono::seconds(std::chrono::seconds)> updateSleepPeriod;
    std::thread thread;
};





template<class T>
class View{
private:
    const T* ptr;
    int nElements;
public:
    View() : View(nullptr, 0){}
    View(const T* p, int n) : ptr(p), nElements(n){}

    const T& operator[](int i) const{
        if(i >= nElements){
            throw std::runtime_error("Out-of-bounds view access!!!");
        }
        return ptr[i];
    }

    const T* begin() const{
        return ptr;
    }

    const T* end() const{
        return ptr + nElements;
    }

    int size() const{
        return int(std::distance(begin(), end()));
    }
};

inline
std::vector<std::string> split(const std::string& str, char c){
	std::vector<std::string> result;

	std::stringstream ss(str);
	std::string s;

	while (std::getline(ss, s, c)) {
		result.emplace_back(s);
	}

	return result;
}



/*
    Performs a set union of multiple ranges into a single output range
*/
template<class T>
struct SetUnionHandle{
    std::vector<T> buffer;
};

template<class T, class OutputIt, class Iter>
OutputIt k_way_set_union(
        SetUnionHandle<T>& handle,
        OutputIt outputbegin, 
        std::vector<std::pair<Iter,Iter>>& ranges){

    using OutputType = typename std::iterator_traits<OutputIt>::value_type;
    using InputType = typename std::iterator_traits<Iter>::value_type;

    static_assert(std::is_same<OutputType, InputType>::value, "");
    static_assert(std::is_same<T, InputType>::value, "");

    //using T = InputType;

    //handle simple cases

    if(ranges.empty()){
        return outputbegin;
    }

    if(ranges.size() == 1){
        return std::copy(ranges[0].first, ranges[0].second, outputbegin);
    }

    if(ranges.size() == 2){
        return std::set_union(ranges[0].first,
                              ranges[0].second,
                              ranges[1].first,
                              ranges[1].second,
                              outputbegin);
    }

    //handle generic case

    //sort ranges by size
    std::sort(ranges.begin(), ranges.end(), [](const auto& l, const auto& r){
        return std::distance(l.first, l.second) < std::distance(r.first, r.second);
    });

    int totalElements = 0;
    for(const auto& range : ranges){
        totalElements += std::distance(range.first, range.second);
    }

    auto& temp = handle.buffer;
    temp.resize(totalElements);

    auto tempbegin = temp.begin();
    auto tempend = tempbegin;
    auto outputend = outputbegin;

    //to avoid a final copy from temp to outputrange, both ranges are swapped in the beginning if number of ranges is odd.
    if(ranges.size() % 2 == 1){
        std::swap(tempbegin, outputbegin);
        std::swap(tempend, outputend);
    }

    for(int k = 0; k < int(ranges.size()); k++){
        tempend = std::set_union(ranges[k].first,
                                  ranges[k].second,
                                  outputbegin,
                                  outputend,
                                  tempbegin);

        std::swap(tempbegin, outputbegin);
        std::swap(tempend, outputend);
    }

    return outputend;
}

template<class T, class OutputIt, class Iter>
OutputIt k_way_set_union(
        OutputIt outputbegin, 
        std::vector<std::pair<Iter,Iter>>& ranges){

    SetUnionHandle<T> handle;

    return k_way_set_union(
        handle,
        outputbegin, 
        ranges
    );
}



template<class OutputIt, class Iter>
OutputIt k_way_set_union_with_priorityqueue(OutputIt outputbegin, std::vector<std::pair<Iter,Iter>>& ranges){
    using OutputType = typename std::iterator_traits<OutputIt>::value_type;
    using InputType = typename std::iterator_traits<Iter>::value_type;

    static_assert(std::is_same<OutputType, InputType>::value, "");

    //handle simple cases

    if(ranges.empty()){
        return outputbegin;
    }

    if(ranges.size() == 1){
        return std::copy(ranges[0].first, ranges[0].second, outputbegin);
    }

    if(ranges.size() == 2){
        return std::set_union(ranges[0].first,
                              ranges[0].second,
                              ranges[1].first,
                              ranges[1].second,
                              outputbegin);
    }

    //handle generic case

    struct PQval{
        int rangeIndex;
        Iter dataIter;

        bool operator<(const PQval& rhs) const{
            return *dataIter > *(rhs.dataIter); //order such that smallest element comes first in priority queue
        }
    };


    std::priority_queue<PQval> pq;

    for(int i = 0; i < int(ranges.size()); i++){
        const auto& range = ranges[i];
        if(std::distance(range.first, range.second) > 0){
            pq.emplace(PQval{i, range.first});
        }
    }

    //position of the previously added output element
    auto currentOutputIter = outputbegin;
    //points behind the last element in output range
    auto outputEnd = outputbegin;

    while(!pq.empty()){
        auto cur = pq.top();
        pq.pop();

        if(currentOutputIter != outputEnd){
            if(*currentOutputIter < *(cur.dataIter)){
                ++currentOutputIter;
                *currentOutputIter = *(cur.dataIter);
                ++outputEnd;
            }
        }else{
            *currentOutputIter = *(cur.dataIter); //the first output element can always be inserted
            ++outputEnd;
        }

         //if there is another element from the same range, add it to the priority queue
        ++cur.dataIter;
        if(cur.dataIter != ranges[cur.rangeIndex].second){
            pq.emplace(cur);
        }        
    }

    return outputEnd;
}



template<class OutputIt, class Iter>
OutputIt k_way_set_union_complicatedsort(OutputIt outputbegin, std::vector<std::pair<Iter,Iter>>& ranges){
    using OutputType = typename std::iterator_traits<OutputIt>::value_type;
    using InputType = typename std::iterator_traits<Iter>::value_type;

    static_assert(std::is_same<OutputType, InputType>::value, "");

    using T = InputType;

    //handle simple cases

    if(ranges.empty()){
        return outputbegin;
    }

    if(ranges.size() == 1){
        return std::copy(ranges[0].first, ranges[0].second, outputbegin);
    }

    if(ranges.size() == 2){
        return std::set_union(ranges[0].first,
                              ranges[0].second,
                              ranges[1].first,
                              ranges[1].second,
                              outputbegin);
    }

    //handle generic case
    auto sortAscending = [](const auto& l, const auto& r){
        return std::distance(l.first, l.second) > std::distance(r.first, r.second);
    };

    //sort ranges by size
    std::sort(ranges.begin(), ranges.end(), sortAscending);

    int totalElements = 0;
    for(const auto& range : ranges){
        totalElements += std::distance(range.first, range.second);
    }

    std::vector<std::vector<T>> tempunions;
    tempunions.resize(ranges.size() - 2);

    int iteration = 0;

    while(ranges.size() > 2){
        auto& range2 = ranges[ranges.size()-1];
        auto& range1 = ranges[ranges.size()-2];

        auto& result = tempunions[iteration];

        result.resize(std::distance(range1.first, range1.second) + std::distance(range2.first, range2.second));

        auto resEnd = std::set_union(range1.first,
                                  range1.second,
                                  range2.first,
                                  range2.second,
                                  result.data());

        auto newRange = std::make_pair(result.data(), resEnd);
        ranges.pop_back();
        ranges.pop_back();
        auto insertpos = std::lower_bound(ranges.begin(), ranges.end(), newRange, sortAscending);
        ranges.insert(insertpos, newRange);

        iteration++;
    }

    auto& range1 = ranges[0];
    auto& range2 = ranges[1];

    auto outputend = std::set_union(range1.first,
                              range1.second,
                              range2.first,
                              range2.second,
                              outputbegin);

    return outputend;
}






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


template<class F>
void print_multiple_sequence_alignment_sorted_by_shift(std::ostream& out, const char* data, int nrows, int ncolumns, std::size_t rowpitch, F get_shift_of_row){
    std::vector<int> indices(nrows);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
            [&](int l, int r){return get_shift_of_row(l) < get_shift_of_row(r);});
    //for(auto i : indices)
    //    out << get_shift_of_row(i) << ' ';
    //out << '\n';
    //assert(std::is_sorted(indices.begin(), indices.end(), [&](int l, int r){return get_shift_of_row(l) < get_shift_of_row(r);}));
    for(int row = 0; row < nrows; row++) {
        int sortedrow = indices[row];
        if(sortedrow == 0)
            out << ">> ";
        else
            out << "   ";
        for(int col = 0; col < ncolumns; col++) {
            const char c = data[sortedrow * rowpitch + col];
            out << (c == '\0' ? '0' : c);
        }
        if(sortedrow == 0)
            out << " <<";
        else
            out << "   ";
        out << '\n';
    }
}

template<class F>
void print_multiple_sequence_alignment_consensusdiff_sorted_by_shift(std::ostream& out, const char* data, const char* consensus,
                                                                        int nrows, int ncolumns, std::size_t rowpitch, F get_shift_of_row){
    std::vector<int> indices(nrows);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
            [&](int l, int r){return get_shift_of_row(l) < get_shift_of_row(r);});
    //for(auto i : indices)
    //    out << get_shift_of_row(i) << ' ';
    //out << '\n';
    //assert(std::is_sorted(indices.begin(), indices.end(), [&](int l, int r){return get_shift_of_row(l) < get_shift_of_row(r);}));
    for(int row = 0; row < nrows; row++) {
        int sortedrow = indices[row];
        if(sortedrow == 0)
            out << ">> ";
        else
            out << "   ";
        for(int col = 0; col < ncolumns; col++) {
            const char c = data[sortedrow * rowpitch + col];
            const char c2 = c == consensus[col] ? '=' : c;
            out << (c2 == '\0' ? '0' : c2);
        }
        if(sortedrow == 0)
            out << " <<";
        else
            out << "   ";
        out << '\n';
    }
}

std::array<int, 5> onehotbase(char base);
void print_multiple_sequence_alignment(std::ostream& out, const char* data, int nrows, int ncolumns, std::size_t rowpitch);

/*
    Bit shifts of bit array
*/

void shiftBitsLeftBy(unsigned char* array, int bytes, int shiftamount);

void shiftBitsRightBy(unsigned char* array, int bytes, int shiftamount);

void shiftBitsBy(unsigned char* array, int bytes, int shiftamount);

int hammingdistanceHiLo(const std::uint8_t* l, const std::uint8_t* r, int length_l, int length_r, int bytes);


#endif
