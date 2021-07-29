#ifndef CARE_CPU_RANGE_GENERATOR_HPP
#define CARE_CPU_RANGE_GENERATOR_HPP

#include <config.hpp>

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <numeric>

namespace care{
namespace cpu{

    template<class Count_t>
    struct RangeGenerator{
    private:
        Count_t begin;
        Count_t end;
        Count_t current;
        bool isEmpty;
        std::mutex mutex;

    public:
        RangeGenerator(Count_t end) : RangeGenerator(Count_t{0}, end, Count_t{0}) {}

        RangeGenerator(Count_t begin, Count_t end, Count_t initial_value)
            : begin(begin), end(end), current(initial_value), isEmpty(initial_value >= end) {}

        bool empty(){
            std::lock_guard<std::mutex> lm(mutex);
            return isEmpty;
        }

        void skip(Count_t n){
            std::lock_guard<std::mutex> lm(mutex);
            Count_t remaining = end - current;
            Count_t resultsize = std::min(remaining, n);
            current += resultsize;

            if(current == end){
                isEmpty = true;
            }
        }

        std::vector<Count_t> next_n(Count_t n){
            std::lock_guard<std::mutex> lm(mutex);
            if(isEmpty)
                return {};

            Count_t remaining = end - current;

            Count_t resultsize = std::min(remaining, n);

            std::vector<Count_t> result(resultsize);
            std::iota(result.begin(), result.end(), current);

            current += resultsize;

            if(current == end)
                isEmpty = true;

            return result;
        }

        //buffer must point to memory location of at least n elements
        //returns past the end iterator
        template<class Iter>
        Iter next_n_into_buffer(Count_t n, Iter buffer){
            std::lock_guard<std::mutex> lm(mutex);
            if(isEmpty)
                return buffer;

            Count_t remaining = end - current;

            Count_t resultsize = std::min(remaining, n);

            std::iota(buffer, buffer + resultsize, current);

            current += resultsize;

            if(current == end)
                isEmpty = true;

            return buffer + resultsize;
        }

        Count_t getBegin() const{
            return begin;
        }

        Count_t getEnd() const{
            return end;
        }

        Count_t getCurrentUnsafe() const{
            return current;
        }
    };

    template<class Count_t>
    struct RangeGeneratorWrapper{
    private:

        const Count_t* begin;
        const Count_t* end;
        const Count_t* current;
        bool isEmpty;
        std::mutex mutex;

    public:
        RangeGeneratorWrapper(const Count_t* begin_, const Count_t* end_) : begin(begin_), end(end_), current(begin_), isEmpty(begin_ >= end_){}

        void reset(const Count_t* begin_, const Count_t* end_){
            std::lock_guard<std::mutex> lm(mutex);
            begin = begin_;
            end = end_;
            current = begin_;
            isEmpty = (begin >= end);
        }

        bool empty(){
            std::lock_guard<std::mutex> lm(mutex);
            return isEmpty;
        }

        void skip(std::size_t n){
            std::lock_guard<std::mutex> lm(mutex);
            const std::size_t remaining = std::distance(current, end);
            const std::size_t resultsize = std::min(remaining, n);
            current += resultsize;

            if(current == end){
                isEmpty = true;
            }
        }

        std::vector<Count_t> next_n(std::size_t n){
            std::lock_guard<std::mutex> lm(mutex);
            if(isEmpty)
                return {};

            const std::size_t remaining = std::distance(current, end);
            const std::size_t resultsize = std::min(remaining, n);

            std::vector<Count_t> result(current, current + resultsize);
            current += resultsize;

            if(current == end)
                isEmpty = true;

            return result;
        }

        //buffer must point to memory location of at least n elements
        //returns past the end iterator
        template<class Iter>
        Iter next_n_into_buffer(std::size_t n, Iter buffer){
            std::lock_guard<std::mutex> lm(mutex);
            if(isEmpty)
                return buffer;

            const std::size_t remaining = std::distance(current, end);
            const std::size_t resultsize = std::min(remaining, n);

            std::copy(current, current + resultsize, buffer);

            current += resultsize;

            if(current == end)
                isEmpty = true;

            return buffer + resultsize;
        }
    };

}
}



#endif
