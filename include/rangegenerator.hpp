#ifndef CARE_CPU_RANGE_GENERATOR_HPP
#define CARE_CPU_RANGE_GENERATOR_HPP

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

}
}



#endif
