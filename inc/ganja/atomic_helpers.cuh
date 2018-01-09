#ifndef GANJA_ATOMIC_HELPERS_CUH
#define GANJA_ATOMIC_HELPERS_CUH

#include <atomic>

template <
    typename T> inline
void atomic_max(
    std::atomic<T>& target,
    const T& other) {

    auto expectation = target.load(std::memory_order_relaxed);
    while (other > expectation && !std::atomic_compare_exchange_weak(
           &target, &expectation, std::max(other, expectation))) {}
}

#endif
