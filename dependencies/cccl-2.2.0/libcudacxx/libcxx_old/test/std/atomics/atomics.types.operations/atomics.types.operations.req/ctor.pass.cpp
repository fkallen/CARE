//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03

// NOTE: atomic<> of a TriviallyCopyable class is wrongly rejected by older
// clang versions. It was fixed right before the llvm 3.5 release. See PR18097.
// XFAIL: apple-clang-6.0, clang-3.4, clang-3.3

// <atomic>

// constexpr atomic<T>::atomic(T value)

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <atomic>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "atomic_helpers.h"

struct UserType {
    int i;

    UserType() noexcept {}
    constexpr explicit UserType(int d) noexcept : i(d) {}

    friend bool operator==(const UserType& x, const UserType& y) {
        return x.i == y.i;
    }
};

template <class Tp>
struct TestFunc {
    void operator()() const {
        typedef std::atomic<Tp> Atomic;
        static_assert(std::is_literal_type<Atomic>::value, "");
        constexpr Tp t(42);
        {
            constexpr Atomic a(t);
            assert(a == t);
        }
        {
            constexpr Atomic a{t};
            assert(a == t);
        }
        {
            constexpr Atomic a = ATOMIC_VAR_INIT(t);
            assert(a == t);
        }
    }
};


int main(int, char**)
{
    TestFunc<UserType>()();
    TestEachIntegralType<TestFunc>()();

  return 0;
}
